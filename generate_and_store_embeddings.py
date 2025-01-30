import os
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_batch
from time import sleep
from tqdm import tqdm
import tiktoken

# Configuration
DB_NAME = "frames_new"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"
BATCH_SIZE = 20  # Number of chunks to process at once for embeddings
MAX_TOKENS = 7600  # Maximum tokens per chunk 

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def chunk_text(text, encoding, max_tokens=7600):
    """Split text into chunks of approximately max_tokens tokens, ending at spaces or newlines."""
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into words and preserve newlines
    words = []
    for line in text.split('\n'):
        words.extend(line.split(' '))
        words.append('\n')
    
    for word in words:
        word_tokens = len(encoding.encode(word))
        
        if current_length + word_tokens > max_tokens:
            # Join the current chunk and add it to chunks
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:  # Only add non-empty chunks
                chunks.append(chunk_text)
            current_chunk = [word]
            current_length = word_tokens
        else:
            current_chunk.append(word)
            current_length += word_tokens
    
    # Add the last chunk if it exists
    final_chunk = ' '.join(current_chunk).strip()
    if final_chunk:
        chunks.append(final_chunk)
    
    return chunks

def get_embeddings(texts, client, encoding):
    """Get embeddings for a batch of texts with automatic token limit handling."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        if "maximum context length" in str(e):
            # If we get a token limit error, process each text individually with smaller chunks
            all_embeddings = []
            for text in texts:
                # Try with half the current MAX_TOKENS
                new_chunks = chunk_text(text, encoding, max_tokens=MAX_TOKENS // 2)
                # Process each smaller chunk
                for chunk in new_chunks:
                    try:
                        response = client.embeddings.create(
                            model="text-embedding-3-small",
                            input=[chunk]
                        )
                        all_embeddings.extend(embedding.embedding for embedding in response.data)
                    except Exception as nested_e:
                        print(f"Error processing smaller chunk: {nested_e}")
                        # If even smaller chunks fail, you might want to skip or handle differently
                        continue
            return all_embeddings if all_embeddings else None
        else:
            print(f"Error getting embeddings: {e}")
            return None

def process_article(article_id, content, client, cursor, conn, encoding):
    """Process a single article: chunk it and generate embeddings."""
    try:
        # Split content into chunks
        chunks = chunk_text(content, encoding, max_tokens=MAX_TOKENS)
        chunks_processed = 0
        
        # Process chunks in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            try:
                batch_chunks = chunks[i:i + BATCH_SIZE]
                embeddings = get_embeddings(batch_chunks, client, encoding)
                
                if embeddings:
                    # Prepare data for batch insert
                    insert_data = []
                    for chunk, embedding in zip(batch_chunks, embeddings):
                        insert_data.append((article_id, chunk, embedding))
                    
                    # Batch insert
                    execute_batch(
                        cursor,
                        """
                        INSERT INTO wikipedia_content_vectors 
                        (article_id, content_chunk, content_chunk_vector)
                        VALUES (%s, %s, %s)
                        """,
                        insert_data
                    )
                    
                    # Commit after each batch
                    conn.commit()
                    chunks_processed += len(batch_chunks)
                    print(f"Article {article_id}: Processed {chunks_processed}/{len(chunks)} chunks")
                    
                sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing batch in article {article_id} at chunk {i}: {e}")
                conn.rollback()
                continue  # Continue with next batch even if current fails
                
    except Exception as e:
        print(f"Error processing article {article_id}: {e}")
        raise

def main():
    client = OpenAI()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Initialize tokenizer
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")

    try:
        # Get total count for progress bar
        cursor.execute("SELECT COUNT(*) FROM wikipedia_content")
        total_records = cursor.fetchone()[0] # type: ignore

        # Find the last processed article_id to resume from there
        cursor.execute("""
            SELECT COALESCE(MAX(article_id), 0)
            FROM wikipedia_content_vectors
        """)
        last_processed_id = cursor.fetchone()[0] # type: ignore
        print(f"Resuming from article_id > {last_processed_id}")

        # Process articles
        with tqdm(total=total_records) as pbar:
            last_id = last_processed_id
            batch_size = 5  # Number of articles to fetch at once
            
            while True:
                # Fetch batch of articles
                cursor.execute(
                    """
                    SELECT id, content 
                    FROM wikipedia_content 
                    WHERE id > %s
                    ORDER BY id 
                    LIMIT %s
                    """,
                    (last_id, batch_size)
                )
                articles = cursor.fetchall()
                
                if not articles:
                    break

                # Process each article
                for article_id, content in articles:
                    try:
                        process_article(article_id, content, client, cursor, conn, encoding)
                        pbar.update(1)
                        last_id = article_id  # Update last_id to the most recently processed article
                    except Exception as e:
                        print(f"Failed to process article {article_id}, continuing with next article. Error: {e}")
                        last_id = article_id  # Even if it fails, update last_id to avoid getting stuck

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()