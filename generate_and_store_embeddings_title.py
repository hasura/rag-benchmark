import os
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

# Configuration
DB_NAME = "frames_new"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"
BATCH_SIZE = 100  # Process more titles at once since they're shorter

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

def get_embeddings(titles, client):
    """Get embeddings for a batch of titles."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=titles
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return None

def main():
    client = OpenAI()
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Get total count for progress bar
        cursor.execute("SELECT COUNT(*) FROM wikipedia_content WHERE title_vector IS NULL")
        total_records = cursor.fetchone()[0] # type: ignore
        print(f"Found {total_records} titles to process")

        # Process titles in batches
        with tqdm(total=total_records) as pbar:
            while True:
                # Fetch batch of unprocessed titles
                cursor.execute("""
                    SELECT id, title 
                    FROM wikipedia_content 
                    WHERE title_vector IS NULL
                    ORDER BY id 
                    LIMIT %s
                """, (BATCH_SIZE,))
                
                records = cursor.fetchall()
                if not records:
                    break

                article_ids, titles = zip(*records)
                
                # Get embeddings for the batch
                embeddings = get_embeddings(titles, client)
                if embeddings:
                    # Prepare data for batch update
                    update_data = [
                        (embedding, article_id)
                        for article_id, embedding in zip(article_ids, embeddings)
                    ]
                    
                    # Batch update
                    execute_batch(
                        cursor,
                        """
                        UPDATE wikipedia_content 
                        SET title_vector = %s
                        WHERE id = %s
                        """,
                        update_data
                    )
                    conn.commit()
                
                pbar.update(len(records))

    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()