import os
import psycopg2
from openai import OpenAI
from anthropic import Anthropic
from typing import List, Dict, Any
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimilaritySearchResult(BaseModel):
    id: int
    title: str
    content: str
    similarity: float


class SemanticSearchClaude:
    def __init__(self):
        # Initialize database connection
        connection_string = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@local.hasura.dev:5433/frames_new')
        self.db_conn = psycopg2.connect(connection_string)
        
        # Initialize API clients
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def format_embedding_for_postgres(self, embedding: List[float]) -> str:
        """Format the embedding array as a PostgreSQL vector literal"""
        return f"[{','.join(map(str, embedding))}]"

    async def search_wikipedia(self, query: str, top_k: int = 5) -> List[SimilaritySearchResult]:
        """Search for similar content using semantic similarity"""
        try:
            # Get embedding for the query
            embedding_response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            
            query_embedding = embedding_response.data[0].embedding
            formatted_embedding = self.format_embedding_for_postgres(query_embedding)
            
            # Perform similarity search
            search_query = """
      SELECT 
        id,
        title,
        content,
        (-1.0) * (title_vector <#> %s::vector) as similarity
      FROM 
        wikipedia_content
      ORDER BY 
        similarity DESC
      LIMIT %s;
            """
            
            with self.db_conn.cursor() as cursor:
                cursor.execute(search_query, (formatted_embedding, top_k))
                results = cursor.fetchall()
            
            return [
                SimilaritySearchResult(
                    id=row[0],
                    title=row[1],
                    content=row[2],
                    similarity=float(row[3]),
                )
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    async def get_claude_response(self, query: str, similar_content: List[SimilaritySearchResult]) -> str:
        """Format similar content as context for Claude"""
        context_text = "\n\n".join([
            f"Related content {i+1} (similarity: {item.similarity:.2f}):\n{item.content}"
            for i, item in enumerate(similar_content)
        ])
        
        # Construct the prompt with context
        prompt = f"""Here is some relevant context that might help answer the query:

{context_text}

Based on the above context and your knowledge, please answer this query:
{query}"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text # type: ignore
            
        except Exception as e:
            logger.error(f"Error getting Claude response: {str(e)}")
            raise

    async def process_query(self, query: str, top_k: int = 5) -> str:
        """Process a query using semantic search and Claude"""
        # Get similar content
        similar_content = await self.search_wikipedia(query, top_k)
        
        # Get Claude's response with context
        response = await self.get_claude_response(query, similar_content)
        
        return response

    def close(self):
        """Close database connection"""
        self.db_conn.close()

async def main():
    # Example usage
    assistant = SemanticSearchClaude()
    
    try:
        while True:
            query = input("\nEnter your question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            response = await assistant.process_query(query)
            print("\nClaude's response:")
            print(response)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        assistant.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())