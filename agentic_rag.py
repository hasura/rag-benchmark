import os
import psycopg2
from openai import OpenAI
from anthropic import Anthropic, BadRequestError
from anthropic.types import MessageParam, ToolParam
from typing import List, Dict, Any, Literal
import logging
from pydantic import BaseModel
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define tool schemas using Pydantic
class SimilaritySearchInput(BaseModel):
    query: str
    top_k: int = 5

class SimilaritySearchResult(BaseModel):
    articleId: int
    chunkContent: str
    similarity: float

class Tool(BaseModel):
    type: Literal["function"]
    function: dict

class SemanticSearchTool:
    def __init__(self):
        # Initialize database connection
        connection_string = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@local.hasura.dev:5433/frames_new')
        self.db_conn = psycopg2.connect(connection_string)
        
        # Initialize OpenAI client for embeddings
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def format_embedding_for_postgres(self, embedding: List[float]) -> str:
        """Format the embedding array as a PostgreSQL vector literal"""
        return f"[{','.join(map(str, embedding))}]"

    def search_wikipedia(self, query: str, top_k: int = 5) -> List[SimilaritySearchResult]:
        """Tool function to search for similar content using semantic similarity"""
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
        article_id,
        content_chunk,
        (-1.0) * (content_chunk_vector <#> %s::vector) as similarity
      FROM 
        wikipedia_content_vectors
      ORDER BY 
        similarity DESC
      LIMIT %s;
            """
            
            with self.db_conn.cursor() as cursor:
                cursor.execute(search_query, (formatted_embedding, top_k))
                results = cursor.fetchall()
            
            return [
                SimilaritySearchResult(
                    articleId=row[0],
                    chunkContent=row[1],
                    similarity=float(row[2])
                )
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise

    @property
    def tool_schema(self) -> ToolParam:
        """Return the tool schema for similarity search"""
        return {
            "name": "search_wikipedia",
            "description": "Search for semantically similar content in the wikipedia knowledge base to help answer questions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant content"
                    },
                },
                "required": ["query"]
            }
        }

class RAGAssistant:
    def __init__(self):
        self.search_tool = SemanticSearchTool()
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def format_context(self, similar_content: List[SimilaritySearchResult]) -> str:
        """Format similar content as context for Claude"""
        return "\n\n".join([
            f"Related content {i+1} (similarity: {item.similarity:.2f}):\n{item.chunkContent}"
            for i, item in enumerate(similar_content)
        ])

    async def process_query(self, query: str) -> str:
        """Process a query using semantic search tool and Claude"""
        messages: List[MessageParam] = [{"role": "user", "content": query}]
        tool_use_count = 0
        MAX_TOOL_USES = 5
        try:
                
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                tools=[self.search_tool.tool_schema],
                messages=messages
            )

            while response.stop_reason == "tool_use" and tool_use_count < MAX_TOOL_USES:
                # Get the tool use block
                tool_use = next(block for block in response.content if block.type == "tool_use")
                tool_input = tool_use.input

                print(f"Tool Input:")
                print(json.dumps(tool_input, indent=2))

                # Execute the search
                similar_content = self.search_tool.search_wikipedia(
                    query=tool_use.input.get("query", query), # type: ignore
                    top_k=tool_use.input.get("top_k", 5) # type: ignore
                )

                # Format context as tool result
                context_text = self.format_context(similar_content)

                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                     "role": "user",
                     "content": [
                         {
                             "type": "tool_result",
                             "tool_use_id": tool_use.id,
                             "content": context_text,
                         }
                     ],
                 })


                tool_use_count += 1

                # Always include tools in the API call, but add a system message if we've hit the limit
                if tool_use_count >= MAX_TOOL_USES:
                     messages.append({
                         "role": "user",
                         "content": "You have reached the maximum number of tool uses. Please provide a final response based on the information you have gathered so far."
                     })

                try: 
                    response = self.anthropic_client.messages.create(
                     model="claude-3-5-sonnet-20241022",
                     max_tokens=4096,
                     tools=[self.search_tool.tool_schema],
                     messages=messages
                    )
                except BadRequestError as e:
                    if "prompt is too long" in str(e):
                        logger.error(f"Token limit exceeded: {str(e)}")
                        return str(e)
                    raise
                except Exception as e:
                    logger.error(f"Error during tool use: {str(e)}")
                    return str(e)

            return response.content[0].text # type: ignore
        
        except BadRequestError as e:
            if "prompt is too long" in str(e):
                logger.error(f"Token limit exceeded: {str(e)}")
                return str(e)
            raise
        except Exception as e:
            logger.error(f"Error during tool use: {str(e)}")
            return str(e)
            
    def close(self):
        """Close tool connections"""
        self.search_tool.db_conn.close()

async def main():
    # Example usage
    assistant = RAGAssistant()
    
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