// npm install pg openai @types/pg
import { Pool } from 'pg';
import OpenAI from 'openai';

// Initialize global clients
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:postgres@local.hasura.dev:5433/frames_new'
});


const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

interface SearchResult {
  articleId: number;
  chunkContent: string;
  similarity: number;
}

function formatEmbeddingForPostgres(embedding: number[]): string {
  // Format the array as a PostgreSQL vector literal: [1,2,3]
  return `[${embedding.join(',')}]`;
}

/**
 * @readonly Exposes the function as an NDC function (the function should only query data without making modifications)
 */
export async function searchSimilarContent(
  query: string,
  topK: number | undefined = 5
): Promise<SearchResult[]> {
  try {
    // Get embedding for the query
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: query
    });
    
    const queryEmbedding = embeddingResponse.data[0].embedding;
    const formattedEmbedding = formatEmbeddingForPostgres(queryEmbedding);

    console.log(formattedEmbedding)

    // Perform similarity search using PostgreSQL vector extension
    const searchQuery = `
      SELECT 
        article_id,
        content_chunk,
        (-1.0) * (content_chunk_vector <#> $1::vector) as similarity
      FROM 
        wikipedia_content_vectors
      ORDER BY 
        similarity DESC
      LIMIT $2;
    `;

    const result = await pool.query(searchQuery, [formattedEmbedding, topK]);

    return result.rows.map(row => ({
      articleId: row.article_id,
      chunkContent: row.content_chunk,
      similarity: parseFloat(row.similarity)
    }));

  } catch (error) {
    console.error('Error in similarity search:', error);
    throw error;
  }
}

interface WikiArticleSearchResult {
  id: number;
  title: string;
  content: string;
  similarity: number;
}

/**
 * @readonly Exposes the function as an NDC function (the function should only query data without making modifications)
 */
export async function getWikipediaArticles(
  query: string,
  topK: number | undefined = 3
): Promise<WikiArticleSearchResult[]> {
  try {
    // Get embedding for the query
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: query
    });
    
    const queryEmbedding = embeddingResponse.data[0].embedding;
    const formattedEmbedding = formatEmbeddingForPostgres(queryEmbedding);

    console.log(formattedEmbedding)

    // Perform similarity search using PostgreSQL vector extension
    const searchQuery = `
      SELECT 
        id,
        title,
        content,
        (-1.0) * (title_vector <#> $1::vector) as similarity
      FROM 
        wikipedia_content
      ORDER BY 
        similarity DESC
      LIMIT $2;
    `;

    const result = await pool.query(searchQuery, [formattedEmbedding, topK]);

    return result.rows.map(row => ({
      id: row.id,
      title: row.title,
      content: row.content,
      similarity: parseFloat(row.similarity)
    }));

  } catch (error) {
    console.error('Error in similarity search:', error);
    throw error;
  }
}

// Example usage:
// const results = await searchSimilarContent("What is the theory of relativity?", 5);