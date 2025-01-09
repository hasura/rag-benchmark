CREATE EXTENSION vector;

CREATE TABLE IF NOT EXISTS wikipedia_content_vectors (             
    chunk_id SERIAL PRIMARY KEY,
    article_id integer REFERENCES wikipedia_content(id),             
    content_chunk_vector vector(1536),
    content_chunk text
);

ALTER TABLE wikipedia_content 
ADD COLUMN title_vector vector(1536);
