CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    embedding VECTOR,
    metadata JSONB,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_id, model, dimension)
);

CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
CREATE INDEX IF NOT EXISTS idx_embeddings_dimension ON embeddings(dimension);
CREATE INDEX IF NOT EXISTS idx_embeddings_embedding ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
