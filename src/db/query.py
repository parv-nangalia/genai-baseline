
from psycopg2.extras import RealDictCursor
from ..utility.helper import get_hf_embedding_textual, get_openai_embedding

from psycopg2.extras import RealDictCursor

def search_similar_chunks(
    conn,
    question,
    model_name,
    top_k=5
):

    if model_name == "openai":
        embedding_vector = get_openai_embedding(question)
    elif model_name == "hugging-face":
        embedding_vector = get_hf_embedding_textual(question)
    model_name = "hf"
    dimension = len(embedding_vector)

    vector_str = "[" + ",".join(
        map(str, embedding_vector)
    ) + "]"

    query = """
    SELECT
        c.id,
        c.doc_id,
        c.content,
        c.metadata AS chunk_metadata,

        e.id AS embedding_id,
        e.model,
        e.dimension,
        e.embedding,
        e.metadata AS embedding_metadata

    FROM chunks c

    JOIN embeddings e
    ON c.id = e.chunk_id

    WHERE
        e.model = %s
        AND e.dimension = %s

    ORDER BY e.embedding <#> %s::vector

    LIMIT %s
    """

    with conn.cursor(
        cursor_factory=RealDictCursor
    ) as cur:

        cur.execute(
            query,
            (
                model_name,
                dimension,
                vector_str,
                top_k
            )
        )

        chunks = cur.fetchall()
    return chunks