"""
=========================================================
DATABASE INGESTION LAYER
=========================================================

This version is broken down by concerns.

Responsibilities:

1. prepare_chunk_rows()
2. prepare_embedding_rows()
3. insert_chunks()
4. insert_embeddings()
5. build_bm25_index()
6. ingest_processed_chunks()

Works with BOTH:
- PDF ingestion
- URL ingestion

as long as they return:

[
    {
        "chunk_id": "...",
        "doc_id": "...",
        "text": "...",
        "metadata": {...},
        "embeddings": {...}
    }
]

=========================================================
"""

from psycopg2.extras import execute_values, Json


# =========================================================
# PREPARE CHUNK ROWS
# =========================================================

def prepare_chunk_rows(processed_chunks):

    """
    Converts processed chunks into DB-ready chunk rows
    """

    chunk_rows = []

    for chunk in processed_chunks:

        chunk_rows.append(
            (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["text"],
                chunk["metadata"]
            )
        )

    return chunk_rows


# =========================================================
# PREPARE EMBEDDING ROWS
# =========================================================

def prepare_embedding_rows(processed_chunks):

    """
    Converts processed chunks into DB-ready embedding rows
    """

    embedding_rows = []

    for chunk in processed_chunks:

        chunk_id = chunk["chunk_id"]

        embeddings = chunk["embeddings"]

        # -------------------------------------------------
        # OPENAI
        # -------------------------------------------------

        if "openai" in embeddings:

            embedding_rows.append(
                (
                    chunk_id,
                    "openai",
                    embeddings["openai"]["vector"],
                    {
                        "embedding_model":
                            embeddings["openai"]["model"]
                    },
                    1536
                )
            )

        # -------------------------------------------------
        # HF
        # -------------------------------------------------

        if "hf" in embeddings:

            embedding_rows.append(
                (
                    chunk_id,
                    "hf",
                    embeddings["hf"]["vector"],
                    {
                        "embedding_model":
                            embeddings["hf"]["model"]
                    },
                    384
                )
            )

    return embedding_rows


# =========================================================
# INSERT CHUNKS
# =========================================================

# def insert_chunks(
#     conn,
#     chunk_rows
# ):

#     """
#     Inserts canonical chunk text
#     """

#     if not chunk_rows:
#         return

#     with conn.cursor() as cur:

#         execute_values(
#             cur,
#             """
#             INSERT INTO chunks
#             (
#                 id,
#                 doc_id,
#                 content,
#                 metadata
#             )
#             VALUES %s
#             """,
#             chunk_rows
#         )

#     conn.commit()


def insert_chunks(conn, chunk_rows):

    if not chunk_rows:
        return

    rows = [
        (
            chunk_id,
            doc_id,
            content,
            Json(metadata)
        )
        for chunk_id, doc_id, content, metadata in chunk_rows
    ]

    with conn.cursor() as cur:

        execute_values(
            cur,
            """
            INSERT INTO chunks
            (
                id,
                doc_id,
                content,
                metadata
            )
            VALUES %s
            ON CONFLICT (id) DO NOTHING
            """,
            rows
        )

    conn.commit()

# =========================================================
# INSERT EMBEDDINGS
# =========================================================

# def insert_embeddings(
#     conn,
#     embedding_rows
# ):

#     """
#     Inserts vector embeddings
#     """

#     if not embedding_rows:
#         return

#     with conn.cursor() as cur:

#         execute_values(
#             cur,
#             """
#             INSERT INTO embeddings
#             (
#                 chunk_id,
#                 model,
#                 embedding,
#                 metadata
#             )
#             VALUES %s
#             """,
#             embedding_rows
#         )

#     conn.commit()

def insert_embeddings(conn, embedding_rows):

    if not embedding_rows:
        return

    rows = [
        (
            chunk_id,
            model,
            embedding,
            Json(metadata),
            dimension
        )
        for chunk_id, model, embedding, metadata, dimension in embedding_rows
    ]

    try:
        with conn.cursor() as cur:

            execute_values(
                cur,
                """
                INSERT INTO embeddings
                (
                    chunk_id,
                    model,
                    embedding,
                    metadata,
                    dimension
                )
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                rows
            )

        conn.commit()
    except Exception as e:
        print("ERROR TYPE:", type(e))
        print("ERROR MESSAGE:", e)
        raise


# =========================================================
# MAIN INGESTION ORCHESTRATOR
# =========================================================

def ingest_processed_chunks(
    conn,
    processed_chunks
):

    """
    Full ingestion pipeline.

    INPUT:
    processed_chunks

    OUTPUT:
    inserts into DB
    """

    # -----------------------------------------------------
    # PREPARE ROWS
    # -----------------------------------------------------

    chunk_rows = prepare_chunk_rows(
        processed_chunks
    )

    embedding_rows = prepare_embedding_rows(
        processed_chunks
    )

    # -----------------------------------------------------
    # INSERT CHUNKS
    # -----------------------------------------------------

    insert_chunks(
        conn,
        chunk_rows
    )

    # -----------------------------------------------------
    # INSERT EMBEDDINGS
    # -----------------------------------------------------

    insert_embeddings(
        conn,
        embedding_rows
    )
    
    return {
        "status": "success",
        "chunks_inserted": len(chunk_rows),
        "embeddings_inserted": len(embedding_rows)
    }