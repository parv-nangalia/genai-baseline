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

from sqlalchemy.orm import Session
from ..db.models import Chunk, Embedding
from typing import List
from ..utility.logging_config import log_function_call


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
                chunk["meta_data"]
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


def insert_chunks(session: Session, chunk_rows: List[tuple]):

    if not chunk_rows:
        return

    for chunk_id, doc_id, content, meta_data in chunk_rows:
        # use merge to insert if not exist, otherwise ignore
        obj = Chunk(
            id=chunk_id,
            doc_id=doc_id,
            content=content,
            meta_data=meta_data,
        )
        session.merge(obj)

    session.commit()


def insert_embeddings(session: Session, embedding_rows: List[tuple]):

    if not embedding_rows:
        return

    for chunk_id, model, embedding, meta_data, dimension in embedding_rows:
        obj = Embedding(
            chunk_id=chunk_id,
            model=model,
            embedding=embedding,
            meta_data=meta_data,
            dimension=dimension,
        )
        session.add(obj)

    try:
        session.commit()
    except Exception:
        session.rollback()
        raise


# =========================================================
# MAIN INGESTION ORCHESTRATOR
# =========================================================

@log_function_call
def ingest_processed_chunks(
    session: Session,
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
    try:
        insert_chunks(
            session,
            chunk_rows
        )

        # -----------------------------------------------------
        # INSERT EMBEDDINGS
        # -----------------------------------------------------

        insert_embeddings(
            session,
            embedding_rows
        )
    except Exception as e:
        raise e
    
    return {
        "status": "success",
        "chunks_inserted": len(chunk_rows),
        "embeddings_inserted": len(embedding_rows)
    }