
from typing import List
from ..utility.helper import get_hf_embedding_textual, get_openai_embedding
from sqlalchemy.orm import Session
from ..db.models import Chunk, Embedding
import math


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return -1.0
    return dot / (mag_a * mag_b)


def search_similar_chunks(
    session: Session,
    question: str,
    model_name: str,
    top_k: int = 5,
):
    """Search similar chunks using ORM session and Python-side similarity.

    This fetches candidate embeddings for the requested model and dimension,
    computes cosine similarity in Python, and returns the top_k matches.
    """
    if model_name == "openai":
        embedding_vector = get_openai_embedding(question)
    elif model_name == "hugging-face":
        embedding_vector = get_hf_embedding_textual(question)
    else:
        embedding_vector = get_hf_embedding_textual(question)

    dimension = len(embedding_vector)

    # fetch embeddings for model and dimension
    results = (
        session.query(Embedding, Chunk)
        .join(Chunk, Embedding.chunk_id == Chunk.id)
        .filter(Embedding.model == ("hf" if model_name.startswith("hugging") else model_name))
        .filter(Embedding.dimension == dimension)
        .all()
    )

    scored = []
    for emb, chunk in results:
        try:
            vec = emb.embedding or []
            score = _cosine_similarity(embedding_vector, vec)
        except Exception:
            score = -1.0
        scored.append((score, chunk, emb))

    scored.sort(key=lambda x: x[0], reverse=True)

    top = []
    for score, chunk, emb in scored[:top_k]:
        top.append(
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "content": chunk.content,
                "chunk_metadata": chunk.meta_data,
                "embedding_id": emb.id,
                "model": emb.model,
                "dimension": emb.dimension,
                "embedding": emb.embedding,
                "embedding_metadata": emb.meta_data,
                "score": score,
            }
        )

    return top