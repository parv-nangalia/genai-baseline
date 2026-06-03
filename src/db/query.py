from typing import List, Dict, Any
from ..utility.helper import get_hf_embedding_textual, get_openai_embedding
from sqlalchemy.orm import Session
from ..db.models import Chunk, Embedding
import math
import re


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return -1.0
    return dot / (mag_a * mag_b)


def tokenize(text: str) -> List[str]:
    """Lowercase and extract alphanumeric word tokens."""
    return re.findall(r'\w+', text.lower())


def score_bm25(query: str, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
    """Score chunks using standard BM25."""
    query_tokens = tokenize(query)
    if not query_tokens or not chunks:
        for c in chunks:
            c["keyword_score"] = 0.0
        return chunks

    doc_tokens = [tokenize(c["content"]) for c in chunks]
    doc_lengths = [len(tokens) for tokens in doc_tokens]
    avg_doc_len = sum(doc_lengths) / len(chunks) if chunks else 1.0

    df: Dict[str, int] = {}
    for tokens in doc_tokens:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            df[token] = df.get(token, 0) + 1

    scored_chunks = []
    N = len(chunks)
    for idx, chunk in enumerate(chunks):
        tokens = doc_tokens[idx]
        doc_len = doc_lengths[idx]
        
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        for q_token in query_tokens:
            if q_token in tf:
                df_t = df.get(q_token, 0)
                idf = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
                tf_val = tf[q_token]
                tf_scaled = (tf_val * (k1 + 1.0)) / (tf_val + k1 * (1.0 - b + b * (doc_len / avg_doc_len)))
                score += idf * tf_scaled

        chunk_copy = dict(chunk)
        chunk_copy["keyword_score"] = score
        scored_chunks.append(chunk_copy)

    scored_chunks.sort(key=lambda x: x["keyword_score"], reverse=True)
    return scored_chunks


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """Fuse vector and keyword results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}
    chunk_map = {}

    for rank, item in enumerate(vector_results, start=1):
        chunk_id = item["id"]
        chunk_map[chunk_id] = item
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, item in enumerate(keyword_results, start=1):
        chunk_id = item["id"]
        if chunk_id not in chunk_map:
            chunk_map[chunk_id] = item
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    
    fused_results = []
    for chunk_id in sorted_chunk_ids[:top_k]:
        item = dict(chunk_map[chunk_id])
        item["score"] = rrf_scores[chunk_id]
        fused_results.append(item)

    return fused_results


def search_chunks(
    session: Session,
    question: str,
    model_name: str,
    search_type: str = "vector",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """General retrieval function supporting vector, keyword, and hybrid modes."""
    if model_name == "openai":
        embedding_vector = get_openai_embedding(question)
    elif model_name == "hugging-face":
        embedding_vector = get_hf_embedding_textual(question)
    else:
        embedding_vector = get_hf_embedding_textual(question)

    dimension = len(embedding_vector)
    db_model_name = "hf" if model_name.startswith("hugging") else model_name

    results = (
        session.query(Embedding, Chunk)
        .join(Chunk, Embedding.chunk_id == Chunk.id)
        .filter(Embedding.model == db_model_name)
        .filter(Embedding.dimension == dimension)
        .all()
    )

    if not results:
        return []

    candidates = []
    for emb, chunk in results:
        candidates.append({
            "id": chunk.id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "chunk_metadata": chunk.meta_data,
            "embedding_id": emb.id,
            "model": emb.model,
            "dimension": emb.dimension,
            "embedding": emb.embedding,
            "embedding_metadata": emb.meta_data,
        })

    if search_type == "vector":
        scored = []
        for c in candidates:
            try:
                score = _cosine_similarity(embedding_vector, c["embedding"] or [])
            except Exception:
                score = -1.0
            c_copy = dict(c)
            c_copy["score"] = score
            scored.append(c_copy)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    elif search_type == "keyword":
        scored = score_bm25(question, candidates)
        for c in scored:
            c["score"] = c["keyword_score"]
        return scored[:top_k]

    elif search_type == "hybrid":
        vector_scored = []
        for c in candidates:
            try:
                score = _cosine_similarity(embedding_vector, c["embedding"] or [])
            except Exception:
                score = -1.0
            c_copy = dict(c)
            c_copy["score"] = score
            vector_scored.append(c_copy)
        vector_scored.sort(key=lambda x: x["score"], reverse=True)

        keyword_scored = score_bm25(question, candidates)

        return reciprocal_rank_fusion(vector_scored, keyword_scored, top_k=top_k)

    else:
        raise ValueError(f"Unknown search_type: {search_type}")


def search_similar_chunks(
    session: Session,
    question: str,
    model_name: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Backward-compatible search using search_chunks in vector mode."""
    return search_chunks(session, question, model_name, "vector", top_k)
