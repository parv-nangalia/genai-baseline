from sentence_transformers import CrossEncoder
from ..utility.logging_config import log_function_call

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    @log_function_call
    def rerank(self, query: str, chunks: list, top_k: int = 5):
        if not chunks:
            return []
        
        # Prepare inputs: list of pairs [query, chunk_text]
        pairs = [[query, chunk["content"]] for chunk in chunks]
        scores = self.model.predict(pairs)
        
        # Add scores to chunks and convert to float (CrossEncoder scores can be float32)
        for idx, score in enumerate(scores):
            chunks[idx]["rerank_score"] = float(score)
            
        chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        return chunks[:top_k]
