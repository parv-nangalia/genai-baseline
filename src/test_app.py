from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_rag_ingestions():
    response = client.post("/ingest", data={"url": "https://en.wikipedia.org/wiki/Chope_(platform)"})
    assert response.status_code == 200

def test_rag_query():
    response = client.post("/query", data={"question": "what is Chope?", "model": "hugging-face", "top_k": 5, "comprehensiveness": 3})
    assert response.status_code == 200

def test_rag_query_hybrid():
    response = client.post("/query", data={"question": "what is Chope?", "model": "hugging-face", "top_k": 5, "comprehensiveness": 3, "search_type": "hybrid"})
    assert response.status_code == 200

def test_rag_query_rerank():
    response = client.post("/query", data={"question": "what is Chope?", "model": "hugging-face", "top_k": 5, "comprehensiveness": 3, "rerank": True})
    assert response.status_code == 200

