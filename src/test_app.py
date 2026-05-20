from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_rag_ingestions():
    response = client.post("/ingest", data={"url": "https://en.wikipedia.org/wiki/Basketball_in_India"})
    assert response.status_code == 200

def test_rag_query():
    response = client.post("/query", json={"question": "What is the history of basketball in India?", "model": "hugging-face", "top_k": 5})
    assert response.status_code == 200
