from fastapi import FastAPI, Request
from dotenv import load_dotenv
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Initialize logging configuration
from .utility.logging_config import logger

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from .routers.rag_urls import router as rag_routers
from .routers.maintenance import router as maintenance_routers

app = FastAPI()

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    path = request.url.path
    method = request.method
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"--> Incoming Request: {method} {path} | Client: {client_ip}")
    
    try:
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        logger.info(f"<-- Outgoing Response: {method} {path} | Status: {response.status_code} | Time: {process_time:.4f}s")
        return response
    except Exception as e:
        process_time = time.perf_counter() - start_time
        logger.error(f"<-- Request Failed: {method} {path} | Error: {type(e).__name__}: {str(e)} | Time: {process_time:.4f}s")
        raise

app.include_router(rag_routers)
app.include_router(maintenance_routers)