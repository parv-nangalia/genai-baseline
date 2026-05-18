from fastapi import FastAPI
from dotenv import load_dotenv
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

from .routers.rag_urls import router as rag_routers
from .routers.maintenance import router as maintenance_routers

app = FastAPI()

app.include_router(rag_routers)
app.include_router(maintenance_routers)