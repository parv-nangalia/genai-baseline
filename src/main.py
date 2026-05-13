from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from dotenv import load_dotenv
import os


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

from .routers.rag_urls import router as rag_routers
from .routers.maintenance import router as maintenance_routers

app = FastAPI()

app.include_router(rag_routers)
app.include_router(maintenance_routers)