from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional

from .routers.rag_urls import router as rag_routers
from .routers.maintenance import router as maintenance_routers

app = FastAPI()

app.include_router(rag_routers)
app.include_router(maintenance_routers)