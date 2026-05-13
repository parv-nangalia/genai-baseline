from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(
    tags=["Maintenance"]
)

@router.get("/health")
async def root():
    return JSONResponse({"status": "ok"})
