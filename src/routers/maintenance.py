from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(
    tags=["Maintenance"]
)

@router.get("/health")
async def root():
    return JSONResponse(
        status_code=200,
        content={"status": "ok"}
    )
