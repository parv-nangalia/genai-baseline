from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from ..services.gateway_service import process_gateway_request

router = APIRouter(
    prefix="/v1",
    tags=["Gateway"],
)

@router.post("/chat/completions")
async def chat_completions(request: Request):
    # Get JSON payload
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    try:
        # Delegate request processing to the shared gateway service
        result = process_gateway_request(body)
        return JSONResponse(status_code=200, content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
