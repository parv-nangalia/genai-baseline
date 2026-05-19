from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from fastapi.responses import JSONResponse
from ..views.rag import ingestionView, ragQueryView

router = APIRouter(
    tags=["RagUrls"],
)

@router.post("/ingest")
async def upload(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)):

    if not url and not file:
        raise HTTPException(
            status_code = 400,
            detail="You need to provide atleast one file or url to be ingested"
        )
    
    if url and file:
        raise HTTPException(
            status_code = 400,
            detail= "Please provide only one at a time"
        )
    
    else:
        if url:
            try:
                await ingestionView(url, "url")
            except Exception as e:
                return HTTPException(
                    status_code = 400,
                    detail= "Some issue ingesting the url"
                )
        else:
            try:
                await ingestionView(file, "file")
            except Exception as e:
                return HTTPException(
                    status_code = 400,
                    detail= "Some issue ingesting the file"
                )
        return JSONResponse(
            status_code = 200,
            content = "successfully created embeddings, text using /query endpoint"
            )

   
@router.post("/query")
async def query_documents(
        question: str,
        model: str = "hugging-face",  # 'openai' or 'hf'
        top_k: int = 5,
        comprehensiveness: int = 3
        ):
    try:
        response = ragQueryView(question, model, top_k, comprehensiveness)
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))