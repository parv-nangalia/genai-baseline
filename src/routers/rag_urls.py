from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from fastapi.responses import JSONResponse
from ..views.rag import ingestionView, ragQueryView
from ..utility.logging_config import log_function_call

router = APIRouter(
    tags=["RagUrls"],
)

@router.post("/ingest")
@log_function_call
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
                raise HTTPException(
                    status_code = 400,
                    detail= "Some issue ingesting the url"
                )
        else:
            try:
                await ingestionView(file, "file")
            except Exception as e:
                raise HTTPException(
                    status_code = 400,
                    detail= "Some issue ingesting the file"
                )
        return JSONResponse(
            status_code = 200,
            content = "successfully created embeddings, text using /query endpoint"
            )

   
@router.post("/query")
@log_function_call
async def query_documents(
        question: str = Form(...),
        model: str = Form("hugging-face"),
        top_k: int = Form(5),
        comprehensiveness: int = Form(3),
        search_type: str = Form("vector"),
        rerank: bool = Form(False),
        llm_type: str = Form("local"),
        llm_model: str = Form("gemini")
        ):
    try:
        response = ragQueryView(
            question=question,
            model=model,
            top_k=top_k,
            comprehensiveness=comprehensiveness,
            search_type=search_type,
            rerank=rerank,
            llm_type=llm_type,
            llm_model=llm_model
        )
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))