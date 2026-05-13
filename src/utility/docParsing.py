"""
UNIFIED DOCUMENT INGESTION FORMAT
---------------------------------

This version works with FastAPI UploadFile objects.
It returns a uniform structure for URL ingestion, PDF, TXT, DOC, etc.
"""

import uuid
from fastapi import UploadFile
from io import BytesIO
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utility.urlParsing import fake_openai_vector

from .helper import get_openai_embedding, get_hf_embedding_textual


# =========================================================
# CONFIG
# =========================================================

OPENAI_MODEL = "text-embedding-3-small"
HF_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# =========================================================
# TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from a PDF given as bytes.
    Works with UploadFile.read() content.
    """
    pdf_stream = BytesIO(pdf_bytes)  # wrap bytes in a file-like object
    reader = PdfReader(pdf_stream)
    
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    
    return "\n".join(pages)

def extract_text_from_txt_bytes(file_bytes: bytes) -> str:
    """
    Extracts text from a TXT file given as bytes.
    """
    return file_bytes.decode("utf-8")


def load_document(file: UploadFile) -> str:
    """
    Reads file content from UploadFile and returns raw text.
    Supports PDF and TXT/MD.\
    """
    try:
        suffix = file.filename.lower().split(".")[-1]
        
        # read file content synchronously
        file.file.seek(0)  # ensure we start at beginning
        content = file.file.read()  # this returns bytes

        if suffix == "pdf":
            return extract_text_from_pdf_bytes(content)
        elif suffix in ["txt", "md"]:
            return extract_text_from_txt_bytes(content)
        else:
            raise ValueError(f"Unsupported file type: {file.filename}")
    except Exception as e:
        print(f"Error loading document: {e}")
        raise


# =========================================================
# CHUNKING
# =========================================================

def chunk_document(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_document(file: UploadFile, doc_id: str):
    """
    Returns a list of chunk dictionaries:

    [
        {
            "chunk_id": "...",
            "doc_id": "...",
            "text": "...",
            "metadata": {...},
            "embeddings": {
                "openai": {...},
                "hf": {...}
            }
        }
    ]
    """

    # -----------------------------------------------------
    # LOAD DOCUMENT
    # -----------------------------------------------------
    document_text = load_document(file)

    # -----------------------------------------------------
    # CHUNK DOCUMENT
    # -----------------------------------------------------
    chunks = chunk_document(document_text)

    # -----------------------------------------------------
    # PROCESS CHUNKS
    # -----------------------------------------------------
    processed_chunks = []

    for idx, chunk_text in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        metadata = {
            "source": file.filename,
            "chunk_index": idx
        }

        # -------------------------------------------------
        # GENERATE EMBEDDINGS
        # -------------------------------------------------
        # openai_embedding = get_openai_embedding(chunk_text)
        openai_embedding = fake_openai_vector(seed=1)
        hf_embedding = get_hf_embedding_textual(chunk_text)

        # -------------------------------------------------
        # FINAL STRUCTURE
        # -------------------------------------------------
        processed_chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": chunk_text,
            "metadata": metadata,
            "embeddings": {
                "openai": {
                    "model": OPENAI_MODEL,
                    "vector": openai_embedding
                },
                "hf": {
                    "model": HF_MODEL_NAME,
                    "vector": hf_embedding
                }
            }
        })

    return processed_chunks