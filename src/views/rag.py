from ..utility.urlParsing import parse_website
from ..utility.docParsing import process_document
from ..db.vectorIngestion import ingest_processed_chunks
from ..db.dbConfig import DBConfig
from ..utility.helper import get_doc_uuid
from ..db.query import search_chunks
from ..services.LLMServiceFactory import LLMServiceFactory
from ..services.reranker import Reranker

_comprehensiveness_instruction_map = None


def _get_comprehensiveness_instructions():
    global _comprehensiveness_instruction_map
    if _comprehensiveness_instruction_map is None:
        _comprehensiveness_instruction_map = {
            1: "Use the fewest words possible and keep the answer direct and to the point.",
            2: "Provide a brief answer with only the most important details.",
            3: "Provide a balanced answer with enough detail to fully address the question.",
            4: "Provide a detailed answer with extra reasoning and useful context.",
            5: "Provide an in-depth, comprehensive answer with full explanation and supporting details."
        }
    return _comprehensiveness_instruction_map


def _get_comprehensiveness_instruction(level: int) -> str:
    level = max(1, min(5, level))
    return _get_comprehensiveness_instructions()[level]


async def ingestionView(target, targetType):
    db_session = DBConfig.get_session()
    if targetType=="url":
        ## logic to scrap using beautifulsoup
        try:
            response = parse_website(target,get_doc_uuid())
        except Exception as e:
            print("Error while parsing the website")
            raise
    else:
        ## logic to scrap using PDF OCR
        try:
            response = process_document(target, get_doc_uuid())
        except Exception as e:
            print("error while parsing the document")
            raise
    try:
        ingest_processed_chunks(db_session, response)
    except Exception as e:
        print(str(e))
        print("Error while inserting the document into the db")
        raise
    finally:
        print("Ingestion Successful, closing the db session")
        db_session.close()
        

def ragQueryView(question, model, top_k, comprehensiveness: int = 3, search_type: str = "vector", rerank: bool = False):
    db_session = DBConfig.get_session()
    try:
        # Fetch more candidates if we need to rerank
        retrieve_k = top_k * 3 if rerank else top_k
        chunks = search_chunks(db_session, question, model, search_type, retrieve_k)
        
        if rerank and chunks:
            
            reranker = Reranker()
            chunks = reranker.rerank(question, chunks, top_k)
    except Exception as e:
        print("Error while querying the db for chunks")
        raise

    if not chunks:
        return {"answer": "No relevant chunks found."}


    context_text = "\n\n".join([f"Chunk {i+1}: {c['content']}" for i, c in enumerate(chunks)])
    instruction = _get_comprehensiveness_instruction(comprehensiveness)

    prompt = f"""
    Use the following document chunks to answer the question.

    Context:
    {context_text}

    Question:
    {question}

    {instruction}
    """

    llm_client = LLMServiceFactory.get_llm_client("gemini")
    answer = llm_client.ask_gpt(prompt)

    return {"answer": answer}