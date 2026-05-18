from ..utility.urlParsing import parse_website
from ..utility.docParsing import process_document
from ..db.vectorIngestion import ingest_processed_chunks
from ..db.dbConfig import DBConfig
from ..utility.helper import get_doc_uuid
from ..db.query import search_similar_chunks
from ..services.LLMServiceFactory import LLMServiceFactory


def ingestionView(target, targetType):
    db_connection = DBConfig.get_connection()
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
        ingest_processed_chunks(db_connection, response)
    except Exception as e:
        print("Error while inserting the document into the db")
        raise
    finally:
        db_connection.close()

        

def ragQueryView(question,model,top_k):
    db_connection = DBConfig.get_connection()
    try:
        chunks = search_similar_chunks(db_connection,question, model, top_k)
    except Exception as e:
        print("Error while querying the db for similar chunks")
        raise

    if not chunks:
        return {"answer": "No relevant chunks found."}


    context_text = "\n\n".join([f"Chunk {i+1}: {c['content']}" for i, c in enumerate(chunks)])

    prompt = f"""
    Use the following document chunks to answer the question.

    Context:
    {context_text}

    Question:
    {question}

    Answer concisely based on the context.
    """

    # -----------------
    # 5. Ask GPT
    # -----------------
    llm_client = LLMServiceFactory.get_llm_client("gemini")
    answer = llm_client.ask_gpt(prompt)

    return {"answer": answer}