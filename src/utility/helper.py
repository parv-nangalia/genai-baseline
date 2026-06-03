from ..services.IngestionServiceFactory import IngestionServiceFactory
import random, string
from .logging_config import log_function_call

@log_function_call
def get_openai_embedding(text):
    service = IngestionServiceFactory.getIngestionService('open-ai')
    return service.create_embeddings(text)


@log_function_call
def get_hf_embedding(text):
    service = IngestionServiceFactory.getIngestionService('hugging-face')
    return service.create_embeddings(text)

@log_function_call
def get_hf_embedding_textual(text):
    service = IngestionServiceFactory.getIngestionService('hugging-face')
    service.set_model("BAAI/bge-small-en-v1.5")
    return service.create_embeddings(text)
    
@log_function_call
def get_doc_uuid():
    res = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return res

