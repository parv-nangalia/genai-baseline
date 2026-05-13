from ..services.IngestionServiceFactory import IngestionServiceFactory
import random, string

def get_openai_embedding(text):
    service = IngestionServiceFactory.getIngestionService('open-ai')
    return service.create_embeddings(text)


def get_hf_embedding(text):
    service = IngestionServiceFactory.getIngestionService('hugging-face')
    return service.create_embeddings(text)

def get_hf_embedding_textual(text):
    service = IngestionServiceFactory.getIngestionService('hugging-face')
    service.set_model("BAAI/bge-small-en-v1.5")
    return service.create_embeddings(text)
    
def get_doc_uuid():
    res = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return res
