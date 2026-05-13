from .ingestionInterface import IngestionService
from .openAIImpl import OpenAIIngestion
from .huggingFaceImpl import HuggingFaceImpl


class IngestionServiceFactory:

    @staticmethod
    def getIngestionService(service: str) -> IngestionService:
        service_mapping = {
            'open-ai' : OpenAIIngestion,
            'hugging-face' : HuggingFaceImpl
        }
        if service in service_mapping:
            return service_mapping[service]()
        else:
            raise ValueError("unsupported service, please raise a request for adding a new service")

