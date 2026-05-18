from .geminillmImpl import geminiLLMClient
from .openAIllmImpl import OpenAIllmImpl

class LLMServiceFactory:
    @staticmethod
    def get_llm_client(service_name: str):
        service_mapping = {
            'open-ai' : OpenAIllmImpl,
            'gemini' : geminiLLMClient
        }
        if service_name in service_mapping:
            return service_mapping[service_name]()
        else:
            raise ValueError("unsupported service, please raise a request for adding a new service")