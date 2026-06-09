import os
from .llmClientInterface import LLMClientInterface
from ..utility.logging_config import log_function_call
from .gateway_service import process_gateway_request

class GatewayLLMClient(LLMClientInterface):
    def __init__(self):
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")

    def get_client(self):
        return self

    @log_function_call
    def ask_gpt(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        try:
            # Call process_gateway_request directly in Python (prevents deadlock)
            result = process_gateway_request(payload)
            choices = result.get("choices", [])
            if not choices:
                raise ValueError(f"Gateway returned no choices in response: {result}")
            return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            print("Error while calling Gateway LLM Service:", e)
            raise
