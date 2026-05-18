
from google import genai

from .llmClientInterface import LLMClientInterface
import os
from src.main import GOOGLE_API_KEY

class geminiLLMClient(LLMClientInterface):

    _GOOGLE_API_KEY = GOOGLE_API_KEY

    def get_client(self):
        api_key = self._GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        client = genai.Client(api_key=api_key)
        return client
    

    def ask_gpt(self, prompt: str):
        client = self.get_client()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text