from openai import OpenAI

from src.main import OPENAI_API_KEY
from .llmClientInterface import LLMClientInterface
import os


class OpenAIllmImpl(LLMClientInterface):

    _OPEN_API_KEY = None

    def __init__(self):
        self._OPEN_API_KEY = os.getenv('OPENAI_API_KEY')

    def get_client(self):
        api_key = self._OPEN_API_KEY
        client = OpenAI(api_key=api_key)
        return client
        
    def ask_gpt(self, prompt: str, model: str = "gpt-3.5-turbo") -> str:
        """
        Sends a prompt to GPT and returns the response text.
        """
        try:
            
            client = self.get_client()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Error while calling OpenAI API:", e)
            raise