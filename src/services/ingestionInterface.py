from abc import ABC
from typing import List

class IngestionService(ABC):

    def create_embeddings() -> List:
        pass

    def check_cosine_similarity() -> List:
        pass
