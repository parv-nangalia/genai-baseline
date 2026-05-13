from .ingestionInterface import IngestionService
from sentence_transformers import SentenceTransformer, util



class HuggingFaceImpl(IngestionService):

    """Input shoud be a list of string seperated by /n or ,"""

    _model = "all-MiniLM-L6-v2"
    _client = None
    
    def get_client(self):
        if self._client == None:
            self._client = SentenceTransformer(self._model)
        return self._client

    def set_model(self, model):
        self._model = model

    def create_embeddings(self, input):
        client = self.get_client()
        embeddings = client.encode(input)
        return embeddings.tolist()

    def check_cosine_similarity(self, input, query):
        self.set_model("BAAI/bge-small-en")
        client = self.get_client()
        doc_embedding = self._model.encode(input)
        query_embedding = self._model.encode(query)
        scores = util.cos_sim(query_embedding, doc_embedding)
        best_idx = scores.argmax()
        # print(input[best_idx])
        return input[best_idx]












