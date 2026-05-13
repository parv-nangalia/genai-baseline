from openai import OpenAI
from .ingestionInterface import IngestionService
from ..main import OPENAI_API_KEY

class OpenAIIngestion(IngestionService):

    """Input shoud be a list of string seperated by /n or ,"""

    _API_KEY = OPENAI_API_KEY
    _client = None
    _model = "text-embedding-3-small"
    
    def get_client(self):
        if self._client == None:
            self._client = OpenAI(api_key=self._API_KEY)
        else:
            return self._client

    def set_model(self, model):
        self._model = model

    def create_embeddings(self, input):
        model = self._model
        client = self.get_client()
        response = client.embeddings.create(
            model = model,
            input = input
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def check_cosine_similarity(self, input, query):
    #     client = self.get_client()
    #     model = self._model
    #     doc_embeddings = client.embeddings.create(
    #         model=model,
    #         input=input
    #     )

    #     query_embedding = client.embeddings.create(
    #         model=model,
    #         input=query
    #     )

    #     doc_vectors = np.array([x.embedding for x in doc_embeddings.data])
    #     query_vector = np.array(query_embedding.data[0].embedding).reshape(1, -1)

    #     scores = cosine_similarity(query_vector, doc_vectors)[0]

    #     best_doc = input[np.argmax(scores)]
    #     print(best_doc)
        pass











    