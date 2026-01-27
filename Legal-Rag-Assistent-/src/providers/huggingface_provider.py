from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
class HuggingFaceProvider:
    def embeddings(self):
        return HuggingFaceEmbeddings(
            model_name=settings.hf_embedding_model
        )

    