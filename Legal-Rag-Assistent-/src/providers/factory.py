from config.settings import settings
from src.providers.huggingface_provider import HuggingFaceProvider
from src.providers.groq_provider import GroqProvider


class ProviderFactory:
    @staticmethod
    def get_provider():
        # 🔹 Embeddings always from HuggingFace
        hf = HuggingFaceProvider()

        # 🔹 LLM from Gemini
        groq = GroqProvider()

        # 🔹 Hybrid provider object
        return HybridProvider(
            embeddings_provider=hf,
            llm_provider=groq,
        )


class HybridProvider:
    def __init__(self, embeddings_provider, llm_provider):
        self._embeddings_provider = embeddings_provider
        self._llm_provider = llm_provider

    def embeddings(self):
        return self._embeddings_provider.embeddings()

    def llm(self):
        return self._llm_provider.llm()
