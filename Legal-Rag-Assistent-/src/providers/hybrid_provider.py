from src.providers.huggingface_provider import HuggingFaceProvider
from src.providers.groq_provider import GroqProvider

class HybridProvider:
    def __init__(self):
        self._emb = HuggingFaceProvider()
        self._llm = GroqProvider()

    def embeddings(self):
        return self._emb.embeddings()

    def llm(self):
        return self._llm.llm()
