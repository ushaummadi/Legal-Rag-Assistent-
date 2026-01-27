from langchain_groq import ChatGroq
from config.settings import settings

class GroqProvider:
    def embeddings(self):
        # Keep embeddings from HF provider (recommended)
        # If your factory uses GroqProvider when API_PROVIDER=groq,
        # you should still return HF embeddings here or use HybridProvider logic.
        raise NotImplementedError("Use HF embeddings via HybridProvider (see below).")

    def llm(self):
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")

        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model=settings.groq_llm_model,
            temperature=settings.temperature,
            max_tokens=settings.max_output_tokens,
        )
