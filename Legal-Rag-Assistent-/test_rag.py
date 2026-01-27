from src.ingestion.vector_store import VectorStoreManager
from src.retrieval.retriever import get_retriever
from src.providers.groq_provider import GroqProvider
from src.generation.rag_pipeline import build_rag_chain

vs = VectorStoreManager()
retriever = get_retriever(vs.vectordb)

llm = GroqProvider().llm()
qa = build_rag_chain(llm, retriever)

query = "Define oral evidence under the Indian Evidence Act"
result = qa.invoke({"query": query})

print("\nANSWER:\n", result["result"])
