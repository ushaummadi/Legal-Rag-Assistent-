from typing import List
from langchain_core.documents import Document
from loguru import logger
from src.ingestion.vector_store import VectorStoreManager
from config.settings import settings

class NativeRetriever:
    def __init__(self):
        self.vs = VectorStoreManager()
        self.collection = self.vs.collection
        self.embeddings = self.vs.embeddings

    def get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.embeddings.embed_query(query)
        
        # Vector Search first (k=20 to get more candidates)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=20,  # More results!
            include=["documents", "metadatas"],
        )
        
        docs = []
        query_lower = query.lower()
        
        # Keyword matching (exact text search)
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            text_lower = text.lower()
            
            # If query words exist in the chunk, add it!
            if all(word in text_lower for word in query_lower.split()):
                docs.append(Document(page_content=text, metadata=meta or {}))
        
        logger.info(f"✅ Found {len(docs)} matching documents")
        return docs[:5]  # Return top 5

def get_retriever():
    return NativeRetriever()
