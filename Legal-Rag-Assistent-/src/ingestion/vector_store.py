from typing import List
from loguru import logger
from langchain_core.documents import Document
import uuid
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import settings
from pathlib import Path
import os

os.environ["OTEL_PYTHON_DISABLED"] = "true"  # Kills telemetry noise


class VectorStoreManager:
    def __init__(self, persist_dir: str | Path | None = None):
        """
        Initialize ChromaDB with a specific persist directory.
        
        Args:
            persist_dir: Path to Chroma DB folder. If None, uses settings.chroma_persist_directory.
        """
        from src.providers.factory import ProviderFactory  # ✅ lazy import
        
        self._provider = ProviderFactory.get_provider()
        
        # ✅ FIX: Accept persist_dir parameter (for Streamlit Cloud)
        if persist_dir is not None:
            chroma_path = str(Path(persist_dir).resolve())
        else:
            chroma_path = settings.chroma_persist_directory
        
        logger.info(f"Initializing Chroma at: {chroma_path}")
        
        self._client = chromadb.PersistentClient(
            path=chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name
        )
        self._embeddings = None
        logger.info(f"Chroma PersistentClient + collection '{settings.chroma_collection_name}' initialized")

    # ✅ REQUIRED for retriever / RAG
    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = self._provider.embeddings()
            logger.info("Embeddings loaded")
        return self._embeddings

    @property
    def collection(self):
        return self._collection

    def add_documents(self, docs: List[Document]) -> List[str]:
        texts, metadatas, ids = [], [], []

        for d in docs:
            if not d.page_content or not d.page_content.strip():
                continue

            texts.append(d.page_content.strip())
            metadatas.append(d.metadata or {})
            ids.append(str(uuid.uuid4()))

        if not texts:
            logger.warning("No valid documents to insert")
            return []

        vectors = self.embeddings.embed_documents(texts)

        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding count mismatch: texts={len(texts)} embeddings={len(vectors)}"
            )

        self._collection.add(
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
            ids=ids,
        )

        logger.info(f"Persisted {len(texts)} chunks to Chroma")
        return ids

    def count(self) -> int:
        return self._collection.count()
