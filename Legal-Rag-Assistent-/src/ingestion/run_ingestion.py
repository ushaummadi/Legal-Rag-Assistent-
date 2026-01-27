from pathlib import Path
from loguru import logger
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from src.ingestion.document_processor import process_file
from src.ingestion.vector_store import VectorStoreManager

def main():
    upload_dir = Path(settings.uploads_dir)
    files = [p for p in upload_dir.iterdir() if p.is_file()]

    if not files:
        raise RuntimeError(f"No files found in {settings.uploads_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    docs = []
    for f in files:
        processed = process_file(str(f))
        chunks = splitter.split_text(processed["text"])
        for i, c in enumerate(chunks):
            docs.append(
                Document(
                    page_content=c,
                    metadata={**processed["metadata"], "chunk": i},
                )
            )

    logger.info(f"Prepared {len(docs)} chunks from {len(files)} files")

    vs = VectorStoreManager()
    vs.add_documents(docs)
    logger.info(f"✅ Done. Collection count ≈ {vs.count()}")


if __name__ == "__main__":
    main()
