📚 LegalRAG – Indian Law Research Assistant (RAG System)

A production-ready Retrieval-Augmented Generation (RAG) system for Indian legal documents (Evidence Act, CPC, CrPC, etc.), built using LangChain, ChromaDB, HuggingFace / Groq LLMs, and designed with clean modular architecture.

🚀 Features

🔍 Semantic search over Indian legal acts & judgments

📄 Chunk-based document ingestion with metadata

🧠 Retrieval-Augmented Generation (RAG)

💾 Persistent vector storage using ChromaDB

🔄 Pluggable LLM providers:

HuggingFace

Groq (fast & free-tier friendly)

🧪 CLI + Streamlit UI support

🏗️ Production-grade folder structure

❌ No hallucination outside uploaded documents

🏗️ Project Structure
legalrag/
│
├── app.py                      # Entry point (CLI / Streamlit launcher)
├── check_chroma.py             # Verify vector DB & stored documents
├── data_cleaning.py            # Optional preprocessing
├── requirements.txt
├── .env                        # API keys & configuration
│
├── config/
│   ├── settings.py
│   └── __init__.py
│
├── data/
│   ├── uploads/                # Raw uploaded documents
│   ├── legal_acts/             # Processed text files
│   └── chroma_db/              # Persistent vector DB
│
├── src/
│   ├── ingestion/
│   │   ├── document_processor.py
│   │   ├── vector_store.py
│   │   ├── run_ingestion.py
│   │   └── __init__.py
│   │
│   ├── retrieval/
│   │   ├── retriever.py
│   │   └── __init__.py
│   │
│   ├── generation/
│   │   ├── rag_pipeline.py
│   │   └── __init__.py
│   │
│   ├── providers/
│   │   ├── factory.py
│   │   ├── groq_provider.py
│   │   ├── huggingface_provider.py
│   │   ├── hybrid_provider.py
│   │   └── __init__.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── __init__.py
│   │
│   ├── ui/
│   │   ├── streamlit_app.py
│   │   └── __init__.py
│   │
│   └── utils/
│       └── __init__.py


✅ Every folder contains __init__.py for stable imports & production readiness.

⚙️ Setup Instructions
1️⃣ Create Virtual Environment
conda create -n legalrag310 python=3.10
conda activate legalrag310

2️⃣ Install Dependencies
pip install -r requirements.txt

🔐 Environment Variables (.env)
API_PROVIDER=groq

GROQ_API_KEY=your_groq_key_here

HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_NAME=legal_documents

📥 Ingest Documents

Put your legal documents (PDF / TXT) inside:

data/uploads/


Then run:

python -m src.ingestion.run_ingestion


✔ Documents are chunked
✔ Embeddings created
✔ Stored persistently in ChromaDB

🔍 Verify Vector Database
python check_chroma.py


Expected output:

Collection name: legal_documents
Document count: XXXX

🤖 Ask Questions (CLI)
python -m src.generation.rag_pipeline


Example:

Ask: Explain Section 58 of the Indian Evidence Act

🧠 RAG Logic (Strict)

Answers are generated ONLY from retrieved context

If relevant context is missing →
"Not available in the uploaded documents."

Prevents hallucinations ❌

🖥️ Streamlit UI (Optional)
streamlit run src/ui/streamlit_app.py


Features:

New Chat

Independent chat history

Source citations

Clean UI

🧪 Tech Stack

Python 3.10

LangChain

ChromaDB

HuggingFace Embeddings

Groq LLM

Streamlit

Loguru

🎯 Use Cases

Legal research assistant

Law student study tool

AI hackathon project

Resume-grade RAG system

Interview-ready architecture demo

🧠 Future Improvements

Section-aware retriever (Section 58 → exact match)

Multi-Act filtering

Citation highlighting

Answer confidence scoring

PDF upload via UI

👤 Author

Usha Rani
AI / Full-Stack Developer
📌 Focus: RAG Systems, LangChain, Agentic AI
