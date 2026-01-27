# debug_simple.py - Fixed for your config
import os
from src.ingestion.vector_store import VectorStoreManager
from config.settings import settings

print("=== LEGAL RAG DEBUGGER (Fixed) ===")
print(f"DB Path: {settings.chroma_persist_directory}")
print(f"Collection: {settings.chroma_collection_name}")
print(f"Docs Dir: {settings.DOCS_DIR}")

# Check folders
print(f"\n✅ Chroma DB: {os.path.exists(settings.chroma_persist_directory)}")
print(f"✅ Docs folder: {os.path.exists(settings.DOCS_DIR)}")

try:
    vs = VectorStoreManager()
    count = vs.collection.count()
    print(f"\n📊 Collection count: **{count}**")
    
    if count == 0:
        print("❌ **EMPTY!** Run ingestion first.")
    else:
        # Test query
        query = "section 125"
        print(f"\n🧪 Testing '{query}'...")
        
        query_vec = vs.embeddings.embed_query(query)
        results = vs.collection.query(
            query_embeddings=[query_vec], 
            n_results=10, 
            include=["documents", "metadatas"]
        )
        
        print(f"Raw results: **{len(results['documents'][0])}**")
        if results["documents"][0]:
            print("✅ **FIRST MATCH:**")
            print(results["documents"][0][0][:200])
        else:
            print("❌ **ZERO RESULTS**")
            
except Exception as e:
    print(f"💥 ERROR: {e}")
