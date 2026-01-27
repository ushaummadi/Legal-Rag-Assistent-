import chromadb
from chromadb.config import Settings

CHROMA_PATH = r"C:\Users\ushar\legalrag\data\chroma_db"

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False),
)

collection = client.get_or_create_collection(
    name="legal_documents"
)

print("Collection name:", collection.name)
print("Document count:", collection.count())

# 🔍 test query
results = collection.query(
    query_texts=["What is Indian Evidence Act?"],
    n_results=3,
)

print("Sample documents:")
for doc in results["documents"][0]:
    print("-", doc[:200])
