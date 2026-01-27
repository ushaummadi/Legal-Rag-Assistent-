from src.retrieval.retriever import get_retriever

def main():
    retriever = get_retriever()

    query = "Define oral evidence under the Indian Evidence Act"
    docs = retriever.get_relevant_documents(query)

    print(f"\nRetrieved {len(docs)} documents:\n")
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.page_content[:300]}...\n")

if __name__ == "__main__":
    main()
