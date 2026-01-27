from typing import List, Dict, Any
from langchain_core.documents import Document

def basic_metrics(query: str, docs: List[Document], answer: str) -> Dict[str, Any]:
    total_chars = sum(len(d.page_content) for d in docs)
    unique_sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    return {
        "query": query,
        "chunks_used": len(docs),
        "unique_sources": unique_sources,
        "context_chars": total_chars,
        "answer_chars": len(answer),
    }

# RAGAS (optional - only if installed)
def ragas_eval(dataset_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        
        ds = Dataset.from_list(dataset_rows)
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        return dict(result)
    except ImportError:
        return {"error": "RAGAS not installed. Run: pip install ragas datasets"}
