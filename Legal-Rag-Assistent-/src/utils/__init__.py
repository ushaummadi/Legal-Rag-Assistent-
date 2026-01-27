from pathlib import Path
from typing import Iterable
from langchain_core.documents import Document

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def format_source(doc: Document) -> str:
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", None)
    if page is None:
        return f"{src}"
    return f"{src} p.{page}"

def join_context(docs: Iterable[Document], max_chars: int = 12000) -> str:
    parts = []
    total = 0
    for i, d in enumerate(docs, start=1):
        chunk = f"[{i}] {format_source(d)}\n{d.page_content}".strip()
        if total + len(chunk) > max_chars:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n".join(parts)
