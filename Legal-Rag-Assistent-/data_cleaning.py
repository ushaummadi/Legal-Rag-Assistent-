import re
import fitz  # pip install pymupdf
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/legal_acts")
OUT_DIR = Path("data/clean_chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 200  # keep meaningful pages only

# ---------- common cleaning ----------
def clean_text_common(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove page-break markers (if you added them previously)
    text = re.sub(r"---\s*PAGE\s*BREAK\s*---", " ", text, flags=re.I)

    # Remove page labels like Page 1 / Py 1 / p. 23
    text = re.sub(r"\b(?:Page|Py|P)\s*\d+\b", " ", text, flags=re.I)
    text = re.sub(r"\bp\.\s*\d+\b", " ", text, flags=re.I)

    # Remove long junk sequences
    text = re.sub(r"[._,\-~]{3,}", " ", text)

    # Fix spaced-out letters ONLY when they are single-letter tokens:
    # "A a d h a a r" -> "Aadhaar"
    text = re.sub(r"(?<=\b[A-Za-z])\s+(?=[A-Za-z]\b)", "", text)

    # Normalize whitespace but keep paragraphs
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Drop empty lines
    text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip()])

    return text.strip()

# ---------- PDF -> DataFrame (page-wise) ----------
def pdf_to_pages_df(pdf_path: Path) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    rows = []
    for page_num in range(len(doc)):
        rows.append({
            "pdf": pdf_path.stem,
            "page": page_num,
            "raw": doc[page_num].get_text("text"),
        })
    doc.close()
    df = pd.DataFrame(rows)
    df["clean"] = df["raw"].map(clean_text_common)
    df["n_chars"] = df["clean"].str.len()
    return df

# ---------- CENTRAL ACTS: 1 act per line WITH year + act no ----------
def parse_central_acts_to_lines(full_text: str):
    """
    Output example:
    1. Aadhaar (...) Act 2016 18
    """
    t = clean_text_common(full_text)

    # Make sure each "N." starts on new line
    t = re.sub(r"(?<!\n)\s+(\d{1,5})\.\s+", r"\n\1. ", t)

    # Strict row-like pattern (S.No + Name + Year + Act No)
    pat = re.compile(r"(?m)^\s*(\d+)\.\s*(.+?)\s+((?:19|20)\d{2})\s+(\d+)\s*$")

    rows = []
    for m in pat.finditer(t):
        s_no = int(m.group(1))
        name = re.sub(r"\s{2,}", " ", m.group(2)).strip()
        year = int(m.group(3))
        act_no = int(m.group(4))
        rows.append((s_no, f"{s_no}. {name} {year} {act_no}"))

    if not rows:
        return []

    out = pd.DataFrame(rows, columns=["s_no", "line"]).sort_values("s_no")
    out = out.drop_duplicates("s_no", keep="first")
    return out["line"].tolist()

def extract_full_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    t = "\n".join(doc[p].get_text("text") for p in range(len(doc)))
    doc.close()
    return t

# ---------- main ----------
def run():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDFs found in data/legal_acts/")
        return

    # 1) Central Acts special output (1 act per line with year)
    central_pdf = RAW_DIR / "Alphabetical_List_Central_Acts.pdf"
    if central_pdf.exists():
        full = extract_full_text(central_pdf)
        lines = parse_central_acts_to_lines(full)
        if lines:
            (OUT_DIR / "Alphabetical_List_Central_Acts_1perline.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            print(f"✅ Central Acts 1-per-line: {len(lines)} lines")
        else:
            print("⚠️ Central Acts: pattern not matched (PDF format differs)")

    # 2) Remaining 4 PDFs (and also Central Acts if you want) -> page-wise chunks
    for pdf_path in pdf_files:
        if pdf_path.name == "Alphabetical_List_Central_Acts.pdf":
            # optional: skip page-chunks for central acts
            continue

        df = pdf_to_pages_df(pdf_path)

        # keep only meaningful pages
        df = df[df["n_chars"] >= MIN_CHARS].copy()

        # ORDER guaranteed: sort by page (so numbering stays in order)
        df = df.sort_values(["page"])  # pandas sort_values [web:384]

        # save chunks: 1 file per page
        saved = 0
        for _, r in df.iterrows():
            out_file = OUT_DIR / f"{r['pdf']}_p{int(r['page'])}_c0.txt"
            out_file.write_text(r["clean"], encoding="utf-8")
            saved += 1

        print(f"✅ {pdf_path.name}: {saved} clean page-chunks")

    print(f"\n🎉 Done. Output folder: {OUT_DIR}")

if __name__ == "__main__":
    run()
