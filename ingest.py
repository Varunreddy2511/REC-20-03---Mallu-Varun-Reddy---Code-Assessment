# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
ingest.py — PDF Ingestion Pipeline for Financial RAG System
============================================================
Extracts text and tables from the AAPL Q3 2022 PDF, chunks them
intelligently, embeds with sentence-transformers, and persists
a FAISS index to disk.

Run once:  python ingest.py
"""

import os
import re
import pickle
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH      = os.path.join(os.path.dirname(__file__), "2022 Q3 AAPL.pdf")
INDEX_DIR     = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, accurate, fully open-source
CHUNK_SIZE    = 512                    # characters per text chunk
CHUNK_OVERLAP = 64                     # overlap to preserve context
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


def table_to_text(table: list[list], page_num: int, table_num: int) -> str:
    """
    Convert a pdfplumber table (list of rows, each row a list of cells)
    into a clean markdown-style string so the LLM can read it clearly.
    """
    if not table or not table[0]:
        return ""

    rows = []
    for row in table:
        # Replace None cells with empty string
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append(" | ".join(cells))

    header_sep = " | ".join(["---"] * len(table[0]))
    markdown = "\n".join([rows[0], header_sep] + rows[1:]) if len(rows) > 1 else rows[0]

    return (
        f"[TABLE — Page {page_num}, Table {table_num}]\n"
        f"{markdown}\n"
        f"[END TABLE]"
    )


def extract_chunks_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract all text and table chunks from the PDF.
    Returns a list of dicts: {text, source, type, page}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    print(f"[PDF] Opening: {pdf_path}")
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"   Total pages : {total_pages}")

        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1

            # ── 1. Extract tables first (they have priority) ──────────────────
            tables = page.extract_tables()
            table_bboxes = []

            for t_idx, table in enumerate(tables, start=1):
                table_text = table_to_text(table, page_num, t_idx)
                if len(table_text.strip()) < 20:
                    continue

                # Store table bbox so we can exclude that area from text
                t_obj = page.find_tables()
                if t_idx - 1 < len(t_obj):
                    table_bboxes.append(t_obj[t_idx - 1].bbox)

                all_chunks.append({
                    "text":   table_text,
                    "source": f"Page {page_num} — Table {t_idx}",
                    "type":   "table",
                    "page":   page_num,
                })

            # ── 2. Extract text (excluding table regions) ─────────────────────
            try:
                if table_bboxes:
                    # Crop out table areas and extract remaining text
                    words = page.extract_words()
                    filtered_words = []
                    for w in words:
                        in_table = False
                        for bbox in table_bboxes:
                            x0, y0, x1, y1 = bbox
                            if (x0 <= float(w["x0"]) <= x1 and
                                    y0 <= float(w["top"]) <= y1):
                                in_table = True
                                break
                        if not in_table:
                            filtered_words.append(w["text"])
                    raw_text = " ".join(filtered_words)
                else:
                    raw_text = page.extract_text() or ""
            except Exception:
                raw_text = page.extract_text() or ""

            raw_text = clean_text(raw_text)
            if len(raw_text) < 30:
                continue

            # Prefix every text chunk with its page number for traceability
            page_header = f"[Page {page_num}] "
            sub_chunks = splitter.split_text(raw_text)
            for chunk in sub_chunks:
                chunk = chunk.strip()
                if len(chunk) < 30:
                    continue
                all_chunks.append({
                    "text":   page_header + chunk,
                    "source": f"Page {page_num}",
                    "type":   "text",
                    "page":   page_num,
                })

    print(f"   [OK] Extracted {len(all_chunks)} chunks "
          f"({sum(1 for c in all_chunks if c['type']=='table')} tables, "
          f"{sum(1 for c in all_chunks if c['type']=='text')} text blocks)")
    return all_chunks


def build_faiss_index(chunks: list[dict], embed_model_name: str, index_dir: str):
    """Embed all chunks and persist a FAISS index + metadata."""
    print(f"\n[EMB] Loading embedding model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)

    texts = [c["text"] for c in chunks]
    print(f"[EMB] Embedding {len(texts)} chunks (this may take a minute)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Inner-product index (cosine after normalization)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"[SAVE] FAISS index saved -> {index_dir}")
    print(f"   Vectors : {index.ntotal} | Dimensions: {dim}")

    # Quick sanity check
    print("\n[SAMPLE] Chunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  [{i}] [{chunk['type'].upper()}] {chunk['source']}")
        print(f"       {chunk['text'][:120]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("  AAPL Q3 2022 - RAG Ingestion Pipeline")
    print("=" * 60)
    chunks = extract_chunks_from_pdf(PDF_PATH)
    if not chunks:
        print("[ERROR] No chunks extracted. Check the PDF path.")
        exit(1)
    build_faiss_index(chunks, EMBED_MODEL, INDEX_DIR)
    print("\n[DONE] Ingestion complete! Run: streamlit run app.py")
