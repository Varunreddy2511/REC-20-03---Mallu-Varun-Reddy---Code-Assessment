# -*- coding: utf-8 -*-
"""
rag_engine.py — Retrieval + Anti-Hallucination RAG Engine
==========================================================
Loads the FAISS index, retrieves the most relevant chunks,
and calls Gemini ONLY when evidence is strong enough.

Anti-hallucination layers:
  1. Similarity threshold — LLM never called if no chunk is confident
  2. Temperature = 0      — fully deterministic Gemini output
  3. Strict system prompt — Gemini forbidden from using outside knowledge
  4. Graceful refusal     — clear "not found" message for the user
"""

import os
import re
import pickle
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_DIR          = os.path.join(os.path.dirname(__file__), "faiss_index")
EMBED_MODEL        = "all-MiniLM-L6-v2"
GEMINI_MODEL       = "gemini-2.5-flash"
TOP_K              = 6      # chunks sent to LLM
SIMILARITY_CUTOFF  = 0.40   # cosine similarity (0–1); below = refuse
GEMINI_TEMPERATURE = 0.0    # deterministic
# ──────────────────────────────────────────────────────────────────────────────

# ── System Prompt (Anti-Hallucination Contract) ───────────────────────────────
SYSTEM_PROMPT = """You are a precise financial document analyst. You ONLY answer questions
using the exact content from the retrieved document excerpts provided to you.

STRICT RULES — you must follow all of these without exception:
1. NEVER use any knowledge outside the provided context excerpts.
2. NEVER guess, infer, or extrapolate beyond what is explicitly stated in the context.
3. If the answer is not clearly present in the context, you MUST respond with exactly:
   "I could not find this information in the provided Apple Q3 2022 document. 
    Please try rephrasing your question or ask about a different topic covered in the report."
4. When you find the answer, quote or closely paraphrase the relevant numbers/text
   and always mention which page or section it came from.
5. For numerical data (revenues, margins, EPS, etc.), reproduce the exact figures as stated.
   Do NOT round, estimate, or recompute unless explicitly asked.
6. Do NOT say "based on my knowledge" or "typically" or "generally". 
   Every sentence must be traceable to the provided context.
7. Keep answers concise and factual. No filler text.
"""
# ──────────────────────────────────────────────────────────────────────────────


class RAGEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "[ERROR] GEMINI_API_KEY not set. "
                "Open the .env file and paste your Gemini API key."
            )
        genai.configure(api_key=api_key)

        print("[EMB] Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("[IDX] Loading FAISS index...")
        index_path  = os.path.join(INDEX_DIR, "index.faiss")
        chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                "[ERROR] FAISS index not found. Run  python ingest.py  first."
            )

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)

        self.gemini = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=1024,
            ),
        )
        print("[OK] RAG engine ready.")

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """
        Embed the query, search FAISS, and return the top-k chunks
        along with their cosine similarity scores.
        """
        q_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, indices = self.index.search(q_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                **self.chunks[idx],
                "score": float(score),
            })
        return results

    # ── Answer ─────────────────────────────────────────────────────────────────

    def answer(self, query: str) -> dict:
        """
        Full RAG pipeline:
          1. Retrieve top-k chunks
          2. Check similarity threshold (anti-hallucination gate)
          3. Build grounded prompt
          4. Call Gemini at temperature=0
          5. Return answer + sources

        Returns:
          {
            "answer":   str,
            "sources":  list[dict],   # chunks with scores
            "refused":  bool,         # True if threshold gate fired
          }
        """
        sources = self.retrieve(query)

        # ── Anti-hallucination gate ────────────────────────────────────────────
        if not sources or sources[0]["score"] < SIMILARITY_CUTOFF:
            return {
                "answer": (
                    "I could not find relevant information about this in the "
                    "Apple Q3 2022 financial report.\n\n"
                    "This could mean:\n"
                    "• The topic isn't covered in this specific document.\n"
                    "• Try rephrasing your question using keywords from the report "
                    "(e.g., 'net sales', 'operating income', 'earnings per share').\n"
                    "• Ask about a specific financial metric, product category, "
                    "or segment mentioned in the quarterly filing."
                ),
                "sources": sources,
                "refused": True,
            }

        # ── Build grounded context ─────────────────────────────────────────────
        context_blocks = []
        for i, chunk in enumerate(sources, start=1):
            context_blocks.append(
                f"--- EXCERPT {i} ({chunk['source']}, type={chunk['type']}, "
                f"similarity={chunk['score']:.3f}) ---\n"
                f"{chunk['text']}\n"
                f"--- END EXCERPT {i} ---"
            )
        context_str = "\n\n".join(context_blocks)

        prompt = (
            f"RETRIEVED DOCUMENT EXCERPTS FROM APPLE Q3 2022 REPORT:\n\n"
            f"{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Answer the question using ONLY the excerpts above. "
            f"If the answer is not present, say so clearly."
        )

        # ── Call Gemini ────────────────────────────────────────────────────────
        try:
            response = self.gemini.generate_content(prompt)
            answer_text = response.text.strip()
        except Exception as e:
            answer_text = f"[ERROR] Gemini API error: {e}"

        return {
            "answer":  answer_text,
            "sources": sources,
            "refused": False,
        }


# ── Singleton loader (cached across Streamlit reruns) ─────────────────────────
_engine_instance: RAGEngine | None = None


def get_engine() -> RAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance
