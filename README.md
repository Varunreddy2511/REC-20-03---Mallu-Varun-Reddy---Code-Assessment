# 📈 Apple Q3 2022 — Financial RAG System

A production-quality **Retrieval-Augmented Generation (RAG)** system that answers questions about Apple's Q3 2022 SEC 10-Q filing with **zero hallucination**. Built entirely with open-source tools for embeddings and vector search, powered by **Google Gemini** as the LLM.

---

## 🧠 How It Works

```
PDF (10-Q Filing)
      │
      ▼
 pdfplumber  ──► Extracts text blocks + tables (page-by-page)
      │
      ▼
 Smart Chunker ──► Text: recursive character splitting (512 chars)
                   Tables: preserved whole as markdown
      │
      ▼
 sentence-transformers ──► Embeds all chunks (all-MiniLM-L6-v2)
      │
      ▼
 FAISS Index ──► Stored locally on disk
      │
      ▼
 User Question ──► Embedded ──► Top-6 chunks retrieved
      │
      ▼
 Similarity Gate ──► Score < 0.40? Refuse (no LLM call)
      │
      ▼
 Gemini 2.5 Flash (T=0) ──► Grounded answer with source citations
      │
      ▼
 Streamlit Chat UI
```

---

## 🛡️ Anti-Hallucination Layers

| Layer | Description |
|---|---|
| **Similarity Gate** | If no chunk scores ≥ 0.40 cosine similarity, the LLM is never called |
| **Temperature = 0** | Fully deterministic Gemini output — no creative generation |
| **Strict System Prompt** | Gemini is contractually forbidden from using knowledge outside the retrieved chunks |
| **Source Citations** | Every answer cites the exact page and section it came from |
| **Graceful Refusal** | Clear, helpful "not found" message when the data isn't in the document |

---

## 🗂️ Tech Stack

| Component | Technology |
|---|---|
| **PDF Parsing** | `pdfplumber` — table-aware extraction |
| **Text Chunking** | `langchain-text-splitters` — RecursiveCharacterTextSplitter |
| **Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` (runs locally, free) |
| **Vector Store** | `FAISS` (CPU) — persisted to disk, no server needed |
| **LLM** | `google-generativeai` — Gemini 2.5 Flash |
| **UI** | `Streamlit` — chat interface with source chunk viewer |
| **Env Management** | `python-dotenv` |

---

## 📁 Project Structure

```
├── 2022 Q3 AAPL.pdf        # Source financial document (SEC 10-Q)
├── ingest.py               # PDF → FAISS index builder (run once)
├── rag_engine.py           # Retrieval + Gemini query logic
├── app.py                  # Streamlit chat UI
├── requirements.txt        # Python dependencies
├── .env                    # Your Gemini API key (not committed)
└── faiss_index/            # Auto-created after running ingest.py
    ├── index.faiss
    └── chunks.pkl
```

---

## ⚙️ Setup & Running

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** If you see a `torch` DLL error, install the  
> [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and retry.

### 4. Add your Gemini API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get a free API key at [https://aistudio.google.com](https://aistudio.google.com)

### 5. Build the FAISS index (run once)

```bash
python ingest.py
```

This will:
- Parse all 28 pages of the PDF
- Extract text blocks and tables separately
- Embed all chunks using `all-MiniLM-L6-v2`
- Save the FAISS index to `faiss_index/`

Expected output:
```
[OK] Extracted 185 chunks (31 tables, 154 text blocks)
[SAVE] FAISS index saved -> faiss_index/
   Vectors : 185 | Dimensions: 384
[DONE] Ingestion complete!
```

### 6. Launch the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 💬 Example Questions

- *What was Apple's net income in Q3 2022?*
- *What were the total net sales?*
- *What is the revenue breakdown by product category?*
- *What was the gross margin percentage?*
- *What were the earnings per share (EPS)?*
- *How did iPhone revenue compare to the previous year?*
- *What was the cash and cash equivalents balance?*

---

## 📌 Notes

- The FAISS index is built locally and persists across sessions — you only need to run `ingest.py` once unless the PDF changes.
- The system will refuse to answer questions not covered in the document rather than guess.
- The `torch.classes` warning in the Streamlit console is a known harmless PyTorch/Streamlit quirk.

---

## 🔒 .gitignore Recommendation

Make sure your `.env` file (which contains your API key) is **not** committed to GitHub. Add this to a `.gitignore` file:

```
.env
venv/
faiss_index/
__pycache__/
*.pyc
```
