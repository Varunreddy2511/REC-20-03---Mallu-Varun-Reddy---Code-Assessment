"""
app.py — Streamlit Chat UI for AAPL Q3 2022 Financial RAG
==========================================================
Run:  streamlit run app.py
"""

import streamlit as st
from rag_engine import get_engine

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Apple Q3 2022 | Financial RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Global Reset ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #111827;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
  }
  section[data-testid="stSidebar"] .stMarkdown p {
    color: #374151;
    font-size: 0.85rem;
  }
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3 {
    color: #111827 !important;
  }

  /* ── Main area ── */
  .main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 860px;
    background-color: #ffffff;
  }

  /* ── Header ── */
  .header-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 24px;
    background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(37,99,235,0.08);
  }
  .header-icon {
    font-size: 2.4rem;
    line-height: 1;
  }
  .header-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1e3a5f;
    margin: 0;
  }
  .header-sub {
    font-size: 0.82rem;
    color: #4b6a8a;
    margin: 2px 0 0 0;
  }
  .badge {
    margin-left: auto;
    padding: 4px 12px;
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.5px;
  }

  /* ── Chat messages ── */
  .user-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
  }
  .user-bubble .bubble {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: #ffffff;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    max-width: 78%;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 2px 12px rgba(37,99,235,0.25);
  }

  .assistant-bubble {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
    gap: 10px;
    align-items: flex-start;
  }
  .avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
  }
  .assistant-bubble .bubble {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #111827;
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 82%;
    font-size: 0.92rem;
    line-height: 1.7;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  }
  .refused-bubble .bubble {
    border-color: #fbbf24;
    background: #fffbeb;
    color: #78350f;
  }

  /* ── Source chips ── */
  .source-chip {
    display: inline-block;
    padding: 3px 10px;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    font-size: 0.72rem;
    color: #475569;
    margin: 3px 3px 0 0;
    font-family: 'JetBrains Mono', monospace;
  }
  .source-chip.table {
    border-color: #93c5fd;
    color: #1d4ed8;
    background: #eff6ff;
  }
  .score-bar-wrap {
    margin-top: 6px;
  }
  .score-label {
    font-size: 0.7rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Confidence badge ── */
  .conf-high  { color: #16a34a; font-weight: 600; }
  .conf-med   { color: #d97706; font-weight: 600; }
  .conf-low   { color: #dc2626; font-weight: 600; }

  /* ── Input ── */
  .stChatInput textarea {
    background: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    color: #111827 !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
  }

  /* ── Divider ── */
  .chat-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 8px 0;
  }

  /* ── Expander ── */
  details summary {
    color: #475569 !important;
    font-size: 0.8rem !important;
  }
  details[open] summary { color: #1e3a5f !important; }

  /* ── Scrollable chunk display ── */
  .chunk-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #374151;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 160px;
    overflow-y: auto;
    margin-top: 6px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Document Info")
    st.markdown("""
**Document:** Apple Inc. — Form 10-Q  
**Period:** Q3 Fiscal 2022 (Jun 25, 2022)  
**Source:** SEC Filing  

""")
    st.divider()
    st.markdown("## 💡 Sample Questions")
    sample_qs = [
        "What was Apple's net income in Q3 2022?",
        "What were the total net sales?",
        "What is the revenue breakdown by product category?",
        "What was the gross margin percentage?",
        "What were the earnings per share (EPS)?",
        "How did iPhone revenue compare to the previous year?",
        "What are the main risk factors mentioned?",
        "What was the cash and cash equivalents balance?",
    ]
    for q in sample_qs:
        st.markdown(f"• {q}")
    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Load Engine ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    return get_engine()

try:
    engine = load_engine()
except (ValueError, FileNotFoundError) as e:
    st.error(str(e))
    st.stop()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="header-icon">📈</div>
  <div>
    <p class="header-title">Apple Q3 2022 — Financial Intelligence</p>
    <p class="header-sub">Grounded RAG </p>
  </div>
  <div class="badge">AAPL · 10-Q · SEC Filing</div>
</div>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hello! I'm your Apple Q3 2022 financial analyst.\n\n"
            "I can answer questions about:\n"
            "• Revenue, net income, earnings per share\n"
            "• Product segment breakdown (iPhone, Mac, iPad, Wearables, Services)\n"
            "• Gross margins, operating expenses, balance sheet\n"
            "• Risk factors, geographic segments, cash flow\n\n"
            "**Important:** I only answer from the actual document. "
            "If the information isn't in the filing, I'll tell you clearly."
        ),
        "sources": [],
        "refused": False,
    })


# ── Render Chat History ───────────────────────────────────────────────────────
def score_class(score: float) -> str:
    if score >= 0.55:
        return "conf-high"
    elif score >= 0.38:
        return "conf-med"
    return "conf-low"


def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(
            f'<div class="user-bubble"><div class="bubble">{content}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        refused = msg.get("refused", False)
        extra_cls = "refused-bubble" if refused else ""
        st.markdown(
            f'<div class="assistant-bubble {extra_cls}">'
            f'<div class="avatar">📊</div>'
            f'<div class="bubble">{content.replace(chr(10), "<br>")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        sources = msg.get("sources", [])
        if sources:
            with st.expander(f"📎 {len(sources)} source chunks retrieved", expanded=False):
                for i, src in enumerate(sources, 1):
                    chip_cls = "table" if src["type"] == "table" else ""
                    sc = src["score"]
                    st.markdown(
                        f'<span class="source-chip {chip_cls}">'
                        f'{"📋 TABLE" if chip_cls else "📄 TEXT"} · {src["source"]}'
                        f'</span>'
                        f'<span class="source-chip">'
                        f'<span class="{score_class(sc)}">similarity: {sc:.3f}</span>'
                        f'</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="chunk-box">{src["text"][:600]}'
                        f'{"..." if len(src["text"]) > 600 else ""}</div>',
                        unsafe_allow_html=True,
                    )
                    if i < len(sources):
                        st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)


for msg in st.session_state.messages:
    render_message(msg)


# ── Chat Input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about Apple's Q3 2022 financials..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message({"role": "user", "content": prompt})

    # Get RAG answer
    with st.spinner("🔍 Searching document & reasoning..."):
        result = engine.answer(prompt)

    # Add assistant response
    assistant_msg = {
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "refused": result["refused"],
    }
    st.session_state.messages.append(assistant_msg)
    render_message(assistant_msg)
    st.rerun()
