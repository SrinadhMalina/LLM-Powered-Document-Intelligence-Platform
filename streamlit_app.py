"""
DocIntel Platform — Streamlit Frontend

Architecture:
  • Tries FastAPI backend first (http://localhost:8000) — shows the REST API layer.
  • Falls back to direct Python service calls if backend is offline
    (works on Streamlit Cloud / single-process hosting).
"""

import streamlit as st
import requests
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocIntel Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:8000/api"
REQUEST_TIMEOUT = 60  # seconds

# ── Detect backend mode ────────────────────────────────────────────────────────
def backend_online() -> bool:
    try:
        r = requests.get("http://localhost:8000/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

USE_API = backend_online()

# ── Load services (only in direct mode) ───────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def get_services():
    from app.services.ingestion import IngestionService
    from app.services.query_engine import QueryEngine
    from app.services.vector_store import VectorStoreService
    from app.core.cache import ResponseCache
    return IngestionService(), QueryEngine(), VectorStoreService(), ResponseCache()

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero h1  { font-size: 2.4rem; font-weight: 700; margin-bottom: 0; }
.hero p   { color: #888; font-size: 1.05rem; margin-top: 0.2rem; }

.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
}
.badge-green { background:#0d3321; color:#4ade80; }
.badge-blue  { background:#0c1a3a; color:#60a5fa; }
.badge-amber { background:#2d1a00; color:#fbbf24; }

.source-card {
    background: #1a1f2e; border: 1px solid #2d3748; border-radius: 8px;
    padding: 10px 14px; margin: 4px 0; font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div class="hero">
        <h1>📚 DocIntel Platform</h1>
        <p>LLM-powered document search & question answering — RAG with source attribution</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    mode_label = "FastAPI + Streamlit" if USE_API else "Streamlit (direct)"
    mode_color = "badge-green" if USE_API else "badge-amber"
    st.markdown(f"""
    <div style="text-align:right; padding-top:1.2rem;">
        <span class="badge {mode_color}">{mode_label}</span><br>
        <span class="badge badge-blue" style="margin-top:4px;">Groq · Qdrant · LangChain</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Manager")

    # API status pill
    if USE_API:
        try:
            info = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
            vc   = info.get("vector_store", {}).get("vectors_count", "?")
            st.success(f"🟢 FastAPI online · {vc} vectors indexed")
        except Exception:
            st.warning("🟡 FastAPI running but health check failed")
    else:
        st.info("🟠 Running in direct mode (no FastAPI backend)")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        if st.button("⚡ Index Document", use_container_width=True, type="primary"):
            with st.spinner(f"Processing **{uploaded_file.name}**…"):
                try:
                    if USE_API:
                        r = requests.post(
                            f"{BACKEND_URL}/upload",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                            timeout=REQUEST_TIMEOUT,
                        )
                        if r.status_code == 200:
                            d = r.json()
                            st.success(f"✅ {d['chunks']} chunks indexed!")
                        else:
                            st.error(f"❌ {r.text}")
                    else:
                        ingestion, _, vector_store, _ = get_services()
                        os.makedirs("data/uploads", exist_ok=True)
                        path = f"data/uploads/{uploaded_file.name}"
                        with open(path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        vectors = ingestion.process_document(path)
                        vector_store.upsert_vectors(vectors)
                        st.success(f"✅ {len(vectors)} chunks indexed!")
                except Exception as e:
                    st.error(f"❌ {e}")

    st.markdown("---")
    st.markdown("**Stack**")
    st.caption("🤖 LLM: Groq (llama-3.3-70b)")
    st.caption("🗄️ Vector DB: Qdrant")
    st.caption("📐 Embeddings: BGE-small-en")
    st.caption("⚡ API: FastAPI / LangChain")
    st.caption("🔒 Cache: In-memory (API cost saver)")

    if USE_API:
        st.markdown("---")
        st.markdown(f"**API Docs** → [localhost:8000/docs](http://localhost:8000/docs)")

# ── Chat interface ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    pct = f"{s['score']*100:.1f}%" if s.get("score") else "—"
                    st.markdown(
                        f'<div class="source-card">📄 <b>{s["source"]}</b> &nbsp;·&nbsp; '
                        f'Page {s["page"]+1} &nbsp;·&nbsp; Relevance: <b>{pct}</b></div>',
                        unsafe_allow_html=True,
                    )
        if msg.get("cached"):
            st.caption("⚡ Served from cache")

# New message
if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            try:
                if USE_API:
                    r = requests.post(
                        f"{BACKEND_URL}/query",
                        params={"question": prompt},
                        timeout=REQUEST_TIMEOUT,
                    )
                    if r.status_code == 200:
                        data = r.json()
                    else:
                        st.error(f"API error: {r.text}")
                        st.stop()
                else:
                    _, engine, _, cache_svc = get_services()
                    cached = cache_svc.get(prompt)
                    if cached:
                        data = {**cached, "cached": True}
                    else:
                        data = engine.answer_question(prompt)
                        cache_svc.set(prompt, data)
                        data["cached"] = False

                answer  = data["answer"]
                sources = data.get("sources", [])
                cached  = data.get("cached", False)

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Sources"):
                        for s in sources:
                            pct = f"{s['score']*100:.1f}%" if s.get("score") else "—"
                            st.markdown(
                                f'<div class="source-card">📄 <b>{s["source"]}</b> &nbsp;·&nbsp; '
                                f'Page {s["page"]+1} &nbsp;·&nbsp; Relevance: <b>{pct}</b></div>',
                                unsafe_allow_html=True,
                            )
                if cached:
                    st.caption("⚡ Served from cache")

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sources": sources, "cached": cached,
                })

            except requests.exceptions.ConnectionError:
                st.error("❌ Lost connection to FastAPI. Refresh to switch to direct mode.")
            except Exception as e:
                st.error(f"❌ {e}")
