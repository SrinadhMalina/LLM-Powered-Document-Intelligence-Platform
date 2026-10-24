# 📚 DocIntel Platform

**LLM-Powered Document Intelligence** — Upload PDFs and ask questions using RAG (Retrieval-Augmented Generation).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | [Groq](https://groq.com) — qwen/qwen3.8-27b | Free, ultra-fast inference |
| **Vector DB** | [Qdrant](https://qdrant.tech) | Production-grade, free local + cloud tier |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Better retrieval accuracy than MiniLM, free |
| **Orchestration** | LangChain | Industry-standard RAG framework |
| **API** | FastAPI | Production REST API with /docs |
| **UI** | Streamlit | Rapid ML app frontend |
| **Caching** | In-memory ResponseCache | Reduces LLM API calls |

---

## Running Locally

### 1. Clone and install
```bash
git clone https://github.com/YOUR_USERNAME/doc-intel-platform.git
cd doc-intel-platform
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Set your API key
Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

### 3. Run (two options)

**Option A — Single command (both servers):**
```bash
python run.py
```

**Option B — Manually in two terminals:**
```bash
# Terminal 1: FastAPI backend
python app/main.py
# → API docs at http://localhost:8000/docs

# Terminal 2: Streamlit frontend
streamlit run streamlit_app.py
# → App UI at http://localhost:8501
```

> **Note:** The Streamlit app works even without the FastAPI backend running — it automatically falls back to direct mode.

---

## Deploying to Streamlit Cloud (Free Public Hosting)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/doc-intel-platform.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository → Branch: `main` → Main file: `streamlit_app.py`
5. Click **"Advanced settings"** → Add secret:
   ```
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
6. Click **Deploy**

Your app will be live at: `https://YOUR_USERNAME-doc-intel-platform-streamlit-app-XXXX.streamlit.app`

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           Streamlit Frontend                │
│         (UI / Chat Interface)               │
└──────────────┬──────────────────────────────┘
               │ HTTP (when FastAPI is running)
               │ Direct calls (Streamlit Cloud)
               ▼
┌─────────────────────────────────────────────┐
│            FastAPI Backend                  │
│   POST /api/upload  POST /api/query         │
│   GET  /api/health  DELETE /api/document    │
└──────┬──────────────────────┬───────────────┘
       │                      │
       ▼                      ▼
┌─────────────┐    ┌─────────────────────────┐
│   Qdrant    │    │       Groq LLM          │
│ Vector DB   │    │  (qwen/llama fallback)  │
│ (local/cloud)│   │   + Response Cache      │
└─────────────┘    └─────────────────────────┘
```

## Features
- **Multi-document RAG** — index multiple PDFs, query across all of them
- **Source attribution** — every answer cites the source file and page number
- **Relevance scoring** — BGE embeddings with cosine similarity [0–1]
- **LLM fallback chain** — tries multiple Groq models automatically
- **Response caching** — identical questions served instantly, reducing API costs
- **REST API** — full FastAPI backend with Swagger docs at `/docs`
