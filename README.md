# 📄 PDF Insight AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-00ADD8?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade RAG application that lets you upload PDFs and ask questions about them — with every answer citing the exact filename and page number it came from.**

[Features](#-features) • [Pipeline](#-how-it-works) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#-configuration) • [Project Structure](#-project-structure)

---

![PDF Insight AI Demo](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

</div>

---

## 🧠 What is PDF Insight AI?

PDF Insight AI is a locally running web application built with Streamlit that transforms your PDF documents into an intelligent, conversational knowledge base. Instead of manually searching through pages, you simply ask questions in plain English and get precise, cited answers in seconds.

What separates this from simply pasting a document into a chatbot is the underlying architecture. It uses **Retrieval Augmented Generation (RAG)** — a technique that finds the most relevant pieces of your document first, then hands only those pieces to the AI. The result is more accurate answers, verifiable citations, and far better handling of long documents.

---

## ✨ Features

### 📁 Multi-PDF Support
- Upload multiple PDFs simultaneously with drag and drop
- All documents are processed into a single unified knowledge base
- Questions can pull answers from across all uploaded files at once

### 🔍 Hybrid Retrieval (BM25 + FAISS)
- **BM25** handles exact keyword matching — great for names, numbers, and technical terms
- **FAISS** handles semantic search — finds relevant content even when you use different words
- Both results are fused using **Reciprocal Rank Fusion (RRF)** for best-of-both-worlds accuracy

### 🏅 BGE Cross-Encoder Reranking
- After retrieval, a cross-encoder model re-scores every candidate chunk by reading the question and chunk together
- Far more precise than embedding similarity alone
- Supports BGE Base, BGE Large, MiniLM Cross-Encoder, and MXBAI Reranker

### 🧩 Query Decomposition
- Compound questions like *"What are the eligibility criteria and required documents?"* are automatically split into sub-questions
- Each sub-question is retrieved independently and results are merged
- Handles multi-part questions that would trip up a naive RAG system

### 📌 Citation-Aware Answers
- Every factual claim is cited inline as `[SOURCE N]`
- Each source shows the exact **filename** and **page number**
- Expandable source panel shows the actual chunk text with confidence scores
- Citation chips appear below every answer for quick reference

### 📝 Document Summaries
- **Short Summary** — 5 to 10 sentence overview generated automatically after processing
- **Detailed Summary** — structured bullet points grouped under Main Topics, Key Points, Important Details, and Conclusions

### 💬 Conversation History
- Full multi-turn conversation maintained across questions
- Prior context is included with every new query
- **Clear Chat** button to start fresh
- **Download Chat** button to export the full conversation as a text file

### ⚙️ Dynamic AI Configuration Sidebar
Switch every component from the UI without touching any code:

| Component | Options |
|-----------|---------|
| LLM | Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 2.5 Pro, GPT-4o, Claude Sonnet |
| Embeddings | MiniLM, BGE Small, BGE Base, BGE Large |
| Vector Store | FAISS, ChromaDB |
| Retrieval | Hybrid, Similarity, MMR, Multi-Query |
| Reranker | None, BGE Base, BGE Large, MiniLM Cross-Encoder, MXBAI |

### 📊 Performance Metrics
After every query the app shows:
- Retrieval time
- Reranking time
- Generation time
- Total response time
- Number of candidates retrieved
- Number of chunks sent to the LLM

### 🏥 Component Health Checks
- Every component shows a live health badge (Loaded / Failed / Not loaded)
- Reranker failures are logged with full diagnostics instead of silently degrading
- Clear error messages guide you to the fix

---

## 🔄 How It Works

### Indexing Phase
When you upload PDFs and click **Process Documents**:

```
PDF Upload → Text Extraction (PyMuPDF) → Chunking (1000 chars, 200 overlap)
    → Embedding (Sentence Transformers) → FAISS Index + BM25 Index → Saved to Disk
    → Summary Generation
```

### Query Phase
When you ask a question:

```
Question → Query Decomposition (if multi-part)
    → Hybrid Retrieval: BM25 + FAISS → RRF Fusion → Top 20 Candidates
    → BGE Reranker → Top 6 Chunks
    → LLM with Citation Prompt → Answer with [SOURCE N] inline
    → Citation Chips + Source Panel + Performance Metrics
```

---

## 🛠 Installation

### Prerequisites
- Python 3.9 or higher
- A Google AI Studio API key — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/yatharth1511/PDF-Insight-AI.git
cd PDF-Insight-AI
```

**2. Create a virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up your API key**
```bash
cp .env.example .env
```
Open `.env` and add your key:
```
GOOGLE_API_KEY=your_google_ai_studio_key_here
```

**5. Run the app**
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## 🚀 Usage

1. **Upload PDFs** — drag and drop one or more PDF files into Section 1
2. **Process** — click **Process Documents** and wait for the spinners to complete
3. **Read Summaries** — short and detailed summaries appear automatically in Section 3
4. **Ask Questions** — type any question in Section 4 and click **Ask**
5. **Check Citations** — every answer shows which file and page each claim came from
6. **Switch Models** — use the sidebar to change LLM, embedding model, or retrieval strategy at any time

---

## ⚙️ Configuration

All configuration lives in `config/settings.py`. The sidebar dropdowns read from this file, so adding a new provider means adding one entry here and implementing the adapter.

Key parameters you can tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Chunk size | 1000 | Characters per chunk |
| Chunk overlap | 200 | Shared characters between chunks |
| Hybrid fetch K | 20 | Candidates retrieved before reranking |
| Rerank top N | 6 | Chunks sent to the LLM after reranking |
| Dense weight | 0.6 | FAISS score weight in RRF fusion |
| Sparse weight | 0.4 | BM25 score weight in RRF fusion |

---

## 📁 Project Structure

```
PDF-Insight-AI/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
│
├── config/
│   └── settings.py               # Central component registry and defaults
│
├── llms/
│   ├── base.py                   # Abstract LLM interface
│   ├── gemini_llm.py             # Google Gemini adapter
│   ├── openai_llm.py             # OpenAI adapter
│   ├── anthropic_llm.py          # Anthropic Claude adapter
│   └── factory.py                # Returns correct LLM from config
│
├── embeddings/
│   └── factory.py                # Loads and caches embedding models
│
├── vectorstores/
│   ├── base.py                   # Abstract vector store interface
│   ├── faiss_store.py            # FAISS implementation with MMR
│   ├── chroma_store.py           # ChromaDB implementation
│   └── factory.py                # Returns correct store from config
│
├── retrievers/
│   └── hybrid.py                 # BM25, FAISS, Hybrid RRF, Multi-Query
│
├── rerankers/
│   ├── base.py                   # BGE, CrossEncoder, None implementations
│   └── factory.py                # Returns correct reranker with health checks
│
└── utils/
    ├── pdf_processor.py          # PDF extraction and chunking
    ├── summarizer.py             # Short and detailed summary generation
    └── qa_engine.py              # Citation-aware answer generation
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | Streamlit |
| PDF Processing | PyMuPDF (fitz) |
| Text Chunking | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Dense Index | FAISS |
| Sparse Index | BM25 (rank-bm25) |
| Reranking | FlagEmbedding (BGE), Sentence Transformers (CrossEncoder) |
| LLM — Default | Google Gemini 2.0 Flash |
| LLM — Optional | OpenAI GPT-4o, Anthropic Claude Sonnet |
| Vector DB Alt | ChromaDB |
| Environment | python-dotenv |

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | Yes (default) | Google AI Studio API key |
| `OPENAI_API_KEY` | Optional | Required only if using GPT-4o |
| `ANTHROPIC_API_KEY` | Optional | Required only if using Claude Sonnet |

---

## 🐛 Common Issues

**`ModuleNotFoundError: langchain.text_splitter`**
```bash
pip install langchain-text-splitters
```

**Gemini 404 model not found**

Run this to see your available models and update `LLM_OPTIONS` in `config/settings.py`:
```bash
python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); [print(m.name) for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]"
```

**UnicodeDecodeError on startup**

Your `.env` file was saved with wrong encoding by Windows Notepad. Recreate it:
```bash
echo GOOGLE_API_KEY=your-key-here > .env
```

**BGE reranker not loading**
```bash
pip install FlagEmbedding
```
The app continues working without it using passthrough ranking. Check the terminal for the full error.

**No text extracted from PDF**

The app works on text-based PDFs only. Scanned image PDFs require an OCR layer which is not included.

---

## 🗺 Roadmap

- [ ] OCR support for scanned PDFs
- [ ] Hugging Face Spaces deployment
- [ ] Qdrant and Pinecone vector store backends
- [ ] Table and figure extraction
- [ ] Multi-language document support
- [ ] Export answers as PDF report

---

## 👤 Author

**Yatharth Sharma**

---

<div align="center">

Built with Python, Streamlit, and Google Gemini

⭐ Star this repo if you found it useful

</div>
