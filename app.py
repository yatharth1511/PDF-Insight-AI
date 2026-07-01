"""
app.py  —  PDF Insight AI v2
Streamlit frontend with:
  • Dynamic AI component sidebar (LLM / Embeddings / VectorStore / Retrieval / Reranker)
  • Hybrid BM25 + FAISS retrieval with RRF fusion
  • BGE / CrossEncoder reranking with health checks
  • Query decomposition for compound questions
  • Citation-aware answers (filename + page number)
  • System status & performance metrics panel
  • Conversation history, Clear Chat, Download Chat

Run:
    streamlit run app.py

"""
import os
import sys
import time
import logging
import datetime
import streamlit as st
from dotenv import load_dotenv

# Logging – visible in terminal; Streamlit shows warnings/errors in UI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")

# Config

load_dotenv(encoding="utf-8")

from config.settings import (
    LLM_OPTIONS, EMBEDDING_OPTIONS, VECTORSTORE_OPTIONS,
    RETRIEVAL_OPTIONS, RERANKER_OPTIONS, PipelineConfig, llm_provider_from_model,
)

FAISS_INDEX_DIR = "faiss_index"

# Page setup

st.set_page_config(
    page_title="PDF Insight AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .citation-chip {
        display:inline-block; background:#EEF2FF; color:#4F6CF7;
        border:1px solid #C7D2FE; border-radius:6px;
        padding:2px 10px; font-size:.78rem; margin:2px 3px; font-weight:500;
    }
    .answer-card {
        background:#FAFBFF; border:1px solid #E2E6F0;
        border-left:4px solid #4F6CF7; border-radius:8px;
        padding:1rem 1.25rem; margin-top:.5rem;
    }
    .chat-user {
        background:#4F6CF7; color:white;
        border-radius:12px 12px 2px 12px;
        padding:.6rem 1rem; margin:.4rem 0;
        max-width:80%; margin-left:auto;
    }
    .chat-assistant {
        background:#F0F2FF; color:#1A1D2E;
        border-radius:12px 12px 12px 2px;
        padding:.6rem 1rem; margin:.4rem 0; max-width:85%;
    }
    .metric-card {
    border:1px solid #E2E6F0;
    border-radius:8px; padding:.6rem .9rem; margin:.3rem 0;
    }
    .health-ok   { color:#16a34a; font-weight:600; }
    .health-fail { color:#dc2626; font-weight:600; }
    .health-warn { color:#d97706; font-weight:600; }
    .label { font-size:.7rem; font-weight:700; letter-spacing:.08em;
             text-transform:uppercase; color:#6B7080; }
</style>
""", unsafe_allow_html=True)

# Session state

def _init():
    defaults = {
        "cfg":                PipelineConfig(),
        "vector_store":       None,
        "bm25":               None,
        "chunks":             [],
        "processed_files":    [],
        "short_summary":      "",
        "detailed_summary":   "",
        "conversation":       [],
        "last_metrics":       {},
        "component_health":   {},
        "llm":                None,
        "embed_model":        None,
        "reranker":           None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
cfg: PipelineConfig = st.session_state.cfg

# Sidebar — Dynamic AI Configuration

with st.sidebar:
    st.title("⚙️ AI Configuration")

    # API Keys 
    with st.expander("🔑 API Keys", expanded=True):
        google_key    = st.text_input("Google AI Studio Key",  type="password",
                                       value=os.getenv("GOOGLE_API_KEY", ""))
        openai_key    = st.text_input("OpenAI API Key",        type="password",
                                       value=os.getenv("OPENAI_API_KEY", ""))
        anthropic_key = st.text_input("Anthropic API Key",     type="password",
                                       value=os.getenv("ANTHROPIC_API_KEY", ""))

    API_KEYS = {"google": google_key, "openai": openai_key, "anthropic": anthropic_key}

    st.divider()

    # LLM 
    st.markdown("<div class='label'>🤖 Language Model</div>", unsafe_allow_html=True)
    llm_display = st.selectbox("LLM", list(LLM_OPTIONS.keys()),
                                index=list(LLM_OPTIONS.keys()).index(cfg.llm_display),
                                label_visibility="collapsed")
    cfg.llm_display  = llm_display
    cfg.llm_model    = LLM_OPTIONS[llm_display]
    cfg.llm_provider = llm_provider_from_model(cfg.llm_model)

    # Embeddings
    st.markdown("<div class='label'>🔍 Embedding Model</div>", unsafe_allow_html=True)
    emb_display = st.selectbox("Embedding", list(EMBEDDING_OPTIONS.keys()),
                                index=list(EMBEDDING_OPTIONS.keys()).index(cfg.embedding_display),
                                label_visibility="collapsed")
    emb_changed = emb_display != cfg.embedding_display
    cfg.embedding_display = emb_display
    cfg.embedding_model   = EMBEDDING_OPTIONS[emb_display]

    # Vector Store
    st.markdown("<div class='label'>📦 Vector Store</div>", unsafe_allow_html=True)
    vs_display = st.selectbox("VectorStore", list(VECTORSTORE_OPTIONS.keys()),
                               index=list(VECTORSTORE_OPTIONS.keys()).index(cfg.vectorstore_display),
                               label_visibility="collapsed")
    cfg.vectorstore_display = vs_display
    cfg.vectorstore_type    = VECTORSTORE_OPTIONS[vs_display]

    # Retrieval
    st.markdown("<div class='label'>🔀 Retrieval Strategy</div>", unsafe_allow_html=True)
    ret_display = st.selectbox("Retrieval", list(RETRIEVAL_OPTIONS.keys()),
                                index=list(RETRIEVAL_OPTIONS.keys()).index(cfg.retrieval_display),
                                label_visibility="collapsed")
    cfg.retrieval_display = ret_display
    cfg.retrieval_method  = RETRIEVAL_OPTIONS[ret_display]

    # Reranker 
    st.markdown("<div class='label'>🏅 Reranker</div>", unsafe_allow_html=True)
    rrk_display = st.selectbox("Reranker", list(RERANKER_OPTIONS.keys()),
                                index=list(RERANKER_OPTIONS.keys()).index(cfg.reranker_display),
                                label_visibility="collapsed")
    cfg.reranker_display = rrk_display
    cfg.reranker_model   = RERANKER_OPTIONS[rrk_display]

    # Chunking 
    st.divider()
    with st.expander("✂️ Chunking Parameters"):
        cfg.chunk_size    = st.slider("Chunk size",    500, 2000, cfg.chunk_size,    50)
        cfg.chunk_overlap = st.slider("Chunk overlap", 50,  500,  cfg.chunk_overlap, 25)
        cfg.hybrid_fetch_k = st.slider("Retrieve top N (pre-rerank)", 10, 50, cfg.hybrid_fetch_k, 5)
        cfg.rerank_top_n   = st.slider("Keep top N (post-rerank)",     3,  15, cfg.rerank_top_n,  1)

    # System Status
    st.divider()
    st.markdown("### 📊 System Status")

    health = st.session_state.component_health
    def _badge(ok: bool | None) -> str:
        if ok is True:  return "<span class='health-ok'>✅ Loaded</span>"
        if ok is False: return "<span class='health-fail'>❌ Failed</span>"
        return "<span class='health-warn'>⏸ Not loaded</span>"

    st.markdown(f"""
    | Component | Status |
    |-----------|--------|
    | LLM | {_badge(health.get('llm'))} |
    | Embeddings | {_badge(health.get('embeddings'))} |
    | Vector Store | {_badge(health.get('vectorstore'))} |
    | BM25 | {_badge(health.get('bm25'))} |
    | Reranker | {_badge(health.get('reranker'))} |
    """, unsafe_allow_html=True)

    # Last metrics
    m = st.session_state.last_metrics
    if m:
        st.divider()
        st.markdown("### ⏱ Last Query Metrics")
        st.markdown(f"""
        <div class='metric-card'>
        🔍 Retrieval: <b>{m.get('retrieval_ms',0):.0f} ms</b><br>
        🏅 Reranking: <b>{m.get('rerank_ms',0):.0f} ms</b><br>
        🤖 Generation: <b>{m.get('gen_ms',0):.0f} ms</b><br>
        ⏱ Total: <b>{m.get('total_ms',0):.0f} ms</b><br>
        📦 Candidates: <b>{m.get('candidates',0)}</b><br>
        ✅ Used: <b>{m.get('final_chunks',0)}</b>
        </div>
        """, unsafe_allow_html=True)

# Header

st.title("📄 PDF Insight AI")
st.markdown(
    "Upload PDFs → Process → Ask questions. "
    "Answers cite the exact **filename and page number** they came from."
)
st.divider()

# Helper: build LLM, embedding model, reranker from current config

def _get_llm():
    from llms.factory import get_llm
    try:
        llm = get_llm(cfg.llm_model, API_KEYS)
        st.session_state.component_health["llm"] = True
        return llm
    except Exception as e:
        st.session_state.component_health["llm"] = False
        st.error(f"LLM load failed: {e}")
        return None

def _get_embed_model():
    from embeddings.factory import get_embedding_model
    try:
        model = get_embedding_model(cfg.embedding_model)
        st.session_state.component_health["embeddings"] = True
        return model
    except Exception as e:
        st.session_state.component_health["embeddings"] = False
        st.error(f"Embedding model load failed: {e}")
        return None

def _get_reranker():
    from rerankers.factory import get_reranker
    r = get_reranker(cfg.reranker_model)
    st.session_state.component_health["reranker"] = r.healthy
    if not r.healthy and cfg.reranker_model != "none":
        st.warning(
            f"⚠️ Reranker **{cfg.reranker_model}** failed to load. "
            "Using passthrough ranking. Check logs for details."
        )
    return r

# Section 1 — Upload

st.header("1 · Upload PDFs")

uploaded_files = st.file_uploader(
    "Drag & drop or click to upload",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    st.success(f"**{len(uploaded_files)} file(s) ready:** {', '.join(f.name for f in uploaded_files)}")

    if emb_changed and st.session_state.vector_store is not None:
        st.warning(
            "⚠️ Embedding model changed. Re-process documents to rebuild the index "
            "with the new model."
        )

st.divider()

# Section 2 — Process

st.header("2 · Process Documents")
process_btn = st.button("⚡ Process Documents", type="primary")

if process_btn:
    if not uploaded_files:
        st.warning("Upload at least one PDF first.")
    else:
        from utils.pdf_processor import extract_text_from_multiple_pdfs, chunk_pages
        from vectorstores.factory import get_vector_store
        from retrievers.hybrid import build_bm25, save_bm25
        from embeddings.factory import embed_texts

        # Extract
        with st.spinner("Extracting text…"):
            pages = extract_text_from_multiple_pdfs(uploaded_files)
        if not pages:
            st.error("No text extracted. Scanned-image PDFs are not supported without OCR.")
            st.stop()

        # Chunk
        with st.spinner(f"Chunking (size={cfg.chunk_size}, overlap={cfg.chunk_overlap})…"):
            chunks = chunk_pages(pages, cfg.chunk_size, cfg.chunk_overlap)

        # Embed
        embed_model = _get_embed_model()
        if not embed_model:
            st.stop()
        with st.spinner(f"Embedding {len(chunks)} chunks with {cfg.embedding_model}…"):
            embeddings = embed_texts(embed_model, [c["text"] for c in chunks])
        st.session_state.embed_model = embed_model

        # Build vector store
        with st.spinner(f"Building {cfg.vectorstore_type.upper()} index…"):
            vs = get_vector_store(cfg.vectorstore_type, FAISS_INDEX_DIR)
            vs.build(chunks, embeddings)
            vs.save(FAISS_INDEX_DIR)
            st.session_state.vector_store = vs
            st.session_state.component_health["vectorstore"] = True

        # Build BM25
        with st.spinner("Building BM25 index…"):
            bm25 = build_bm25(chunks)
            save_bm25(bm25, FAISS_INDEX_DIR)
            st.session_state.bm25 = bm25
            st.session_state.component_health["bm25"] = True

        st.session_state.chunks           = chunks
        st.session_state.processed_files  = list({c["filename"] for c in chunks})

        # Summarize
        llm = _get_llm()
        if llm:
            from utils.summarizer import generate_summaries
            with st.spinner("Generating summaries…"):
                short_s, detailed_s = generate_summaries(chunks, llm)
                st.session_state.short_summary    = short_s
                st.session_state.detailed_summary = detailed_s

        st.success(
            f"✅ **{len(chunks)} chunks** indexed from "
            f"**{len(st.session_state.processed_files)} file(s)**."
        )
        logger.info(
            f"Processing complete: {len(chunks)} chunks, "
            f"files={st.session_state.processed_files}"
        )

st.divider()

# Section 3 — Summaries

st.header("3 · Document Summaries")

if st.session_state.short_summary:
    with st.expander("📝 Short Summary", expanded=True):
        st.write(st.session_state.short_summary)
    with st.expander("📋 Detailed Summary"):
        st.markdown(st.session_state.detailed_summary)
else:
    st.info("Process your documents above to generate summaries.")

st.divider()

# Section 4 — Q&A

st.header("4 · Ask Questions")

# Chat controls
c1, c2 = st.columns(2)
with c1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

with c2:
    if st.session_state.conversation:
        lines = []
        for t in st.session_state.conversation:
            role = "You" if t["role"] == "user" else "AI"
            lines.append(f"[{role}]\n{t['content']}\n")
            if t.get("citations"):
                lines += [f"  • {c.get('filename','?')} p.{c.get('page_number','?')}\n"
                          for c in t["citations"]]
            lines.append("")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Download Chat",
            data="".join(lines),
            file_name=f"pdf_insight_{ts}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.button("⬇️ Download Chat", disabled=True, use_container_width=True)

st.markdown("---")

# Render history
if st.session_state.conversation:
    st.markdown("<div class='label'>Conversation</div>", unsafe_allow_html=True)
    for turn in st.session_state.conversation:
        if turn["role"] == "user":
            st.markdown(f"<div class='chat-user'>🧑 {turn['content']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-assistant'>🤖 {turn['content']}</div>",
                        unsafe_allow_html=True)
            cits = turn.get("citations", [])
            if cits:
                seen, html = set(), ""
                for c in cits:
                    key = (c.get("filename", "?"), c.get("page_number", "?"))
                    if key not in seen:
                        seen.add(key)
                        html += f"<span class='citation-chip'>📄 {key[0]} · p.{key[1]}</span>"
                st.markdown(f"<div style='margin-top:4px'>{html}</div>",
                            unsafe_allow_html=True)
    st.markdown("---")

# Question input

if "input_counter" not in st.session_state:
    st.session_state.input_counter = 0

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What are the eligibility criteria and required documents?",
    key=f"q_input_{st.session_state.input_counter}",
    label_visibility="collapsed",
)
ask_btn = st.button("Ask →", type="primary")

# Answer pipeline

if ask_btn and question.strip():
    if not st.session_state.vector_store:
        st.warning("Process your documents first (Section 2).")
    else:
        from retrievers.hybrid import Retriever
        from utils.qa_engine   import answer_question

        total_start = time.perf_counter()

        # Load components
        llm         = _get_llm()
        embed_model = st.session_state.embed_model or _get_embed_model()
        reranker    = _get_reranker()

        if not llm or not embed_model:
            st.stop()

        vs   = st.session_state.vector_store
        bm25 = st.session_state.bm25
        ret  = Retriever(vs, embed_model, bm25)

        # Retrieval
        t0 = time.perf_counter()
        with st.spinner(f"Retrieving via {cfg.retrieval_method}…"):
            candidates = ret.retrieve(
                question,
                method=cfg.retrieval_method,
                k=cfg.hybrid_fetch_k,
                llm=llm if cfg.retrieval_method == "multiquery" else None,
            )
        retrieval_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Retrieval: {len(candidates)} candidates in {retrieval_ms:.0f}ms")

        # Reranking
        t1 = time.perf_counter()
        with st.spinner(f"Reranking with {cfg.reranker_model}…"):
            top_chunks = reranker.rerank(question, candidates, cfg.rerank_top_n)
        rerank_ms = (time.perf_counter() - t1) * 1000
        logger.info(f"Reranking: {len(top_chunks)} chunks in {rerank_ms:.0f}ms")

        # Generation
        history = [{"role": t["role"], "content": t["content"]}
                   for t in st.session_state.conversation]
        t2 = time.perf_counter()
        with st.spinner("Generating answer…"):
            answer, cited = answer_question(question, top_chunks, history, llm)
        gen_ms = (time.perf_counter() - t2) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000

        # Save metrics
        st.session_state.last_metrics = {
            "retrieval_ms": retrieval_ms,
            "rerank_ms":    rerank_ms,
            "gen_ms":       gen_ms,
            "total_ms":     total_ms,
            "candidates":   len(candidates),
            "final_chunks": len(top_chunks),
        }
        logger.info(
            f"Query complete | retrieval={retrieval_ms:.0f}ms "
            f"rerank={rerank_ms:.0f}ms gen={gen_ms:.0f}ms total={total_ms:.0f}ms"
        )

        # Save to conversation
        st.session_state.conversation.append({"role": "user",      "content": question})
        st.session_state.conversation.append({"role": "assistant",  "content": answer, "citations": cited})

        # Display
        st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
        st.markdown(answer)
        st.markdown("</div>", unsafe_allow_html=True)

        if cited:
            with st.expander(f"📚 {len(cited)} source(s) cited", expanded=True):
                for i, c in enumerate(cited, 1):
                    score = c.get("rerank_score")
                    score_str = f" · confidence {score:.2f}" if score is not None else ""
                    st.markdown(f"**[{i}] {c.get('filename','?')} — Page {c.get('page_number','?')}**{score_str}")
                    st.caption(c["text"][:450] + ("…" if len(c["text"]) > 450 else ""))
                    if i < len(cited):
                        st.markdown("<hr style='margin:.3rem 0;border-color:#E2E6F0'>",
                                    unsafe_allow_html=True)

        # Quick metrics bar under answer
        col_m = st.columns(4)
        col_m[0].metric("Retrieval", f"{retrieval_ms:.0f} ms")
        col_m[1].metric("Reranking", f"{rerank_ms:.0f} ms")
        col_m[2].metric("Generation", f"{gen_ms:.0f} ms")
        col_m[3].metric("Total", f"{total_ms:.0f} ms")

        st.session_state.input_counter += 1
        st.rerun()

elif ask_btn and not question.strip():
    st.warning("Please enter a question.")

# Footer
st.markdown(
    "<br><div style='text-align:center;color:#6B7080;font-size:.8rem'>"
    "PDF Insight AI v2 · Hybrid RAG · BGE Reranker · Multi-query decomposition · Citation-aware"
    "</div>",
    unsafe_allow_html=True,
)
