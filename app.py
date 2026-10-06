#LabMate: An AI-Powered Interactive Laboratory Assistant
from __future__ import annotations

# ---- Make ChromaDB work on hosts with an old system SQLite (e.g. Streamlit Cloud) ----
# Harmless locally: if pysqlite3 isn't installed this is simply skipped.
try:
    import sys
    import pysqlite3  # type: ignore  # noqa: F401
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import hashlib
import io
import math
import os
import random
import re
import time
import traceback
import uuid
from typing import Optional

import chromadb
import pandas as pd
import streamlit as st
from chromadb.config import Settings
from dotenv import load_dotenv
from pypdf import PdfReader

try:  # Ollama is optional (not needed on a Gemini-only deployment)
    import ollama
except Exception:  # pragma: no cover
    ollama = None  # type: ignore

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ---------- Hugging Face Local Cache ----------
# Store downloaded models inside the LabMate project
# instead of the inaccessible Windows user cache.
try:
    HF_CACHE_DIR = os.path.join(PROJECT_ROOT, "hf_cache")
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
except Exception:
    pass

st.set_page_config(page_title="LabMate", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join(PROJECT_ROOT, "chroma_db"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_SECTION_CHARS = 6000
LEVELS = ["Easy", "Medium", "Hard"]

# Gemini settings (used when GEMINI_API_KEY is available)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_FALLBACKS = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3.5-flash",
                    "gemini-3-flash-preview", "gemini-2.5-flash"]
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

SUBJECTS = {
    "SQL / DBMS": {"kind": "sql", "lang": "sql"},
    "Python Programming": {"kind": "code", "lang": "python"},
    "Java Programming": {"kind": "code", "lang": "java"},
    "Machine Learning": {"kind": "code", "lang": "python"},
    "Artificial Intelligence": {"kind": "code", "lang": "python"},
    "Data Science": {"kind": "code", "lang": "python"},
    "Other": {"kind": "code", "lang": "text"},
}


class LabMateError(Exception):
    """User-friendly application error."""


# ---------------------------------------------------------------------------
# Backend selection (Gemini when a key exists, otherwise local Ollama)
# ---------------------------------------------------------------------------
def get_gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            key = ""
    return str(key).strip().strip('"').strip("'")


def llm_backend() -> str:
    """'gemini' or 'ollama'. Set LLM_BACKEND=ollama in your LOCAL .env to force Ollama."""
    pref = os.getenv("LLM_BACKEND", "auto").strip().lower()
    if pref in ("ollama", "gemini"):
        return pref
    return "gemini" if get_gemini_key() else "ollama"


def embed_backend() -> str:
    """'gemini' or 'local' (sentence-transformers). Override with EMBED_BACKEND."""
    pref = os.getenv("EMBED_BACKEND", "auto").strip().lower()
    if pref in ("local", "gemini"):
        return pref
    return "gemini" if llm_backend() == "gemini" else "local"


@st.cache_resource(show_spinner=False)
def gemini_runtime() -> dict:
    """Remembers which Gemini models worked so we don't re-probe every call."""
    return {"model": None, "embed_model": None}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are LabMate, an Academic Laboratory Assistant for college students.

Rules you must always follow:
1. Help students understand laboratory experiments.
2. Explain concepts clearly.
3. Prefer information from the uploaded laboratory material (the passages given to you).
4. Use the retrieved context whenever the question relates to the lab manual.
5. Do NOT invent information that the document does not support.
6. If the information is not in the material, clearly say: "The uploaded document does not contain enough information about this."
7. Do not just hand over answers when an explanation would help the student learn.
8. When debugging, explain the underlying issue, not only the fix.
9. In Viva Mode, behave like an examiner and ask ONE question at a time.
10. Keep explanations appropriate for a college student.
11. Encourage understanding rather than blind copying.
12. Clearly separate what comes from the lab manual (label it "From the lab manual") from general explanations (label it "General explanation").
You have NOT executed any code or query. Never claim that you ran anything; reason by reading it."""

ANALYZER_INSTR = """Analyze the experiment and present it using EXACTLY these markdown headings, in this order:
### Experiment Number & Title
### Objective
### Concepts Involved
### Required Inputs / Requirements
### Procedure
### Code / Query
### Expected Output
### Important Terminology
### Questions in the Manual
### Viva-Related Concepts
Use bullet points. If something is not present in the material, write "Not found in the uploaded material." Do not make things up."""

UNDERSTAND_INSTR = """Explain the experiment to a college student using EXACTLY these markdown headings:
### Experiment Overview
### Objective
### Concepts
### Procedure (step by step)
### Example
### Common Mistakes
### Key Takeaways
Base the explanation primarily on the lab manual material. A simple example may be a general explanation; label it as such."""

ASK_INSTR = """Answer the student's question using the lab manual passages provided. Give a clear, student-friendly explanation.
If the passages do not contain the answer, say so plainly, then (optionally) give a short general explanation labelled "General explanation".
Use the earlier conversation to resolve references like "it" or "that"."""

DEBUG_INSTR = """Analyze the student's code/query for problems. Use EXACTLY these headings:
### Problem
### Why It Happens
### Suggested Fix
### Explanation
Check for syntax errors, logical issues, incorrect concepts, missing conditions and causes of unexpected output.
Remember: you did not run it. If the code looks correct, say so and mention what to double check."""

WORKSPACE_ACTIONS = {
    "analyze": "Analyze this query: what it does, tables/columns it assumes, the shape of the result, possible issues, and related lab concepts.",
    "explain": "Explain this clause by clause / line by line in simple language, then summarize what it produces.",
    "debug": DEBUG_INSTR,
    "errors": "List possible syntax errors, logical errors and edge cases. For each, say why it matters. Do not claim it was executed.",
    "improve": "Suggest improvements (readability, correctness, efficiency, best practices) and explain why each helps.",
    "practice": "Create ONE similar practice problem (new scenario, same concept) with a hint. Put a model solution at the very end under the heading '### Solution (try first!)'.",
}

PRACTICE_Q_INSTR = """Write exactly ONE practice question based on the lab material.
Output ONLY the question (include any code/query/table needed). Do NOT include the answer, hints, or any preface."""

PRACTICE_EVAL_INSTR = """Evaluate the student's answer to the practice question using the lab manual passages.
Your reply MUST start with a first line exactly like:
VERDICT: Correct   (or: VERDICT: Partially correct   /   VERDICT: Incorrect)
Then use these headings:
### Explanation
### Relevant Concept
### Reference
In Reference, point to the lab manual passage (experiment/page) if possible, otherwise say it is a general concept."""

VIVA_Q_INSTR = """You are a strict but fair laboratory viva examiner.
Ask exactly ONE short viva question at the requested difficulty, grounded in the lab material.
Do not repeat earlier questions. Output ONLY the question text. No answer, no hints."""

VIVA_EVAL_INSTR = """You are a laboratory viva examiner evaluating one student answer using the lab passages.
Reply in EXACTLY this format and nothing else:
VERDICT: <Correct | Partial | Incorrect>
TOPIC: <2-4 word concept being tested>
FEEDBACK: <2-4 sentences: what was right/wrong and the expected answer>"""

VIVA_SUMMARY_INSTR = """Based on the viva results provided, suggest what the student should revise.
Give 3-5 short bullet points naming topics/concepts (and experiments if known) to review. Be encouraging and concise."""

CSS = """
<style>
.lm-hero{padding:1.4rem 1.6rem;border-radius:16px;background:linear-gradient(135deg,#4f46e5 0%,#0ea5e9 100%);color:#fff;margin-bottom:1.2rem}
.lm-hero h1{margin:0;font-size:2.1rem;color:#fff;padding:0}
.lm-hero p{margin:.3rem 0 0;opacity:.95;font-size:1.05rem}
.lm-card{border:1px solid rgba(128,128,128,.3);border-radius:14px;padding:1rem;height:100%;background:rgba(128,128,128,.06);margin-bottom:.6rem}
.lm-card h4{margin:0 0 .3rem 0}
.lm-card p{margin:0;font-size:.9rem;opacity:.85}
</style>
"""


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def new_viva() -> dict:
    return {"active": False, "finished": False, "turns": [], "current_q": None,
            "difficulty": "Medium", "max_q": 8, "exp": 0, "summary": None}


def new_practice() -> dict:
    return {"question": None, "sources": [], "evaluation": None, "verdict": None,
            "eval_sources": [], "qid": 0, "attempted": 0, "correct": 0}


def init_state() -> None:
    defaults = {
        "doc": None, "chunks": [], "chat": [], "results": {},
        "practice": new_practice(), "viva": new_viva(),
        "load_error": None, "uploader_key": 0, "sid": uuid.uuid4().hex[:8],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def selected_exp() -> int:
    return int(st.session_state.get("selected_exp", 0) or 0) if st.session_state.doc else 0


def subject_info() -> dict:
    return SUBJECTS.get(st.session_state.get("subject", "SQL / DBMS"), SUBJECTS["Other"])


def exp_label(exp: int) -> str:
    doc = st.session_state.doc
    if not exp or not doc:
        return "Entire document"
    e = doc["experiments"].get(exp)
    title = e["title"] if e else ""
    return f"Experiment {exp}" + (f" — {title}" if title else "")


def current_model() -> str:
    return st.session_state.get("model", DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ollama_client():
    if ollama is None:
        raise LabMateError("The 'ollama' package is not installed. Run: pip install ollama")
    return ollama.Client(host=OLLAMA_HOST)


@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)…")
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


# ---------------------------------------------------------------------------
# PDF processing and chunking
# ---------------------------------------------------------------------------
EXP_RE = re.compile(
    r"^\s*(?:experiment|expt|exp|practical|program|lab)\s*(?:no|number|#)?\s*\.?\s*[:\-–]?\s*(\d{1,2})\s*[:\-–.)]?\s*(.*)$",
    re.IGNORECASE,
)


def clean_text(t: str) -> str:
    t = t.replace("\x00", " ").replace("\r", "\n")
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r" ?\n ?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_pdf_pages(data: bytes) -> list[dict]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        raise LabMateError("This file doesn't look like a valid PDF. Please upload a different file.") from e
    if reader.is_encrypted:
        try:
            ok = reader.decrypt("")
        except Exception:
            ok = 0
        if not ok:
            raise LabMateError("This PDF is password protected. Please upload an unlocked copy.")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        pages.append({"page": i, "text": clean_text(raw)})
    if sum(len(p["text"]) for p in pages) < 50:
        raise LabMateError("No readable text found. The PDF may be empty or a scanned image (OCR is not supported yet). "
                           "Try pasting the text instead.")
    return pages


def segment_document(pages: list[dict]) -> tuple[list[dict], dict]:
    """Split pages into segments tagged with page + experiment; detect experiments."""
    segments: list[dict] = []
    experiments: dict[int, dict] = {}
    current = 0

    def flush(buf: list[str], page: int, exp: int) -> None:
        text = "\n".join(buf).strip()
        if not text:
            return
        segments.append({"page": page, "experiment": exp, "text": text})
        if exp:
            e = experiments.setdefault(exp, {"title": "", "text": "", "pages": set()})
            e["text"] += ("\n" if e["text"] else "") + text
            e["pages"].add(page)

    for p in pages:
        buf: list[str] = []
        seg_exp = current
        for line in p["text"].split("\n"):
            m = EXP_RE.match(line) if len(line.strip()) <= 120 else None
            if m:
                num = int(m.group(1))
                title = re.sub(r"\.{2,}\s*\d+\s*$", "", m.group(2)).strip(" :-–—.")
                if num != current:
                    flush(buf, p["page"], seg_exp)
                    buf = []
                    current = num
                    seg_exp = num
                e = experiments.setdefault(num, {"title": "", "text": "", "pages": set()})
                if len(title) > len(e["title"]):
                    e["title"] = title
            buf.append(line)
        flush(buf, p["page"], seg_exp)

    for e in experiments.values():
        e["pages"] = sorted(e["pages"])
    return segments, experiments


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------------------------------------------------------------------------
# Gemini helpers (errors, retries)
# ---------------------------------------------------------------------------
def _err_text(e: Exception) -> str:
    return f"{getattr(e, 'status', '')} {getattr(e, 'message', '')} {e}".lower()


def _is_not_found(e: Exception) -> bool:
    msg = _err_text(e)
    return (getattr(e, "code", None) == 404 or "not_found" in msg or "not found" in msg
            or "no longer available" in msg or "is not supported" in msg or "shut down" in msg)


def _is_retryable(e: Exception) -> bool:
    msg = _err_text(e)
    return (getattr(e, "code", None) in (429, 500, 503) or "unavailable" in msg
            or "resource_exhausted" in msg or "overloaded" in msg)


def _translate_gemini_error(e: Optional[Exception]) -> LabMateError:
    if e is None:
        return LabMateError("None of the Gemini models were available for your API key. "
                            "Set GEMINI_MODEL to a model listed in Google AI Studio.")
    msg = _err_text(e)
    code = getattr(e, "code", None)
    if "api key" in msg or "api_key" in msg or code in (401, 403) or "permission_denied" in msg:
        return LabMateError("The Gemini API key is missing, invalid or not allowed. Check GEMINI_API_KEY.")
    if code == 429 or "quota" in msg or "resource_exhausted" in msg:
        return LabMateError("Gemini rate limit or quota reached. Wait a minute and try again.")
    if _is_not_found(e):
        return LabMateError("The selected Gemini model was not found. Set GEMINI_MODEL=gemini-flash-latest.")
    if any(w in msg for w in ("connect", "timeout", "network", "unreachable", "name resolution")):
        return LabMateError("Could not reach Gemini. Check your internet connection.")
    if "safety" in msg or "blocked" in msg:
        return LabMateError("Gemini blocked that request for safety reasons. Try rephrasing it.")
    return LabMateError("Something went wrong while talking to Gemini. Please try again.")


def _import_genai():
    try:
        from google import genai
        from google.genai import types
        return genai, types
    except Exception as e:
        raise LabMateError("The 'google-genai' package is not installed. Run: pip install google-genai") from e


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


# ---------------------------------------------------------------------------
# Embeddings + ChromaDB
# ---------------------------------------------------------------------------
def _local_embed(texts: list[str], batch: int = 32, progress=None) -> list[list[float]]:
    try:
        model = get_embedder()
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            vecs = model.encode(texts[i:i + batch], normalize_embeddings=True, show_progress_bar=False)
            out.extend(vecs.tolist())
            if progress:
                progress(min(1.0, (i + batch) / max(len(texts), 1)))
        return out
    except Exception as e:
        raise LabMateError("Could not generate embeddings locally. Install sentence-transformers, "
                           "or set GEMINI_API_KEY to use Gemini embeddings.") from e


def _gemini_embed(texts: list[str], task: str, batch: int = 50, progress=None) -> list[list[float]]:
    key = get_gemini_key()
    if not key:
        raise LabMateError("Gemini API key missing. Add GEMINI_API_KEY to your .env file or hosting secrets.")
    genai, types = _import_genai()
    client = genai.Client(api_key=key)
    rt = gemini_runtime()
    models: list[str] = []
    for m in (rt.get("embed_model"), GEMINI_EMBED_MODEL, "gemini-embedding-001", "gemini-embedding-2"):
        if m and m not in models:
            models.append(m)
    cfg = types.EmbedContentConfig(task_type=task, output_dimensionality=GEMINI_EMBED_DIM)
    out: list[list[float]] = []
    model_ok: Optional[str] = None
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        vecs = None
        last: Optional[Exception] = None
        for model in ([model_ok] if model_ok else models):
            for attempt in range(5):
                try:
                    resp = client.models.embed_content(model=model, contents=part, config=cfg)
                    vecs = [list(e.values) for e in resp.embeddings]
                    break
                except Exception as e:
                    last = e
                    if _is_not_found(e):
                        break
                    if _is_retryable(e) and attempt < 4:
                        time.sleep(8 * (attempt + 1))
                        continue
                    raise _translate_gemini_error(e) from e
            if vecs is not None:
                model_ok = model
                break
        if vecs is None:
            raise _translate_gemini_error(last)
        out.extend(_normalize(v) for v in vecs)
        if progress:
            progress(min(1.0, (i + batch) / max(len(texts), 1)))
    rt["embed_model"] = model_ok
    return out


def embed_texts(texts: list[str], backend: str, task: str = "document", progress=None) -> list[list[float]]:
    if backend == "gemini":
        return _gemini_embed(texts, "RETRIEVAL_DOCUMENT" if task == "document" else "RETRIEVAL_QUERY",
                             progress=progress)
    return _local_embed(texts, progress=progress)


def build_knowledge_base(name: str, pages: list[dict], doc_hash: str, progress) -> tuple[dict, list[dict]]:
    segments, experiments = segment_document(pages)
    chunks: list[dict] = []
    for seg in segments:
        for c in chunk_text(seg["text"]):
            n = len(chunks)
            chunks.append({
                "id": f"{doc_hash[:8]}-{n}",
                "text": c,
                "metadata": {"source": name, "page": seg["page"], "experiment": str(seg["experiment"]), "chunk": n},
            })
    if not chunks:
        raise LabMateError("The document has no usable text to index.")

    texts = [c["text"] for c in chunks]
    preferred = embed_backend()
    order = [preferred] + [b for b in ("gemini", "local") if b != preferred]
    embeddings, used, last_err = None, None, None
    for b in order:
        if b == "gemini" and not get_gemini_key():
            continue
        try:
            embeddings = embed_texts(texts, b, "document", progress=lambda f: progress(0.8 * f))
            used = b
            break
        except LabMateError as e:
            last_err = e
    if embeddings is None or used is None:
        raise last_err or LabMateError("No embedding method is available. Set GEMINI_API_KEY or install sentence-transformers.")

    collection_name = f"labmate_{doc_hash[:10]}_{used[0]}_{st.session_state.get('sid', 'x')}"
    try:
        client = get_chroma_client()
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        col = client.create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        step = 256
        for i in range(0, len(chunks), step):
            part = chunks[i:i + step]
            col.add(
                ids=[c["id"] for c in part],
                documents=[c["text"] for c in part],
                embeddings=embeddings[i:i + step],
                metadatas=[c["metadata"] for c in part],
            )
            progress(0.8 + 0.2 * min(1.0, (i + step) / len(chunks)))
    except Exception as e:
        raise LabMateError("The vector database (ChromaDB) failed while storing the document.") from e

    info = {
        "name": name, "hash": doc_hash, "pages": len(pages),
        "chars": sum(len(p["text"]) for p in pages), "n_chunks": len(chunks),
        "collection": collection_name, "experiments": experiments, "status": "Ready",
        "embed_backend": used,
    }
    return info, chunks


def fmt_source(meta: dict) -> str:
    exp = meta.get("experiment", "0")
    prefix = f"Experiment {exp}, " if exp not in ("0", "", None) else ""
    return f"{meta.get('source', 'document')} — {prefix}Page {meta.get('page', '?')}"


def hits_to_context(hits: list[tuple[str, dict]]) -> tuple[str, list[dict]]:
    parts, sources, seen = [], [], set()
    for i, (text, meta) in enumerate(hits, 1):
        label = fmt_source(meta)
        parts.append(f"[Passage {i} | {label}]\n{text}")
        key = (label, text[:80])
        if key not in seen:
            seen.add(key)
            sources.append({"label": label, "snippet": text[:300].replace("\n", " ")})
    return "\n\n".join(parts), sources


def retrieve(query: str, k: int = TOP_K, exp: int = 0) -> tuple[str, list[dict]]:
    doc = st.session_state.doc
    try:
        col = get_chroma_client().get_collection(doc["collection"])
        emb = embed_texts([query], doc.get("embed_backend", "local"), "query")

        def run(where: Optional[dict]) -> list[tuple[str, dict]]:
            kw = {"query_embeddings": emb, "n_results": min(k, max(col.count(), 1)),
                  "include": ["documents", "metadatas"]}
            if where:
                kw["where"] = where
            r = col.query(**kw)
            return list(zip(r["documents"][0], r["metadatas"][0]))

        hits: list[tuple[str, dict]] = []
        if exp:
            hits = run({"experiment": str(exp)})
        if len(hits) < min(k, 3):
            seen = {h[0] for h in hits}
            for h in run(None):
                if h[0] not in seen:
                    hits.append(h)
                    seen.add(h[0])
        return hits_to_context(hits[:k])
    except LabMateError:
        raise
    except Exception as e:
        raise LabMateError("Searching the knowledge base failed. Try re-uploading the lab manual.") from e


def get_context(query: str, exp: int = 0, k: int = TOP_K, whole_section: bool = False):
    """Returns (context, sources) or (None, None) after showing an error."""
    try:
        doc = st.session_state.doc
        if whole_section and exp and exp in doc["experiments"]:
            e = doc["experiments"][exp]
            if len(e["text"]) <= MAX_SECTION_CHARS:
                pr = e["pages"]
                pages = f"pages {pr[0]}–{pr[-1]}" if pr and pr[0] != pr[-1] else f"page {pr[0] if pr else '?'}"
                return e["text"], [{"label": f"{doc['name']} — Experiment {exp}, {pages}",
                                    "snippet": e["text"][:300].replace("\n", " ")}]
            k = max(k, 8)
        ctx, sources = retrieve(query, k, exp)
        return (ctx or "(No relevant passages were found in the document.)"), sources
    except LabMateError as err:
        st.error(str(err))
        return None, None


def sample_context(exp: int, n: int = 3, limit: int = 3500) -> tuple[str, list[dict]]:
    chunks = st.session_state.chunks
    pool = [c for c in chunks if not exp or c["metadata"]["experiment"] == str(exp)] or chunks
    picks = random.sample(pool, min(n, len(pool)))
    picks.sort(key=lambda c: c["metadata"]["chunk"])
    ctx, sources = hits_to_context([(c["text"], c["metadata"]) for c in picks])
    return ctx[:limit], sources


# ---------------------------------------------------------------------------
# LLM backends (Ollama + Gemini)
# ---------------------------------------------------------------------------
def _translate_error(e: Exception) -> LabMateError:
    if ollama is not None and isinstance(e, ollama.ResponseError):
        if getattr(e, "status_code", None) == 404:
            return LabMateError(f"Model '{current_model()}' is not available. Run `ollama pull {current_model()}` and try again.")
        return LabMateError("Ollama returned an error. Please check the model and try again.")
    msg = str(e).lower()
    if any(w in msg for w in ("connect", "refused", "connection", "unreachable")):
        return LabMateError("Ollama is not running. Please start Ollama and make sure the selected model is available.")
    return LabMateError("Something went wrong while talking to the model. Please try again.")


def _stream_ollama(messages: list[dict], temperature: float):
    try:
        stream = get_ollama_client().chat(
            model=current_model(), messages=messages, stream=True,
            options={"temperature": temperature, "num_ctx": NUM_CTX},
        )
        for chunk in stream:
            piece = chunk["message"]["content"]
            if piece:
                yield piece
    except LabMateError:
        raise
    except Exception as e:
        raise _translate_error(e) from e


def _stream_gemini(messages: list[dict], temperature: float):
    key = get_gemini_key()
    if not key:
        raise LabMateError("Gemini API key missing. Add GEMINI_API_KEY to your .env file or hosting secrets.")
    genai, types = _import_genai()
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        types.Content(role="model" if m["role"] == "assistant" else "user",
                      parts=[types.Part(text=m["content"])])
        for m in messages if m["role"] != "system"
    ]
    client = genai.Client(api_key=key)
    # temperature is deprecated on the newest Gemini models, so it is intentionally not sent.
    config = types.GenerateContentConfig(system_instruction=system or None)
    rt = gemini_runtime()
    candidates: list[str] = []
    for m in (rt.get("model"), GEMINI_MODEL, *GEMINI_FALLBACKS):
        if m and m not in candidates:
            candidates.append(m)

    last: Optional[Exception] = None
    for model in candidates:
        for attempt in range(3):
            started = False
            try:
                for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
                    text = getattr(chunk, "text", None)
                    if text:
                        started = True
                        yield text
                rt["model"] = model
                return
            except Exception as e:
                last = e
                if started:
                    raise _translate_gemini_error(e) from e
                if _is_not_found(e):
                    break  # try the next model
                if _is_retryable(e) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                raise _translate_gemini_error(e) from e
    raise _translate_gemini_error(last)


def _stream_llm(messages: list[dict], temperature: float):
    if llm_backend() == "gemini":
        yield from _stream_gemini(messages, temperature)
    else:
        yield from _stream_ollama(messages, temperature)


def run_llm(messages: list[dict], placeholder=None, temperature: float = 0.3) -> Optional[str]:
    """Run the LLM. Streams into placeholder if given. Returns text or None on error."""
    text = ""
    try:
        for piece in _stream_llm(messages, temperature):
            text += piece
            if placeholder is not None:
                placeholder.markdown(text + "▌")
        if not text.strip():
            raise LabMateError("The model returned an empty response. Please try again.")
        if placeholder is not None:
            placeholder.markdown(text)
        return text.strip()
    except LabMateError as e:
        if placeholder is not None:
            placeholder.error(str(e))
        else:
            st.error(str(e))
        return None


def build_messages(task_instr: str, user_content: str, history: Optional[list[dict]] = None) -> list[dict]:
    system = SYSTEM_PROMPT + (f"\n\nCURRENT TASK INSTRUCTIONS:\n{task_instr}" if task_instr else "")
    msgs = [{"role": "system", "content": system}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user_content})
    return msgs


def list_models() -> Optional[list[str]]:
    try:
        resp = get_ollama_client().list()
        models = getattr(resp, "models", None)
        if models is None and isinstance(resp, dict):
            models = resp.get("models", [])
        names = []
        for m in models or []:
            n = getattr(m, "model", None)
            if n is None and isinstance(m, dict):
                n = m.get("model") or m.get("name")
            if n:
                names.append(n)
        return names
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def show_sources(sources: Optional[list[dict]]) -> None:
    if not sources:
        return
    with st.expander("📎 Sources used"):
        for s in sources:
            st.markdown(f"**{s['label']}**")
            if s.get("snippet"):
                st.caption(s["snippet"])


def show_result(slot: str) -> None:
    r = st.session_state.results.get(slot)
    if r:
        if r.get("title"):
            st.markdown(f"##### {r['title']}")
        st.markdown(r["text"])
        show_sources(r.get("sources"))


def generate_into_slot(slot: str, messages: list[dict], sources: list[dict], title: str = "") -> None:
    ph = st.empty()
    text = run_llm(messages, ph)
    if text:
        st.session_state.results[slot] = {"text": text, "sources": sources, "title": title}
        st.rerun()


def require_doc() -> bool:
    if st.session_state.doc:
        return True
    st.info("📄 Upload a lab manual PDF (or paste its text) in the sidebar to unlock this section.")
    return False


def page_range(pages: list[int]) -> str:
    if not pages:
        return "—"
    return str(pages[0]) if pages[0] == pages[-1] else f"{pages[0]}–{pages[-1]}"


# ---------------------------------------------------------------------------
# Sidebar + ingestion
# ---------------------------------------------------------------------------
def ingest(name: str, pages: list[dict], doc_hash: str) -> None:
    old = st.session_state.doc
    bar = st.sidebar.progress(0.0, text="Building knowledge base…")
    try:
        info, chunks = build_knowledge_base(
            name, pages, doc_hash,
            lambda f: bar.progress(min(max(f, 0.0), 1.0), text="Building knowledge base…"),
        )
        if old and old["collection"] != info["collection"]:
            try:
                get_chroma_client().delete_collection(old["collection"])
            except Exception:
                pass
        st.session_state.doc = info
        st.session_state.chunks = [{"text": c["text"], "metadata": c["metadata"]} for c in chunks]
        st.session_state.results = {}
        st.session_state.chat = []
        st.session_state.practice = new_practice()
        st.session_state.viva = new_viva()
        st.session_state.load_error = None
        st.session_state.pop("selected_exp", None)
    except LabMateError as e:
        st.session_state.load_error = {"hash": doc_hash, "msg": str(e)}
    finally:
        bar.empty()


def reset_session() -> None:
    doc = st.session_state.get("doc")
    if doc:
        try:
            get_chroma_client().delete_collection(doc["collection"])
        except Exception:
            pass
    n = st.session_state.get("uploader_key", 0) + 1
    st.session_state.clear()
    init_state()
    st.session_state.uploader_key = n
    st.rerun()


def render_sidebar() -> None:
    sb = st.sidebar
    sb.title("🧪 LabMate")
    sb.caption("Your interactive laboratory assistant")

    sb.subheader("1 · Lab manual")
    uploaded = sb.file_uploader("Upload PDF", type=["pdf"], key=f"uploader_{st.session_state.uploader_key}")
    with sb.expander("…or paste text instead"):
        pasted = st.text_area("Paste experiment text", height=150, key="pasted_text")
        use_pasted = st.button("Use pasted text", key="use_pasted")

    err = st.session_state.load_error
    if uploaded is not None:
        data = uploaded.getvalue()
        h = hashlib.sha1(data).hexdigest()
        doc = st.session_state.doc
        if not (doc and doc["hash"] == h):
            if err and err["hash"] == h:
                sb.error(err["msg"])
            else:
                try:
                    pages = extract_pdf_pages(data)
                    ingest(uploaded.name, pages, h)
                    if st.session_state.load_error:
                        sb.error(st.session_state.load_error["msg"])
                except LabMateError as e:
                    st.session_state.load_error = {"hash": h, "msg": str(e)}
                    sb.error(str(e))
    elif use_pasted:
        if len(pasted.strip()) < 50:
            sb.warning("Please paste at least a few sentences of lab material.")
        else:
            h = hashlib.sha1(pasted.encode("utf-8")).hexdigest()
            ingest("Pasted text", [{"page": 1, "text": clean_text(pasted)}], h)
            if st.session_state.load_error:
                sb.error(st.session_state.load_error["msg"])

    sb.subheader("2 · Subject & experiment")
    sb.selectbox("Subject", list(SUBJECTS), key="subject")
    doc = st.session_state.doc
    if doc:
        options = [0] + sorted(doc["experiments"])
        sb.selectbox("Experiment", options, key="selected_exp", format_func=exp_label)
    else:
        sb.selectbox("Experiment", ["Upload a manual first"], disabled=True)

    sb.subheader("3 · AI model")
    sb.caption(f"Backend: {llm_backend()} · key found: {bool(get_gemini_key())}")
    if llm_backend() == "gemini":
        active = gemini_runtime().get("model") or GEMINI_MODEL
        st.session_state["model"] = active
        if get_gemini_key():
            sb.success(f"Using Gemini ({active})", icon="✅")
        else:
            sb.error("Gemini API key missing. Add GEMINI_API_KEY to your .env file or hosting secrets.")
    else:
        names = list_models()
        if names is None:
            sb.error("Ollama is not running. Please start Ollama and make sure the selected model is available.")
            st.session_state["model"] = sb.text_input("Model name", value=DEFAULT_MODEL, key="model_text")
        elif not names:
            sb.warning(f"No models installed. Run `ollama pull {DEFAULT_MODEL}`.")
            st.session_state["model"] = DEFAULT_MODEL
        else:
            idx = 0
            for i, n in enumerate(names):
                if n == DEFAULT_MODEL or n.split(":")[0] == DEFAULT_MODEL.split(":")[0]:
                    idx = i
                    break
            st.session_state["model"] = sb.selectbox("Ollama model", names, index=idx, key="model_select")
            sb.success("Ollama connected", icon="✅")

    sb.divider()
    if sb.button("🗑️ Clear conversations"):
        st.session_state.chat = []
        st.session_state.viva = new_viva()
        st.rerun()
    if sb.button("♻️ Reset entire session"):
        reset_session()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
def tab_dashboard() -> None:
    st.markdown(
        '<div class="lm-hero"><h1>🧪 LabMate</h1><p>Your Interactive Laboratory Assistant — '
        "turn a lab manual into a workspace for understanding, debugging, practice and viva prep.</p></div>",
        unsafe_allow_html=True,
    )
    doc = st.session_state.doc
    if not doc:
        st.info("**Welcome to LabMate.** Upload your laboratory manual (PDF) or paste the text in the sidebar to begin.")
    st.caption("Workflow: Upload → Analyze → Understand → Work in Workspace → Debug → Practice → Viva")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Document", doc["name"] if doc else "None")
    c2.metric("Experiments detected", len(doc["experiments"]) if doc else 0)
    c3.metric("Knowledge base", doc["status"] if doc else "Empty")
    v = st.session_state.viva
    session = "Viva in progress" if v["active"] else f"{len(st.session_state.chat)} chat messages"
    c4.metric("Session", session)
    if doc:
        st.success(f"**Lab Manual Loaded** — {doc['name']} · {doc['pages']} pages · {doc['chars']:,} characters · "
                   f"{doc['n_chunks']} chunks · {len(doc['experiments'])} experiments identified · Knowledge base: Ready")

    cards = [
        ("📄 Lab Analyzer", "Extract objective, procedure, code, outputs and terminology."),
        ("📚 Understand", "Step-by-step explanation with examples and common mistakes."),
        ("💬 Ask LabMate", "Context-aware Q&A grounded in your manual, with sources."),
        ("💻 Workspace", "Explain, analyze and improve your code or SQL queries."),
        ("🐞 Debug", "Find the problem, understand why, and learn the fix."),
        ("🧠 Practice", "Generate questions by difficulty and get graded feedback."),
        ("🎤 Viva Mode", "An examiner that asks one question at a time and adapts."),
    ]
    for row in (cards[:4], cards[4:]):
        cols = st.columns(4)
        for col, (title, desc) in zip(cols, row):
            col.markdown(f'<div class="lm-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)


def tab_analyzer() -> None:
    st.header("📄 Lab Analyzer")
    if not require_doc():
        return
    doc = st.session_state.doc
    exps = doc["experiments"]
    if exps:
        rows = [{"Experiment": n, "Title": e["title"] or "—", "Pages": page_range(e["pages"]),
                 "Characters": len(e["text"])} for n, e in sorted(exps.items())]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.warning("No experiment headings were detected, so LabMate will analyze the document as a whole.")

    exp = selected_exp()
    st.subheader(exp_label(exp))
    slot = f"analyze-{exp}"
    show_result(slot)
    if exp and exp in exps:
        with st.expander("View raw extracted text"):
            st.text(exps[exp]["text"][:4000])
    if st.button("🔍 Analyze this experiment", key="btn_analyze"):
        ctx, sources = get_context("objective procedure code query expected output viva questions",
                                   exp, k=8, whole_section=True)
        if ctx is None:
            return
        msgs = build_messages(ANALYZER_INSTR,
                              f"Subject: {st.session_state.get('subject')}\nScope: {exp_label(exp)}\n\n"
                              f"LAB MANUAL MATERIAL:\n{ctx}\n\nAnalyze this experiment.")
        generate_into_slot(slot, msgs, sources)


def tab_understand() -> None:
    st.header("📚 Understand")
    if not require_doc():
        return
    exp = selected_exp()
    st.subheader(exp_label(exp))
    focus = st.text_input("Anything specific to focus on? (optional)", key="understand_focus")
    slot = f"understand-{exp}"
    show_result(slot)
    if st.button("📖 Explain this experiment", key="btn_understand"):
        ctx, sources = get_context(focus or "objective concepts procedure example", exp, k=8, whole_section=True)
        if ctx is None:
            return
        extra = f"\nThe student especially wants help with: {focus}" if focus.strip() else ""
        msgs = build_messages(UNDERSTAND_INSTR,
                              f"Subject: {st.session_state.get('subject')}\nScope: {exp_label(exp)}{extra}\n\n"
                              f"LAB MANUAL MATERIAL:\n{ctx}\n\nExplain this experiment.")
        generate_into_slot(slot, msgs, sources)


def tab_ask() -> None:
    st.header("💬 Ask LabMate")
    if not require_doc():
        return
    exp = selected_exp()
    st.caption(f"Answers are grounded in your lab manual · Focus: {exp_label(exp)}")
    if st.button("🗑️ Clear this conversation", key="clear_chat"):
        st.session_state.chat = []
        st.rerun()

    box = st.container()
    prompt = st.chat_input("Ask about the experiment, a concept, your code…", key="ask_input")
    chat = st.session_state.chat
    with box:
        for m in chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m["role"] == "assistant":
                    show_sources(m.get("sources"))
        if prompt is None:
            return
        if not prompt.strip():
            st.warning("Please type a question.")
            return
        with st.chat_message("user"):
            st.markdown(prompt)
        query = prompt
        prev = [m["content"] for m in chat if m["role"] == "user"]
        if len(prompt.split()) < 8 and prev:
            query = prev[-1] + " " + prompt  # short follow-up: keep topic
        ctx, sources = get_context(query, exp)
        if ctx is None:
            return
        history = [{"role": m["role"], "content": m["content"]} for m in chat[-8:]]
        user_content = (f"Subject: {st.session_state.get('subject')}\n\nLAB MANUAL PASSAGES:\n{ctx}\n\n"
                        f"STUDENT QUESTION:\n{prompt}")
        with st.chat_message("assistant"):
            ph = st.empty()
            text = run_llm(build_messages(ASK_INSTR, user_content, history), ph)
            if text:
                show_sources(sources)
        if text:
            chat.append({"role": "user", "content": prompt})
            chat.append({"role": "assistant", "content": text, "sources": sources})


def tab_workspace() -> None:
    info = subject_info()
    is_sql = info["kind"] == "sql"
    st.header("💻 SQL Workspace" if is_sql else "💻 Code Workspace")
    if not require_doc():
        return
    exp = selected_exp()
    st.caption("LabMate analyzes by reading — it does not execute your code or queries.")
    code = st.text_area("Enter SQL query" if is_sql else f"Enter {info['lang']} code", height=220, key="ws_input")
    notes = st.text_input("Notes / what are you trying to do? (optional)", key="ws_notes")
    actions = ([("Analyze Query", "analyze"), ("Explain Query", "explain"), ("Debug Query", "debug"),
                ("Generate Similar Practice", "practice")] if is_sql else
               [("Explain Code", "explain"), ("Find Possible Errors", "errors"),
                ("Suggest Improvements", "improve"), ("Generate Practice Problem", "practice")])
    cols = st.columns(len(actions))
    pressed = None
    for col, (label, key) in zip(cols, actions):
        if col.button(label, key=f"ws_{key}"):
            pressed = (label, key)
    show_result("workspace")
    if pressed:
        if not code.strip():
            st.warning("Please enter some code or a query first.")
            return
        label, key = pressed
        ctx, sources = get_context(code[:400] + " " + notes, exp, k=3)
        if ctx is None:
            return
        user = (f"Subject: {st.session_state.get('subject')}\n\nRELATED LAB MANUAL PASSAGES:\n{ctx}\n\n"
                f"STUDENT {'QUERY' if is_sql else 'CODE'}:\n```{info['lang']}\n{code}\n```\n"
                f"Student's notes: {notes or '(none)'}")
        generate_into_slot("workspace", build_messages(WORKSPACE_ACTIONS[key], user), sources, title=label)


def tab_debug() -> None:
    st.header("🐞 Debug")
    if not require_doc():
        return
    info = subject_info()
    exp = selected_exp()
    code = st.text_area("Paste your code / query", height=200, key="dbg_code")
    c1, c2 = st.columns(2)
    err = c1.text_area("Error message or unexpected output (optional)", height=120, key="dbg_err")
    expected = c2.text_area("What did you expect? (optional)", height=120, key="dbg_expected")
    show_result("debug")
    if st.button("🔧 Debug it", key="btn_debug"):
        if not code.strip():
            st.warning("Please paste some code or a query to debug.")
            return
        ctx, sources = get_context(code[:400] + " " + err, exp, k=3)
        if ctx is None:
            return
        user = (f"Subject: {st.session_state.get('subject')}\n\nRELATED LAB MANUAL PASSAGES:\n{ctx}\n\n"
                f"STUDENT CODE:\n```{info['lang']}\n{code}\n```\n"
                f"Error / unexpected output: {err or '(none given)'}\nExpected behaviour: {expected or '(none given)'}")
        generate_into_slot("debug", build_messages(DEBUG_INSTR, user), sources)


def parse_verdict(text: str) -> str:
    m = re.search(r"VERDICT:\s*\**\s*([A-Za-z]+)", text, re.I)
    word = m.group(1).lower() if m else "partial"
    if word.startswith("corr"):
        return "correct"
    if word.startswith("inc") or word.startswith("wrong"):
        return "incorrect"
    return "partial"


def tab_practice() -> None:
    st.header("🧠 Practice")
    if not require_doc():
        return
    p = st.session_state.practice
    exp = selected_exp()
    is_sql = subject_info()["kind"] == "sql"

    c1, c2 = st.columns(2)
    difficulty = c1.selectbox("Difficulty", LEVELS, index=1, key="prac_diff")
    qtypes = ["Mixed", "Conceptual", "Short answer", "Code-based"] + (["SQL query"] if is_sql else []) + \
             ["Output prediction", "Debugging", "Scenario-based"]
    qtype = c2.selectbox("Question type", qtypes, key="prac_type")

    if p["attempted"]:
        m1, m2 = st.columns(2)
        m1.metric("Attempted", p["attempted"])
        m2.metric("Fully correct", p["correct"])
        st.progress(p["correct"] / p["attempted"])

    if st.button("🎲 Generate question", key="prac_gen"):
        ctx, src = sample_context(exp)
        user = (f"Subject: {st.session_state.get('subject')}\nDifficulty: {difficulty}\nQuestion type: {qtype}\n"
                f"Scope: {exp_label(exp)}\n\nLAB MANUAL MATERIAL:\n{ctx}")
        ph = st.empty()
        q = run_llm(build_messages(PRACTICE_Q_INSTR, user), ph, temperature=0.7)
        if q:
            qid = p["qid"] + 1
            p.update(question=q, sources=src, evaluation=None, verdict=None, eval_sources=[], qid=qid)
            st.rerun()

    if p["question"]:
        st.markdown("#### Practice Question")
        st.markdown(p["question"])
        show_sources(p["sources"])
        ans = st.text_area("Your answer", height=140, key=f"prac_answer_{p['qid']}")
        if st.button("✅ Check my answer", key="prac_check"):
            if not ans.strip():
                st.warning("Please write an answer first.")
            else:
                ctx, src = get_context(p["question"] + " " + ans, exp, k=4)
                if ctx is not None:
                    user = (f"LAB MANUAL PASSAGES:\n{ctx}\n\nPRACTICE QUESTION:\n{p['question']}\n\n"
                            f"STUDENT ANSWER:\n{ans}")
                    ph = st.empty()
                    text = run_llm(build_messages(PRACTICE_EVAL_INSTR, user), ph, temperature=0.2)
                    if text:
                        verdict = parse_verdict(text)
                        p["attempted"] += 1
                        p["correct"] += verdict == "correct"
                        p["verdict"] = verdict
                        p["evaluation"] = re.sub(r"^\s*VERDICT:.*\n?", "", text, count=1, flags=re.I).strip()
                        p["eval_sources"] = src
                        st.rerun()
        if p["evaluation"]:
            banner = {"correct": st.success, "partial": st.warning, "incorrect": st.error}[p["verdict"]]
            banner({"correct": "✅ Correct", "partial": "🟡 Partially correct", "incorrect": "❌ Incorrect"}[p["verdict"]])
            st.markdown(p["evaluation"])
            show_sources(p["eval_sources"])
    else:
        st.info("Choose a difficulty and click **Generate question**.")


# ---------------------------------------------------------------------------
# Viva
# ---------------------------------------------------------------------------
def adapt_difficulty(current: str, turns: list[dict]) -> str:
    tail = []
    for t in reversed(turns):
        if t["difficulty"] != current:
            break
        tail.append(t["verdict"])
    last = tail[:2]
    i = LEVELS.index(current)
    if len(last) == 2 and all(v == "correct" for v in last):
        i = min(i + 1, len(LEVELS) - 1)
    elif len(last) == 2 and all(v == "incorrect" for v in last):
        i = max(i - 1, 0)
    return LEVELS[i]


def viva_next_question(first: bool = False) -> Optional[str]:
    v = st.session_state.viva
    ctx, _ = sample_context(v["exp"], n=3)
    asked = [t["question"] for t in v["turns"]] + ([v["current_q"]] if v["current_q"] else [])
    prior = "\n".join(f"- {q}" for q in asked[-8:]) or "(none)"
    ask = ("Ask the FIRST question: a gentle opener about the objective of the experiment."
           if first else "Ask the next question.")
    user = (f"Subject: {st.session_state.get('subject')}\nScope: {exp_label(v['exp'])}\n"
            f"Difficulty: {v['difficulty']}\nQuestions already asked (do not repeat):\n{prior}\n\n"
            f"LAB MANUAL MATERIAL:\n{ctx}\n\n{ask}")
    q = run_llm(build_messages(VIVA_Q_INSTR, user), temperature=0.6)
    return re.sub(r"^\s*(question|q)\s*\d*\s*[:.\-]\s*", "", q, flags=re.I).strip() if q else None


def parse_viva_eval(text: str) -> dict:
    topic_m = re.search(r"TOPIC:\s*(.+)", text, re.I)
    fb_m = re.search(r"FEEDBACK:\s*(.*)", text, re.I | re.S)
    feedback = fb_m.group(1).strip() if fb_m else re.sub(r"^\s*(VERDICT|TOPIC):.*$", "", text, flags=re.I | re.M).strip()
    topic = topic_m.group(1).strip().strip("*<>") if topic_m else "General"
    return {"verdict": parse_verdict(text), "topic": topic.title()[:40] or "General", "feedback": feedback}


def finish_viva() -> None:
    v = st.session_state.viva
    turns = v["turns"]
    counts = {k: sum(t["verdict"] == k for t in turns) for k in ("correct", "partial", "incorrect")}
    by_topic: dict[str, list[str]] = {}
    for t in turns:
        by_topic.setdefault(t["topic"], []).append(t["verdict"])
    strong = [k for k, vs in by_topic.items() if all(x == "correct" for x in vs)]
    weak = [k for k, vs in by_topic.items() if any(x != "correct" for x in vs)]
    score = round(100 * (counts["correct"] + 0.5 * counts["partial"]) / len(turns)) if turns else 0
    recommend = None
    if turns:
        summary_txt = "\n".join(f"- {t['topic']}: {t['verdict']} (Q: {t['question'][:120]})" for t in turns)
        with st.spinner("Preparing your viva summary…"):
            recommend = run_llm(build_messages(VIVA_SUMMARY_INSTR,
                                               f"Scope: {exp_label(v['exp'])}\nResults:\n{summary_txt}"), temperature=0.3)
    v["summary"] = {"attempted": len(turns), "counts": counts, "strong": strong, "weak": weak,
                    "score": score, "recommend": recommend}
    v["active"], v["finished"] = False, True


def tab_viva() -> None:
    st.header("🎤 Viva Mode")
    if not require_doc():
        return
    v = st.session_state.viva

    if not v["active"] and not v["finished"]:
        st.write("LabMate will act as your examiner: one question at a time, evaluating each answer and adapting difficulty.")
        c1, c2 = st.columns(2)
        diff = c1.selectbox("Starting difficulty", LEVELS, index=1, key="viva_diff")
        n = c2.slider("Number of questions", 3, 15, 8, key="viva_n")
        st.caption(f"Scope: {exp_label(selected_exp())} (change in the sidebar)")
        if st.button("▶️ Start viva", key="viva_start"):
            st.session_state.viva = new_viva()
            v = st.session_state.viva
            v.update(difficulty=diff, max_q=n, exp=selected_exp())
            with st.spinner("The examiner is preparing the first question…"):
                q = viva_next_question(first=True)
            if q:
                v["current_q"], v["active"] = q, True
                st.rerun()
        return

    if v["finished"]:
        s = v["summary"]
        st.markdown("### Viva Performance")
        st.write(f"**Questions attempted:** {s['attempted']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Correct", s["counts"]["correct"])
        c2.metric("Partial", s["counts"]["partial"])
        c3.metric("Incorrect", s["counts"]["incorrect"])
        c4.metric("Practice score", f"{s['score']}%")
        st.markdown("**Strong areas:**")
        st.markdown("\n".join(f"- {t}" for t in s["strong"]) or "- None yet")
        st.markdown("**Needs revision:**")
        st.markdown("\n".join(f"- {t}" for t in s["weak"]) or "- Nothing flagged 🎉")
        if s["recommend"]:
            st.markdown("**Recommended topics:**")
            st.markdown(s["recommend"])
        st.caption("This is an educational practice assessment, not an official academic grade.")
        with st.expander("Full transcript"):
            for t in v["turns"]:
                st.markdown(f"**Q ({t['difficulty']}):** {t['question']}")
                st.markdown(f"**You:** {t['answer']}")
                st.markdown(f"**Feedback ({t['verdict']}):** {t['feedback']}")
                st.divider()
        if st.button("🔁 Start a new viva", key="viva_again"):
            st.session_state.viva = new_viva()
            st.rerun()
        return

    # Active viva
    done = len(v["turns"])
    st.progress(done / v["max_q"], text=f"Question {min(done + 1, v['max_q'])} of {v['max_q']} · Difficulty: {v['difficulty']}")
    icons = {"correct": "✅ Correct", "partial": "🟡 Partial", "incorrect": "❌ Incorrect"}
    for t in v["turns"]:
        with st.chat_message("assistant"):
            st.markdown(t["question"])
        with st.chat_message("user"):
            st.markdown(t["answer"])
        with st.chat_message("assistant"):
            st.markdown(f"**{icons[t['verdict']]}** · _{t['topic']}_\n\n{t['feedback']}")
    with st.chat_message("assistant"):
        st.markdown(v["current_q"])

    with st.form("viva_form", clear_on_submit=True):
        ans = st.text_area("Your answer", height=120)
        submitted = st.form_submit_button("Submit answer")
    if submitted:
        if not ans.strip():
            st.warning("Please type an answer before submitting.")
        else:
            with st.spinner("The examiner is evaluating your answer…"):
                ctx, _ = get_context(v["current_q"] + " " + ans, v["exp"], k=4)
                if ctx is None:
                    return
                user = (f"LAB MANUAL PASSAGES:\n{ctx}\n\nVIVA QUESTION:\n{v['current_q']}\n\nSTUDENT ANSWER:\n{ans}")
                text = run_llm(build_messages(VIVA_EVAL_INSTR, user), temperature=0.2)
                if not text:
                    return
                ev = parse_viva_eval(text)
                v["turns"].append({"question": v["current_q"], "answer": ans.strip(),
                                   "difficulty": v["difficulty"], **ev})
                if len(v["turns"]) >= v["max_q"]:
                    finish_viva()
                else:
                    v["difficulty"] = adapt_difficulty(v["difficulty"], v["turns"])
                    q = viva_next_question()
                    v["current_q"] = q or "In your own words, explain one more key concept from this experiment."
            st.rerun()

    if st.button("🏁 End viva & see summary", key="viva_end"):
        finish_viva()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def safe_render(fn) -> None:
    """Show a friendly message instead of a raw stack trace if a section fails."""
    try:
        fn()
    except LabMateError as e:
        st.error(str(e))
    except Exception as e:  # StreamlitRerun/Stop are BaseExceptions, so they pass through
        st.error("Something unexpected went wrong in this section. Please try again, "
                 "or use ♻️ Reset entire session in the sidebar.")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception_only(type(e), e)))


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    safe_render(render_sidebar)
    tabs = st.tabs(["🏠 Dashboard", "📄 Lab Analyzer", "📚 Understand", "💬 Ask LabMate",
                    "💻 Workspace", "🐞 Debug", "🧠 Practice", "🎤 Viva"])
    for tab, fn in zip(tabs, (tab_dashboard, tab_analyzer, tab_understand, tab_ask,
                              tab_workspace, tab_debug, tab_practice, tab_viva)):
        with tab:
            safe_render(fn)


main()