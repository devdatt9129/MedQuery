import os
import re
import json
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# NEW: hybrid retrieval helper
from . import retrieval

# LangChain core
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings / LLM
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Vector store (modern)
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

# Chat histories (Redis)
from langchain_community.chat_message_histories import RedisChatMessageHistory

from .settings import settings

# ---------------------------
# Globals
# ---------------------------
_STORE: Chroma | None = None
_MEMORY_STORES: dict[str, Chroma] = {}
_CHAT_STORES: dict[str, Chroma] = {}

# One telemetry setting for all Chroma clients
_CHROMA_CLIENT_SETTINGS = ChromaSettings(anonymized_telemetry=False)

# ---------------------------
# Utilities
# ---------------------------
def _require_redis_url() -> str:
    url = os.getenv("REDIS_URL")
    if not url:
        raise RuntimeError(
            "REDIS_URL is not set. Example local: redis://localhost:6379 "
            "Example Upstash: rediss://default:<PASSWORD>@<HOST>:6379"
        )
    try:
        import redis  # type: ignore
        r = redis.from_url(url)
        r.ping()
    except Exception as e:
        raise RuntimeError(
            f"Cannot connect to Redis at {url}. Start Redis or fix REDIS_URL. "
            f"Mac: 'brew services start redis' | Docker: 'docker run -p 6379:6379 redis:7'. "
            f"Original error: {e}"
        )
    return url

def _load_sources() -> List[Document]:
    base = Path(__file__).resolve().parents[1] / "data"
    transcript = json.loads((base / "transcript.json").read_text())
    soap = json.loads((base / "soap_note.json").read_text())

    # Flatten transcript turns into one doc (keep time + speaker)
    turns = transcript["content"]
    joined = "\n".join([f"[{t['time']}] {t['speaker']}: {t['text']}" for t in turns])

    docs: List[Document] = [
        Document(
            page_content=joined,
            metadata={"doc_id": "tx", "title": transcript.get("title", "Transcript")},
        )
    ]
    # SOAP sections as distinct docs
    for section_key, section_name in [
        ("subjective", "S"),
        ("objective", "O"),
        ("assessment", "A"),
        ("plan", "P"),
    ]:
        if section_key in soap:
            docs.append(
                Document(
                    page_content=json.dumps(soap[section_key], ensure_ascii=False),
                    metadata={"doc_id": "soap", "section": section_name},
                )
            )
    return docs

def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def _get_embeddings():
    if settings.EMBEDDINGS_PROVIDER.lower() == "e5":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.HF_E5_MODEL)
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBED_MODEL, api_key=settings.OPENAI_API_KEY
    )

# ---------------------------
# Main clinical vector store
# ---------------------------
def get_vectorstore() -> Chroma:
    global _STORE
    if _STORE:
        return _STORE
    docs = _split_docs(_load_sources())
    # Build BM25 over the same split docs
    retrieval.init_bm25(docs)
    embeddings = _get_embeddings()
    _STORE = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DIR,
        client_settings=_CHROMA_CLIENT_SETTINGS,
    )
    return _STORE

# ---------------------------
# Chat history (Redis REQUIRED)
# ---------------------------
def get_history(session_id: str) -> RedisChatMessageHistory:
    url = _require_redis_url()
    return RedisChatMessageHistory(session_id=session_id, url=url)

# ---------------------------
# Fact memory (Layer 2)
# ---------------------------
def get_memory_store(session_id: str) -> Chroma:
    if session_id in _MEMORY_STORES:
        return _MEMORY_STORES[session_id]
    emb = _get_embeddings()
    store = Chroma(
        collection_name=f"mem-{session_id}",
        embedding_function=emb,
        persist_directory=f"{settings.CHROMA_DIR}/mem",
        client_settings=_CHROMA_CLIENT_SETTINGS,
    )
    _MEMORY_STORES[session_id] = store
    return store

def extract_facts(answer_text: str) -> list[str]:
    if not answer_text or not answer_text.strip():
        return []
    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    )
    prompt = (
        "Extract 1 to 3 concise factual statements from the following assistant answer. "
        "Focus on stable clinical details (symptom duration, triggers, allergies, medications, tests, plan). "
        "Return ONLY a JSON array of strings.\n\n"
        f"{answer_text}"
    )
    try:
        out = llm.invoke(prompt).content
        facts = json.loads(out)
        facts = [f.strip() for f in facts if isinstance(f, str) and 3 <= len(f) <= 200]
        return facts
    except Exception:
        return []

# ---------------------------
# Chat memory (semantic over prior turns)
# ---------------------------
def get_chat_store(session_id: str) -> Chroma:
    if session_id in _CHAT_STORES:
        return _CHAT_STORES[session_id]
    emb = _get_embeddings()
    store = Chroma(
        collection_name=f"chat-{session_id}",
        embedding_function=emb,
        persist_directory=f"{settings.CHROMA_DIR}/chat",
        client_settings=_CHROMA_CLIENT_SETTINGS,
    )
    _CHAT_STORES[session_id] = store
    return store

def index_chat_turn(session_id: str, role: str, content: str):
    if not content or not content.strip():
        return
    store = get_chat_store(session_id)
    ts = datetime.utcnow().isoformat(timespec="seconds")
    doc = Document(
        page_content=content,
        metadata={"doc_id": "chat", "role": role, "ts": ts, "turn_id": str(uuid4())},
    )
    store.add_documents([doc])

def search_chat(session_id: str, query: str, k: int = 5) -> List[Document]:
    store = get_chat_store(session_id)
    return store.similarity_search(query, k=k)

def format_chat_hits(hits: List[Document]) -> str:
    lines = []
    for d in hits:
        role = d.metadata.get("role", "?").capitalize()
        ts = d.metadata.get("ts", "")
        lines.append(f"- {ts} {role}: {d.page_content}")
    return "\n".join(lines)

def _chroma_count(store: Chroma) -> int:
    try:
        return store._collection.count()  # type: ignore[attr-defined]
    except Exception:
        return 0

# ---------------------------
# Control intents
# ---------------------------
REPEAT_Q = re.compile(
    r"(?:\brepeat\b|\bsay that again\b|\bagain please\b|\bplease repeat\b|\brepeat that\b|\bwhat did you say\b|\bcan you say that again\b)",
    re.IGNORECASE,
)
SUMMARY_Q = re.compile(
    r"(?:\bsummarize\b|\bsummary\b|\btl;dr\b|\bwhat have we discussed\b|\bso far\b)",
    re.IGNORECASE,
)
HISTORY_Q = re.compile(
    r"(what did (?:i|we|you) (?:say|discuss|decide|mention)|"
    r"remind me|earlier|previous answer|previously|before|again about|"
    r"what did you answer|what did i ask|list (?:my|the) (?:questions|asks))",
    re.IGNORECASE,
)
# NEW: deterministic question-order intents
FIRST_Q_RE = re.compile(r"\b(first|very first|initial)\s+(question|ask|message)\b|\bfirst question i asked\b", re.IGNORECASE)
LAST_Q_RE  = re.compile(r"\b(last|most recent|previous)\s+(question|ask|message)\b|\blast question i asked\b", re.IGNORECASE)
LIST_Q_RE  = re.compile(r"\blist (?:my|the) (?:questions|asks)\b|\bwhat have i asked\b", re.IGNORECASE)

# --- helpers for deterministic intents (+ snapshot first question) ---
def _user_questions_from_history(history: RedisChatMessageHistory) -> List[str]:
    return [m.content for m in history.messages if getattr(m, "type", "") == "human"]

def _redis_client():
    import redis  # type: ignore
    url = _require_redis_url()
    r = redis.from_url(url)
    r.ping()
    return r

def remember_first_question(session_id: str, question: str):
    """Set the first question once (idempotent)."""
    try:
        r = _redis_client()
        key = f"firstq:{session_id}"
        r.setnx(key, question)
    except Exception:
        pass

def get_remembered_first_question(session_id: str) -> Optional[str]:
    try:
        r = _redis_client()
        val = r.get(f"firstq:{session_id}")
        if val:
            return val.decode()
    except Exception:
        pass
    return None

# ---------------------------
# Main entry
# ---------------------------
def ask(question: str, session_id: str) -> Tuple[str, List[Dict]]:
    """
    - Handles repeat / summarize / history queries directly from chat memory.
    - Adds deterministic first/last/list question handlers (order-based).
    - Otherwise does omni-retrieval (chat + facts + main docs) with BM25 hybrid and answers with a grounded prompt.
    - Indexes every turn into chat memory and updates fact memory.
    """
    _require_redis_url()  # ensure Redis works

    llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.0,
    )
    history = get_history(session_id)

    # snapshot the first question early
    remember_first_question(session_id, question)

    # --------- REPEAT ----------
    if REPEAT_Q.search(question.lower()):
        from langchain_core.messages import AIMessage
        last = None
        for msg in reversed(history.messages):
            if isinstance(msg, AIMessage):
                last = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
        if last:
            history.add_user_message(question)
            history.add_ai_message(last)
            index_chat_turn(session_id, "user", question)
            index_chat_turn(session_id, "assistant", last)
            return last, [{"doc_id": "history", "snippet": "Repeated previous assistant answer"}]

    # --------- SUMMARIZE (chat) ----------
    if SUMMARY_Q.search(question.lower()):
        convo = []
        for m in history.messages:
            t = getattr(m, "type", "")
            c = getattr(m, "content", "")
            if t == "human": convo.append(f"User: {c}")
            elif t == "ai":  convo.append(f"Assistant: {c}")
        convo_text = "\n".join(convo)
        if convo_text.strip():
            prompt = (
                "Summarize the following conversation between a clinician and the assistant. "
                "Be concise and factual, focusing on questions and discussed facts.\n\n"
                f"{convo_text}"
            )
            summary = llm.invoke(prompt).content
            history.add_user_message(question)
            history.add_ai_message(summary)
            index_chat_turn(session_id, "user", question)
            index_chat_turn(session_id, "assistant", summary)
            return summary, [{"doc_id": "history", "snippet": "Summary of chat conversation"}]

    # --------- DETERMINISTIC QUESTION ORDER INTENTS ----------
    q_lower = question.lower()

    if FIRST_Q_RE.search(q_lower):
        snap = get_remembered_first_question(session_id)
        if not snap:
            qs = _user_questions_from_history(history)
            snap = qs[0] if qs else None
        answer_text = f'The first question you asked was:\n\n“{snap}”.' if snap else "I don't have any earlier question recorded yet."
        history.add_user_message(question); history.add_ai_message(answer_text)
        index_chat_turn(session_id, "user", question); index_chat_turn(session_id, "assistant", answer_text)
        return answer_text, [{"doc_id": "history", "snippet": "First user question"}]

    if LAST_Q_RE.search(q_lower):
        qs = _user_questions_from_history(history)
        last_q = qs[-1] if qs else None
        answer_text = f'The last question you asked was:\n\n“{last_q}”.' if last_q else "I don't have any earlier question recorded yet."
        history.add_user_message(question); history.add_ai_message(answer_text)
        index_chat_turn(session_id, "user", question); index_chat_turn(session_id, "assistant", answer_text)
        return answer_text, [{"doc_id": "history", "snippet": "Last user question"}]

    if LIST_Q_RE.search(q_lower):
        qs = _user_questions_from_history(history)
        if qs:
            bullets = "\n".join(f"- {x}" for x in qs[-50:])  # cap for readability
            answer_text = f"Here are your previous questions (oldest → newest):\n\n{bullets}"
        else:
            answer_text = "I don't have any earlier question recorded yet."
        history.add_user_message(question); history.add_ai_message(answer_text)
        index_chat_turn(session_id, "user", question); index_chat_turn(session_id, "assistant", answer_text)
        return answer_text, [{"doc_id": "history", "snippet": "Listed previous user questions"}]

    # --------- HISTORY QUERIES (semantic over prior turns) ----------
    if HISTORY_Q.search(question.lower()):
        hits = search_chat(session_id, question, k=6)
        if hits:
            answer_text = "Here’s what we discussed earlier that matches your request:\n\n" + format_chat_hits(hits)
            history.add_user_message(question); history.add_ai_message(answer_text)
            index_chat_turn(session_id, "user", question); index_chat_turn(session_id, "assistant", answer_text)
            cits = [
                {"doc_id": d.metadata.get("doc_id", "chat"),
                 "section": d.metadata.get("role"),
                 "snippet": d.page_content[:200]}
                for d in hits
            ]
            return answer_text, cits

    # --------- DEFAULT: omni-retrieval (chat + facts + main docs) ----------
    # a) clinical docs (dense)
    main_ret = get_vectorstore().as_retriever(search_kwargs={"k": 4})
    dense_main = main_ret.invoke(question)  # modern API

    # b) distilled fact memory (dynamic k)
    mem_k  = 3 if _chroma_count(get_memory_store(session_id)) >= 3 else 2
    chat_k = 3 if _chroma_count(get_chat_store(session_id))   >= 3 else 2

    mem_docs  = get_memory_store(session_id).similarity_search(question, k=mem_k)
    chat_docs = search_chat(session_id, question, k=chat_k)

    # c) hybrid + hierarchical ensemble (BM25 + dense + chat + memory)
    all_docs = retrieval.ensemble_retrieve(
        question=question,
        dense_main_docs=dense_main,
        fact_mem_docs=mem_docs,
        chat_docs=chat_docs,
        bm25_k=6,
        final_k=8,
    )

    # Evidence gate
    is_ok, _reason = retrieval.answerability_check(question, all_docs, min_docs=1, min_unique_sources=1)
    if not is_ok:
        answer_text = "Not stated in the transcript/SOAP or prior chat."
        return answer_text, [{"doc_id": "tx", "snippet": "No sufficient evidence found"}]

    # grounded prompt
    context_parts = []
    for d in all_docs[:8]:
        tag = d.metadata.get("doc_id", "doc")
        role = d.metadata.get("role")
        header = f"[{tag}{f'/{role}' if role else ''}]"
        context_parts.append(f"{header} {d.page_content}")
    context_text = "\n\n---\n\n".join(context_parts) if context_parts else "None."

    prompt = (
        "You are a clinical QA assistant. Answer strictly from the provided context.\n"
        "Context may include prior chat turns [chat/*], distilled facts [memory], and clinical docs [tx|soap]. "
        "Prefer tx/soap for medical facts; chat/memory reflect this session. "
        "If unknown, reply: 'Not stated in the transcript/SOAP or prior chat.'\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )
    answer_text = llm.invoke(prompt).content

    # citations (mirror what we sent)
    cits: List[Dict] = []
    for d in all_docs[:8]:
        cits.append({
            "doc_id": d.metadata.get("doc_id", "doc"),
            "section": d.metadata.get("role") or d.metadata.get("section"),
            "snippet": d.page_content[:200],
        })

    # persist chat turn + index chat memory
    history.add_user_message(question)
    history.add_ai_message(answer_text)
    index_chat_turn(session_id, "user", question)
    index_chat_turn(session_id, "assistant", answer_text)

    # update fact memory
    facts = extract_facts(answer_text)
    if facts:
        mem_store = get_memory_store(session_id)
        docs_to_add = [Document(page_content=f, metadata={"doc_id": "memory"}) for f in facts]
        mem_store.add_documents(docs_to_add)

    return answer_text, cits