# server/main.py  (replace your existing main with this)
import logging
import re
import time
import threading
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# NOTE: import heavy libs lazily inside functions (see get_embeddings/get_db/get_llm)
# so FastAPI can boot quickly without waiting for models to load.

# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 3
SESSION_TTL = 6 * 60 * 60  # 6 hours
SESSION_CLEAN_INTERVAL = 60 * 10  # 10 minutes

# -------------------------------
# LOGGING
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_rag")
# optionally reduce noisy libs if desired:
# logging.getLogger("chromadb").setLevel(logging.WARNING)
# logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# -------------------------------
# SIMPLE IN-MEMORY SESSIONS
# -------------------------------
user_sessions: Dict[str, Dict[str, Any]] = {}


def create_session() -> str:
    sid = str(uuid.uuid4())
    user_sessions[sid] = {
        "memory": None,  # lazy-create ConversationBufferMemory when needed
        "last_active": time.time(),
    }
    logger.info(f"🆕 Created session: {sid}")
    return sid


def get_session(sid: str) -> Optional[Dict[str, Any]]:
    info = user_sessions.get(sid)
    if info is not None:
        info["last_active"] = time.time()
    return info


def clear_session(sid: str):
    if sid in user_sessions:
        del user_sessions[sid]
        logger.info(f"Cleared session: {sid}")


def cleanup_sessions_once():
    now = time.time()
    expired = [sid for sid, info in user_sessions.items() if now - info["last_active"] > SESSION_TTL]
    for sid in expired:
        del user_sessions[sid]
        logger.info(f"🧹 Cleaned expired session: {sid}")


def schedule_session_cleanup():
    def loop():
        while True:
            try:
                cleanup_sessions_once()
            except Exception:
                logger.exception("Session cleanup failed")
            time.sleep(SESSION_CLEAN_INTERVAL)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


# -------------------------------
# LAZY LOAD MODELS (caching)
# -------------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    # lazy import heavy libraries
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Loading HuggingFace embeddings...")
        emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        logger.info("✅ Embeddings loaded")
        return emb
    except Exception as e:
        logger.exception("Failed to load embeddings")
        raise e


@lru_cache(maxsize=1)
def get_db():
    try:
        from langchain_chroma import Chroma

        logger.info("Opening Chroma DB (persist dir=%s)...", DB_DIR)
        emb = get_embeddings()
        db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
        logger.info("✅ Chroma DB opened")
        return db
    except Exception as e:
        logger.exception("Failed to open Chroma DB")
        raise e


@lru_cache(maxsize=1)
def get_llm():
    try:
        from langchain_community.llms import Ollama

        logger.info("Loading Ollama LLM (model=%s)...", LLM_MODEL)
        llm = Ollama(model=LLM_MODEL)
        logger.info("✅ Ollama LLM ready")
        return llm
    except Exception as e:
        logger.exception("Failed to load Ollama model")
        raise e


# -------------------------------
# PROMPT / HELPERS
# -------------------------------
def build_prompt(context: str, question: str, chat_history: str) -> str:
    template = """
SYSTEM:
You are "CampusGuide," a friendly AI assistant for VIT-AP students.
Respond naturally, like a fellow student, and answer neatly in 3 lines max with 2-3 emojies.

Rules:
- Answer only what is asked and don't unnecessarily say anything.
- Keep tone conversational and helpful.
- Use CONTEXT for VIT-AP related questions.
- If CONTEXT is empty, say you don’t have info and suggest official sources.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER'S QUESTION:
{question}

YOUR ANSWER:
"""
    return template.format(context=context, question=question, chat_history=chat_history)


def get_filter_from_question(question: str) -> Optional[Dict[str, str]]:
    q = (question or "").lower()
    if re.search(r"\bleave\b|\bouting\b|\bweekend\b", q):
        return {"category": "VTOP"}
    if re.search(r"\bplacement\b|\bpat\b", q):
        return {"category": "Placements"}
    if "hostel" in q:
        return {"category": "Hostels"}
    if "startup" in q:
        return {"category": "Startups"}
    if "guest" in q:
        return {"category": "GuestHouse"}
    return None


def sanitize_response(text: Optional[str]) -> str:
    if not text:
        return ""
    t = str(text)
    t = t.replace("undefined", "").replace("None", "")
    return t.strip()


# -------------------------------
# QUERY HANDLER (main RAG logic)
# -------------------------------
def run_query(query: str, memory) -> str:
    q = (query or "").strip()
    if not q:
        return ""

    q_lower = q.lower()

    # quick greetings/farewells
    greetings = {"hi", "hello", "hey", "hola", "how are you", "what's up"}
    farewells = {"bye", "goodbye", "see you", "adios", "tata", "fine", "thank you", "thanks", "okay", "okie", "ok"}
    if q_lower in greetings:
        try:
            llm = get_llm()
            return llm.invoke(f"System: Respond casually and warmly to this greeting with 2-3 emojis: '{q}'")
        except Exception:
            return "Hey! 👋"

    if q_lower in farewells:
        try:
            llm = get_llm()
            return llm.invoke(f"System: Respond with a friendly 2-line farewell message: '{q}'")
        except Exception:
            return "Bye! 🙌"

    # retrieval stage
    try:
        db = get_db()
    except Exception as e:
        logger.error("Chroma DB unavailable: %s", e)
        # fallback to LLM only
        try:
            llm = get_llm()
            return llm.invoke(build_prompt(context="", question=q, chat_history=""))
        except Exception:
            return "Sorry, the service is temporarily unavailable."

    # prepare retriever kwargs
    top_k = TOP_K
    search_kwargs = {"k": top_k, "fetch_k": max(20, top_k * 5)}
    filt = get_filter_from_question(q)
    if filt:
        search_kwargs["filter"] = filt

    retriever = db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    # Use .invoke if present (newer langchain), otherwise try legacy method
    docs = []
    try:
        # invoke may return a list of docs, or a LangChain result object; handle flexibly
        result = retriever.invoke(q)  # preferred (newer API)
        # try to interpret different return shapes
        if isinstance(result, list):
            docs = result
        elif hasattr(result, "documents"):
            docs = getattr(result, "documents")
        elif hasattr(result, "text"):
            # some retriever wrappers return a text field
            docs = [result]
        else:
            # fallback: try calling similarity search if available
            docs = retriever.get_relevant_documents(q) if hasattr(retriever, "get_relevant_documents") else []
    except Exception:
        # older langchain installations might not have invoke; fallback gracefully
        try:
            docs = retriever.get_relevant_documents(q)
        except Exception:
            logger.exception("Retriever failed")
            docs = []

    # if filtered search returned nothing, retry unfiltered once
    if not docs and filt:
        try:
            retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": search_kwargs["fetch_k"]})
            try:
                result = retriever.invoke(q)
                if isinstance(result, list):
                    docs = result
                elif hasattr(result, "documents"):
                    docs = getattr(result, "documents")
            except Exception:
                if hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(q)
        except Exception:
            docs = []

    context = ""
    if docs:
        # assemble context text safely
        parts = []
        for d in docs[: TOP_K]:
            # `d` may be a LangChain Document-like object or a simple dict
            if hasattr(d, "page_content"):
                parts.append(d.page_content)
            elif isinstance(d, dict) and "page_content" in d:
                parts.append(d["page_content"])
            elif isinstance(d, dict) and "content" in d:
                parts.append(d["content"])
            elif isinstance(d, str):
                parts.append(d)
            else:
                # generic fallback
                parts.append(str(d))
        context = "\n\n".join(parts)

    # if no context, return safe fallback instead of hallucinating
    if not context.strip():
        return "Hmm, I don’t have exact info on that. You can check the official VIT-AP website or contact the admin for the most accurate details."

    # build prompt and call llm
    chat_history = ""
    try:
        # attempt to read memory messages if available (backwards-compatible)
        if memory is not None and hasattr(memory, "chat_memory"):
            msgs = getattr(memory.chat_memory, "messages", []) or []
            chat_history = "\n".join([f"{'User' if getattr(m, 'type', 'human') == 'human' else 'CampusGuide'}: {getattr(m, 'content', str(m))}" for m in msgs])
    except Exception:
        chat_history = ""

    prompt = build_prompt(context=context, question=q, chat_history=chat_history)
    try:
        llm = get_llm()
        resp = llm.invoke(prompt)
        return sanitize_response(resp)
    except Exception:
        logger.exception("LLM invoke failed")
        return sanitize_response("Sorry, I couldn't generate an answer right now.")


# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI(title="VIT-AP CampusGuide (Optimized RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskPayload(BaseModel):
    question: str


@app.on_event("startup")
def on_startup():
    # schedule periodic cleanup of sessions
    schedule_session_cleanup()

    # start a non-blocking warm-up thread to reduce cold-start latency
    def warmup():
        try:
            # try to lazy-init models quietly
            logger.info("Warm-up: lazy-loading embeddings and DB in background...")
            try:
                _ = get_embeddings()
                _ = get_db()
            except Exception:
                logger.info("DB/embeddings warmup skipped (error).")
            # LLM warmup (light)
            try:
                llm = get_llm()
                llm.invoke("Hello. Warmup.")
            except Exception:
                logger.info("LLM warmup skipped (error).")
            logger.info("🔥 Warm-up completed.")
        except Exception:
            logger.exception("Warmup thread failed.")

    t = threading.Thread(target=warmup, daemon=True)
    t.start()


@app.post("/ask")
def ask(payload: AskPayload, request: Request):
    # quick session housekeeping
    sid = request.headers.get("X-Session-ID")
    clear_chat = request.headers.get("X-Clear-Chat", "false").lower() == "true"

    # create session if not provided
    if not sid or sid not in user_sessions:
        sid = create_session()

    # clear session when requested
    if clear_chat:
        # if session exists, reset its memory
        if sid in user_sessions:
            user_sessions[sid]["memory"] = None
        return {"answer": "Chat history has been cleared!", "session_id": sid}

    # ensure memory object exists lazily (to avoid import-time deprecation noise)
    if user_sessions[sid]["memory"] is None:
        try:
            # lazy import for ConversationBufferMemory
            from langchain.memory import ConversationBufferMemory

            user_sessions[sid]["memory"] = ConversationBufferMemory(return_messages=True)
        except Exception:
            # if memory cannot be created, store None and continue; chat_history will be empty
            user_sessions[sid]["memory"] = None

    memory = user_sessions[sid]["memory"]

    question = (payload.question or "").strip()
    if not question:
        return {"answer": "", "session_id": sid}

    # run main RAG logic
    answer = run_query(question, memory)

    # update memory if available
    try:
        if memory is not None:
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
    except Exception:
        logger.exception("Failed to update memory; continuing.")

    user_sessions[sid]["last_active"] = time.time()
    return {"answer": sanitize_response(answer), "session_id": sid}


@app.get("/")
def root():
    return {"message": "✅ VIT-AP CampusGuide (Optimized RAG) is running."}


# If you want to measure boot time, start script outside (optional)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
