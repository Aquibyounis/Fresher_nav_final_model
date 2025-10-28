from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
import re
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 3
SESSION_TTL = 6 * 60 * 60  # 6 hours

# -------------------------------
# PER-USER MEMORY
# -------------------------------
# In-memory session storage. For production, use Redis or another store.
user_sessions = {}

# -------------------------------
# INITIALIZE MODELS (load once)
# -------------------------------
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = Ollama(model=LLM_MODEL)
    logger.info("✅ Models and Chroma DB loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load models or DB: {e}")
    exit()

# -------------------------------
# HELPERS
# -------------------------------
def get_filter_from_question(question: str) -> dict | None:
    """Return metadata filter based on keywords."""
    q = question.lower()
    if re.search(r"\bleave\b|\bouting\b|\bweekend\b", q):
        return {"category": "VTOP"}
    elif re.search(r"\bplacement\b|\bpat\b", q):
        return {"category": "Placements"}
    elif "hostel" in q:
        return {"category": "Hostels"}
    elif "startup" in q:
        return {"category": "Startups"}
    elif "guest" in q:
        return {"category": "GuestHouse"}
    return None


def build_prompt(context: str, question: str, chat_history: str) -> str:
    """Builds a concise prompt for 5-line friendly answers."""
    template = """
SYSTEM:
You are "CampusGuide," a friendly AI assistant for VIT-AP students.
Respond naturally, like a fellow student, and answer neatly in 3 lines max.

Rules:
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
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )
    return prompt.format(context=context, question=question, chat_history=chat_history)


def run_query(query: str, memory: ConversationBufferMemory) -> str:
    """Handles query and returns a 5-line friendly answer."""
    q_lower = query.lower().strip()

    # Handle greetings and farewells
    greetings = ["hi", "hello", "hey", "hola", "how are you", "what's up"]
    farewells = ["bye", "goodbye", "see you", "adios", "tata"]

    if q_lower in greetings:
        return llm.invoke(f"System: Respond casually and warmly to this greeting: '{query}'")
    if q_lower in farewells:
        return llm.invoke("System: Respond with a friendly 2-line farewell message.")

    # Retrieve relevant context
    filter_metadata = get_filter_from_question(query)
    search_kwargs = {"k": TOP_K, "fetch_k": 10}
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata

    retriever = db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    docs = retriever.get_relevant_documents(query)

    if not docs and filter_metadata:
        logger.info(f"No results for '{query}' with filter. Retrying unfiltered.")
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": 10})
        docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    history_messages = memory.chat_memory.messages
    chat_history = "\n".join(
        [f"{'User' if m.type == 'human' else 'CampusGuide'}: {m.content}" for m in history_messages]
    )

    # Generate response
    if not context.strip():
        response = (
            "Hmm, I don’t have exact info on that. You can check the official VIT-AP website "
            "or contact the admin for the most accurate details."
        )
    else:
        prompt = build_prompt(context=context, question=query, chat_history=chat_history)
        response = llm.invoke(prompt)

    # Update memory
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response)
    return response


# -------------------------------
# SESSION MANAGEMENT
# -------------------------------
def cleanup_sessions():
    """Removes sessions inactive beyond TTL."""
    now = time.time()
    expired_sessions = [
        sid for sid, info in user_sessions.items()
        if now - info["last_active"] > SESSION_TTL
    ]
    for sid in expired_sessions:
        del user_sessions[sid]
        logger.info(f"🧹 Cleaned expired session: {sid}")


# -------------------------------
# FASTAPI SETUP
# -------------------------------
app = FastAPI(title="VIT-AP CampusGuide API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(q: Question, request: Request):
    """Main endpoint for user queries and session control."""
    cleanup_sessions()

    session_id = request.headers.get("X-Session-ID")
    clear_chat = request.headers.get("X-Clear-Chat", "false").lower() == "true"

    # Handle chat clearing
    if session_id in user_sessions and clear_chat:
        del user_sessions[session_id]
        logger.info(f"Cleared chat for session: {session_id}")

    # Create or get session
    if not session_id or session_id not in user_sessions:
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {
            "memory": ConversationBufferMemory(return_messages=True),
            "last_active": time.time(),
        }
        logger.info(f"🆕 Created new session: {session_id}")

    session_info = user_sessions[session_id]

    if clear_chat and not q.question.strip():
        return {"answer": "Chat history has been cleared!", "session_id": session_id}

    memory = session_info["memory"]

    # Run the main logic
    answer = run_query(q.question, memory)

    session_info["last_active"] = time.time()

    return {"answer": answer, "session_id": session_id}


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "✅ VIT-AP CampusGuide API is running."}
