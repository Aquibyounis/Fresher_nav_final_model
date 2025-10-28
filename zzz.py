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
SESSION_TTL = 6 * 60 * 60  # 6 hours in seconds

# -------------------------------
# PER-USER MEMORY
# -------------------------------
# In-memory session storage. For production, consider Redis or another persistent store.
# session_id -> {"memory": ConversationBufferMemory(), "last_active": timestamp, "last_question": str}
user_sessions = {}

# -------------------------------
# INITIALIZE MODELS (Load once on startup for efficiency)
# -------------------------------
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = Ollama(model=LLM_MODEL)
    logger.info("Models and Chroma DB loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load models or DB: {e}")
    # Exit if core components fail to load
    exit()

# -------------------------------
# HELPERS
# -------------------------------
def get_filter_from_question(question: str) -> dict | None:
    """Determines a metadata filter based on question keywords."""
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
    """Builds the final prompt for the LLM with clear instructions."""
    template = """
SYSTEM:
You are "CampusGuide," a friendly and helpful AI assistant for students at VIT-AP.
Your tone should be like talking to a fellow student—approachable, and supportive.

Rules:
- Answer any question in 5 lines.
- For greetings, farewells, or small talk, respond naturally and casually in 2 lines.
- For questions about VIT-AP, use the provided "CONTEXT" to form your answer.
  - If the context is empty, state that you don't have specific information and suggest checking official VIT-AP sources.
- You must remember the user's last question. If the user asks "What was the last question I asked?", you will respond with only that question.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER'S QUESTION:
{question}

YOUR FRIENDLY ANSWER:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )
    return prompt.format(context=context, question=question, chat_history=chat_history)

# -------------------------------
# QUERY HANDLER
# -------------------------------
def run_query(query: str, memory: ConversationBufferMemory, last_question: str = None) -> str:
    """Handles the full logic for responding to a user query."""
    q_lower = query.lower().strip()

    # Handle special case: "What was the last question I asked?"
    if "what was the last question i asked" in q_lower:
        return last_question or "You haven't asked any questions yet in this session."

    # Handle simple greetings and farewells without hitting the DB
    greetings = ["hi", "hello", "hey", "hola", "how are you", "what's up"]
    farewells = ["bye", "adios", "good bye", "goodbye", "see you later", "tata"]

    if q_lower in greetings:
        return llm.invoke(f"System: Respond casually and warmly to this greeting: '{query}'")
    if q_lower in farewells:
        return llm.invoke("System: Respond with a friendly farewell message.")

    # Main logic: Retrieve context and generate response
    filter_metadata = get_filter_from_question(query)
    
    search_kwargs = {"k": TOP_K, "fetch_k": 10}
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata

    retriever = db.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    docs = retriever.get_relevant_documents(query)

    # If filtered search yields no results, try a general search as a fallback
    if not docs and filter_metadata:
        logger.info(f"Filtered search for '{query}' found no results. Retrying without filter.")
        general_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": 10})
        docs = general_retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    
    if not context.strip():
        # Fallback response if no context is found at all
        response = "Hmm, I don't have specific information on that topic. It might be best to check the official VIT-AP website or contact the administration for the most accurate details."
    else:
        # Generate response using context and history
        history_messages = memory.chat_memory.messages
        chat_history = "\n".join([f"{'User' if m.type == 'human' else 'CampusGuide'}: {m.content}" for m in history_messages])
        
        prompt = build_prompt(context=context, question=query, chat_history=chat_history)
        response = llm.invoke(prompt)

    # Update memory with the current exchange
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response)
    
    return response

# -------------------------------
# SESSION MANAGEMENT
# -------------------------------
def cleanup_sessions():
    """Removes sessions that have been inactive for longer than SESSION_TTL."""
    now = time.time()
    expired_sessions = [
        sid for sid, info in user_sessions.items() 
        if now - info["last_active"] > SESSION_TTL
    ]
    for sid in expired_sessions:
        del user_sessions[sid]
        logger.info(f"Cleaned up expired session: {sid}")

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
    """Main endpoint to handle user questions, session management, and chat clearing."""
    cleanup_sessions()

    session_id = request.headers.get("X-Session-ID")
    clear_chat = request.headers.get("X-Clear-Chat", "false").lower() == "true"

    # If a clear request comes for an existing session, delete it.
    if session_id in user_sessions and clear_chat:
        del user_sessions[session_id]
        logger.info(f"Cleared chat history for session: {session_id}")
        # A new session will be created below.
        
    # Get or create a session for the user.
    if not session_id or session_id not in user_sessions:
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = {
            "memory": ConversationBufferMemory(return_messages=True),
            "last_active": time.time(),
            "last_question": None,
        }
        logger.info(f"Created new session: {session_id}")
    
    session_info = user_sessions[session_id]
    
    # Don't process an empty question if it was just for clearing the chat.
    if clear_chat and not q.question.strip():
        return {"answer": "Chat history has been cleared!", "session_id": session_id}

    # Retrieve memory and last question for the current session.
    memory = session_info["memory"]
    last_question = session_info.get("last_question")
    
    # Run the query logic.
    answer = run_query(q.question, memory, last_question)
    
    # Update session state after the query.
    session_info["last_active"] = time.time()
    session_info["last_question"] = q.question
    
    return {"answer": answer, "session_id": session_id}

@app.get("/")
def root():
    """Root endpoint for API health check."""
    return {"message": "VIT-AP CampusGuide API is running."}
