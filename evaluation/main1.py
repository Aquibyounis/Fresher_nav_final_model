"""
CampusGuide: Conversational RAG Assistant
Using LLaMA (via Ollama) + Chroma retriever
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory


# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"    # Change if your Ollama model name differs
TOP_K = 3                 # Number of docs to retrieve


# -------------------------------
# MEMORY
# -------------------------------
memory = ConversationBufferMemory(return_messages=True)


# -------------------------------
# SETUP RETRIEVER
# -------------------------------
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    # Hybrid-style retrieval (MMR = diverse + relevant)
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 10}
    )


# -------------------------------
# PROMPT TEMPLATE
# -------------------------------
def build_prompt(context, question, chat_history):
    template = """
SYSTEM:
You are "CampusGuide" — a warm, natural, human-like assistant for VIT-AP.

Rules:
- If it's small-talk (hi, hello, how are you, what's up, hola, etc.) → respond casually, like a real person. Do NOT repeat "I'm CampusGuide" every time.
- If it's a factual or general knowledge question → answer directly, no need for context.
- If it's a college-related interrogative question → use this context:
{context}
  * If context has numbered steps, show them in order.
- If context doesn't contain the answer → say: "I don't see that in the info I have — here's a safe next step."
- Always keep a friendly, student-to-student tone. Short sentences, no stiff policy language.
- Use chat history to stay consistent and personal.

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

Answer:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )
    return prompt.format(context=context, question=question, chat_history=chat_history)


# -------------------------------
# RUN QUERY
# -------------------------------
def run_query(query: str) -> str:
    retriever = load_retriever()
    llm = OllamaLLM(model=LLM_MODEL)

    # Retrieve context
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""

    # Get chat history
    history = "\n".join([f"{m.type}: {m.content}" for m in memory.chat_memory.messages])

    # Build prompt
    prompt = build_prompt(context, query, history)

    # If no context retrieved and it's casual chit-chat → fallback
    if not context.strip() and query.lower() in ["hi", "hello", "hey", "hola", "how are you", "what's up"]:
        response = llm.invoke(f"Respond casually to: {query}")
    else:
        response = llm.invoke(prompt)

    # Save interaction in memory
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response)

    return response


# -------------------------------
# MAIN LOOP
# -------------------------------
def main():
    print("🫂 CampusGuide is ready! Type 'exit' to quit.")
    while True:
        query = input("\n❓ Ask a question (or 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\n🤖 CampusGuide:\n", run_query(query))


if __name__ == "__main__":
    main()
