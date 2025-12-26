"""
CampusGuide: Conversational RAG Assistant
Model: LLaMA 3.2 (via Ollama)
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
TOP_K = 3

# Context window control (ADDED)
MAX_CONTEXT_CHARS = 2000

# -------------------------------
# RETRIEVER
# -------------------------------
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 10}
    )

# -------------------------------
# PROMPT
# -------------------------------
def build_prompt(context, question):
    template = """
SYSTEM:
You are CampusGuide, a helpful assistant for college students.

Rules:
- Use CONTEXT if available for college-related questions.
- If CONTEXT is empty, say you don't have exact info and suggest official sources.
- Keep answers clear and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    return prompt.format(context=context, question=question)

# -------------------------------
# RUN QUERY (returns answer, context)
# -------------------------------
def run_query(query: str):
    retriever = load_retriever()
    llm = OllamaLLM(model=LLM_MODEL)

    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""

    # ---- CONTEXT WINDOWING (IMPORTANT) ----
    context = context[:MAX_CONTEXT_CHARS]

    prompt = build_prompt(context, query)
    response = llm.invoke(prompt)

    return response, context

# -------------------------------
# OPTIONAL LOCAL TEST
# -------------------------------
def main():
    while True:
        q = input("\n❓ Question (or exit): ")
        if q.lower() in ["exit", "quit"]:
            break
        ans, _ = run_query(q)
        print("\n🤖 LLaMA:\n", ans)

if __name__ == "__main__":
    main()
