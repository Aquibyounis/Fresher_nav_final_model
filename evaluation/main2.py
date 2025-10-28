"""
Chatbot using Mistral + Chroma retriever.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"   # change if your Ollama mistral model has a different name
TOP_K = 2


# -------------------------------
# SETUP
# -------------------------------
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": TOP_K})


def build_prompt(context, question):
    template = """
SYSTEM:
You are "CampusGuide" — a friendly, human-sounding assistant for [College Name].  
Rules:
- If it is simple question or general knowledge question answer it even without context
- If it is interrogative Question then use the provided {context} (procedures, policies, contacts) to answer the {question}. If context contains numbered steps, reproduce them in order.
- Tone: conversational (use contractions, short sentences, no legalese). Avoid robotic templates like "as per policy". Use plain language.
- If context lacks required info, say: "I don't have X in the context — here's a safe suggested next step." then follow the structure.
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    return prompt.format(context=context, question=question)


# -------------------------------
# RUN QUERY (for eval.py import)
# -------------------------------
def run_query(query: str) -> str:
    retriever = load_retriever()
    llm = OllamaLLM(model=LLM_MODEL)

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = build_prompt(context, query)
    return llm.invoke(prompt)


# -------------------------------
# MAIN (interactive)
# -------------------------------
def main():
    while True:
        query = input("\n❓ Ask a question (or 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        print("\n🌊 Mistral Answer:\n", run_query(query))


if __name__ == "__main__":
    main()
