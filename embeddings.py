import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# CONFIG
# -------------------------------
DATASET_FOLDER = "dataset"
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embeddings and Chroma DB
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# -------------------------------
# LOAD JSON FILE NAMES FROM names.txt
# -------------------------------
with open("names.txt", "r", encoding="utf-8") as f:
    json_files = [line.strip() for line in f if line.strip()]

print(f"Found {len(json_files)} JSON files:", json_files)

# -------------------------------
# PROCESS EACH JSON FILE
# -------------------------------
for json_file in json_files:
    path = os.path.join(DATASET_FOLDER, json_file)
    print(f"Processing {json_file}...")

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

        texts = [entry["text"] for entry in entries]
        metadatas = [
            {
                "title": entry.get("title", ""),
                "category": entry.get("category", ""),
                "type": entry.get("type", ""),
                "tags": entry.get("tags", "")
            }
            for entry in entries
        ]

        db.add_texts(texts, metadatas=metadatas)

# -------------------------------
# DATABASE IS AUTOMATICALLY PERSISTED
# -------------------------------
print("✅ All embeddings created and stored in new_db!")
