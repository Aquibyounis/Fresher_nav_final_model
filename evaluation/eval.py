import os
import time
import csv
import json
import subprocess
from typing import Dict

from main1 import run_query as llama_query
from main2 import run_query as mistral_query

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "evaluation_results.csv")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
JUDGE_MODEL = "llama2"

# -------------------------------
# QUESTIONS
# -------------------------------
questions = [
    "What is the difference between applying for leave through VTOP and applying for leave manually?",
    "What are the restrictions for weekend outings on VTOP?",
    "If VTOP does not allow leave due to technical issues, what can a student do?",
    "How does placement work at VIT-AP?",
    "How do students register for courses at VIT-AP?",
    "What happens if I lose my student ID card?",
    "Are there technical clubs at VIT-AP?",
    "I had a bad day. Can you cheer me up?",
    "Hi",
    "Can you tell me something funny about college life?"
]

# -------------------------------
# TOKEN COUNTER (simple & safe)
# -------------------------------
def count_tokens(text: str) -> int:
    if not text:
        return 0
    # simple approximation (acceptable for reporting)
    return len(text.split())

# -------------------------------
# LLM JUDGE
# -------------------------------
def ollama_judge_score(question: str, answer: str) -> Dict[str, int]:
    prompt = f"""
You are an evaluator. Score STRICTLY from 1 (poor) to 5 (excellent).

Criteria:
1. Relevance – does the answer directly address the question?
2. Fluency – is the language natural and human-like?
3. Completeness – does it cover all required points?

Return ONLY valid JSON like:
{{"relevance":X,"fluency":Y,"completeness":Z}}

Question: {question}
Answer: {answer}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", JUDGE_MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        text = result.stdout.decode("utf-8").strip()
        json_str = text[text.find("{"):text.rfind("}") + 1]
        return json.loads(json_str)
    except Exception:
        return {"relevance": 0, "fluency": 0, "completeness": 0}

# -------------------------------
# EVALUATE ONE MODEL
# -------------------------------
def evaluate_model(model_name: str, query_fn):
    rows = []
    totals = {
        "latency_ms": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "context_tokens": 0,
        "relevance": 0,
        "fluency": 0,
        "completeness": 0
    }

    for q in questions:
        start = time.time()
        answer, context = query_fn(q)  # IMPORTANT: return (answer, context)
        end = time.time()

        latency_ms = int((end - start) * 1000)

        in_tokens = count_tokens(q)
        out_tokens = count_tokens(answer)
        ctx_tokens = count_tokens(context)

        scores = ollama_judge_score(q, answer)
        avg_score = round(
            (scores["relevance"] + scores["fluency"] + scores["completeness"]) / 3, 2
        )

        row = {
            "Model": model_name,
            "Question": q,
            "Latency_ms": latency_ms,
            "Input_Tokens": in_tokens,
            "Output_Tokens": out_tokens,
            "Context_Tokens": ctx_tokens,
            "Relevance": scores["relevance"],
            "Fluency": scores["fluency"],
            "Completeness": scores["completeness"],
            "Avg_Quality": avg_score
        }
        rows.append(row)

        totals["latency_ms"] += latency_ms
        totals["input_tokens"] += in_tokens
        totals["output_tokens"] += out_tokens
        totals["context_tokens"] += ctx_tokens
        totals["relevance"] += scores["relevance"]
        totals["fluency"] += scores["fluency"]
        totals["completeness"] += scores["completeness"]

    n = len(questions)
    averages = {
        "Model": model_name,
        "Question": "AVERAGE",
        "Latency_ms": round(totals["latency_ms"] / n, 2),
        "Input_Tokens": round(totals["input_tokens"] / n, 2),
        "Output_Tokens": round(totals["output_tokens"] / n, 2),
        "Context_Tokens": round(totals["context_tokens"] / n, 2),
        "Relevance": round(totals["relevance"] / n, 2),
        "Fluency": round(totals["fluency"] / n, 2),
        "Completeness": round(totals["completeness"] / n, 2),
        "Avg_Quality": round(
            (totals["relevance"] + totals["fluency"] + totals["completeness"]) / (3 * n),
            2
        )
    }

    return rows, averages

# -------------------------------
# RUN EVALUATION
# -------------------------------
print("⏳ Evaluating LLaMA...")
llama_rows, llama_avg = evaluate_model("LLaMA-3.2", llama_query)

print("⏳ Evaluating Mistral...")
mistral_rows, mistral_avg = evaluate_model("Mistral", mistral_query)

# -------------------------------
# SAVE CSV
# -------------------------------
fieldnames = [
    "Model",
    "Question",
    "Latency_ms",
    "Input_Tokens",
    "Output_Tokens",
    "Context_Tokens",
    "Relevance",
    "Fluency",
    "Completeness",
    "Avg_Quality"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in llama_rows:
        writer.writerow(r)
    writer.writerow({})

    for r in mistral_rows:
        writer.writerow(r)
    writer.writerow({})

    writer.writerow(llama_avg)
    writer.writerow(mistral_avg)

print(f"✅ Evaluation complete. Results saved to {OUTPUT_FILE}")
