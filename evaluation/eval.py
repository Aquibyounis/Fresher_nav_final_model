import os
import time
import csv
import json
import subprocess
from main1 import run_query as llama_query
from main2 import run_query as mistral_query

# -------------------------------
# CONFIG
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "evaluation_results.csv")

# -------------------------------
# Evaluation Questions
# -------------------------------
questions = [
    #General & random
    "What is the difference between applying for leave through VTOP and applying for leave manually?",
    "What are the restrictions for weekend outings on VTOP, and what happens if a student returns late?",
    "If VTOP does not allow me to apply for leave due to technical issues, what alternative method is available?",
    "Hi",
    "how are you?",
    "See you soon",
    "I had a very bad day",
    "I wish I were an Astronaut",
    "Can you tell me something funny about VIT-AP joke",
    "Can you tell me hi",
    "I didn't score well in examinations, Can you console me with something funny"
    
    # Guest House
    "I heard there’s a guest house in VIT-AP. What is it for?",
    "Then where exactly is this guest house located on campus?",
    "Does guest house provide proper facilities for the visitors?",
    
    # Lost ID Card
    "What happens if I lose my student ID card?",
    "While I’m waiting, can I still attend classes and use the library without the card?",
    "After I get the new card, can I use it everywhere again?",
    
    # VIT AP Events
    "Does VIT-AP have any big events or festivals for students?",
    "Are technical events also conducted at VIT-AP?",
    "Are there clubs we can join at VIT-AP?",
    
    # Course Registration
    "How do we register for courses at VIT-AP?",
    "Is there anything I should check before registering for a course?",
    "What happens if I miss the registration deadline in course registration?",
    "If I want to withdraw from a course later in the semester, how do I do it?",
    
    # Placement details
    "How does placement work at VIT-AP?",
    "Do we need to register separately for placements?",
    "How much is the fee for B.Tech students for placements?",
    "What is the eligibility to register for placements at VIT-AP?"
]


# -------------------------------
# Ollama Judge Function
# -------------------------------
def ollama_judge_score(question, answer, model_name):
    """
    Uses local llama2 via Ollama to score a model's answer.
    Returns a dict: correctness, relevance, fluency, helpfulness
    """
    prompt = f"""
You are an expert evaluator. Evaluate the model's answer to the question.
Score from 1 to 10 for:

1. Correctness: factual accuracy
2. Relevance: how well it answers the question
3. Fluency: readability and human-likeness
4. Helpfulness: usefulness to the user

Return ONLY JSON like: {{"correctness":X,"relevance":Y,"fluency":Z,"helpfulness":W}}

Question: {question}
Answer: {answer}
Model: {model_name}
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "llama2"],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        content = result.stdout.decode("utf-8").strip()
        # Extract JSON safely
        start = content.find("{")
        end = content.rfind("}") + 1
        json_str = content[start:end]
        scores = json.loads(json_str)
        return scores
    except Exception as e:
        print(f"⚠️ LLM scoring failed for {model_name}: {e}")
        return {"correctness": 0, "relevance": 0, "fluency": 0, "helpfulness": 0}

# -------------------------------
# Evaluate One Model
# -------------------------------
def evaluate_model(model_name, query_func):
    results = []
    total_time = 0
    total_scores = {"correctness":0, "relevance":0, "fluency":0, "helpfulness":0}

    for q in questions:
        start = time.time()
        try:
            answer = query_func(q)
        except Exception as e:
            answer = f"Error: {e}"
        end = time.time()
        duration = round(end - start, 2)

        scores = ollama_judge_score(q, answer, model_name)
        avg_score = round(sum(scores.values())/4, 3)

        results.append({
            "Question": q,
            "Answer": answer,
            "Response Time (s)": duration,
            "Correctness": scores["correctness"],
            "Relevance": scores["relevance"],
            "Fluency": scores["fluency"],
            "Helpfulness": scores["helpfulness"],
            "Avg Score": avg_score
        })

        total_time += duration
        for k in total_scores:
            total_scores[k] += scores[k]

    # Avoid division by zero
    num_questions = max(len(questions), 1)
    avg = {k: round(total_scores[k]/num_questions, 3) for k in total_scores}
    avg["avg_time"] = round(total_time / num_questions, 2)
    return results, avg

# -------------------------------
# Run Evaluations
# -------------------------------
print("⏳ Evaluating LLaMA...")
llama_results, llama_avg = evaluate_model("LLaMA", llama_query)

print("⏳ Evaluating Mistral...")
mistral_results, mistral_avg = evaluate_model("Mistral", mistral_query)

# -------------------------------
# Save CSV (Side by Side)
# -------------------------------
# -------------------------------
# Save CSV (All LLaMA first, then blank line, then Mistral)
# -------------------------------
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "Question",
        "Answer",
        "Response Time (s)",
        "Correctness",
        "Relevance",
        "Fluency",
        "Helpfulness",
        "Avg Score"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Write all LLaMA results
    for r in llama_results:
        writer.writerow({
            "Question": r["Question"],
            "Answer": r["Answer"],
            "Response Time (s)": r["Response Time (s)"],
            "Correctness": r["Correctness"],
            "Relevance": r["Relevance"],
            "Fluency": r["Fluency"],
            "Helpfulness": r["Helpfulness"],
            "Avg Score": r["Avg Score"]
        })

    # Blank line between models
    writer.writerow({})

    # Write all Mistral results
    for r in mistral_results:
        writer.writerow({
            "Question": r["Question"],
            "Answer": r["Answer"],
            "Response Time (s)": r["Response Time (s)"],
            "Correctness": r["Correctness"],
            "Relevance": r["Relevance"],
            "Fluency": r["Fluency"],
            "Helpfulness": r["Helpfulness"],
            "Avg Score": r["Avg Score"]
        })

    # Blank line
    writer.writerow({})

    # Write averages
    writer.writerow({
        "Question": "AVERAGES",
        "Answer": "",
        "Response Time (s)": llama_avg["avg_time"],
        "Correctness": llama_avg["correctness"],
        "Relevance": llama_avg["relevance"],
        "Fluency": llama_avg["fluency"],
        "Helpfulness": llama_avg["helpfulness"],
        "Avg Score": round(sum([llama_avg[k] for k in ["correctness","relevance","fluency","helpfulness"]])/4, 3)
    })
    writer.writerow({
        "Question": "AVERAGES",
        "Answer": "",
        "Response Time (s)": mistral_avg["avg_time"],
        "Correctness": mistral_avg["correctness"],
        "Relevance": mistral_avg["relevance"],
        "Fluency": mistral_avg["fluency"],
        "Helpfulness": mistral_avg["helpfulness"],
        "Avg Score": round(sum([mistral_avg[k] for k in ["correctness","relevance","fluency","helpfulness"]])/4, 3)
    })

print(f"✅ Evaluation complete! Results saved to {OUTPUT_FILE}")
