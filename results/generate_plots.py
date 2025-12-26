import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# CONFIG
# -------------------------------
CSV_FILE = "../evaluation/evaluation_results.csv"
OUTPUT_DIR = "."
DPI = 300

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(CSV_FILE)

# Remove empty rows
df = df.dropna(subset=["Model"])

# -------------------------------
# AGGREGATE METRICS
# -------------------------------
grouped = df.groupby("Model").mean(numeric_only=True)

models = grouped.index.tolist()

# -------------------------------
# FIGURE 1: QUALITY SCORES
# -------------------------------
plt.figure(figsize=(8, 5))

metrics = ["Relevance", "Fluency", "Completeness"]
x = range(len(models))

for i, metric in enumerate(metrics):
    plt.bar(
        [p + i*0.25 for p in x],
        grouped[metric],
        width=0.25,
        label=metric
    )

plt.xticks([p + 0.25 for p in x], models)
plt.ylabel("Average Score (1–5)")
plt.xlabel("Model")
plt.title(
    "Figure 1: Qualitative Evaluation Comparison\n"
    "(Relevance, Fluency, Completeness)"
)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("figure1_quality_scores.png", dpi=DPI)
plt.close()

# -------------------------------
# FIGURE 2: LATENCY
# -------------------------------
plt.figure(figsize=(6, 4))

plt.bar(models, grouped["Latency_ms"])
plt.ylabel("Latency (ms)")
plt.xlabel("Model")
plt.title(
    "Figure 2: Average Response Latency\n"
    "(Lower is Better)"
)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("figure2_latency.png", dpi=DPI)
plt.close()

# -------------------------------
# FIGURE 3: OUTPUT LENGTH
# -------------------------------
plt.figure(figsize=(6, 4))

plt.bar(models, grouped["Output_Tokens"])
plt.ylabel("Average Output Tokens")
plt.xlabel("Model")
plt.title(
    "Figure 3: Output Length Comparison\n"
    "(Tokens per Response)"
)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("figure3_output_length.png", dpi=DPI)
plt.close()

print("✅ Plots generated successfully in /results folder")
