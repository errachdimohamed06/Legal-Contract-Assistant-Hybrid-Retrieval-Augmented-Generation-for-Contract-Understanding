import sys
from pathlib import Path

sys.path.append(".")
from src.pipeline.rag_hybrid import process_contract_text

txt_path = list(Path("data/raw/full_contract_txt").glob("*.txt"))[0]
print(f"Contrat testé : {txt_path.name}")

with open(txt_path, "r", encoding="utf-8") as f:
    text = f.read()

questions = [
    "What is the expiration date?",
    "Explain indemnification clause simply.",
    "What is the governing law?",
    "Explain what a confidentiality clause means.",
    "How can this contract be terminated?"
]

for q in questions:
    print(f"\n{'='*80}")
    print(f"Question: {q}")
    print(f"{'='*80}")
    try:
        answer = process_contract_text(text, q, k=5)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 80)