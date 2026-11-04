import sys
sys.path.append(".")

from src.pipeline.rag_chain import ask_question

questions = [
    "What is the governing law of the contract?",
    "What is the expiration date?",
    "What are the confidentiality obligations?",
    "How can the contract be terminated?"
]

for q in questions:
    print(f"\n{'='*80}")
    print(f"Question: {q}")
    print(f"{'='*80}")
    try:
        response = ask_question(q)
        print(f"Answer: {response}")
    except Exception as e:
        print(f"Error: {e}")
    print("-" * 80)