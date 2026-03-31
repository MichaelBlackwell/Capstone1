import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

from langchain.evaluation.qa import QAEvalChain

from src.chains import ask, get_llm

load_dotenv()

# Ground-truth Q&A pairs derived from the actual dataset
GROUND_TRUTH = [
    {
        "question": "What is the total number of transactions in the dataset?",
        "answer": "There are 2,500 transactions in the dataset.",
    },
    {
        "question": "What is the total sales amount across all transactions?",
        "answer": "The total sales amount is 1,383,220.",
    },
    {
        "question": "Which product has the highest total sales?",
        "answer": "Widget A has the highest total sales at 375,235.",
    },
    {
        "question": "Which product has the lowest total sales?",
        "answer": "Widget D has the lowest total sales at 326,854.",
    },
    {
        "question": "Which region has the highest total sales?",
        "answer": "The West region has the highest total sales at 361,383.",
    },
    {
        "question": "Which region has the lowest total sales?",
        "answer": "The East region has the lowest total sales at 320,296.",
    },
    {
        "question": "What are the total sales in Q1 2023?",
        "answer": "Total sales in Q1 2023 are 46,086 from 90 transactions.",
    },
    {
        "question": "What is the average sales amount per transaction?",
        "answer": "The average sales amount per transaction is approximately 553.29.",
    },
    {
        "question": "What is the average customer satisfaction score?",
        "answer": "The average customer satisfaction score is approximately 3.03 out of 5.",
    },
    {
        "question": "Which product has the highest average customer satisfaction?",
        "answer": "Widget D has the highest average customer satisfaction at 3.07.",
    },
    {
        "question": "Do female or male customers have higher average sales?",
        "answer": "Female customers have higher average sales at 558.96 compared to male customers at 547.56.",
    },
    {
        "question": "Which age group has the highest average sales?",
        "answer": "The 18-25 age group has the highest average sales at 572.29.",
    },
    {
        "question": "How many female customers are in the dataset?",
        "answer": "There are 1,256 female customers in the dataset.",
    },
    {
        "question": "What is the minimum and maximum sales value?",
        "answer": "The minimum sales value is 100 and the maximum is 999.",
    },
    {
        "question": "What is the customer age range in the dataset?",
        "answer": "Customer ages range from 18 to 69 years old.",
    },
]


def generate_predictions(ground_truth=GROUND_TRUTH, model="openai/gpt-oss-120b"):
    """Run the QA chain on each ground-truth question and collect predictions."""
    predictions = []
    for i, qa in enumerate(ground_truth, 1):
        print(f"  [{i}/{len(ground_truth)}] {qa['question'][:60]}...")
        pred = ask(qa["question"], model=model)
        predictions.append({"question": qa["question"], "result": pred})
        if i < len(ground_truth):
            time.sleep(2)
    return predictions


def evaluate(ground_truth=GROUND_TRUTH, predictions=None, model="openai/gpt-oss-120b"):
    """Evaluate predictions against ground truth using QAEvalChain."""
    if predictions is None:
        print("Generating predictions...")
        predictions = generate_predictions(ground_truth, model)

    eval_llm = get_llm(model, temperature=0)
    eval_chain = QAEvalChain.from_llm(eval_llm)

    # Format inputs for the eval chain
    examples = [{"query": qa["question"], "answer": qa["answer"]} for qa in ground_truth]
    results = eval_chain.evaluate(examples, predictions)

    # Compile results
    detailed = []
    correct = 0
    for i, (qa, pred, res) in enumerate(zip(ground_truth, predictions, results)):
        grade = res["results"].strip().upper()
        is_correct = "CORRECT" in grade
        if is_correct:
            correct += 1
        detailed.append({
            "question": qa["question"],
            "ground_truth": qa["answer"],
            "prediction": pred["result"],
            "grade": grade,
            "correct": is_correct,
        })

    total = len(ground_truth)
    accuracy = correct / total if total > 0 else 0

    summary = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "detailed": detailed,
    }
    return summary


def print_report(summary):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"  Total Questions:  {summary['total']}")
    print(f"  Correct:          {summary['correct']}")
    print(f"  Incorrect:        {summary['incorrect']}")
    print(f"  Accuracy:         {summary['accuracy']:.1%}")
    print(f"{'='*60}\n")

    for i, d in enumerate(summary["detailed"], 1):
        status = "PASS" if d["correct"] else "FAIL"
        print(f"  [{status}] Q{i}: {d['question']}")
        if not d["correct"]:
            print(f"         Expected: {d['ground_truth'][:80]}")
            print(f"         Got:      {d['prediction'][:80]}")
    print()


if __name__ == "__main__":
    summary = evaluate()
    print_report(summary)
