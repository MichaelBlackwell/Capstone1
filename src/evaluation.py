import os
import time
import logging

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

from langchain.evaluation.qa import QAEvalChain

from src.chains import ask, get_llm

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    logger.info(f"Starting predictions with model={model}, {len(ground_truth)} questions")
    predictions = []
    for i, qa in enumerate(ground_truth, 1):
        logger.info(f"  PREDICT [{i}/{len(ground_truth)}] {qa['question'][:60]}...")
        try:
            pred = ask(qa["question"], model=model)
            logger.info(f"  PREDICT [{i}] OK — {len(pred)} chars")
            predictions.append({"question": qa["question"], "result": pred})
        except Exception as e:
            logger.error(f"  PREDICT [{i}] FAILED: {e}")
            predictions.append({"question": qa["question"], "result": f"ERROR: {e}"})
        if i < len(ground_truth):
            time.sleep(2)
    logger.info(f"Predictions complete: {len(predictions)} results")
    return predictions


def evaluate(ground_truth=GROUND_TRUTH, predictions=None, model="openai/gpt-oss-120b"):
    """Evaluate predictions against ground truth using QAEvalChain."""
    logger.info(f"=== EVALUATION START (model={model}) ===")

    if predictions is None:
        logger.info("No predictions provided, generating...")
        predictions = generate_predictions(ground_truth, model)

    logger.info("Building eval chain...")
    eval_llm = get_llm(model, temperature=0)
    eval_chain = QAEvalChain.from_llm(eval_llm)
    logger.info("Eval chain ready, starting grading...")

    # Grade one at a time with delays to avoid Groq rate limits
    detailed = []
    correct = 0
    for i, (qa, pred) in enumerate(zip(ground_truth, predictions)):
        logger.info(f"  GRADE [{i+1}/{len(ground_truth)}] {qa['question'][:50]}...")
        try:
            example = [{"query": qa["question"], "answer": qa["answer"]}]
            prediction = [pred]
            res = eval_chain.evaluate(example, prediction)
            grade = res[0]["results"].strip().upper()
            logger.info(f"  GRADE [{i+1}] result: {grade}")
        except Exception as e:
            logger.error(f"  GRADE [{i+1}] FAILED: {e}")
            grade = "ERROR"
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
        if i < len(ground_truth) - 1:
            time.sleep(2)

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
