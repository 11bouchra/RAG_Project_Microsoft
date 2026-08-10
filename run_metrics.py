import csv
from collections import Counter
import os
from pathlib import Path
import re

import numpy as np
from sentence_transformers import SentenceTransformer


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_PATH = Path(
    os.environ.get(
        "RAG_PROJECT_PATH",
        r"C:\Users\Bouchra\Desktop\Self_Guided_Microsoft\RAG_Project",
    )
)

EVALUATION_FILE = PROJECT_PATH / "evaluation_results_final.csv"
RESULTS_DIR = PROJECT_PATH / "Results"
REPORT_FILE = RESULTS_DIR / "final_evaluation_report.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NOT_FOUND = "Not found in the company documents."

# Operational thresholds used by the original evaluation script.
SEMANTIC_THRESHOLD = 0.55
TOKEN_F1_THRESHOLD = 0.25


# =====================================================
# TEXT PROCESSING
# =====================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
    "was", "were", "with", "as", "by", "for", "on", "at", "from",
    "that", "this", "it", "they", "their", "its", "we", "our",
    "who", "what", "where", "why", "how", "does", "do", "did",
    "company", "microsoft", "document", "documents",
}


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize(text):
    text = clean(text).lower()
    text = re.sub(r"[^a-z0-9.%\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        token
        for token in normalize(text).split()
        if token not in STOPWORDS
    ]


def is_refusal(text):
    return "not found in the company documents" in normalize(text)


def token_metrics(gold, answer):
    """Calculate multiset token precision, recall, and F1."""
    gold_tokens = tokenize(gold)
    answer_tokens = tokenize(answer)

    if not gold_tokens or not answer_tokens:
        return 0.0, 0.0, 0.0

    overlap = Counter(gold_tokens) & Counter(answer_tokens)
    common = sum(overlap.values())

    precision = common / len(answer_tokens)
    recall = common / len(gold_tokens)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    return precision, recall, f1


def leading_polarity(text):
    normalized = normalize(text)
    if re.match(r"^(no|not|never)\b", normalized):
        return "negative"
    if re.match(r"^(yes|certainly|definitely)\b", normalized):
        return "positive"
    return None


def extract_numbers(text):
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", normalize(text)))


# =====================================================
# AUTOMATIC ANSWER CLASSIFICATION
# =====================================================

def classify_answer(gold, answer, semantic_score, token_f1):
    gold_text = normalize(gold)
    answer_text = normalize(answer)

    gold_refusal = is_refusal(gold)
    answer_refusal = is_refusal(answer) or not answer_text

    if gold_refusal and answer_refusal:
        return "Correct abstention"
    if gold_refusal and not answer_refusal:
        return "Answered unanswerable item"
    if not gold_refusal and answer_refusal:
        return "Incorrect abstention"

    # Similar wording can conceal an opposite yes/no answer.
    gold_polarity = leading_polarity(gold)
    answer_polarity = leading_polarity(answer)
    if (
        gold_polarity
        and answer_polarity
        and gold_polarity != answer_polarity
    ):
        return "Incorrect"

    # Avoid accepting a response with a conflicting quantity only because the
    # surrounding wording is semantically similar.
    gold_numbers = extract_numbers(gold)
    answer_numbers = extract_numbers(answer)
    if gold_numbers and answer_numbers and not gold_numbers.issubset(answer_numbers):
        return "Incorrect"

    if gold_text == answer_text or (gold_text and gold_text in answer_text):
        return "Correct"

    gold_tokens = set(tokenize(gold))
    answer_tokens = set(tokenize(answer))
    if gold_tokens and gold_tokens.issubset(answer_tokens):
        return "Correct"

    # Preserve the operational decision rule used in the original experiment:
    # either semantic agreement or sufficient token overlap can mark an answer
    # as correct. The result is reported as automatic/estimated accuracy.
    if semantic_score >= SEMANTIC_THRESHOLD or token_f1 >= TOKEN_F1_THRESHOLD:
        return "Correct"

    return "Incorrect"


def is_correct_status(status):
    return status in {"Correct", "Correct abstention"}


def average(values):
    return float(np.mean(values)) if values else 0.0


# =====================================================
# LOAD SAVED ANSWERS
# =====================================================

def load_evaluation_results():
    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"evaluation_results_final.csv was not found here: "
            f"{EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])

        required = {"Question", "Gold_Answer"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        baseline_column = next(
            (
                name
                for name in ("Baseline", "Baseline_Answer")
                if name in fieldnames
            ),
            None,
        )
        rag_column = next(
            (name for name in ("RAG", "RAG_Answer") if name in fieldnames),
            None,
        )

        if baseline_column is None or rag_column is None:
            raise ValueError(
                "The final CSV must contain Baseline and RAG answer columns. "
                f"Found: {reader.fieldnames}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("evaluation_results_final.csv contains no data rows.")

    return rows, baseline_column, rag_column


# =====================================================
# FAST BATCHED SEMANTIC SIMILARITY
# =====================================================

def batch_encode(model, texts):
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def semantic_scores(gold_embeddings, answer_embeddings, answers):
    scores = np.sum(gold_embeddings * answer_embeddings, axis=1)

    for index, answer in enumerate(answers):
        if not clean(answer) or is_refusal(answer):
            scores[index] = 0.0

    return scores.tolist()


def calculate_metrics():
    source_rows, baseline_column, rag_column = load_evaluation_results()

    gold_answers = [clean(row.get("Gold_Answer")) for row in source_rows]
    baseline_answers = [clean(row.get(baseline_column)) for row in source_rows]
    rag_answers = [clean(row.get(rag_column)) for row in source_rows]

    print(f"Loaded {len(source_rows)} saved questions.")
    print("Loading the embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding all gold answers in one batch...")
    gold_embeddings = batch_encode(model, gold_answers)

    print("Encoding all Baseline answers in one batch...")
    baseline_embeddings = batch_encode(model, baseline_answers)

    print("Encoding all RAG answers in one batch...")
    rag_embeddings = batch_encode(model, rag_answers)

    baseline_similarities = semantic_scores(
        gold_embeddings,
        baseline_embeddings,
        baseline_answers,
    )
    rag_similarities = semantic_scores(
        gold_embeddings,
        rag_embeddings,
        rag_answers,
    )

    results = []

    for position, row in enumerate(source_rows):
        gold = gold_answers[position]
        baseline = baseline_answers[position]
        rag = rag_answers[position]

        bp, br, bf1 = token_metrics(gold, baseline)
        rp, rr, rf1 = token_metrics(gold, rag)

        baseline_semantic = float(baseline_similarities[position])
        rag_semantic = float(rag_similarities[position])

        results.append(
            {
                "ID": clean(row.get("ID")) or str(position + 1),
                "Document": clean(row.get("Document")) or "Not specified",
                "Category": clean(row.get("Category")) or "Uncategorized",
                "Question": clean(row.get("Question")),
                "Gold Answer": gold,
                "Baseline Answer": baseline,
                "RAG Answer": rag,
                "Baseline Precision": bp,
                "Baseline Recall": br,
                "Baseline F1": bf1,
                "RAG Precision": rp,
                "RAG Recall": rr,
                "RAG F1": rf1,
                "Baseline Semantic": baseline_semantic,
                "RAG Semantic": rag_semantic,
                "Baseline Status": classify_answer(
                    gold,
                    baseline,
                    baseline_semantic,
                    bf1,
                ),
                "RAG Status": classify_answer(
                    gold,
                    rag,
                    rag_semantic,
                    rf1,
                ),
            }
        )

    return results


# =====================================================
# SUMMARY TABLES
# =====================================================

def system_summary(rows, prefix):
    total = len(rows)
    statuses = [row[f"{prefix} Status"] for row in rows]
    answers = [row[f"{prefix} Answer"] for row in rows]

    correct = sum(is_correct_status(status) for status in statuses)
    refused = sum(is_refusal(answer) or not clean(answer) for answer in answers)

    return {
        "Token Precision": average([row[f"{prefix} Precision"] for row in rows]),
        "Token Recall": average([row[f"{prefix} Recall"] for row in rows]),
        "Token F1": average([row[f"{prefix} F1"] for row in rows]),
        "Semantic Similarity": average(
            [row[f"{prefix} Semantic"] for row in rows]
        ),
        "Estimated Accuracy": correct / total,
        "Answer Coverage": (total - refused) / total,
        "Refusal Rate": refused / total,
        "Correct Count": correct,
        "Refusal Count": refused,
        "Incorrect Answer Candidates": sum(
            status == "Incorrect" for status in statuses
        ),
    }


def outcome_counts(rows):
    statuses = [
        "Correct",
        "Incorrect",
        "Correct abstention",
        "Incorrect abstention",
        "Answered unanswerable item",
    ]

    table = [
        {
            "Outcome": status,
            "Baseline Count": sum(
                row["Baseline Status"] == status for row in rows
            ),
            "Baseline Rate": sum(
                row["Baseline Status"] == status for row in rows
            ) / len(rows),
            "RAG Count": sum(
                row["RAG Status"] == status for row in rows
            ),
            "RAG Rate": sum(
                row["RAG Status"] == status for row in rows
            ) / len(rows),
        }
        for status in statuses
    ]
    table.append(
        {
            "Outcome": "Total",
            "Baseline Count": len(rows),
            "Baseline Rate": 1.0,
            "RAG Count": len(rows),
            "RAG Rate": 1.0,
        }
    )
    return table


def grouped_statistics(rows, group_key):
    groups = sorted({row[group_key] for row in rows})
    output = []

    for group in groups:
        group_rows = [row for row in rows if row[group_key] == group]
        baseline = system_summary(group_rows, "Baseline")
        rag = system_summary(group_rows, "RAG")

        output.append(
            {
                group_key: group,
                "Questions": len(group_rows),
                "Baseline Accuracy": baseline["Estimated Accuracy"],
                "RAG Accuracy": rag["Estimated Accuracy"],
                "Difference": (
                    rag["Estimated Accuracy"]
                    - baseline["Estimated Accuracy"]
                ),
                "Baseline Precision": baseline["Token Precision"],
                "RAG Precision": rag["Token Precision"],
                "Baseline Recall": baseline["Token Recall"],
                "RAG Recall": rag["Token Recall"],
                "Baseline F1": baseline["Token F1"],
                "RAG F1": rag["Token F1"],
                "Baseline Semantic": baseline["Semantic Similarity"],
                "RAG Semantic": rag["Semantic Similarity"],
                "RAG Refusals": rag["Refusal Count"],
            }
        )

    return output


# =====================================================
# MARKDOWN-STYLE TEXT TABLES
# =====================================================

def percentage(value):
    return f"{100 * value:.1f}%"


def decimal(value):
    return f"{value:.3f}"


def signed_points(value):
    return f"{100 * value:+.1f} pp"


def markdown_table(headers, rows):
    text_rows = [[clean(row.get(header, "")) for header in headers] for row in rows]
    widths = [len(header) for header in headers]

    for row in text_rows:
        for column, value in enumerate(row):
            widths[column] = max(widths[column], len(value))

    lines = []
    lines.append(
        "| "
        + " | ".join(
            header.ljust(widths[index])
            for index, header in enumerate(headers)
        )
        + " |"
    )
    lines.append(
        "| "
        + " | ".join("-" * widths[index] for index in range(len(headers)))
        + " |"
    )

    for row in text_rows:
        lines.append(
            "| "
            + " | ".join(
                value.ljust(widths[index])
                for index, value in enumerate(row)
            )
            + " |"
        )

    return "\n".join(lines)


def overall_report_rows(baseline, rag):
    metrics = [
        ("Accuracy", "Estimated Accuracy", percentage),
        ("Precision", "Token Precision", decimal),
        ("Recall", "Token Recall", decimal),
        ("F1 Score", "Token F1", decimal),
        ("Semantic Similarity", "Semantic Similarity", decimal),
        ("Answer Coverage", "Answer Coverage", percentage),
        ("Refusal Rate", "Refusal Rate", percentage),
    ]

    rows = []
    for display_name, metric, formatter in metrics:
        rows.append(
            {
                "Metric": display_name,
                "Baseline LLM": formatter(baseline[metric]),
                "RAG-Enhanced LLM": formatter(rag[metric]),
                "Difference": (
                    signed_points(rag[metric] - baseline[metric])
                    if metric
                    in {"Estimated Accuracy", "Answer Coverage", "Refusal Rate"}
                    else f"{rag[metric] - baseline[metric]:+.3f}"
                ),
            }
        )
    return rows


def formatted_group_rows(rows, group_key):
    return [
        {
            group_key: row[group_key],
            "Questions": row["Questions"],
            "Baseline Accuracy": percentage(row["Baseline Accuracy"]),
            "RAG Accuracy": percentage(row["RAG Accuracy"]),
            "Difference": signed_points(row["Difference"]),
            "Baseline Precision": decimal(row["Baseline Precision"]),
            "RAG Precision": decimal(row["RAG Precision"]),
            "Baseline Recall": decimal(row["Baseline Recall"]),
            "RAG Recall": decimal(row["RAG Recall"]),
            "Baseline F1": decimal(row["Baseline F1"]),
            "RAG F1": decimal(row["RAG F1"]),
            "Baseline Semantic": decimal(row["Baseline Semantic"]),
            "RAG Semantic": decimal(row["RAG Semantic"]),
            "RAG Refusals": row["RAG Refusals"],
        }
        for row in rows
    ]


# =====================================================
# WRITE ONE TXT REPORT
# =====================================================

def write_report(rows):
    baseline = system_summary(rows, "Baseline")
    rag = system_summary(rows, "RAG")
    outcomes = outcome_counts(rows)
    documents = grouped_statistics(rows, "Document")

    sections = []
    sections.append("FINAL QUANTITATIVE EVALUATION RESULTS")
    sections.append("=" * 100)

    sections.append("TABLE 1. OVERALL PERFORMANCE COMPARISON")
    sections.append(
        markdown_table(
            ["Metric", "Baseline LLM", "RAG-Enhanced LLM", "Difference"],
            overall_report_rows(baseline, rag),
        )
    )

    sections.append("TABLE 2. AUTOMATIC ANSWER OUTCOMES")
    sections.append(
        markdown_table(
            [
                "Outcome", "Baseline Count", "Baseline Rate",
                "RAG Count", "RAG Rate",
            ],
            [
                {
                    **row,
                    "Baseline Rate": percentage(row["Baseline Rate"]),
                    "RAG Rate": percentage(row["RAG Rate"]),
                }
                for row in outcomes
            ],
        )
    )

    document_rows = formatted_group_rows(documents, "Document")
    sections.append("TABLE 3. DOCUMENT-LEVEL QUANTITATIVE RESULTS")
    sections.append(
        markdown_table(
            [
                "Document", "Questions", "Baseline Accuracy", "RAG Accuracy",
                "Difference", "Baseline Precision", "RAG Precision",
                "Baseline Recall", "RAG Recall", "Baseline F1", "RAG F1",
                "Baseline Semantic", "RAG Semantic", "RAG Refusals",
            ],
            document_rows,
        )
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n\n".join(sections) + "\n", encoding="utf-8")

    return baseline, rag


# =====================================================
# MAIN
# =====================================================

def main():
    print("=" * 70)
    print("FINAL METRICS FROM SAVED ANSWERS")
    print("=" * 70)
    print(f"Input:  {EVALUATION_FILE}")
    print(f"Output: {REPORT_FILE}")
    print("Ollama calls: 0")
    print("ChromaDB retrieval calls: 0\n")

    rows = calculate_metrics()
    baseline, rag = write_report(rows)

    print("\n" + "=" * 70)
    print("METRICS COMPLETED")
    print("=" * 70)
    print(f"Questions evaluated: {len(rows)}")
    print(f"Baseline estimated accuracy: {percentage(baseline['Estimated Accuracy'])}")
    print(f"RAG estimated accuracy: {percentage(rag['Estimated Accuracy'])}")
    print(f"Baseline refusals: {baseline['Refusal Count']}")
    print(f"RAG refusals: {rag['Refusal Count']}")
    print("Files created: 1")
    print(f"Final TXT report: {REPORT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()