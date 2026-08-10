import argparse
from collections import Counter
import csv
from pathlib import Path
import re


# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_PATH = Path(
    r"C:\Users\Bouchra\Desktop\Self_Guided_Microsoft\RAG_Project"
)

DEFAULT_INPUT = PROJECT_PATH / "evaluation_results_final.csv"
DEFAULT_OUTPUT = PROJECT_PATH / "Results" / "hallucination_results.txt"

MODEL_NAME = "all-MiniLM-L6-v2"
SEMANTIC_THRESHOLD = 0.55
F1_THRESHOLD = 0.25
NOT_FOUND_TEXT = "not found in the company documents"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are",
    "was", "were", "with", "as", "by", "for", "on", "at", "from",
    "that", "this", "it", "they", "their", "its", "we", "our",
    "who", "what", "where", "why", "how", "does", "do", "did",
    "company", "microsoft", "document", "documents",
}


# =====================================================
# TEXT AND CSV HELPERS
# =====================================================

def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize(text):
    normalized = clean(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(text):
    return [
        word
        for word in normalize(text).split()
        if word not in STOPWORDS
    ]


def token_f1(gold, answer):
    gold_tokens = set(tokenize(gold))
    answer_tokens = set(tokenize(answer))

    if not gold_tokens or not answer_tokens:
        return 0.0

    overlap = len(gold_tokens & answer_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def is_refusal(text):
    return NOT_FOUND_TEXT in normalize(text)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def resolve_answer_columns(fieldnames):
    baseline_column = next(
        (
            column
            for column in ("Baseline", "Baseline_Answer")
            if column in fieldnames
        ),
        None,
    )
    rag_column = next(
        (
            column
            for column in ("RAG", "RAG_Answer")
            if column in fieldnames
        ),
        None,
    )

    if baseline_column is None or rag_column is None:
        raise ValueError(
            "The CSV must contain Baseline and RAG answer columns. "
            f"Found: {fieldnames}"
        )

    return baseline_column, rag_column


# =====================================================
# AUTOMATIC HALLUCINATION PROXY
# =====================================================

def classify_answer(gold_answer, generated_answer, semantic_score):
    gold = normalize(gold_answer)
    answer = normalize(generated_answer)

    if not answer or is_refusal(answer):
        return "Refusal"

    # If the benchmark says that the answer is unavailable but the model
    # provides an answer, the generated answer is an unsupported response.
    if is_refusal(gold):
        return "Hallucination proxy"

    if gold and gold in answer:
        return "Supported"

    gold_tokens = set(tokenize(gold))
    answer_tokens = set(tokenize(answer))

    if gold_tokens and gold_tokens.issubset(answer_tokens):
        return "Supported"

    if round(semantic_score, 3) >= SEMANTIC_THRESHOLD:
        return "Supported"

    if round(token_f1(gold, answer), 3) >= F1_THRESHOLD:
        return "Supported"

    # This is an operational proxy: an incorrect non-refusal answer is
    # counted as potentially hallucinated, not as human-verified fabrication.
    return "Hallucination proxy"


def evaluate_all_answers(rows, baseline_column, rag_column):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers is required. Install it with: "
            "pip install sentence-transformers"
        ) from error

    valid_rows = [
        row
        for row in rows
        if clean(row.get("Question")) and clean(row.get("Gold_Answer"))
    ]

    if not valid_rows:
        raise ValueError("No questions with gold answers were found.")

    gold_answers = [clean(row.get("Gold_Answer")) for row in valid_rows]
    baseline_answers = [clean(row.get(baseline_column)) for row in valid_rows]
    rag_answers = [clean(row.get(rag_column)) for row in valid_rows]

    print("Loading the cached embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding all saved answers in one batch...")
    all_texts = gold_answers + baseline_answers + rag_answers
    embeddings = model.encode(
        all_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    total = len(valid_rows)
    gold_embeddings = embeddings[:total]
    baseline_embeddings = embeddings[total : 2 * total]
    rag_embeddings = embeddings[2 * total :]

    outcomes = {"baseline": [], "rag": []}

    for index in range(total):
        baseline_similarity = float(
            (gold_embeddings[index] * baseline_embeddings[index]).sum()
        )
        rag_similarity = float(
            (gold_embeddings[index] * rag_embeddings[index]).sum()
        )

        outcomes["baseline"].append(
            classify_answer(
                gold_answers[index],
                baseline_answers[index],
                baseline_similarity,
            )
        )
        outcomes["rag"].append(
            classify_answer(
                gold_answers[index],
                rag_answers[index],
                rag_similarity,
            )
        )

    return outcomes, total


# =====================================================
# ONE TXT TABLE
# =====================================================

def summarize(system_name, outcomes, total):
    counts = Counter(outcomes)
    supported = counts["Supported"]
    hallucination_proxy = counts["Hallucination proxy"]
    refusals = counts["Refusal"]
    answered = total - refusals
    proxy_rate = (
        100 * hallucination_proxy / answered
        if answered
        else 0.0
    )

    return {
        "System": system_name,
        "Questions": total,
        "Supported": supported,
        "Hallucination proxy": hallucination_proxy,
        "Refusals": refusals,
        "Proxy rate": f"{proxy_rate:.1f}%",
    }


def make_table(rows):
    headers = [
        "System",
        "Questions",
        "Supported",
        "Hallucination proxy",
        "Refusals",
        "Proxy rate",
    ]
    widths = {
        header: max(
            len(header),
            *(len(str(row[header])) for row in rows),
        )
        for header in headers
    }

    header_line = "| " + " | ".join(
        header.ljust(widths[header]) for header in headers
    ) + " |"
    separator_line = "| " + " | ".join(
        ("-" * widths[header])
        if header == "System"
        else ("-" * (widths[header] - 1) + ":")
        for header in headers
    ) + " |"

    data_lines = []
    for row in rows:
        values = []
        for header in headers:
            value = str(row[header])
            if header == "System":
                values.append(value.ljust(widths[header]))
            else:
                values.append(value.rjust(widths[header]))
        data_lines.append("| " + " | ".join(values) + " |")

    return "\n".join([header_line, separator_line, *data_lines])


def write_report(output_path, outcomes, total):
    rows = [
        summarize("Baseline LLM", outcomes["baseline"], total),
        summarize("RAG-Enhanced LLM", outcomes["rag"], total),
    ]

    text = "\n".join(
        [
            "AUTOMATIC HALLUCINATION PROXY RESULTS",
            "=" * 90,
            "",
            make_table(rows),
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return rows


# =====================================================
# RUN
# =====================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate an automatic hallucination proxy from already saved "
            "Baseline and RAG answers."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_arguments()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fieldnames, rows = read_csv(input_path)
    required = {"Question", "Gold_Answer"}
    missing = required - set(fieldnames)

    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    baseline_column, rag_column = resolve_answer_columns(fieldnames)

    print("=" * 70)
    print("AUTOMATIC HALLUCINATION PROXY EVALUATION")
    print("=" * 70)
    print(f"Input file: {input_path}")
    print("Manual work required: 0")
    print("New Ollama calls: 0")

    outcomes, total = evaluate_all_answers(
        rows,
        baseline_column,
        rag_column,
    )
    results = write_report(output_path, outcomes, total)

    print("\nEvaluation completed.")
    for row in results:
        print(
            f"{row['System']}: supported={row['Supported']}, "
            f"proxy={row['Hallucination proxy']}, "
            f"refusals={row['Refusals']}, "
            f"rate={row['Proxy rate']}"
        )
    print("Files created: 1")
    print(f"TXT result: {output_path}")


if __name__ == "__main__":
    main()