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

SEMANTIC_THRESHOLD = 0.55
TOKEN_F1_THRESHOLD = 0.25


# =====================================================
# TEXT PROCESSING
# =====================================================

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "to",
    "is", "are", "was", "were", "with", "as", "by",
    "for", "on", "at", "from", "that", "this", "it",
    "they", "their", "its", "we", "our", "who", "what",
    "where", "why", "how", "does", "do", "did",
    "company", "microsoft", "document", "documents",
}


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize(text):
    text = clean(text).lower()
    text = re.sub(r"[^a-z0-9.%\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return [
        token
        for token in normalize(text).split()
        if token not in STOPWORDS
    ]


# =====================================================
# IDENTIFY HALLUCINATION TEST QUESTIONS
# =====================================================

def is_hallucination_test(row):

    category = normalize(
        row.get("Category", "")
    )

    qid = clean(
        row.get("ID", "")
    ).upper()

    gold = normalize(
        row.get("Gold_Answer", "")
    )

    return (
        category == "hallucination test"
        or qid.startswith("H")
        or "not found in the company documents" in gold
    )


# =====================================================
# GENERAL ABSTENTION PHRASES
# =====================================================

ABSTENTION_PHRASES = [
    "not found in the company documents",
    "i don't know",
    "i do not know",
    "i'm not aware",
    "i am not aware",
    "i don't have access",
    "i do not have access",
    "i cannot determine",
    "i can't determine",
    "cannot determine",
    "cannot be determined",
    "unable to determine",
    "i cannot answer",
    "i can't answer",
    "unable to answer",
    "not publicly disclosed",
    "not publicly available",
    "no information is available",
    "there is no information",
    "not provided",
    "not specified",
    "there is no report",
    "no report mentioned",
    "didn't receive a report",
    "did not receive a report",
]


def contains_abstention(text):

    normalized = normalize(text)

    if not normalized:
        return True

    return any(
        phrase in normalized
        for phrase in ABSTENTION_PHRASES
    )


# =====================================================
# HALLUCINATION CLASSIFICATION
# =====================================================

def classify_hallucination_answer(answer):
    """
    Used only for H1-H8.

    Correct abstention:
        The model clearly states that it does not know
        or does not have enough information, without
        giving an unsupported answer.

    Hallucination:
        The model gives a company-specific claim,
        guess, estimate, list of possibilities,
        policy statement, or unsupported factual answer.
    """

    normalized = normalize(answer)

    if not normalized:
        return "Correct abstention"

    # Exact RAG fallback
    if normalized == "not found in the company documents":
        return "Correct abstention"


    # -------------------------------------------------
    # Clear unsupported yes/no or policy claim
    # -------------------------------------------------

    direct_claim_patterns = [
        r"^yes\b",
        r"^no\b",
        r"^microsoft requires\b",
        r"^microsoft does not require\b",
        r"^the company requires\b",
        r"^the company provides\b",
        r"^employees are entitled\b",
        r"^employees receive\b",
    ]

    for pattern in direct_claim_patterns:

        if re.match(
            pattern,
            normalized
        ):
            return "Hallucination"


    # -------------------------------------------------
    # Indicators that the model continued guessing
    # after admitting uncertainty
    # -------------------------------------------------

    unsupported_continuation_markers = [
        "some common",
        "common colors",
        "for example",
        "for instance",
        "such as",
        "in general",
        "that being said",
        "salary range",
        "salary ranges",
        "average salary",
        "median salary",
        "according to the bureau",
        "according to glassdoor",
        "according to shrm",
        "may offer",
        "can vary widely",
        "can vary greatly",
        "typically offer",
        "common in",
        "generally",
        "it depends on",
    ]


    # -------------------------------------------------
    # Clean abstention
    # -------------------------------------------------

    if contains_abstention(answer):

        if any(
            marker in normalized
            for marker in unsupported_continuation_markers
        ):
            return "Hallucination"

        return "Correct abstention"


    # -------------------------------------------------
    # Any substantive answer to an intentionally
    # unsupported question counts as hallucination.
    # -------------------------------------------------

    return "Hallucination"


# =====================================================
# TOKEN METRICS
# =====================================================

def token_metrics(gold, answer):

    gold_tokens = tokenize(gold)
    answer_tokens = tokenize(answer)

    if not gold_tokens or not answer_tokens:
        return 0.0, 0.0, 0.0

    overlap = (
        Counter(gold_tokens)
        & Counter(answer_tokens)
    )

    common = sum(
        overlap.values()
    )

    precision = (
        common / len(answer_tokens)
    )

    recall = (
        common / len(gold_tokens)
    )

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    return precision, recall, f1


# =====================================================
# YES / NO AND NUMBER CHECKS
# =====================================================

def leading_polarity(text):

    normalized = normalize(text)

    if re.match(
        r"^(no|not|never)\b",
        normalized
    ):
        return "negative"

    if re.match(
        r"^(yes|certainly|definitely)\b",
        normalized
    ):
        return "positive"

    return None


def extract_numbers(text):

    return set(
        re.findall(
            r"\b\d+(?:\.\d+)?%?\b",
            normalize(text)
        )
    )


# =====================================================
# CLASSIFY ANSWERABLE QUESTIONS
# =====================================================

def classify_answer(
    gold,
    answer,
    semantic_score,
    token_f1
):

    gold_text = normalize(gold)
    answer_text = normalize(answer)


    # -------------------------------------------------
    # Refusal on an answerable question
    # -------------------------------------------------

    if contains_abstention(answer):
        return "Incorrect abstention"


    # -------------------------------------------------
    # Yes / No contradiction
    # -------------------------------------------------

    gold_polarity = leading_polarity(
        gold
    )

    answer_polarity = leading_polarity(
        answer
    )

    if (
        gold_polarity
        and answer_polarity
        and gold_polarity != answer_polarity
    ):
        return "Incorrect"


    # -------------------------------------------------
    # Numeric contradiction
    # -------------------------------------------------

    gold_numbers = extract_numbers(
        gold
    )

    answer_numbers = extract_numbers(
        answer
    )

    if (
        gold_numbers
        and answer_numbers
        and not gold_numbers.issubset(
            answer_numbers
        )
    ):
        return "Incorrect"


    # -------------------------------------------------
    # Exact match / contained answer
    # -------------------------------------------------

    if (
        gold_text == answer_text
        or (
            gold_text
            and gold_text in answer_text
        )
    ):
        return "Correct"


    # -------------------------------------------------
    # Gold tokens contained in answer
    # -------------------------------------------------

    gold_tokens = set(
        tokenize(gold)
    )

    answer_tokens = set(
        tokenize(answer)
    )

    if (
        gold_tokens
        and gold_tokens.issubset(
            answer_tokens
        )
    ):
        return "Correct"


    # -------------------------------------------------
    # Semantic or token-based agreement
    # -------------------------------------------------

    if (
        semantic_score >= SEMANTIC_THRESHOLD
        or token_f1 >= TOKEN_F1_THRESHOLD
    ):
        return "Correct"


    return "Incorrect"


# =====================================================
# LOAD FINAL EVALUATION CSV
# =====================================================

def load_evaluation_results():

    if not EVALUATION_FILE.exists():

        raise FileNotFoundError(
            f"Could not find:\n{EVALUATION_FILE}"
        )


    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(
            file
        )

        fieldnames = set(
            reader.fieldnames or []
        )


        required = {
            "Question",
            "Gold_Answer",
            "Category",
        }

        missing = (
            required
            - fieldnames
        )


        if missing:

            raise ValueError(
                f"Missing columns: "
                f"{sorted(missing)}"
            )


        baseline_column = next(
            (
                name
                for name in (
                    "Baseline",
                    "Baseline_Answer"
                )
                if name in fieldnames
            ),
            None
        )


        rag_column = next(
            (
                name
                for name in (
                    "RAG",
                    "RAG_Answer"
                )
                if name in fieldnames
            ),
            None
        )


        if (
            baseline_column is None
            or rag_column is None
        ):

            raise ValueError(
                "Baseline and RAG columns are required."
            )


        rows = list(
            reader
        )


    if not rows:

        raise ValueError(
            "evaluation_results_final.csv is empty."
        )


    return (
        rows,
        baseline_column,
        rag_column
    )


# =====================================================
# EMBEDDINGS
# =====================================================

def batch_encode(model, texts):

    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def semantic_scores(
    gold_embeddings,
    answer_embeddings,
    answers
):

    scores = np.sum(
        gold_embeddings
        * answer_embeddings,
        axis=1
    )


    for index, answer in enumerate(
        answers
    ):

        if (
            not clean(answer)
            or contains_abstention(answer)
        ):
            scores[index] = 0.0


    return scores.tolist()


# =====================================================
# CALCULATE METRICS
# =====================================================

def calculate_metrics():

    (
        source_rows,
        baseline_column,
        rag_column
    ) = load_evaluation_results()


    answerable_source = [
        row
        for row in source_rows
        if not is_hallucination_test(row)
    ]


    hallucination_source = [
        row
        for row in source_rows
        if is_hallucination_test(row)
    ]


    print(
        f"Total questions loaded: {len(source_rows)}"
    )

    print(
        f"Answerable questions: {len(answerable_source)}"
    )

    print(
        f"Hallucination-test questions: "
        f"{len(hallucination_source)}"
    )


    # =================================================
    # STANDARD QA — 303 QUESTIONS
    # =================================================

    gold_answers = [
        clean(row.get("Gold_Answer"))
        for row in answerable_source
    ]


    baseline_answers = [
        clean(row.get(baseline_column))
        for row in answerable_source
    ]


    rag_answers = [
        clean(row.get(rag_column))
        for row in answerable_source
    ]


    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )


    print(
        "Encoding gold answers..."
    )

    gold_embeddings = batch_encode(
        model,
        gold_answers
    )


    print(
        "Encoding Baseline answers..."
    )

    baseline_embeddings = batch_encode(
        model,
        baseline_answers
    )


    print(
        "Encoding RAG answers..."
    )

    rag_embeddings = batch_encode(
        model,
        rag_answers
    )


    baseline_similarities = semantic_scores(
        gold_embeddings,
        baseline_embeddings,
        baseline_answers
    )


    rag_similarities = semantic_scores(
        gold_embeddings,
        rag_embeddings,
        rag_answers
    )


    answerable_results = []


    for position, row in enumerate(
        answerable_source
    ):

        gold = gold_answers[
            position
        ]

        baseline = baseline_answers[
            position
        ]

        rag = rag_answers[
            position
        ]


        bp, br, bf1 = token_metrics(
            gold,
            baseline
        )


        rp, rr, rf1 = token_metrics(
            gold,
            rag
        )


        baseline_semantic = float(
            baseline_similarities[
                position
            ]
        )


        rag_semantic = float(
            rag_similarities[
                position
            ]
        )


        answerable_results.append(
            {
                "ID":
                    clean(row.get("ID")),

                "Document":
                    clean(row.get("Document")),

                "Question":
                    clean(row.get("Question")),

                "Gold Answer":
                    gold,

                "Baseline Answer":
                    baseline,

                "RAG Answer":
                    rag,

                "Baseline Precision":
                    bp,

                "Baseline Recall":
                    br,

                "Baseline F1":
                    bf1,

                "RAG Precision":
                    rp,

                "RAG Recall":
                    rr,

                "RAG F1":
                    rf1,

                "Baseline Semantic":
                    baseline_semantic,

                "RAG Semantic":
                    rag_semantic,

                "Baseline Status":
                    classify_answer(
                        gold,
                        baseline,
                        baseline_semantic,
                        bf1
                    ),

                "RAG Status":
                    classify_answer(
                        gold,
                        rag,
                        rag_semantic,
                        rf1
                    ),
            }
        )


    # =================================================
    # H1-H8
    # =================================================

    hallucination_results = []


    for row in hallucination_source:

        baseline = clean(
            row.get(baseline_column)
        )

        rag = clean(
            row.get(rag_column)
        )


        hallucination_results.append(
            {
                "ID":
                    clean(row.get("ID")),

                "Document":
                    clean(row.get("Document")),

                "Question":
                    clean(row.get("Question")),

                "Baseline Status":
                    classify_hallucination_answer(
                        baseline
                    ),

                "RAG Status":
                    classify_hallucination_answer(
                        rag
                    ),

                "Baseline Answer":
                    baseline,

                "RAG Answer":
                    rag,
            }
        )


    return (
        answerable_results,
        hallucination_results
    )


# =====================================================
# SUMMARY FUNCTIONS
# =====================================================

def average(values):

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def standard_summary(
    rows,
    prefix
):

    total = len(rows)


    correct = sum(
        row[f"{prefix} Status"]
        == "Correct"
        for row in rows
    )


    incorrect = sum(
        row[f"{prefix} Status"]
        == "Incorrect"
        for row in rows
    )


    incorrect_abstentions = sum(
        row[f"{prefix} Status"]
        == "Incorrect abstention"
        for row in rows
    )


    return {
        "Questions":
            total,

        "Correct":
            correct,

        "Incorrect":
            incorrect,

        "Incorrect Abstentions":
            incorrect_abstentions,

        "Accuracy":
            (
                correct / total
                if total
                else 0.0
            ),

        "Precision":
            average([
                row[f"{prefix} Precision"]
                for row in rows
            ]),

        "Recall":
            average([
                row[f"{prefix} Recall"]
                for row in rows
            ]),

        "F1":
            average([
                row[f"{prefix} F1"]
                for row in rows
            ]),

        "Semantic":
            average([
                row[f"{prefix} Semantic"]
                for row in rows
            ]),
    }


def hallucination_summary(
    rows,
    prefix
):

    total = len(rows)


    hallucinations = sum(
        row[f"{prefix} Status"]
        == "Hallucination"
        for row in rows
    )


    correct_abstentions = sum(
        row[f"{prefix} Status"]
        == "Correct abstention"
        for row in rows
    )


    return {
        "Questions":
            total,

        "Hallucinations":
            hallucinations,

        "Correct Abstentions":
            correct_abstentions,

        "Hallucination Rate":
            (
                hallucinations / total
                if total
                else 0.0
            ),

        "Correct Abstention Rate":
            (
                correct_abstentions / total
                if total
                else 0.0
            ),
    }


# =====================================================
# OVERALL 311 ACCURACY
# =====================================================

def overall_accuracy(
    qa_summary,
    hallucination_data
):

    correct = (
        qa_summary["Correct"]
        +
        hallucination_data[
            "Correct Abstentions"
        ]
    )

    total = (
        qa_summary["Questions"]
        +
        hallucination_data[
            "Questions"
        ]
    )

    return (
        correct / total
        if total
        else 0.0
    )


# =====================================================
# FORMATTING
# =====================================================

def percentage(value):

    return f"{value * 100:.1f}%"


def decimal(value):

    return f"{value:.3f}"


def signed_points(value):

    return f"{value * 100:+.1f} pp"


def shorten(text, length=70):

    text = clean(
        text
    ).replace(
        "\n",
        " "
    )

    if len(text) <= length:
        return text

    return (
        text[:length - 3]
        + "..."
    )


def markdown_table(
    headers,
    rows
):

    text_rows = [
        [
            clean(
                row.get(
                    header,
                    ""
                )
            )
            for header in headers
        ]
        for row in rows
    ]


    widths = [
        len(header)
        for header in headers
    ]


    for row in text_rows:

        for index, value in enumerate(
            row
        ):

            widths[index] = max(
                widths[index],
                len(value)
            )


    lines = []


    lines.append(
        "| "
        +
        " | ".join(
            header.ljust(
                widths[index]
            )
            for index, header
            in enumerate(headers)
        )
        +
        " |"
    )


    lines.append(
        "| "
        +
        " | ".join(
            "-" * width
            for width in widths
        )
        +
        " |"
    )


    for row in text_rows:

        lines.append(
            "| "
            +
            " | ".join(
                value.ljust(
                    widths[index]
                )
                for index, value
                in enumerate(row)
            )
            +
            " |"
        )


    return "\n".join(
        lines
    )


# =====================================================
# WRITE FINAL REPORT
# =====================================================

def write_report(
    answerable_rows,
    hallucination_rows
):

    baseline = standard_summary(
        answerable_rows,
        "Baseline"
    )

    rag = standard_summary(
        answerable_rows,
        "RAG"
    )


    baseline_h = hallucination_summary(
        hallucination_rows,
        "Baseline"
    )

    rag_h = hallucination_summary(
        hallucination_rows,
        "RAG"
    )


    baseline_overall = overall_accuracy(
        baseline,
        baseline_h
    )


    rag_overall = overall_accuracy(
        rag,
        rag_h
    )


    sections = []


    # =================================================
    # TITLE
    # =================================================

    sections.append(
        "FINAL EVALUATION REPORT"
    )

    sections.append(
        "=" * 100
    )


    sections.append(
        f"Total benchmark questions: "
        f"{len(answerable_rows) + len(hallucination_rows)}"
    )


    sections.append(
        f"Answerable questions: "
        f"{len(answerable_rows)}"
    )


    sections.append(
        f"Unsupported hallucination-test questions: "
        f"{len(hallucination_rows)}"
    )


    # =================================================
    # TABLE 1
    # =================================================

    sections.append(
        "\nTABLE 1. OVERALL BENCHMARK ACCURACY — ALL 311 QUESTIONS"
    )


    overall_rows = [
        {
            "System":
                "Baseline LLM",

            "Correct":
                baseline["Correct"]
                +
                baseline_h[
                    "Correct Abstentions"
                ],

            "Total":
                len(answerable_rows)
                +
                len(hallucination_rows),

            "Accuracy":
                percentage(
                    baseline_overall
                ),
        },

        {
            "System":
                "RAG-Enhanced LLM",

            "Correct":
                rag["Correct"]
                +
                rag_h[
                    "Correct Abstentions"
                ],

            "Total":
                len(answerable_rows)
                +
                len(hallucination_rows),

            "Accuracy":
                percentage(
                    rag_overall
                ),
        },
    ]


    sections.append(
        markdown_table(
            [
                "System",
                "Correct",
                "Total",
                "Accuracy",
            ],
            overall_rows
        )
    )


    # =================================================
    # TABLE 2
    # =================================================

    sections.append(
        "\nTABLE 2. QUESTION-ANSWERING METRICS — 303 ANSWERABLE QUESTIONS"
    )


    qa_rows = [
        {
            "Metric":
                "Accuracy",

            "Baseline LLM":
                percentage(
                    baseline["Accuracy"]
                ),

            "RAG-Enhanced LLM":
                percentage(
                    rag["Accuracy"]
                ),

            "Difference":
                signed_points(
                    rag["Accuracy"]
                    -
                    baseline["Accuracy"]
                ),
        },

        {
            "Metric":
                "Precision",

            "Baseline LLM":
                decimal(
                    baseline["Precision"]
                ),

            "RAG-Enhanced LLM":
                decimal(
                    rag["Precision"]
                ),

            "Difference":
                f"{rag['Precision'] - baseline['Precision']:+.3f}",
        },

        {
            "Metric":
                "Recall",

            "Baseline LLM":
                decimal(
                    baseline["Recall"]
                ),

            "RAG-Enhanced LLM":
                decimal(
                    rag["Recall"]
                ),

            "Difference":
                f"{rag['Recall'] - baseline['Recall']:+.3f}",
        },

        {
            "Metric":
                "F1 Score",

            "Baseline LLM":
                decimal(
                    baseline["F1"]
                ),

            "RAG-Enhanced LLM":
                decimal(
                    rag["F1"]
                ),

            "Difference":
                f"{rag['F1'] - baseline['F1']:+.3f}",
        },

        {
            "Metric":
                "Semantic Similarity",

            "Baseline LLM":
                decimal(
                    baseline["Semantic"]
                ),

            "RAG-Enhanced LLM":
                decimal(
                    rag["Semantic"]
                ),

            "Difference":
                f"{rag['Semantic'] - baseline['Semantic']:+.3f}",
        },
    ]


    sections.append(
        markdown_table(
            [
                "Metric",
                "Baseline LLM",
                "RAG-Enhanced LLM",
                "Difference",
            ],
            qa_rows
        )
    )


    # =================================================
    # TABLE 3
    # =================================================

    sections.append(
        "\nTABLE 3. ANSWER OUTCOMES — 303 ANSWERABLE QUESTIONS"
    )


    outcome_rows = [
        {
            "Outcome":
                "Correct",

            "Baseline":
                baseline[
                    "Correct"
                ],

            "RAG":
                rag[
                    "Correct"
                ],
        },

        {
            "Outcome":
                "Incorrect",

            "Baseline":
                baseline[
                    "Incorrect"
                ],

            "RAG":
                rag[
                    "Incorrect"
                ],
        },

        {
            "Outcome":
                "Incorrect abstention",

            "Baseline":
                baseline[
                    "Incorrect Abstentions"
                ],

            "RAG":
                rag[
                    "Incorrect Abstentions"
                ],
        },
    ]


    sections.append(
        markdown_table(
            [
                "Outcome",
                "Baseline",
                "RAG",
            ],
            outcome_rows
        )
    )


    # =================================================
    # TABLE 4 — H1-H8
    # =================================================

    sections.append(
        "\nTABLE 4. HALLUCINATION TEST — INDIVIDUAL QUESTIONS"
    )


    hallucination_question_rows = []


    for row in hallucination_rows:

        hallucination_question_rows.append(
            {
                "ID":
                    row["ID"],

                "Document":
                    row["Document"],

                "Question":
                    shorten(
                        row["Question"]
                    ),

                "Baseline Status":
                    row[
                        "Baseline Status"
                    ],

                "RAG Status":
                    row[
                        "RAG Status"
                    ],
            }
        )


    sections.append(
        markdown_table(
            [
                "ID",
                "Document",
                "Question",
                "Baseline Status",
                "RAG Status",
            ],
            hallucination_question_rows
        )
    )


    # =================================================
    # TABLE 5 — HALLUCINATION SUMMARY
    # =================================================

    sections.append(
        "\nTABLE 5. HALLUCINATION / ABSTENTION SUMMARY"
    )


    hallucination_summary_rows = [
        {
            "System":
                "Baseline LLM",

            "Hallucinations":
                baseline_h[
                    "Hallucinations"
                ],

            "Correct Abstentions":
                baseline_h[
                    "Correct Abstentions"
                ],

            "Hallucination Rate":
                percentage(
                    baseline_h[
                        "Hallucination Rate"
                    ]
                ),

            "Correct Abstention Rate":
                percentage(
                    baseline_h[
                        "Correct Abstention Rate"
                    ]
                ),
        },

        {
            "System":
                "RAG-Enhanced LLM",

            "Hallucinations":
                rag_h[
                    "Hallucinations"
                ],

            "Correct Abstentions":
                rag_h[
                    "Correct Abstentions"
                ],

            "Hallucination Rate":
                percentage(
                    rag_h[
                        "Hallucination Rate"
                    ]
                ),

            "Correct Abstention Rate":
                percentage(
                    rag_h[
                        "Correct Abstention Rate"
                    ]
                ),
        },
    ]


    sections.append(
        markdown_table(
            [
                "System",
                "Hallucinations",
                "Correct Abstentions",
                "Hallucination Rate",
                "Correct Abstention Rate",
            ],
            hallucination_summary_rows
        )
    )


    # =================================================
    # NOTES
    # =================================================

    sections.append(
        "\nEVALUATION NOTES"
    )


    sections.append(
        "- Overall accuracy includes all 311 benchmark questions."
    )


    sections.append(
        "- Precision, Recall, F1-score, and Semantic Similarity "
        "are calculated on the 303 answerable questions."
    )


    sections.append(
        "- H1-H8 are intentionally unsupported questions used "
        "to evaluate hallucination and abstention behavior."
    )


    sections.append(
        "- A correct abstention means the model recognizes that "
        "the answer is unavailable and does not continue with "
        "unsupported guesses, estimates, or company-specific claims."
    )


    # =================================================
    # SAVE ONLY ONE FILE
    # =================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    REPORT_FILE.write_text(
        "\n\n".join(
            sections
        )
        + "\n",
        encoding="utf-8"
    )


    return (
        baseline,
        rag,
        baseline_h,
        rag_h,
        baseline_overall,
        rag_overall
    )


# =====================================================
# MAIN
# =====================================================

def main():

    print(
        "=" * 70
    )

    print(
        "FINAL EVALUATION FROM SAVED ANSWERS"
    )

    print(
        "=" * 70
    )


    print(
        f"Input: {EVALUATION_FILE}"
    )

    print(
        "Ollama calls: 0"
    )

    print(
        "ChromaDB calls: 0\n"
    )


    (
        answerable_rows,
        hallucination_rows
    ) = calculate_metrics()


    (
        baseline,
        rag,
        baseline_h,
        rag_h,
        baseline_overall,
        rag_overall
    ) = write_report(
        answerable_rows,
        hallucination_rows
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION COMPLETED"
    )

    print(
        "=" * 70
    )


    print(
        "Total questions:",
        len(answerable_rows)
        +
        len(hallucination_rows)
    )


    print(
        "Answerable questions:",
        len(answerable_rows)
    )


    print(
        "Hallucination questions:",
        len(hallucination_rows)
    )


    print(
        "\nOverall Baseline accuracy:",
        percentage(
            baseline_overall
        )
    )


    print(
        "Overall RAG accuracy:",
        percentage(
            rag_overall
        )
    )


    print(
        "\nBaseline hallucination rate:",
        percentage(
            baseline_h[
                "Hallucination Rate"
            ]
        )
    )


    print(
        "RAG hallucination rate:",
        percentage(
            rag_h[
                "Hallucination Rate"
            ]
        )
    )


    print(
        "\nOnly one report created:"
    )


    print(
        REPORT_FILE
    )


    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()