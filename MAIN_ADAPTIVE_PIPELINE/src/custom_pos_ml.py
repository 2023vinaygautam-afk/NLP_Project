
"""
Assignment Exercise 12:
ML-based POS tagger using a supervised Hidden Markov Model.

1. Train on NLTK Penn Treebank.
2. Evaluate using a held-out file-level split.
3. Compare HMM and default NLTK performance.
4. Apply both taggers to the healthcare domain corpus.
5. Preserve punctuation during tokenization.
6. Report domain agreement separately from gold accuracy.

Domain agreement with NLTK is NOT domain-specific POS accuracy.
"""

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

import nltk
from nltk.corpus import treebank
from nltk.tag import HiddenMarkovModelTrainer, pos_tag
from nltk.probability import LidstoneProbDist


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "cleaned_corpus.json"

OUTPUT_CSV = ROOT / "results" / "custom_pos_ml_comparison.csv"

OUTPUT_JSON = ROOT / "results" / "custom_pos_ml_comparison.json"

METRICS_JSON = ROOT / "results" / "custom_pos_ml_metrics.json"


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

SEED = 42
TEST_FRACTION = 0.20
GAMMA = 0.1


# ============================================================
# 3. PUNCTUATION-AWARE TOKENIZATION
# ============================================================

TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?%?"
    r"|[A-Za-z]+(?:[-'][A-Za-z0-9]+)*"
    r"|[^\w\s]"
)


def tokenize(text):
    """
    Extract words, hyphenated terms, numbers, and punctuation.
    Punctuation is preserved as individual tokens.
    """
    return TOKEN_PATTERN.findall(text)


def tokenize_sentences(text):
    """
    Split the document into sentences and tokenize each sentence.

    Sentence boundaries are preserved so the HMM does not
    process an entire document as one continuous sequence.
    """
    return [
        tokens
        for sentence in nltk.sent_tokenize(text)
        if (tokens := tokenize(sentence))
    ]


# ============================================================
# 4. NLTK RESOURCE VALIDATION
# ============================================================

def ensure_treebank():

    try:
        treebank.fileids()

    except LookupError:
        if not nltk.download("treebank", quiet=False):
            raise RuntimeError(
                "Could not download the NLTK Treebank corpus."
            )

    for resource, package in (
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        (
            "taggers/averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger_eng",
        ),
    ):

        try:
            nltk.data.find(resource)

        except LookupError:
            if not nltk.download(package, quiet=False):
                raise RuntimeError(
                    f"Could not download NLTK resource: {package}."
                )

    if not treebank.fileids():
        raise RuntimeError("The NLTK Treebank corpus is empty.")


# ============================================================
# 5. LOAD PROJECT CORPUS
# ============================================================

def load_corpus(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing project corpus: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("documents", [])

    if not isinstance(raw, list):
        raise TypeError(
            "Expected a list of document records "
            "or {'documents': [...]}."
        )

    docs = []

    for i, item in enumerate(raw, start=1):

        if not isinstance(item, dict):
            raise TypeError(
                f"Document record {i} is not an object."
            )

        doc_id = (
            item.get("doc_id")
            or item.get("id")
            or item.get("document_id")
            or f"D{i:02d}"
        )

        name = (
            item.get("document_name")
            or item.get("name")
            or item.get("filename")
            or doc_id
        )

        text = (
            item.get("cleaned_text")
            or item.get("cleaned")
            or item.get("text")
            or item.get("content")
            or item.get("original_text")
            or ""
        )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"No text found for document {doc_id}."
            )

        docs.append({
            "doc_id": str(doc_id),
            "document_name": str(name),
            "text": text,
        })

    return docs


# ============================================================
# 6. TRAIN AND EVALUATE HMM
# ============================================================

def train_and_evaluate():

    file_ids = list(treebank.fileids())

    if len(file_ids) < 2:
        raise ValueError(
            "NLTK Treebank must contain at least two source files."
        )

    rng = random.Random(SEED)

    rng.shuffle(file_ids)

    split = min(
        max(
            int(len(file_ids) * (1 - TEST_FRACTION)),
            1
        ),
        len(file_ids) - 1,
    )

    train_files = file_ids[:split]
    test_files = file_ids[split:]

    train_sents = list(
        treebank.tagged_sents(fileids=train_files)
    )

    test_sents = list(
        treebank.tagged_sents(fileids=test_files)
    )

    if not train_sents or not test_sents:
        raise ValueError(
            "Treebank split produced an empty training or test set."
        )

    print(f"Training sentences: {len(train_sents)}")
    print(f"Testing sentences: {len(test_sents)}")

    print("\nTraining supervised HMM...")

    trainer = HiddenMarkovModelTrainer()

    tagger = trainer.train_supervised(
        train_sents,
        estimator=lambda fd, bins: LidstoneProbDist(
            fd,
            GAMMA,
            bins
        ),
    )

    # --------------------------------------------------------
    # Evaluate HMM and default NLTK on the same test sentences
    # --------------------------------------------------------

    correct = 0
    default_correct = 0
    total = 0

    for sentence in test_sents:

        words = [
            word for word, _ in sentence
        ]

        gold = [
            tag for _, tag in sentence
        ]

        predicted = [
            tag for _, tag in tagger.tag(words)
        ]

        default_predicted = [
            tag for _, tag in pos_tag(words)
        ]

        if (
            len(gold) != len(predicted)
            or len(gold) != len(default_predicted)
        ):
            raise RuntimeError(
                "A POS tagger returned an unexpected sequence length."
            )

        for expected, actual in zip(gold, predicted):

            total += 1

            correct += int(expected == actual)

        default_correct += sum(
            expected == actual
            for expected, actual in zip(
                gold,
                default_predicted
            )
        )

    if total == 0:
        raise ValueError(
            "Treebank test split contains no tokens."
        )

    metrics = {
        "model": "Supervised Hidden Markov Model (HMM)",
        "training_data": "NLTK Penn Treebank",
        "random_seed": SEED,
        "test_fraction": TEST_FRACTION,
        "split_strategy": "Randomized file-level split",
        "smoothing_gamma": GAMMA,
        "training_files": len(train_files),
        "test_files": len(test_files),
        "training_sentences": len(train_sents),
        "test_sentences": len(test_sents),
        "test_tokens": total,
        "hmm_correct_tokens": correct,
        "hmm_heldout_accuracy": correct / total,
        "nltk_default_correct_tokens": default_correct,
        "nltk_default_heldout_accuracy": (
            default_correct / total
        ),
        "accuracy_difference_percentage_points": (
            (correct - default_correct) / total * 100
        ),
        "domain_accuracy": None,
        "domain_accuracy_note": (
            "Domain accuracy requires manually annotated gold POS labels."
        ),
    }

    return tagger, metrics


# ============================================================
# 7. DOMAIN POS TAGGING
# ============================================================

def tag_domain_documents(docs, tagger):

    rows = []

    for doc in docs:

        sentence_groups = tokenize_sentences(doc["text"])

        for sentence_number, tokens in enumerate(
            sentence_groups,
            start=1
        ):

            nltk_tags = pos_tag(tokens)

            hmm_tags = tagger.tag(tokens)

            if (
                len(tokens) != len(nltk_tags)
                or len(tokens) != len(hmm_tags)
            ):
                raise RuntimeError(
                    f"Tagger output length mismatch for "
                    f"{doc['doc_id']}, sentence {sentence_number}."
                )

            for token, (_, nltk_pos), (_, hmm_pos) in zip(
                tokens,
                nltk_tags,
                hmm_tags
            ):

                rows.append({
                    "document_id": doc["doc_id"],
                    "document_name": doc["document_name"],
                    "sentence_number": sentence_number,
                    "token": token,
                    "nltk_default_pos": nltk_pos,
                    "hmm_ml_pos": hmm_pos,
                    "taggers_agree": nltk_pos == hmm_pos,
                })

    if not rows:
        raise ValueError(
            "No tokens were produced from the project documents."
        )

    return rows


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_results(rows, metrics):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = list(rows[0].keys())

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "token_comparison": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    METRICS_JSON.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("ML-BASED POS TAGGING: SUPERVISED HMM")
    print("=" * 60)

    ensure_treebank()

    docs = load_corpus(INPUT_FILE)

    print(f"\nProject documents loaded: {len(docs)}")

    tagger, metrics = train_and_evaluate()

    rows = tag_domain_documents(docs, tagger)

    agreement_count = sum(
        row["taggers_agree"]
        for row in rows
    )

    metrics.update({
        "domain_documents_tagged": len(docs),
        "domain_tokens_tagged": len(rows),
        "domain_nltk_hmm_agreement_count": agreement_count,
        "domain_nltk_hmm_agreement": (
            agreement_count / len(rows)
        ),
        "domain_nltk_hmm_disagreement_count": (
            len(rows) - agreement_count
        ),
        "comparison_csv": str(OUTPUT_CSV),
        "comparison_json": str(OUTPUT_JSON),
        "metrics_json": str(METRICS_JSON),
        "tokenization": (
            "Sentence-wise tokenization with punctuation preserved"
        ),
        "interpretation": (
            "Held-out Treebank accuracy compares both taggers "
            "against Treebank gold labels. NLTK-HMM agreement "
            "on project documents is not domain-specific accuracy; "
            "that requires manually verified labels."
        ),
    })

    save_results(rows, metrics)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("HMM MACHINE-LEARNING POS TAGGING COMPLETE")
    print("=" * 60)

    print(f"Documents tagged: {len(docs)}")

    print(f"Domain tokens tagged: {len(rows)}")

    print(
        "Treebank test accuracy (HMM): "
        f"{metrics['hmm_heldout_accuracy']:.4f}"
    )

    print(
        "Treebank test accuracy (NLTK default): "
        f"{metrics['nltk_default_heldout_accuracy']:.4f}"
    )

    print(
        "Accuracy difference (percentage points): "
        f"{metrics['accuracy_difference_percentage_points']:.2f}"
    )

    print(
        "Domain NLTK-HMM agreement: "
        f"{metrics['domain_nltk_hmm_agreement']:.4f}"
    )

    print(
        "Domain disagreements: "
        f"{metrics['domain_nltk_hmm_disagreement_count']}"
    )

    print(
        "\nNote: Domain agreement is not domain accuracy."
    )

    print(f"\nComparison CSV: {OUTPUT_CSV}")

    print(f"Comparison JSON: {OUTPUT_JSON}")

    print(f"Evaluation metrics: {METRICS_JSON}")


if __name__ == "__main__":
    main()
