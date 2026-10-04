
"""
Assignment Exercise 12:
Machine Learning-Based POS Tagger Using a Supervised Hidden Markov Model.

Pipeline:
1. Train an HMM using the NLTK Penn Treebank.
2. Evaluate it using a randomized 80:20 sentence-level split.
3. Apply the trained HMM to the project's healthcare corpus.
4. Compare HMM predictions with the default NLTK POS tagger.
5. Report held-out accuracy and domain agreement separately.

Important:
Domain agreement with NLTK is NOT domain-specific POS accuracy.
Actual domain accuracy requires manually annotated gold POS labels.
"""

import csv
import json
import random
import re
from pathlib import Path

import nltk
from nltk.corpus import treebank
from nltk.tag import HiddenMarkovModelTrainer, pos_tag
from nltk.probability import LidstoneProbDist
from nltk.tokenize import sent_tokenize


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
# 3. TOKENIZATION
# ============================================================

TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:[-'][A-Za-z0-9]+)*\b|\d+(?:\.\d+)?%?"
)


def tokenize(text):
    """
    Extract words, hyphenated terms, and numeric expressions.
    """
    return TOKEN_PATTERN.findall(text)


# ============================================================
# 4. DOWNLOAD REQUIRED NLTK RESOURCES
# ============================================================

def ensure_resources():

    resources = [
        ("corpora/treebank", "treebank"),
        (
            "taggers/averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger_eng",
        ),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]

    for resource_path, package_name in resources:

        try:
            nltk.data.find(resource_path)

        except LookupError:
            print(f"Downloading NLTK resource: {package_name}")
            nltk.download(package_name, quiet=False)

    # Verify that the Treebank corpus is accessible.
    treebank.fileids()


# ============================================================
# 5. LOAD PROJECT CORPUS
# ============================================================

def load_corpus(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing project corpus: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if isinstance(raw, dict):
        raw = raw.get("documents", [])

    if not isinstance(raw, list):
        raise TypeError(
            "Expected a list of document records "
            "or a dictionary containing 'documents'."
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
# 6. TRAIN AND EVALUATE SUPERVISED HMM
# ============================================================

def train_and_evaluate():

    print("\nLoading Penn Treebank dataset...")

    tagged_sentences = list(treebank.tagged_sents())

    rng = random.Random(SEED)

    rng.shuffle(tagged_sentences)

    split_index = max(
        1,
        int(len(tagged_sentences) * (1 - TEST_FRACTION))
    )

    train_sentences = tagged_sentences[:split_index]

    test_sentences = tagged_sentences[split_index:]

    print(f"Training sentences: {len(train_sentences)}")
    print(f"Testing sentences: {len(test_sentences)}")

    print("\nTraining supervised Hidden Markov Model...")

    trainer = HiddenMarkovModelTrainer()

    hmm = trainer.train_supervised(
        train_sentences,
        estimator=lambda frequency_distribution, bins: (
            LidstoneProbDist(
                frequency_distribution,
                GAMMA,
                bins
            )
        ),
    )

    # --------------------------------------------------------
    # Evaluate on held-out Treebank sentences
    # --------------------------------------------------------

    correct = 0
    total = 0

    for sentence in test_sentences:

        words = [
            word for word, tag in sentence
        ]

        gold_tags = [
            tag for word, tag in sentence
        ]

        predicted_tags = [
            tag for word, tag in hmm.tag(words)
        ]

        for expected, predicted in zip(
            gold_tags,
            predicted_tags
        ):

            total += 1

            if expected == predicted:
                correct += 1

    accuracy = correct / total if total else None

    metrics = {
        "training_corpus": "NLTK Penn Treebank",
        "algorithm": "Supervised Hidden Markov Model (HMM)",
        "smoothing": f"Lidstone gamma={GAMMA}",
        "random_seed": SEED,
        "split": "Randomized sentence-level 80:20 holdout",
        "training_sentences": len(train_sentences),
        "test_sentences": len(test_sentences),
        "test_tokens": total,
        "correct_test_tokens": correct,
        "heldout_treebank_accuracy": accuracy,
        "domain_accuracy": None,
        "domain_accuracy_note": (
            "Domain accuracy was not measured because the "
            "project corpus does not contain manually annotated "
            "gold POS labels. NLTK/HMM agreement is reported "
            "separately and must not be interpreted as accuracy."
        ),
    }

    return hmm, metrics


# ============================================================
# 7. SENTENCE-WISE DOMAIN TAGGING
# ============================================================

def tag_domain_documents(docs, hmm):

    rows = []

    detailed_docs = []

    domain_total = 0

    agreement_total = 0

    for doc in docs:

        # Split the document into sentences before HMM tagging.
        sentences = sent_tokenize(doc["text"])

        token_rows = []

        sentence_count = 0

        for sentence_number, sentence in enumerate(
            sentences,
            start=1
        ):

            tokens = tokenize(sentence)

            if not tokens:
                continue

            sentence_count += 1

            # Both taggers receive exactly the same tokens.
            nltk_tags = pos_tag(tokens)

            hmm_tags = hmm.tag(tokens)

            for (word_n, tag_n), (word_h, tag_h) in zip(
                nltk_tags,
                hmm_tags
            ):

                same_token = word_n == word_h

                agree = (
                    tag_n == tag_h
                    if same_token
                    else False
                )

                domain_total += 1

                agreement_total += int(agree)

                row = {
                    "doc_id": doc["doc_id"],
                    "document_name": doc["document_name"],
                    "sentence_number": sentence_number,
                    "token": word_n,
                    "nltk_default_pos": tag_n,
                    "hmm_ml_pos": tag_h,
                    "taggers_agree": agree,
                }

                rows.append(row)

                token_rows.append(row)

        if not token_rows:
            continue

        detailed_docs.append({
            "doc_id": doc["doc_id"],
            "document_name": doc["document_name"],
            "sentence_count": sentence_count,
            "tokens_tagged": len(token_rows),
            "token_comparison": token_rows,
        })

    return (
        rows,
        detailed_docs,
        domain_total,
        agreement_total,
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_results(
    rows,
    detailed_docs,
    metrics
):

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "doc_id",
        "document_name",
        "sentence_number",
        "token",
        "nltk_default_pos",
        "hmm_ml_pos",
        "taggers_agree",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(rows)

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "metrics": metrics,
                "documents": detailed_docs,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    with METRICS_JSON.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("ML-BASED POS TAGGING: SUPERVISED HMM")
    print("=" * 60)

    ensure_resources()

    docs = load_corpus(INPUT_FILE)

    print(f"\nProject documents loaded: {len(docs)}")

    hmm, metrics = train_and_evaluate()

    (
        rows,
        detailed_docs,
        domain_total,
        agreement_total,
    ) = tag_domain_documents(docs, hmm)

    agreement_rate = (
        agreement_total / domain_total
        if domain_total
        else None
    )

    metrics.update({
        "domain_documents_tagged": len(detailed_docs),
        "domain_tokens_tagged": domain_total,
        "nltk_hmm_agreement_tokens": agreement_total,
        "nltk_hmm_agreement_rate": agreement_rate,
        "nltk_hmm_disagreement_tokens": (
            domain_total - agreement_total
        ),
        "domain_agreement_caveat": (
            "Agreement with NLTK is not domain-specific POS "
            "accuracy. A manually annotated gold-standard "
            "sample is required to calculate domain accuracy."
        ),
        "input_file": str(INPUT_FILE),
        "output_csv": str(OUTPUT_CSV),
        "output_json": str(OUTPUT_JSON),
    })

    save_results(
        rows,
        detailed_docs,
        metrics
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("HMM MACHINE-LEARNING POS TAGGING COMPLETE")
    print("=" * 60)

    print(
        f"Documents tagged: {len(detailed_docs)}"
    )

    print(
        f"Domain tokens tagged: {domain_total}"
    )

    print(
        "Held-out Treebank accuracy: "
        f"{metrics['heldout_treebank_accuracy']:.4f}"
    )

    print(
        "NLTK/HMM domain agreement: "
        f"{agreement_rate:.4f}"
    )

    print(
        f"Agreement tokens: {agreement_total}"
    )

    print(
        f"Disagreement tokens: "
        f"{domain_total - agreement_total}"
    )

    print(
        "\nNote: Domain agreement is not domain accuracy."
    )

    print(f"\nCSV: {OUTPUT_CSV}")

    print(f"Detailed JSON: {OUTPUT_JSON}")

    print(f"Metrics JSON: {METRICS_JSON}")


if __name__ == "__main__":
    main()
