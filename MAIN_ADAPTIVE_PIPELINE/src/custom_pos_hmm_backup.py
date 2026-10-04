"""
Assignment Exercise 12: ML-based POS tagger using a supervised HMM.

Trains on NLTK Penn Treebank, evaluates on a held-out sentence split,
then applies the trained tagger to the project's domain corpus.
Domain agreement with NLTK is reported separately from gold accuracy.
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

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "results" / "cleaned_corpus.json"
OUTPUT_CSV = ROOT / "results" / "custom_pos_ml_comparison.csv"
OUTPUT_JSON = ROOT / "results" / "custom_pos_ml_comparison.json"
METRICS_JSON = ROOT / "results" / "custom_pos_ml_metrics.json"
SEED = 42
TEST_FRACTION = 0.20
GAMMA = 0.1

TOKEN_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:[-'][A-Za-z0-9]+)*\b|\d+(?:\.\d+)?%?"
)


def ensure_treebank():
    try:
        treebank.fileids()
    except LookupError:
        nltk.download("treebank", quiet=False)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=False)


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def load_corpus(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing project corpus: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("documents", [])
    if not isinstance(raw, list):
        raise TypeError("Expected a list of document records or {'documents': [...]}.")

    docs = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Document record {i} is not an object.")
        doc_id = item.get("doc_id") or item.get("id") or item.get("document_id") or f"D{i:02d}"
        name = item.get("document_name") or item.get("name") or item.get("filename") or doc_id
        text = (
            item.get("cleaned_text") or item.get("cleaned") or item.get("text")
            or item.get("content") or item.get("original_text") or ""
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"No text found for document {doc_id}.")
        docs.append({"doc_id": str(doc_id), "document_name": str(name), "text": text})
    return docs


def train_and_evaluate():
    tagged = list(treebank.tagged_sents())
    rng = random.Random(SEED)
    rng.shuffle(tagged)
    split = max(1, int(len(tagged) * (1 - TEST_FRACTION)))
    train_sents, test_sents = tagged[:split], tagged[split:]

    trainer = HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(
        train_sents,
        estimator=lambda fd, bins: LidstoneProbDist(fd, GAMMA, bins),
    )

    correct = total = 0
    for sentence in test_sents:
        words = [word for word, _ in sentence]
        gold = [tag for _, tag in sentence]
        predicted = [tag for _, tag in tagger.tag(words)]
        for expected, actual in zip(gold, predicted):
            total += 1
            correct += int(expected == actual)

    return tagger, {
        "training_corpus": "NLTK Penn Treebank",
        "algorithm": "Supervised Hidden Markov Model (HMM)",
        "smoothing": f"Lidstone gamma={GAMMA}",
        "random_seed": SEED,
        "split": "Randomized sentence-level holdout",
        "training_sentences": len(train_sents),
        "test_sentences": len(test_sents),
        "test_tokens": total,
        "correct_test_tokens": correct,
        "heldout_treebank_accuracy": correct / total if total else None,
        "domain_accuracy": None,
        "domain_accuracy_note": (
            "Not measured: project-domain tokens have no manually annotated gold POS labels. "
            "NLTK/HMM agreement is reported only as agreement, not accuracy."
        ),
    }


def main():
    ensure_treebank()
    docs = load_corpus(INPUT_FILE)
    hmm, metrics = train_and_evaluate()

    rows = []
    detailed_docs = []
    domain_total = 0
    agreement_total = 0

    for doc in docs:
        tokens = tokenize(doc["text"])
        if not tokens:
            continue
        nltk_tags = pos_tag(tokens)
        hmm_tags = hmm.tag(tokens)
        token_rows = []
        for (word_n, tag_n), (word_h, tag_h) in zip(nltk_tags, hmm_tags):
            # Both taggers receive the exact same token sequence.
            same_token = word_n == word_h
            agree = (tag_n == tag_h) if same_token else False
            domain_total += 1
            agreement_total += int(agree)
            row = {
                "doc_id": doc["doc_id"],
                "document_name": doc["document_name"],
                "token": word_n,
                "nltk_default_pos": tag_n,
                "hmm_ml_pos": tag_h,
                "taggers_agree": agree,
            }
            rows.append(row)
            token_rows.append(row)

        detailed_docs.append({
            "doc_id": doc["doc_id"],
            "document_name": doc["document_name"],
            "tokens_tagged": len(token_rows),
            "token_comparison": token_rows,
        })

    metrics.update({
        "domain_documents_tagged": len(detailed_docs),
        "domain_tokens_tagged": domain_total,
        "nltk_hmm_agreement_tokens": agreement_total,
        "nltk_hmm_agreement_rate": agreement_total / domain_total if domain_total else None,
        "nltk_hmm_disagreement_tokens": domain_total - agreement_total,
        "domain_agreement_caveat": (
            "Agreement with NLTK is not domain-specific POS accuracy. To calculate domain "
            "accuracy, manually annotate a representative sample and compare predictions "
            "against those gold labels."
        ),
        "input_file": str(INPUT_FILE),
        "output_csv": str(OUTPUT_CSV),
        "output_json": str(OUTPUT_JSON),
    })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "doc_id", "document_name", "token",
            "nltk_default_pos", "hmm_ml_pos", "taggers_agree"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "documents": detailed_docs}, f, indent=2, ensure_ascii=False)

    with METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\nHMM MACHINE-LEARNING POS TAGGING COMPLETE")
    print(f"Documents tagged: {len(detailed_docs)}")
    print(f"Domain tokens tagged: {domain_total}")
    print(f"Held-out Treebank accuracy: {metrics['heldout_treebank_accuracy']:.4f}")
    print(f"NLTK/HMM domain agreement: {metrics['nltk_hmm_agreement_rate']:.4f}")
    print("Note: domain agreement is not domain accuracy.")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Detailed JSON: {OUTPUT_JSON}")
    print(f"Metrics JSON: {METRICS_JSON}")


if __name__ == "__main__":
    main()
