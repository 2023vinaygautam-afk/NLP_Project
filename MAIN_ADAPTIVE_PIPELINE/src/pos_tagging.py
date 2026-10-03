
import json
import csv
import re
from pathlib import Path

import nltk
import spacy

from nltk import pos_tag, word_tokenize


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "cleaned_corpus.json"

OUTPUT_JSON = ROOT / "results" / "pos_tagging_comparison.json"
OUTPUT_CSV = ROOT / "results" / "pos_tagging_comparison.csv"

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD NLP MODELS
# --------------------------------------------------

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model missing. Run:\n"
        "python -m spacy download en_core_web_sm"
    )


# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def tokenize_text(text):
    return re.findall(
        r"\b[A-Za-z]+(?:[-'][A-Za-z0-9]+)*\b|\d+(?:\.\d+)?%?",
        text
    )


# --------------------------------------------------
# NLTK POS TAGGING
# --------------------------------------------------

def nltk_pos_tagging(text):

    tokens = tokenize_text(text)

    if not tokens:
        return []

    return [
        {
            "token": word,
            "pos": tag
        }
        for word, tag in pos_tag(tokens)
    ]


# --------------------------------------------------
# SPACY POS TAGGING
# --------------------------------------------------

def spacy_pos_tagging(text):

    doc = nlp(text)

    return [
        {
            "token": token.text,
            "pos": token.pos_,
            "tag": token.tag_
        }
        for token in doc
        if not token.is_space and not token.is_punct
    ]


# --------------------------------------------------
# POS TAGGING COMPARISON
# --------------------------------------------------

def compare_pos(nltk_results, spacy_results):

    nltk_tags = {}

    for item in nltk_results:
        token = item["token"].lower()
        nltk_tags.setdefault(token, []).append(item["pos"])

    spacy_tags = {}

    for item in spacy_results:
        token = item["token"].lower()
        spacy_tags.setdefault(token, []).append(item["pos"])

    common_words = set(nltk_tags) & set(spacy_tags)

    comparison = []

    for word in sorted(common_words):

        comparison.append({
            "word": word,
            "nltk_tags": nltk_tags[word],
            "spacy_tags": spacy_tags[word],
            "same_tag": (
                nltk_tags[word][0] == spacy_tags[word][0]
            )
        })

    return comparison


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\n"
            "Run text_preprocessing.py first."
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    results = []
    csv_rows = []

    total_nltk_tokens = 0
    total_spacy_tokens = 0

    for document in corpus:

        doc_id = document["doc_id"]
        text = document["cleaned_text"]

        nltk_results = nltk_pos_tagging(text)
        spacy_results = spacy_pos_tagging(text)

        comparison = compare_pos(
            nltk_results,
            spacy_results
        )

        total_nltk_tokens += len(nltk_results)
        total_spacy_tokens += len(spacy_results)

        results.append({
            "doc_id": doc_id,
            "document_name": document["document_name"],
            "nltk_token_count": len(nltk_results),
            "spacy_token_count": len(spacy_results),
            "nltk_pos": nltk_results,
            "spacy_pos": spacy_results,
            "comparison": comparison
        })

        for item in comparison:

            csv_rows.append({
                "doc_id": doc_id,
                "word": item["word"],
                "nltk_tags": ", ".join(item["nltk_tags"]),
                "spacy_tags": ", ".join(item["spacy_tags"]),
                "same_tag": item["same_tag"]
            })

    summary = {
        "total_documents": len(results),
        "total_nltk_tokens": total_nltk_tokens,
        "total_spacy_tokens": total_spacy_tokens,
        "documents": results
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "doc_id",
            "word",
            "nltk_tags",
            "spacy_tags",
            "same_tag"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nPOS TAGGING COMPLETED")
    print("----------------------")
    print("Documents:", len(results))
    print("NLTK tokens:", total_nltk_tokens)
    print("spaCy tokens:", total_spacy_tokens)
    print("JSON:", OUTPUT_JSON)
    print("CSV:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
