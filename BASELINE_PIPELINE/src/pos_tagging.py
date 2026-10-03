
from pathlib import Path
import pandas as pd
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk import pos_tag


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD NLP MODELS
# --------------------------------------------------

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# DOMAIN-SPECIFIC POS RULES
# --------------------------------------------------

DOMAIN_NOUNS = {
    "covid-19",
    "sars-cov-2",
    "vaccination",
    "vaccine",
    "epidemiology",
    "surveillance",
    "transmission",
    "immunity",
    "quarantine",
    "outbreak",
    "infection",
    "incidence",
    "prevalence",
    "contact tracing",
    "antimicrobial resistance"
}


DOMAIN_VERBS = {
    "vaccinate",
    "immunize",
    "transmit",
    "surveil",
    "quarantine",
    "isolate",
    "monitor",
    "prevent"
}


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents():

    documents = {}

    for path in sorted(DOCUMENTS_DIR.glob("*.txt")):

        documents[path.stem] = path.read_text(
            encoding="utf-8"
        )

    if not documents:
        raise ValueError("No documents found.")

    return documents


# --------------------------------------------------
# NLTK POS
# --------------------------------------------------

def nltk_pos(text):

    tokens = word_tokenize(text)

    return pos_tag(tokens)


# --------------------------------------------------
# SPACY POS
# --------------------------------------------------

def spacy_pos(text):

    doc = nlp(text)

    return [
        (token.text, token.pos_)
        for token in doc
        if not token.is_space
    ]


# --------------------------------------------------
# CUSTOM POS
# --------------------------------------------------

def custom_pos(text):

    tokens = word_tokenize(text)

    tagged = []

    for token, default_tag in pos_tag(tokens):

        word = token.lower()

        if word in DOMAIN_NOUNS:
            tag = "NN"

        elif word in DOMAIN_VERBS:
            tag = "VB"

        else:
            tag = default_tag

        tagged.append((token, tag))

    return tagged


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def process_documents(documents):

    rows = []

    for doc_id, text in documents.items():

        methods = {
            "NLTK": nltk_pos(text),
            "spaCy": spacy_pos(text),
            "Custom_Rule_Based": custom_pos(text)
        }

        for method, tagged_tokens in methods.items():

            for token, tag in tagged_tokens:

                rows.append({
                    "Document_ID": doc_id,
                    "Method": method,
                    "Token": token,
                    "POS_Tag": tag
                })

    return pd.DataFrame(rows)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BASELINE PIPELINE")
    print("MODULE 4: POS TAGGING")
    print("=" * 60)

    documents = load_documents()

    results = process_documents(documents)

    summary = (
        results.groupby(["Method", "POS_Tag"])
        .size()
        .reset_index(name="Frequency")
    )

    print("\nPOS TAG DISTRIBUTION")
    print(summary.to_string(index=False))

    results.to_csv(
        RESULTS_DIR / "pos_tagging_comparison.csv",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "pos_tagging_summary.csv",
        index=False
    )

    print("\nSample POS Tags:")
    print(results.head(25).to_string(index=False))

    print("\nResults saved successfully!")


if __name__ == "__main__":
    main()
