
from pathlib import Path
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# NLP TOOLS
# --------------------------------------------------

STOP_WORDS = set(stopwords.words("english"))

porter = PorterStemmer()
snowball = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# DOMAIN TERMS
# --------------------------------------------------

DOMAIN_TERMS = {
    "covid-19",
    "sars-cov-2",
    "vaccination",
    "vaccine",
    "epidemiology",
    "surveillance",
    "transmission",
    "infection",
    "immunity",
    "outbreak",
    "quarantine"
}


# --------------------------------------------------
# PREPROCESSING FUNCTIONS
# --------------------------------------------------

def remove_stopwords(tokens):

    return [
        token for token in tokens
        if token.lower() not in STOP_WORDS
    ]


def apply_porter(tokens):

    return [porter.stem(token) for token in tokens]


def apply_snowball(tokens):

    return [snowball.stem(token) for token in tokens]


def apply_lemmatization(tokens):

    return [
        lemmatizer.lemmatize(token.lower())
        for token in tokens
    ]


# --------------------------------------------------
# DOCUMENT LOADING
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
# COMPARISON
# --------------------------------------------------

def compare_preprocessing(documents):

    rows = []

    for doc_id, text in documents.items():

        tokens = word_tokenize(text.lower())

        variants = {
            "Original": tokens,
            "Stopword_Removal": remove_stopwords(tokens),
            "Porter_Stemming": apply_porter(
                remove_stopwords(tokens)
            ),
            "Snowball_Stemming": apply_snowball(
                remove_stopwords(tokens)
            ),
            "WordNet_Lemmatization": apply_lemmatization(
                remove_stopwords(tokens)
            )
        }

        for method, processed in variants.items():

            domain_preserved = sum(
                1 for token in processed
                if token.lower() in DOMAIN_TERMS
            )

            rows.append({
                "Document_ID": doc_id,
                "Method": method,
                "Token_Count": len(processed),
                "Vocabulary_Size": len(set(processed)),
                "Domain_Term_Count": domain_preserved,
                "Sample_Tokens": " | ".join(processed[:20])
            })

    return pd.DataFrame(rows)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BASELINE PIPELINE")
    print("MODULE 3: STOPWORDS, STEMMING AND LEMMATIZATION")
    print("=" * 60)

    documents = load_documents()

    results = compare_preprocessing(documents)

    summary = (
        results.groupby("Method")
        .agg(
            Total_Tokens=("Token_Count", "sum"),
            Average_Tokens=("Token_Count", "mean"),
            Total_Document_Vocabulary=(
                "Vocabulary_Size", "sum"
            ),
            Domain_Term_Count=("Domain_Term_Count", "sum")
        )
        .reset_index()
    )

    print("\nPREPROCESSING COMPARISON")
    print(summary.to_string(index=False))

    results.to_csv(
        RESULTS_DIR / "preprocessing_comparison.csv",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "preprocessing_summary.csv",
        index=False
    )

    print("\nResults saved successfully!")


if __name__ == "__main__":
    main()
