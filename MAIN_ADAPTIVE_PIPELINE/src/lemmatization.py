
from pathlib import Path
import json
import csv
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy


# ==================================================
# 1. PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULT_DIR / "cleaned_corpus.json"

OUTPUT_JSON = RESULT_DIR / "lemmatization_comparison.json"
OUTPUT_CSV = RESULT_DIR / "lemmatization_comparison.csv"


# ==================================================
# 2. INITIALIZE MODELS
# ==================================================

try:
    wordnet_lemmatizer = WordNetLemmatizer()

    # Check WordNet resource
    wordnet_lemmatizer.lemmatize("testing")

except LookupError:

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

    wordnet_lemmatizer = WordNetLemmatizer()


nlp = spacy.load("en_core_web_sm")


# ==================================================
# 3. TOKENIZATION
# ==================================================

def tokenize(text):

    return re.findall(
        r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b",
        text.lower()
    )


# ==================================================
# 4. WORDNET POS MAPPING
# ==================================================

def get_wordnet_pos(tag):

    if tag.startswith("J"):
        return wordnet.ADJ

    elif tag.startswith("V"):
        return wordnet.VERB

    elif tag.startswith("N"):
        return wordnet.NOUN

    elif tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


# ==================================================
# 5. NLTK LEMMATIZATION
# ==================================================

def nltk_lemmatize(tokens):

    tagged_tokens = nltk.pos_tag(tokens)

    results = []

    for word, tag in tagged_tokens:

        pos = get_wordnet_pos(tag)

        lemma = wordnet_lemmatizer.lemmatize(
            word,
            pos=pos
        )

        results.append(lemma)

    return results


# ==================================================
# 6. SPACY LEMMATIZATION
# ==================================================

def spacy_lemmatize(text):

    doc = nlp(text)

    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_space
    ]


# ==================================================
# 7. CUSTOM HEALTHCARE LEMMATIZATION
# ==================================================

CUSTOM_LEMMAS = {

    "vaccines": "vaccine",
    "vaccinated": "vaccinate",
    "vaccinating": "vaccinate",
    "vaccinations": "vaccination",

    "infections": "infection",
    "infected": "infect",
    "infecting": "infect",

    "diseases": "disease",
    "variants": "variant",
    "outbreaks": "outbreak",

    "cases": "case",
    "studies": "study",
    "analyses": "analysis",

    "children": "child",
    "people": "person",
    "mice": "mouse"
}


def custom_lemmatize(tokens):

    results = []

    for token in tokens:

        lemma = CUSTOM_LEMMAS.get(
            token.lower(),
            token.lower()
        )

        results.append(lemma)

    return results


# ==================================================
# 8. PROCESS CORPUS
# ==================================================

def process_corpus(documents):

    methods = {
        "NLTK": nltk_lemmatize,
        "Custom": custom_lemmatize
    }

    summary = {}
    document_results = []
    word_examples = {}

    for method_name, method in methods.items():

        all_lemmas = []

        for document in documents:

            tokens = tokenize(
                document["cleaned_text"]
            )

            lemmas = method(tokens)

            all_lemmas.extend(lemmas)

            document_results.append({

                "document_id": document["document_id"],
                "method": method_name,
                "token_count": len(lemmas),
                "vocabulary_size": len(set(lemmas)),
                "lemmatized_tokens": lemmas
            })

            for original, lemma in zip(tokens, lemmas):

                if original != lemma:

                    word_examples.setdefault(
                        original, {}
                    )

                    word_examples[original][method_name] = lemma

        summary[method_name] = {

            "total_tokens": len(all_lemmas),

            "vocabulary_size": len(set(all_lemmas)),

            "average_tokens_per_document": round(
                len(all_lemmas) / len(documents), 2
            ) if documents else 0
        }

    # Process spaCy separately because it uses full text
    all_spacy_lemmas = []

    for document in documents:

        lemmas = spacy_lemmatize(
            document["cleaned_text"]
        )

        all_spacy_lemmas.extend(lemmas)

        document_results.append({

            "document_id": document["document_id"],
            "method": "spaCy",
            "token_count": len(lemmas),
            "vocabulary_size": len(set(lemmas)),
            "lemmatized_tokens": lemmas
        })

    summary["spaCy"] = {

        "total_tokens": len(all_spacy_lemmas),

        "vocabulary_size": len(set(all_spacy_lemmas)),

        "average_tokens_per_document": round(
            len(all_spacy_lemmas) / len(documents), 2
        ) if documents else 0
    }

    return summary, document_results, word_examples


# ==================================================
# 9. SAVE RESULTS
# ==================================================

def save_results(summary, document_results, word_examples):

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "summary": summary,
                "word_examples": word_examples,
                "document_results": document_results
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "document_id",
                "method",
                "token_count",
                "vocabulary_size",
                "lemmatized_tokens"
            ]
        )

        writer.writeheader()

        for row in document_results:

            writer.writerow({

                **row,

                "lemmatized_tokens": " ".join(
                    row["lemmatized_tokens"]
                )
            })


# ==================================================
# 10. MAIN
# ==================================================

def main():

    print("=" * 65)
    print("MODULE 6: LEMMATIZATION COMPARISON")
    print("=" * 65)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "cleaned_corpus.json not found. "
            "Run previous modules first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    summary, document_results, word_examples = process_corpus(
        documents
    )

    save_results(
        summary,
        document_results,
        word_examples
    )

    print("\nLEMMATIZATION COMPARISON")
    print("-" * 65)

    for method, stats in summary.items():

        print(f"\nMethod: {method}")

        print(f"Total Tokens: {stats['total_tokens']}")

        print(f"Vocabulary Size: {stats['vocabulary_size']}")

        print(
            "Average Tokens per Document:",
            stats["average_tokens_per_document"]
        )

    print("\nWORD LEMMATIZATION EXAMPLES")
    print("-" * 65)

    for word, methods in list(word_examples.items())[:20]:

        print(f"\nOriginal: {word}")

        for method, lemma in methods.items():

            print(f"{method}: {lemma}")

    print("\nOUTPUT FILES")
    print(OUTPUT_JSON)
    print(OUTPUT_CSV)

    print("\nMODULE 6 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
