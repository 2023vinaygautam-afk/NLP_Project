
import json
import csv
import re
from pathlib import Path

import nltk
import spacy

from nltk import pos_tag


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "cleaned_corpus.json"

OUTPUT_JSON = ROOT / "results" / "custom_pos_comparison.json"

OUTPUT_CSV = ROOT / "results" / "custom_pos_comparison.csv"

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. HEALTHCARE DOMAIN VOCABULARY
# ==================================================

HEALTHCARE_NOUNS = {

    "covid-19",
    "covid",
    "coronavirus",
    "virus",
    "infection",
    "infections",
    "disease",
    "diseases",
    "vaccine",
    "vaccines",
    "vaccination",
    "vaccinations",
    "epidemiology",
    "epidemiologist",
    "transmission",
    "outbreak",
    "outbreaks",
    "pandemic",
    "epidemic",
    "incidence",
    "prevalence",
    "surveillance",
    "immunity",
    "antibody",
    "antibodies",
    "variant",
    "variants",
    "pathogen",
    "pathogens",
    "testing",
    "diagnosis",
    "diagnostic",
    "quarantine",
    "isolation",
    "symptom",
    "symptoms",
    "hospital",
    "hospitals",
    "patient",
    "patients",
    "healthcare",
    "immunization",
    "immunisation",
    "mortality",
    "morbidity",
    "prevention",
    "protection",
    "contact",
    "tracing",
    "risk",
    "population",
    "populations",
    "variant",
    "variants",
    "clinical",
    "epidemiological",
    "vaccine-safety",
    "reproduction-number",
    "rt-pcr",
    "pcr",
    "who",
    "cdc"

}


HEALTHCARE_VERBS = {

    "transmit",
    "transmits",
    "transmitted",
    "transmitting",
    "infect",
    "infects",
    "infected",
    "infecting",
    "vaccinate",
    "vaccinates",
    "vaccinated",
    "vaccinating",
    "prevent",
    "prevents",
    "prevented",
    "preventing",
    "monitor",
    "monitors",
    "monitored",
    "monitoring",
    "detect",
    "detects",
    "detected",
    "detecting",
    "spread",
    "spreads",
    "spreading",
    "reduce",
    "reduces",
    "reduced",
    "reducing",
    "protect",
    "protects",
    "protected",
    "protecting",
    "test",
    "tests",
    "tested",
    "testing",
    "track",
    "tracks",
    "tracked",
    "tracking",
    "isolate",
    "isolates",
    "isolated",
    "isolating"

}


HEALTHCARE_ADJECTIVES = {

    "infectious",
    "infected",
    "epidemiological",
    "clinical",
    "viral",
    "vaccine-related",
    "preventive",
    "preventable",
    "immune",
    "immunological",
    "contagious",
    "severe",
    "mild",
    "asymptomatic",
    "symptomatic",
    "effective",
    "effective",
    "respiratory",
    "public-health"

}


# ==================================================
# 3. LOAD MODELS
# ==================================================

def setup_nltk():

    try:
        nltk.data.find(
            "taggers/averaged_perceptron_tagger_eng"
        )

    except LookupError:
        nltk.download(
            "averaged_perceptron_tagger_eng"
        )


def load_spacy():

    try:
        return spacy.load("en_core_web_sm")

    except OSError:

        raise RuntimeError(
            "Install spaCy model using:\n"
            "python -m spacy download en_core_web_sm"
        )


# ==================================================
# 4. TOKENIZATION
# ==================================================

def tokenize_text(text):

    pattern = (
        r"\b[A-Za-z]+(?:[-'][A-Za-z0-9]+)*\b"
        r"|\d+(?:\.\d+)?%?"
    )

    return re.findall(pattern, text)


# ==================================================
# 5. CUSTOM POS RULES
# ==================================================

def custom_pos_tagging(text):

    tokens = tokenize_text(text)

    results = []

    for word in tokens:

        lower_word = word.lower()

        if lower_word in HEALTHCARE_VERBS:

            tag = "DOMAIN_VERB"

        elif lower_word in HEALTHCARE_ADJECTIVES:

            tag = "DOMAIN_ADJ"

        elif lower_word in HEALTHCARE_NOUNS:

            tag = "DOMAIN_NOUN"

        elif re.fullmatch(r"\d+(?:\.\d+)?%?", word):

            tag = "NUM"

        elif word.istitle():

            tag = "PROPN"

        else:

            tag = "GENERAL"

        results.append({

            "token": word,

            "pos": tag,

            "method": "Custom Healthcare Rules"

        })

    return results


# ==================================================
# 6. NLTK POS TAGGING
# ==================================================

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


# ==================================================
# 7. SPACY POS TAGGING
# ==================================================

def spacy_pos_tagging(text, nlp):

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


# ==================================================
# 8. CORPUS NORMALIZATION
# ==================================================

def normalize_corpus(corpus):

    if isinstance(corpus, dict):

        corpus = corpus.get("documents", [])

    if not isinstance(corpus, list):

        raise TypeError(
            "Expected a list of documents."
        )

    normalized = []

    for index, document in enumerate(corpus, start=1):

        if not isinstance(document, dict):

            raise TypeError(
                f"Document {index} is not a dictionary."
            )

        doc_id = (
            document.get("doc_id")
            or document.get("id")
            or document.get("document_id")
            or f"D{index:02d}"
        )

        name = (
            document.get("document_name")
            or document.get("name")
            or document.get("filename")
            or document.get("document")
            or doc_id
        )

        text = (
            document.get("cleaned_text")
            or document.get("cleaned")
            or document.get("text")
            or document.get("content")
            or document.get("original_text")
            or ""
        )

        if not isinstance(text, str) or not text.strip():

            raise ValueError(
                f"Missing text for {doc_id}. "
                f"Fields found: {list(document.keys())}"
            )

        normalized.append({

            "doc_id": str(doc_id),

            "document_name": str(name),

            "text": text

        })

    return normalized


# ==================================================
# 9. COMPARE TAGGING METHODS
# ==================================================

def compare_methods(
    nltk_results,
    spacy_results,
    custom_results
):

    comparison = []

    total = min(
        len(nltk_results),
        len(spacy_results),
        len(custom_results)
    )

    for index in range(total):

        nltk_item = nltk_results[index]

        spacy_item = spacy_results[index]

        custom_item = custom_results[index]

        comparison.append({

            "token": custom_item["token"],

            "nltk_pos": nltk_item["pos"],

            "spacy_pos": spacy_item["pos"],

            "custom_pos": custom_item["pos"]

        })

    return comparison


# ==================================================
# 10. MAIN EXECUTION
# ==================================================

def main():

    print("\nCUSTOM HEALTHCARE POS TAGGING")
    print("=============================")

    setup_nltk()

    nlp = load_spacy()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        raw_corpus = json.load(file)

    corpus = normalize_corpus(raw_corpus)

    all_results = []

    csv_rows = []

    total_tokens = 0

    domain_nouns = 0

    domain_verbs = 0

    domain_adjectives = 0

    for document in corpus:

        doc_id = document["doc_id"]

        name = document["document_name"]

        text = document["text"]

        nltk_results = nltk_pos_tagging(text)

        spacy_results = spacy_pos_tagging(text, nlp)

        custom_results = custom_pos_tagging(text)

        comparison = compare_methods(
            nltk_results,
            spacy_results,
            custom_results
        )

        total_tokens += len(custom_results)

        for item in custom_results:

            if item["pos"] == "DOMAIN_NOUN":
                domain_nouns += 1

            elif item["pos"] == "DOMAIN_VERB":
                domain_verbs += 1

            elif item["pos"] == "DOMAIN_ADJ":
                domain_adjectives += 1

        for item in comparison:

            csv_rows.append({

                "doc_id": doc_id,

                "document_name": name,

                "token": item["token"],

                "nltk_pos": item["nltk_pos"],

                "spacy_pos": item["spacy_pos"],

                "custom_pos": item["custom_pos"]

            })

        all_results.append({

            "doc_id": doc_id,

            "document_name": name,

            "nltk_pos": nltk_results,

            "spacy_pos": spacy_results,

            "custom_pos": custom_results,

            "comparison": comparison

        })

        print(
            f"Processed: {doc_id} - {name}"
        )

    # ==================================================
    # 11. SUMMARY
    # ==================================================

    summary = {

        "total_documents": len(corpus),

        "total_tokens": total_tokens,

        "domain_nouns": domain_nouns,

        "domain_verbs": domain_verbs,

        "domain_adjectives": domain_adjectives,

        "documents": all_results

    }

    # ==================================================
    # 12. SAVE JSON
    # ==================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ==================================================
    # 13. SAVE CSV
    # ==================================================

    fieldnames = [

        "doc_id",

        "document_name",

        "token",

        "nltk_pos",

        "spacy_pos",

        "custom_pos"

    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(csv_rows)

    # ==================================================
    # 14. FINAL OUTPUT
    # ==================================================

    print("\nCUSTOM POS TAGGING COMPLETED")
    print("============================")

    print("Documents:", len(corpus))

    print("Total tokens:", total_tokens)

    print("Domain nouns:", domain_nouns)

    print("Domain verbs:", domain_verbs)

    print("Domain adjectives:", domain_adjectives)

    print("\nGenerated files:")

    print(OUTPUT_JSON)

    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
