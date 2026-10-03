
import json
import csv
import re
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "cleaned_corpus.json"

OUTPUT_JSON = ROOT / "results" / "ner_results.json"

OUTPUT_CSV = ROOT / "results" / "ner_results.csv"

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. HEALTHCARE ENTITY DICTIONARY
# ==================================================

HEALTHCARE_ENTITIES = {

    "DISEASE": [
        "COVID-19",
        "coronavirus",
        "infectious disease",
        "respiratory infection"
    ],

    "MEDICAL_TERM": [
        "vaccination",
        "vaccine",
        "epidemiology",
        "contact tracing",
        "wastewater surveillance",
        "public health surveillance",
        "infection prevention",
        "vaccine safety",
        "reproduction number",
        "antiviral treatment",
        "outbreak investigation"
    ],

    "ORGANIZATION": [
        "World Health Organization",
        "WHO",
        "Centers for Disease Control and Prevention",
        "CDC"
    ],

    "TECHNOLOGY": [
        "RT-PCR",
        "PCR",
        "rapid antigen test"
    ]

}


# ==================================================
# 3. LOAD SPACY
# ==================================================

def load_spacy_model():

    try:
        return spacy.load("en_core_web_sm")

    except OSError:

        raise RuntimeError(
            "spaCy model is missing.\n"
            "Run:\n"
            "python -m spacy download en_core_web_sm"
        )


# ==================================================
# 4. LOAD CORPUS
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
                f"Available fields: {list(document.keys())}"
            )

        normalized.append({

            "doc_id": str(doc_id),

            "document_name": str(name),

            "text": text

        })

    return normalized


# ==================================================
# 5. CUSTOM HEALTHCARE MATCHER
# ==================================================

def create_healthcare_matcher(nlp):

    matcher = PhraseMatcher(
        nlp.vocab,
        attr="LOWER"
    )

    for entity_type, terms in HEALTHCARE_ENTITIES.items():

        patterns = [
            nlp.make_doc(term)
            for term in terms
        ]

        matcher.add(entity_type, patterns)

    return matcher


# ==================================================
# 6. SPACY NER
# ==================================================

def extract_spacy_entities(doc):

    entities = []

    for ent in doc.ents:

        entities.append({

            "text": ent.text,

            "label": ent.label_,

            "start": ent.start_char,

            "end": ent.end_char,

            "source": "spaCy"

        })

    return entities


# ==================================================
# 7. CUSTOM HEALTHCARE NER
# ==================================================

def extract_custom_entities(doc, matcher):

    matches = matcher(doc)

    entities = []

    for match_id, start, end in matches:

        span = doc[start:end]

        entity_type = doc.vocab.strings[match_id]

        entities.append({

            "text": span.text,

            "label": entity_type,

            "start": span.start_char,

            "end": span.end_char,

            "source": "Custom Healthcare Rules"

        })

    return entities


# ==================================================
# 8. ENTITY COMPARISON
# ==================================================

def compare_entities(spacy_entities, custom_entities):

    spacy_set = {

        (
            item["text"].lower(),
            item["label"]
        )

        for item in spacy_entities

    }

    custom_set = {

        (
            item["text"].lower(),
            item["label"]
        )

        for item in custom_entities

    }

    return {

        "spacy_entity_count": len(spacy_entities),

        "custom_entity_count": len(custom_entities),

        "common_entities": [
            list(item)
            for item in sorted(spacy_set & custom_set)
        ],

        "spacy_only": [
            list(item)
            for item in sorted(spacy_set - custom_set)
        ],

        "custom_only": [
            list(item)
            for item in sorted(custom_set - spacy_set)
        ]

    }


# ==================================================
# 9. MAIN EXECUTION
# ==================================================

def main():

    print("\nHEALTHCARE NER")
    print("==============")

    nlp = load_spacy_model()

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        raw_corpus = json.load(file)

    corpus = normalize_corpus(raw_corpus)

    matcher = create_healthcare_matcher(nlp)

    all_results = []

    csv_rows = []

    total_spacy_entities = 0

    total_custom_entities = 0

    for document in corpus:

        doc_id = document["doc_id"]

        document_name = document["document_name"]

        text = document["text"]

        doc = nlp(text)

        spacy_entities = extract_spacy_entities(doc)

        custom_entities = extract_custom_entities(
            doc,
            matcher
        )

        comparison = compare_entities(
            spacy_entities,
            custom_entities
        )

        total_spacy_entities += len(spacy_entities)

        total_custom_entities += len(custom_entities)

        for entity in spacy_entities:

            csv_rows.append({

                "doc_id": doc_id,

                "document_name": document_name,

                "entity": entity["text"],

                "label": entity["label"],

                "start": entity["start"],

                "end": entity["end"],

                "source": entity["source"]

            })

        for entity in custom_entities:

            csv_rows.append({

                "doc_id": doc_id,

                "document_name": document_name,

                "entity": entity["text"],

                "label": entity["label"],

                "start": entity["start"],

                "end": entity["end"],

                "source": entity["source"]

            })

        all_results.append({

            "doc_id": doc_id,

            "document_name": document_name,

            "spacy_entities": spacy_entities,

            "custom_entities": custom_entities,

            "comparison": comparison

        })

        print(
            f"Processed: {doc_id} - {document_name}"
        )

    # ==================================================
    # 10. SUMMARY
    # ==================================================

    summary = {

        "total_documents": len(corpus),

        "total_spacy_entities": total_spacy_entities,

        "total_custom_entities": total_custom_entities,

        "entity_categories": list(
            HEALTHCARE_ENTITIES.keys()
        ),

        "documents": all_results

    }

    # ==================================================
    # 11. SAVE JSON
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
    # 12. SAVE CSV
    # ==================================================

    fieldnames = [

        "doc_id",

        "document_name",

        "entity",

        "label",

        "start",

        "end",

        "source"

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
    # 13. FINAL OUTPUT
    # ==================================================

    print("\nNER COMPLETED")
    print("==============")

    print("Documents:", len(corpus))

    print("spaCy entities:", total_spacy_entities)

    print("Custom healthcare entities:", total_custom_entities)

    print("\nGenerated files:")

    print(OUTPUT_JSON)

    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
