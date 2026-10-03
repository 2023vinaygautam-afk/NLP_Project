
from pathlib import Path
import pandas as pd
import spacy


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# LOAD SPACY MODEL
# --------------------------------------------------

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# ENTITY TYPES REQUIRED BY ASSIGNMENT
# --------------------------------------------------

REQUIRED_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "DATE",
    "MONEY",
    "PRODUCT",
    "EVENT"
}


# --------------------------------------------------
# HEALTHCARE DOMAIN TERMS
# --------------------------------------------------

DOMAIN_TERMS = {
    "COVID-19",
    "SARS-CoV-2",
    "vaccination",
    "vaccine",
    "epidemiology",
    "surveillance",
    "immunity",
    "transmission",
    "outbreak",
    "infection",
    "quarantine",
    "contact tracing",
    "public health",
    "antimicrobial resistance"
}


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents():

    documents = {}

    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):

        documents[file_path.stem] = file_path.read_text(
            encoding="utf-8"
        )

    if not documents:
        raise ValueError("No healthcare documents found.")

    return documents


# --------------------------------------------------
# EXTRACT ENTITIES
# --------------------------------------------------

def extract_entities(documents):

    rows = []

    for doc_id, text in documents.items():

        doc = nlp(text)

        for entity in doc.ents:

            rows.append({
                "Document_ID": doc_id,
                "Entity": entity.text,
                "Entity_Label": entity.label_,
                "Start_Position": entity.start_char,
                "End_Position": entity.end_char,
                "Required_Label": (
                    entity.label_ in REQUIRED_ENTITY_TYPES
                )
            })

    return pd.DataFrame(
        rows,
        columns=[
            "Document_ID",
            "Entity",
            "Entity_Label",
            "Start_Position",
            "End_Position",
            "Required_Label"
        ]
    )


# --------------------------------------------------
# DOMAIN TERM ANALYSIS
# --------------------------------------------------

def analyze_domain_terms(documents):

    rows = []

    for doc_id, text in documents.items():

        text_lower = text.lower()

        for term in sorted(DOMAIN_TERMS):

            count = text_lower.count(term.lower())

            if count > 0:

                rows.append({
                    "Document_ID": doc_id,
                    "Domain_Term": term,
                    "Frequency": count,
                    "Detected_By_Spacy_NER": False
                })

    return pd.DataFrame(rows)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BASELINE PIPELINE")
    print("MODULE 5: NAMED ENTITY RECOGNITION")
    print("=" * 60)

    documents = load_documents()

    entities_df = extract_entities(documents)

    domain_df = analyze_domain_terms(documents)

    if entities_df.empty:

        summary = pd.DataFrame(
            columns=["Entity_Label", "Frequency"]
        )

    else:

        summary = (
            entities_df.groupby("Entity_Label")
            .size()
            .reset_index(name="Frequency")
            .sort_values("Frequency", ascending=False)
        )

    print("\nENTITY LABEL DISTRIBUTION")

    print(summary.to_string(index=False))

    print("\nSAMPLE ENTITIES")

    print(entities_df.head(30).to_string(index=False))

    print("\nHEALTHCARE DOMAIN TERMS")

    print(domain_df.head(30).to_string(index=False))

    entities_df.to_csv(
        RESULTS_DIR / "ner_entities.csv",
        index=False
    )

    summary.to_csv(
        RESULTS_DIR / "ner_summary.csv",
        index=False
    )

    domain_df.to_csv(
        RESULTS_DIR / "ner_domain_terms.csv",
        index=False
    )

    print("\nNER results saved successfully!")


if __name__ == "__main__":
    main()
