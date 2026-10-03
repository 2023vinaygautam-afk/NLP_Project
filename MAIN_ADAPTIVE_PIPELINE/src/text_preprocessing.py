
from pathlib import Path
import json
import re


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULT_DIR / "processed_corpus.json"

OUTPUT_FILE = RESULT_DIR / "cleaned_corpus.json"


# ==================================================
# 2. TEXT CLEANING
# ==================================================

def clean_text(text):

    original_text = text

    # Convert HTML entities and remove HTML tags
    text = re.sub(r"&amp;", "and", text)
    text = re.sub(r"&lt;", " less than ", text)
    text = re.sub(r"&gt;", " greater than ", text)

    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Normalize Unicode dash characters
    text = re.sub(r"[–—−]", "-", text)

    # Lowercase text
    text = text.lower()

    # Preserve letters, digits, spaces, and hyphens
    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text
    )

    # Remove standalone hyphens
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return {
        "cleaned_text": text,
        "original_character_count": len(original_text),
        "cleaned_character_count": len(text)
    }


# ==================================================
# 3. PROCESS CORPUS
# ==================================================

def preprocess_documents(documents):

    cleaned_documents = []

    for document in documents:

        result = clean_text(document["text"])

        cleaned_document = {
            "document_id": document["document_id"],
            "file_name": document["file_name"],
            "document_name": document["document_name"],
            "original_text": document["text"],
            "cleaned_text": result["cleaned_text"],
            "original_character_count":
                result["original_character_count"],
            "cleaned_character_count":
                result["cleaned_character_count"]
        }

        cleaned_documents.append(cleaned_document)

    return cleaned_documents


# ==================================================
# 4. SAVE CLEANED CORPUS
# ==================================================

def save_cleaned_corpus(documents):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==================================================
# 5. MAIN EXECUTION
# ==================================================

def main():

    print("=" * 65)
    print("MAIN ADAPTIVE PIPELINE")
    print("MODULE 2: TEXT CLEANING AND PREPROCESSING")
    print("=" * 65)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "processed_corpus.json not found. "
            "Run document_loader.py first."
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    cleaned_documents = preprocess_documents(documents)

    save_cleaned_corpus(cleaned_documents)

    print(f"\nDocuments processed: {len(cleaned_documents)}")

    print("\nCLEANING COMPARISON")
    print("-" * 65)

    for document in cleaned_documents[:3]:

        print(f"\nDocument ID: {document['document_id']}")

        print(
            "Original characters:",
            document["original_character_count"]
        )

        print(
            "Cleaned characters:",
            document["cleaned_character_count"]
        )

        print(
            "Cleaned preview:",
            document["cleaned_text"][:200]
        )

    print("\nOUTPUT")
    print(f"Cleaned corpus saved at: {OUTPUT_FILE}")

    print("\nMODULE 2 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
