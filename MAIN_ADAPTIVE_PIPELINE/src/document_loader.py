
from pathlib import Path
import json
import re


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_DIR = PROJECT_ROOT / "data" / "documents"
RESULT_DIR = PROJECT_ROOT / "results"

CORPUS_FILE = RESULT_DIR / "processed_corpus.json"
STATISTICS_FILE = RESULT_DIR / "document_statistics.json"


# ==================================================
# 2. TEXT STATISTICS
# ==================================================

def calculate_document_statistics(text):

    words = re.findall(r"\b\w+\b", text)

    sentences = re.split(r"[.!?]+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences)
    }


# ==================================================
# 3. LOAD DOCUMENTS
# ==================================================

def load_documents():

    if not DOCUMENT_DIR.exists():
        raise FileNotFoundError(
            f"Document directory not found: {DOCUMENT_DIR}"
        )

    files = sorted(
        DOCUMENT_DIR.glob("*.txt")
    )

    if not files:
        raise FileNotFoundError(
            "No TXT documents found."
        )

    documents = []
    empty_documents = []

    for index, file_path in enumerate(files, start=1):

        document_id = f"D{index:02d}"

        text = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

        if not text:
            empty_documents.append(file_path.name)
            print(f"WARNING: Empty document: {file_path.name}")
            continue

        stats = calculate_document_statistics(text)

        document = {
            "document_id": document_id,
            "file_name": file_path.name,
            "document_name": file_path.stem,
            "text": text,
            **stats
        }

        documents.append(document)

    if not documents:
        raise ValueError("No valid documents were loaded.")

    return documents, empty_documents


# ==================================================
# 4. CORPUS STATISTICS
# ==================================================

def calculate_corpus_statistics(documents):

    all_text = " ".join(
        doc["text"] for doc in documents
    )

    all_words = re.findall(
        r"\b\w+\b",
        all_text.lower()
    )

    vocabulary = set(all_words)

    total_documents = len(documents)

    total_words = len(all_words)

    total_sentences = sum(
        doc["sentence_count"] for doc in documents
    )

    total_characters = sum(
        doc["character_count"] for doc in documents
    )

    statistics = {
        "total_documents": total_documents,
        "total_sentences": total_sentences,
        "total_tokens": total_words,
        "total_characters": total_characters,
        "vocabulary_size": len(vocabulary),
        "average_document_length": round(
            total_words / total_documents, 2
        )
    }

    return statistics


# ==================================================
# 5. SAVE OUTPUT
# ==================================================

def save_output(documents, statistics, empty_documents):

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CORPUS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    output = {
        "corpus_statistics": statistics,
        "empty_documents": empty_documents
    }

    with open(
        STATISTICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==================================================
# 6. MAIN
# ==================================================

def main():

    print("=" * 65)
    print("MAIN ADAPTIVE PIPELINE")
    print("MODULE 1: DOCUMENT LOADING")
    print("=" * 65)

    documents, empty_documents = load_documents()

    statistics = calculate_corpus_statistics(documents)

    save_output(
        documents,
        statistics,
        empty_documents
    )

    print("\nDOCUMENT DETAILS")
    print("-" * 65)

    for doc in documents:

        print(
            f"{doc['document_id']} | "
            f"{doc['file_name']} | "
            f"Words: {doc['word_count']} | "
            f"Sentences: {doc['sentence_count']}"
        )

    print("\nCORPUS STATISTICS")
    print("-" * 65)

    for key, value in statistics.items():

        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\nVALIDATION")
    print("-" * 65)

    print(f"Valid documents: {len(documents)}")
    print(f"Empty documents: {len(empty_documents)}")

    if len(documents) != 20:
        print("WARNING: Expected 20 valid documents.")

    print("\nOUTPUT FILES")
    print(f"Corpus: {CORPUS_FILE}")
    print(f"Statistics: {STATISTICS_FILE}")

    print("\nMODULE 1 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
