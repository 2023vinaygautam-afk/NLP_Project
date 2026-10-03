
import json
import csv
import re
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "cleaned_corpus.json"

OUTPUT_JSON = ROOT / "results" / "bpe_comparison.json"

OUTPUT_CSV = ROOT / "results" / "bpe_token_comparison.csv"

MODEL_FILE = ROOT / "results" / "healthcare_bpe_tokenizer.json"

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. TOKENIZATION
# ==================================================

def word_tokenize(text):

    pattern = (
        r"\b[A-Za-z]+(?:[-'][A-Za-z0-9]+)*\b"
        r"|\d+(?:\.\d+)?%?"
    )

    return re.findall(pattern, text.lower())


# ==================================================
# 3. CORPUS NORMALIZATION
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
# 4. TRAIN BPE TOKENIZER
# ==================================================

def train_bpe(corpus):

    tokenizer = Tokenizer(
        BPE(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(

        vocab_size=500,

        min_frequency=2,

        special_tokens=[
            "[UNK]",
            "[PAD]",
            "[CLS]",
            "[SEP]",
            "[MASK]"
        ]

    )

    training_texts = [
        document["text"]
        for document in corpus
    ]

    tokenizer.train_from_iterator(
        training_texts,
        trainer=trainer
    )

    tokenizer.save(str(MODEL_FILE))

    return tokenizer


# ==================================================
# 5. ANALYZE BPE
# ==================================================

def analyze_bpe(tokenizer, text):

    encoding = tokenizer.encode(text)

    return {

        "tokens": encoding.tokens,

        "token_ids": encoding.ids,

        "token_count": len(encoding.tokens)

    }


# ==================================================
# 6. MAIN EXECUTION
# ==================================================

def main():

    print("\nBPE TOKENIZATION")
    print("================")

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

    print("Training BPE tokenizer...")

    tokenizer = train_bpe(corpus)

    document_results = []

    csv_rows = []

    total_word_tokens = 0

    total_bpe_tokens = 0

    for document in corpus:

        doc_id = document["doc_id"]

        name = document["document_name"]

        text = document["text"]

        word_tokens = word_tokenize(text)

        bpe_result = analyze_bpe(
            tokenizer,
            text
        )

        word_count = len(word_tokens)

        bpe_count = bpe_result["token_count"]

        total_word_tokens += word_count

        total_bpe_tokens += bpe_count

        document_results.append({

            "doc_id": doc_id,

            "document_name": name,

            "word_token_count": word_count,

            "bpe_token_count": bpe_count,

            "word_tokens": word_tokens,

            "bpe_tokens": bpe_result["tokens"],

            "bpe_token_ids": bpe_result["token_ids"]

        })

        csv_rows.append({

            "doc_id": doc_id,

            "document_name": name,

            "word_token_count": word_count,

            "bpe_token_count": bpe_count,

            "word_tokens": " | ".join(word_tokens),

            "bpe_tokens": " | ".join(
                bpe_result["tokens"]
            )

        })

        print(
            f"Processed: {doc_id} | "
            f"Word tokens={word_count} | "
            f"BPE tokens={bpe_count}"
        )

    # ==================================================
    # 7. SUMMARY
    # ==================================================

    vocabulary_size = tokenizer.get_vocab_size()

    summary = {

        "total_documents": len(corpus),

        "bpe_vocabulary_size": vocabulary_size,

        "total_word_tokens": total_word_tokens,

        "total_bpe_tokens": total_bpe_tokens,

        "token_count_difference": (
            total_bpe_tokens - total_word_tokens
        ),

        "documents": document_results

    }

    # ==================================================
    # 8. SAVE JSON
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
    # 9. SAVE CSV
    # ==================================================

    fieldnames = [

        "doc_id",

        "document_name",

        "word_token_count",

        "bpe_token_count",

        "word_tokens",

        "bpe_tokens"

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
    # 10. FINAL OUTPUT
    # ==================================================

    print("\nBPE TOKENIZATION COMPLETED")
    print("==========================")

    print("Documents:", len(corpus))

    print("BPE vocabulary size:", vocabulary_size)

    print("Total word tokens:", total_word_tokens)

    print("Total BPE tokens:", total_bpe_tokens)

    print("Difference:", total_bpe_tokens - total_word_tokens)

    print("\nGenerated files:")

    print(OUTPUT_JSON)

    print(OUTPUT_CSV)

    print(MODEL_FILE)


if __name__ == "__main__":
    main()
