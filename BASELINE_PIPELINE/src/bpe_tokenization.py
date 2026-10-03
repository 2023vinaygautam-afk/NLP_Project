
from pathlib import Path
from collections import Counter
import re
import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "documents"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_documents():
    documents = {}

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        documents[file_path.stem] = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    return documents


def word_tokens(text):
    return re.findall(
        r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b",
        text.lower()
    )


def main():
    print("\nBASELINE PIPELINE")
    print("MODULE 7: BPE TOKENIZATION")

    documents = load_documents()

    if not documents:
        print("No documents found.")
        return

    corpus = list(documents.values())

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=500,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]"]
    )

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    comparison = []
    subword_counts = Counter()

    for doc_id, text in documents.items():
        normal_tokens = word_tokens(text)
        bpe_output = tokenizer.encode(text)
        bpe_tokens = bpe_output.tokens

        comparison.append({
            "Document_ID": doc_id,
            "Word_Token_Count": len(normal_tokens),
            "BPE_Token_Count": len(bpe_tokens),
            "Word_Vocabulary": len(set(normal_tokens)),
            "BPE_Unique_Tokens": len(set(bpe_tokens))
        })

        subword_counts.update(bpe_tokens)

    comparison_df = pd.DataFrame(comparison)

    summary_df = pd.DataFrame([{
        "Documents": len(documents),
        "BPE_Vocabulary_Size": tokenizer.get_vocab_size(),
        "Total_Word_Tokens": comparison_df["Word_Token_Count"].sum(),
        "Total_BPE_Tokens": comparison_df["BPE_Token_Count"].sum(),
        "BPE_Unique_Tokens_Observed": len(subword_counts)
    }])

    subword_df = pd.DataFrame(
        subword_counts.most_common(100),
        columns=["Subword_Token", "Frequency"]
    )

    comparison_df.to_csv(
        RESULTS_DIR / "bpe_document_comparison.csv",
        index=False
    )

    summary_df.to_csv(
        RESULTS_DIR / "bpe_summary.csv",
        index=False
    )

    subword_df.to_csv(
        RESULTS_DIR / "bpe_subword_frequencies.csv",
        index=False
    )

    print("\nBPE SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nTOP 15 BPE TOKENS")
    print(subword_df.head(15).to_string(index=False))

    print("\nBPE tokenization completed successfully.")
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
