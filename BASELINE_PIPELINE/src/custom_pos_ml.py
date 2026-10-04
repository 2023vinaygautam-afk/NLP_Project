
def tokenize_sentences(text: str) -> list[list[str]]:
    output = []
    for sentence in nltk.sent_tokenize(text):
        tokens = TOKEN_PATTERN.findall(sentence)
        if tokens:
            output.append(tokens)
    return output


def load_treebank_split():
    file_ids = list(treebank.fileids())
    if len(file_ids) < 2:
        raise ValueError("NLTK Treebank must contain at least two source files.")

    random.Random(SEED).shuffle(file_ids)
    split_index = min(max(int(0.9 * len(file_ids)), 1), len(file_ids) - 1)
    train_file_ids = file_ids[:split_index]
    test_file_ids = file_ids[split_index:]
    train_sentences = list(treebank.tagged_sents(fileids=train_file_ids))
    test_sentences = list(treebank.tagged_sents(fileids=test_file_ids))
    if not train_sentences or not test_sentences:
        raise ValueError("Treebank split produced an empty train or test set.")
    return train_sentences, test_sentences, train_file_ids, test_file_ids


def main():
    docs, input_source = load_documents()

    ensure_nltk_resource("corpora/treebank", "treebank")
    ensure_nltk_resource("tokenizers/punkt_tab/english", "punkt_tab")
    ensure_nltk_resource(
        "taggers/averaged_perceptron_tagger_eng",
        "averaged_perceptron_tagger_eng",
    )

    train, test, train_file_ids, test_file_ids = load_treebank_split()
    hmm = HiddenMarkovModelTrainer().train_supervised(
        train,
        estimator=lambda fd, bins: LidstoneProbDist(
            fd, SMOOTHING_GAMMA, bins
        ),
    )

    total = correct = 0
    for gold_sentence in test:
        words = [word for word, _tag in gold_sentence]
        gold_tags = [tag for _word, tag in gold_sentence]
        predicted_tags = [tag for _word, tag in hmm.tag(words)]
        if len(gold_tags) != len(predicted_tags):
            raise RuntimeError("HMM output length differs from gold sequence.")
        for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
            total += 1
            correct += int(gold_tag == predicted_tag)

    if total == 0:
        raise ValueError("Treebank test split contains no tokens.")
    accuracy = correct / total

    rows = []
    for doc_id, text in docs.items():
        for sentence_no, tokens in enumerate(tokenize_sentences(text), start=1):
            default_tags = nltk.pos_tag(tokens)
            hmm_tags = hmm.tag(tokens)
            if len(tokens) != len(default_tags) or len(tokens) != len(hmm_tags):
                raise RuntimeError(
                    f"Tagger output length mismatch for {doc_id}, "
                    f"sentence {sentence_no}."
                )
            for token, (_, default_pos), (_, hmm_pos) in zip(
                tokens, default_tags, hmm_tags
            ):
                rows.append({
                    "document_id": doc_id,
                    "sentence_number": sentence_no,
                    "token": token,
                    "nltk_default_pos": default_pos,
                    "hmm_ml_pos": hmm_pos,
                    "taggers_agree": default_pos == hmm_pos,
                })

    if not rows:
        raise ValueError("No tokens were produced from the project documents.")

    OUT.mkdir(parents=True, exist_ok=True)
    csv_file = OUT / "custom_pos_ml_comparison.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "model": "Supervised Hidden Markov Model (HMM)",
        "training_data": "NLTK Penn Treebank",
        "input_source": input_source,
        "random_seed": SEED,
        "split_strategy": "Randomized file-level 90/10 split",
        "smoothing_gamma": SMOOTHING_GAMMA,
        "training_sentences": len(train),
        "test_sentences": len(test),
        "training_files": len(train_file_ids),
        "test_files": len(test_file_ids),
        "test_tokens": total,
        "correct_tokens": correct,
        "heldout_treebank_accuracy": accuracy,
        "domain_documents_tagged": len(docs),
        "domain_tokens_tagged": len(rows),
        "comparison_csv": str(csv_file),
        "interpretation": (
            "Held-out Treebank accuracy is not domain-specific accuracy. "
            "Agreement between NLTK and HMM on project documents is not "
            "gold-label accuracy. Manually annotate a domain sample to "
            "measure true domain POS accuracy."
        ),
    }

    metrics_file = OUT / "custom_pos_ml_metrics.json"
    metrics_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Input source: {input_source}")
    print(f"Documents tagged: {len(docs)}")
    print(f"Held-out Treebank token accuracy: {accuracy:.4f}")
    print(
        f"Treebank split: {len(train_file_ids)} training files, "
        f"{len(test_file_ids)} held-out files"
    )
