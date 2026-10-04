"""
Exercise 12: Custom POS Tagger using Machine Learning.

Algorithm:
    Feature-based LinearSVC with greedy left-to-right decoding.

Training:
    NLTK Penn Treebank, deterministic file-level 80:20 split.
    Original-case plus lowercased training sentences.

Features:
    Prefix/suffix, word shape, punctuation, digits, hyphenation,
    medical morphology, neighboring words, and previous POS tags.

Run:
    python src/custom_pos_ml.py all
    python src/custom_pos_ml.py train
    python src/custom_pos_ml.py make-gold-template
    python src/custom_pos_ml.py evaluate-gold

Domain agreement with NLTK is not domain accuracy.
Domain accuracy is computed only after manually filling gold_pos.
"""

import csv
import json
import pickle
import random
import re
import sys

from collections import Counter
from pathlib import Path

import nltk

from nltk.corpus import treebank
from nltk.tag import pos_tag

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from scipy.sparse import csr_matrix


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

CORPUS_JSON = RESULTS / "cleaned_corpus.json"

CORPUS_TXT_DIR = ROOT / "data" / "documents"

MODEL_PKL = RESULTS / "custom_pos_ml_model.pkl"

METRICS_JSON = RESULTS / "custom_pos_ml_metrics.json"

COMPARISON_CSV = RESULTS / "custom_pos_ml_comparison.csv"

DOMAIN_TERMS_CSV = RESULTS / "custom_pos_ml_domain_terms.csv"

GOLD_CSV = RESULTS / "pos_gold_sample.csv"

GOLD_METRICS_JSON = RESULTS / "custom_pos_domain_accuracy.json"


# ============================================================
# HYPERPARAMETERS
# ============================================================

SEED = 42

TEST_FRACTION = 0.20

RARE_MASK_PROB = 0.50

GOLD_SAMPLE_TOKENS = 200

MIN_GOLD_LABELS = 150

MAX_SENT = 40

WINDOW = 25


# ============================================================
# DOMAIN-SPECIFIC TERMS
# ============================================================

DOMAIN_TERMS = [
    "covid-19",
    "sars-cov-2",
    "antiviral",
    "vaccination",
    "oseltamivir",
    "pandemic",
    "epidemiology",
    "asymptomatic",
    "surveillance",
    "influenza",
]


# ============================================================
# TOKENIZER
# ============================================================

TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?%?"
    r"|[A-Za-z]+(?:[-'][A-Za-z0-9]+)*"
    r"|[^\w\s]"
)


# ============================================================
# MEDICAL MORPHOLOGY
# ============================================================

MED_SUFFIXES = (
    "itis",
    "emia",
    "osis",
    "pathy",
    "ectomy",
    "ology",
    "oma",
    "virus",
    "viral",
    "cidal",
    "mycin",
    "cillin",
    "vir",
    "ine",
    "ion",
    "ity",
    "ment",
    "ness",
    "ance",
    "ence",
    "ism",
    "ist",
    "ic",
    "al",
    "ous",
    "ive",
    "able",
    "ible",
    "ful",
    "less",
    "ing",
    "ed",
    "ly",
    "es",
    "s",
)


# ============================================================
# DOWNLOAD REQUIRED NLTK RESOURCES
# ============================================================

def ensure_nltk():

    resources = (
        ("corpora/treebank", "treebank"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
        (
            "taggers/averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger_eng",
        ),
    )

    for resource, package in resources:

        try:
            nltk.data.find(resource)

        except LookupError:
            nltk.download(package, quiet=True)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_sentences(text):

    """Tokenize words and punctuation."""

    sentences = []

    for sentence in nltk.sent_tokenize(text):

        tokens = TOKEN_PATTERN.findall(sentence)

        if not tokens:
            continue

        if len(tokens) > MAX_SENT:

            sentences.extend(
                tokens[i:i + WINDOW]
                for i in range(0, len(tokens), WINDOW)
            )

        else:
            sentences.append(tokens)

    return sentences


# ============================================================
# WORD SHAPE
# ============================================================

def word_shape(word):

    shape = re.sub(r"[A-Z]", "X", word)

    shape = re.sub(r"[a-z]", "x", shape)

    shape = re.sub(r"\d", "d", shape)

    return re.sub(r"(.)\1{2,}", r"\1\1", shape)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def token_features(
    tokens,
    index,
    previous_tag,
    previous2_tag,
    mask_word=False,
):

    word = tokens[index]

    lower = word.lower()

    previous_word = (
        tokens[index - 1].lower()
        if index > 0
        else "<s>"
    )

    next_word = (
        tokens[index + 1].lower()
        if index + 1 < len(tokens)
        else "</s>"
    )

    features = {

        "bias": 1,

        "suffix1": lower[-1:],

        "suffix2": lower[-2:],

        "suffix3": lower[-3:],

        "suffix4": lower[-4:],

        "prefix1": lower[:1],

        "prefix2": lower[:2],

        "prefix3": lower[:3],

        "shape": word_shape(word),

        "is_title": word.istitle(),

        "is_upper": word.isupper(),

        "has_hyphen": "-" in word,

        "has_digit": any(
            char.isdigit() for char in word
        ),

        "is_punctuation": not any(
            char.isalnum() for char in word
        ),

        "length_bucket": min(len(word), 12),

        "previous_tag": previous_tag,

        "previous2_tag": previous2_tag,

        "previous_tag_pair": (
            previous2_tag + "|" + previous_tag
        ),

        "is_first": index == 0,

        "is_last": index == len(tokens) - 1,

        "previous_word": previous_word,

        "next_word": next_word,

        "previous_suffix2": previous_word[-2:],

        "next_suffix2": next_word[-2:],

    }

    # Medical suffix features

    for suffix in MED_SUFFIXES:

        if (
            lower.endswith(suffix)
            and len(lower) > len(suffix) + 2
        ):

            features["medical_suffix_" + suffix] = True

    # Hyphenated medical terms

    if "-" in lower:

        parts = lower.split("-")

        features["hyphen_first"] = parts[0]

        features["hyphen_last"] = parts[-1]

    # Word identity

    if not mask_word:

        features["word"] = lower

    return features


# ============================================================
# CUSTOM MACHINE LEARNING POS TAGGER
# ============================================================

class MLPosTagger:

    def __init__(self):

        self.vectorizer = DictVectorizer()

        self.classifier = LinearSVC(
            C=0.5,
            max_iter=3000,
        )

    # --------------------------------------------------------
    # FIXED SPARSE MATRIX CONVERSION
    # --------------------------------------------------------

    @staticmethod
    def prepare_matrix(matrix):

        """
        Convert sparse matrix to CSR format and use
        32-bit integer indices for compatibility.
        """

        matrix = matrix.tocsr()

        matrix = csr_matrix(
            (
                matrix.data,
                matrix.indices.astype("int32", copy=False),
                matrix.indptr.astype("int32", copy=False),
            ),
            shape=matrix.shape,
        )

        matrix.indices = matrix.indices.astype(
            "int32",
            copy=False,
        )

        matrix.indptr = matrix.indptr.astype(
            "int32",
            copy=False,
        )

        return matrix

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    def fit(self, tagged_sentences, seed=SEED):

        rng = random.Random(seed)

        counts = Counter(
            word.lower()
            for sentence in tagged_sentences
            for word, _tag in sentence
        )

        feature_rows = []

        labels = []

        for sentence in tagged_sentences:

            words = [
                word for word, _tag in sentence
            ]

            previous_tag = "<s>"

            previous2_tag = "<s>"

            for index, (word, gold_tag) in enumerate(sentence):

                mask = (
                    counts[word.lower()] <= 1
                    and rng.random() < RARE_MASK_PROB
                )

                feature_rows.append(
                    token_features(
                        words,
                        index,
                        previous_tag,
                        previous2_tag,
                        mask_word=mask,
                    )
                )

                labels.append(gold_tag)

                # Teacher-forced tag history

                previous2_tag, previous_tag = (
                    previous_tag,
                    gold_tag,
                )

        matrix = self.vectorizer.fit_transform(feature_rows)

        matrix = self.prepare_matrix(matrix)

        self.classifier.fit(matrix, labels)

        return self

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def tag(self, tokens):

        predicted = []

        previous_tag = "<s>"

        previous2_tag = "<s>"

        for index in range(len(tokens)):

            feature_row = token_features(
                tokens,
                index,
                previous_tag,
                previous2_tag,
            )

            matrix = self.vectorizer.transform(
                [feature_row]
            )

            matrix = self.prepare_matrix(matrix)

            tag = str(
                self.classifier.predict(matrix)[0]
            )

            predicted.append(
                (tokens[index], tag)
            )

            previous2_tag, previous_tag = (
                previous_tag,
                tag,
            )

        return predicted


# ============================================================
# DATASET SPLIT
# ============================================================

def split_treebank():

    file_ids = list(treebank.fileids())

    random.Random(SEED).shuffle(file_ids)

    split_at = int(
        len(file_ids) * (1 - TEST_FRACTION)
    )

    train_ids = file_ids[:split_at]

    test_ids = file_ids[split_at:]

    train = list(
        treebank.tagged_sents(fileids=train_ids)
    )

    test = list(
        treebank.tagged_sents(fileids=test_ids)
    )

    return (
        train,
        test,
        len(train_ids),
        len(test_ids),
    )


# ============================================================
# LOWERCASE DATA AUGMENTATION
# ============================================================

def lowercase_sentences(sentences):

    return [
        [
            (word.lower(), tag)
            for word, tag in sentence
        ]
        for sentence in sentences
    ]


# ============================================================
# ACCURACY EVALUATION
# ============================================================

def score(tagger_function, sentences):

    correct = 0

    total = 0

    for sentence in sentences:

        words = [
            word for word, _tag in sentence
        ]

        predicted = tagger_function(words)

        if len(predicted) != len(sentence):

            raise ValueError(
                "Tagger returned a different number of tokens."
            )

        correct += sum(
            predicted_tag == gold_tag
            for (
                _word,
                predicted_tag,
            ), (
                _gold_word,
                gold_tag,
            ) in zip(predicted, sentence)
        )

        total += len(sentence)

    return correct, total


# ============================================================
# OUT-OF-VOCABULARY EVALUATION
# ============================================================

def score_oov(
    tagger_function,
    sentences,
    training_vocabulary,
):

    correct = 0

    total = 0

    for sentence in sentences:

        words = [
            word for word, _tag in sentence
        ]

        predicted = tagger_function(words)

        for (
            _predicted_word,
            predicted_tag,
        ), (
            gold_word,
            gold_tag,
        ) in zip(predicted, sentence):

            if gold_word.lower() not in training_vocabulary:

                total += 1

                correct += predicted_tag == gold_tag

    return correct, total


# ============================================================
# SAFE RATIO
# ============================================================

def ratio(correct, total):

    return (
        round(correct / total, 4)
        if total
        else None
    )


# ============================================================
# LOAD PROJECT DOCUMENTS
# ============================================================

def load_project_documents():

    documents = []

    if CORPUS_JSON.exists():

        raw = json.loads(
            CORPUS_JSON.read_text(encoding="utf-8")
        )

        if isinstance(raw, dict):

            raw = raw.get("documents", [])

        for index, document in enumerate(raw, 1):

            text = (
                document.get("original_text")
                or document.get("cleaned_text")
                or document.get("text")
                or ""
            )

            doc_id = (
                document.get("document_id")
                or document.get("doc_id")
                or f"D{index:02d}"
            )

            documents.append(
                (str(doc_id), text)
            )

    elif CORPUS_TXT_DIR.exists():

        for index, path in enumerate(
            sorted(CORPUS_TXT_DIR.glob("*.txt")),
            1,
        ):

            documents.append(
                (
                    f"D{index:02d}",
                    path.read_text(encoding="utf-8"),
                )
            )

    if not documents:

        raise FileNotFoundError(
            "No project documents found: expected "
            "results/cleaned_corpus.json or "
            "data/documents/*.txt."
        )

    return documents


# ============================================================
# TRAIN AND EVALUATE MODEL
# ============================================================

def train_and_evaluate():

    ensure_nltk()

    train, test, train_file_count, test_file_count = (
        split_treebank()
    )

    train_vocabulary = {
        word.lower()
        for sentence in train
        for word, _tag in sentence
    }

    print(
        f"Training sentences: {len(train)} "
        f"| test sentences: {len(test)}"
    )

    tagger = MLPosTagger().fit(
        train + lowercase_sentences(train)
    )

    test_lowercase = lowercase_sentences(test)

    metrics = {

        "algorithm": (
            "Feature-based LinearSVC with "
            "greedy left-to-right decoding"
        ),

        "training_data": (
            "NLTK Penn Treebank, "
            "original and lowercase copies"
        ),

        "random_seed": SEED,

        "split_strategy": (
            "Randomized file-level 80:20"
        ),

        "training_files": train_file_count,

        "test_files": test_file_count,

        "training_sentences": len(train),

        "test_sentences": len(test),

    }

    # Evaluate original and lowercase test sets

    for split_name, sentences in (

        ("heldout_original_case", test),

        ("heldout_lowercase", test_lowercase),

    ):

        ml_correct, total = score(
            tagger.tag,
            sentences,
        )

        nltk_correct, _ = score(
            pos_tag,
            sentences,
        )

        metrics[f"{split_name}_tokens"] = total

        metrics[f"{split_name}_ml_accuracy"] = ratio(
            ml_correct,
            total,
        )

        metrics[f"{split_name}_nltk_accuracy"] = ratio(
            nltk_correct,
            total,
        )

        metrics[
            f"{split_name}_improvement_percentage_points"
        ] = (
            round(
                (ml_correct - nltk_correct) / total * 100,
                2,
            )
            if total
            else None
        )

    # OOV evaluation

    ml_oov_correct, oov_total = score_oov(
        tagger.tag,
        test_lowercase,
        train_vocabulary,
    )

    nltk_oov_correct, _ = score_oov(
        pos_tag,
        test_lowercase,
        train_vocabulary,
    )

    metrics["unseen_word_tokens_lowercase"] = oov_total

    metrics["unseen_word_ml_accuracy"] = ratio(
        ml_oov_correct,
        oov_total,
    )

    metrics["unseen_word_nltk_accuracy"] = ratio(
        nltk_oov_correct,
        oov_total,
    )

    # Project document tagging

    documents = load_project_documents()

    comparison_rows = []

    agreement_count = 0

    for document_id, text in documents:

        for sentence_number, tokens in enumerate(
            tokenize_sentences(text),
            1,
        ):

            nltk_tags = pos_tag(tokens)

            ml_tags = tagger.tag(tokens)

            if (
                len(nltk_tags) != len(tokens)
                or len(ml_tags) != len(tokens)
            ):

                raise ValueError(
                    f"Token/tag length mismatch in "
                    f"{document_id}, "
                    f"sentence {sentence_number}"
                )

            for token, nltk_pair, ml_pair in zip(
                tokens,
                nltk_tags,
                ml_tags,
            ):

                nltk_tag = nltk_pair[1]

                ml_tag = ml_pair[1]

                agrees = nltk_tag == ml_tag

                agreement_count += int(agrees)

                comparison_rows.append({

                    "document_id": document_id,

                    "sentence_number": sentence_number,

                    "token": token,

                    "nltk_default_pos": nltk_tag,

                    "ml_pos": ml_tag,

                    "taggers_agree": agrees,

                })

    metrics.update({

        "domain_documents_tagged": len(documents),

        "domain_tokens_tagged": len(comparison_rows),

        "domain_nltk_ml_agreement": ratio(
            agreement_count,
            len(comparison_rows),
        ),

        "domain_accuracy": None,

        "domain_accuracy_note": (
            "Not measured yet. Hand-label gold_pos "
            "in pos_gold_sample.csv, then run evaluate-gold. "
            "NLTK/ML agreement is not accuracy."
        ),

    })

    # Ensure output directory exists

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save comparison CSV

    if comparison_rows:

        with COMPARISON_CSV.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(comparison_rows[0]),
            )

            writer.writeheader()

            writer.writerows(comparison_rows)

    # Domain-specific term comparison

    domain_rows = []

    for term in DOMAIN_TERMS:

        matches = [
            row
            for row in comparison_rows
            if row["token"].lower() == term
        ]

        if matches:

            domain_rows.append({

                "term": term,

                "occurrences": len(matches),

                "nltk_tags": dict(
                    Counter(
                        row["nltk_default_pos"]
                        for row in matches
                    )
                ),

                "ml_tags": dict(
                    Counter(
                        row["ml_pos"]
                        for row in matches
                    )
                ),

            })

    with DOMAIN_TERMS_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        fields = [
            "term",
            "occurrences",
            "nltk_tags",
            "ml_tags",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(domain_rows)

    # Save metrics

    METRICS_JSON.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Save trained model

    with MODEL_PKL.open("wb") as handle:

        pickle.dump(tagger, handle)

    print("\nML POS TAGGING EVALUATION")

    print(
        json.dumps(metrics, indent=2)
    )

    print(
        f"\nTagged project tokens: {len(comparison_rows)}"
    )

    print(
        f"Domain-term rows: {len(domain_rows)}"
    )

    print(
        f"Metrics: {METRICS_JSON}"
    )

    print(
        f"Comparison: {COMPARISON_CSV}"
    )

    print(
        f"Domain terms: {DOMAIN_TERMS_CSV}"
    )

    return tagger


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if MODEL_PKL.exists():

        with MODEL_PKL.open("rb") as handle:

            return pickle.load(handle)

    return train_and_evaluate()


# ============================================================
# CREATE GOLD-LABEL TEMPLATE
# ============================================================

def make_gold_template():

    ensure_nltk()

    tagger = load_model()

    rng = random.Random(SEED)

    candidate_sentences = []

    for document_id, text in load_project_documents():

        for sentence_number, tokens in enumerate(
            tokenize_sentences(text),
            1,
        ):

            if 6 <= len(tokens) <= 30:

                candidate_sentences.append(
                    (
                        document_id,
                        sentence_number,
                        tokens,
                    )
                )

    rng.shuffle(candidate_sentences)

    domain_set = set(DOMAIN_TERMS)

    candidate_sentences.sort(
        key=lambda item: not any(
            token.lower() in domain_set
            for token in item[2]
        )
    )

    rows = []

    token_count = 0

    for document_id, sentence_number, tokens in candidate_sentences:

        if token_count >= GOLD_SAMPLE_TOKENS:
            break

        nltk_output = pos_tag(tokens)

        ml_output = tagger.tag(tokens)

        for token, nltk_pair, ml_pair in zip(
            tokens,
            nltk_output,
            ml_output,
        ):

            rows.append({

                "document_id": document_id,

                "sentence_number": sentence_number,

                "token": token,

                "gold_pos": "",

                "nltk_pos": nltk_pair[1],

                "ml_pos": ml_pair[1],

            })

        token_count += len(tokens)

    if not rows:

        raise ValueError(
            "No suitable sentences found for the gold-label template."
        )

    with GOLD_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Created {GOLD_CSV} with {len(rows)} tokens."
    )

    print(
        "Manually fill gold_pos using the appropriate "
        "Penn Treebank POS tags."
    )


# ============================================================
# EVALUATE DOMAIN GOLD LABELS
# ============================================================

def evaluate_gold():

    if not GOLD_CSV.exists():

        raise FileNotFoundError(
            "Run make-gold-template first."
        )

    with GOLD_CSV.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        rows = list(csv.DictReader(handle))

    labelled = [
        row
        for row in rows
        if row.get("gold_pos", "").strip()
    ]

    if len(labelled) < MIN_GOLD_LABELS:

        raise ValueError(
            f"Only {len(labelled)} labelled tokens. "
            f"Label at least {MIN_GOLD_LABELS} tokens "
            "before evaluating."
        )

    output = {

        "labelled_tokens": len(labelled),

        "accuracy": {},

        "error_examples": {},

    }

    for column in ("nltk_pos", "ml_pos"):

        correct = sum(
            row[column].strip() == row["gold_pos"].strip()
            for row in labelled
        )

        output["accuracy"][column] = ratio(
            correct,
            len(labelled),
        )

        output["error_examples"][column] = [

            {

                "token": row["token"],

                "gold": row["gold_pos"],

                "predicted": row[column],

            }

            for row in labelled
            if row[column].strip() != row["gold_pos"].strip()

        ][:10]

    if (
        "ml_pos" in output["accuracy"]
        and "nltk_pos" in output["accuracy"]
    ):

        output[
            "ml_improvement_over_nltk_percentage_points"
        ] = round(

            (
                output["accuracy"]["ml_pos"]
                - output["accuracy"]["nltk_pos"]
            ) * 100,

            2,

        )

    GOLD_METRICS_JSON.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(output, indent=2)
    )

    print(
        f"Gold evaluation saved to: {GOLD_METRICS_JSON}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    command = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "all"
    )

    if command == "train":

        train_and_evaluate()

    elif command == "make-gold-template":

        make_gold_template()

    elif command == "evaluate-gold":

        evaluate_gold()

    elif command == "all":

        train_and_evaluate()

        if not GOLD_CSV.exists():

            make_gold_template()

        else:

            try:

                evaluate_gold()

            except ValueError as error:

                print(
                    f"Gold evaluation pending: {error}"
                )

    else:

        print(__doc__)


if __name__ == "__main__":

    main()



