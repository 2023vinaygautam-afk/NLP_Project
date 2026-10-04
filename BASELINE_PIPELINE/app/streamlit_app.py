import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "documents"
RESULTS_DIR = BASE_DIR / "results"
MAIN_RESULTS_DIR = BASE_DIR.parent / "MAIN_ADAPTIVE_PIPELINE" / "results"
QUERY_FILE = DATA_DIR / "Query" / "queries_400.csv"

st.set_page_config(
    page_title="Baseline NLP Retrieval Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 32px;
            font-weight: 700;
            color: #2563eb;
        }
        .subtitle {
            color: #64748b;
            font-size: 15px;
        }
        .section-title {
            font-size: 21px;
            font-weight: 600;
            margin-top: 15px;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(128,128,128,0.08);
            padding: 15px;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_documents():
    docs = {}
    for file in sorted(DATA_DIR.glob("*.txt")):
        docs[file.stem] = file.read_text(encoding="utf-8", errors="ignore")
    return docs


def tokenize(text):
    return re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", str(text).lower())


@st.cache_data
def build_index(docs):
    index = {}
    for doc_id, text in docs.items():
        for token in set(tokenize(text)):
            index.setdefault(token, set()).add(doc_id)
    return index


def retrieve(query, method, docs, index):
    terms = tokenize(query)
    if not terms:
        return []

    if method == "KEYWORD":
        result = set()
        for term in terms:
            result.update(index.get(term, set()))
    elif method == "PHRASE":
        phrase = " ".join(terms)
        result = {
            doc_id
            for doc_id, text in docs.items()
            if phrase in " ".join(tokenize(text))
        }
    elif method == "AND":
        result = set(index.get(terms[0], set()))
        for term in terms[1:]:
            result.intersection_update(index.get(term, set()))
    elif method == "OR":
        result = set()
        for term in terms:
            result.update(index.get(term, set()))
    elif method == "NOT":
        result = set(docs.keys())
        for term in terms:
            result.difference_update(index.get(term, set()))
    else:
        result = set()

    return sorted(result)


def read_csv(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json(filename, results_dir=RESULTS_DIR):
    path = results_dir / filename
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def title_of(text, default):
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return default
    sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0]
    if len(sentence) > 120:
        return sentence[:117].rstrip() + "..."
    return sentence


documents = load_documents()
inverted_index = build_index(documents)

st.sidebar.title("NLP Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Document Explorer",
        "Information Retrieval",
        "Evaluation Results",
        "Query Dataset",
        "NLP Analysis Artifacts",
        "Pipeline and Methodology",
    ],
)

st.sidebar.divider()
st.sidebar.write("**Project Domain**")
st.sidebar.write("Healthcare and Public Health")
st.sidebar.write("**Corpus**")
st.sidebar.write(f"{len(documents)} documents")

st.markdown(
    '<div class="main-title">Baseline NLP Information Retrieval System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Healthcare | Infectious Diseases | COVID-19 | Vaccination | Epidemiology</div>',
    unsafe_allow_html=True,
)
st.divider()

all_text = " ".join(documents.values())
all_tokens = tokenize(all_text)
vocabulary = set(all_tokens)
total_sentences = sum(len(re.findall(r"[.!?]+", text)) for text in documents.values())
total_characters = len(all_text)
average_length = len(all_tokens) / len(documents) if documents else 0

if page == "Dashboard":
    st.header("Corpus Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Documents", len(documents))
    c2.metric("Total Tokens", f"{len(all_tokens):,}")
    c3.metric("Vocabulary", f"{len(vocabulary):,}")
    c4.metric("Average Length", f"{average_length:.2f}")
    c5.metric("Sentences", total_sentences)
    c6.metric("Characters", f"{total_characters:,}")

    st.divider()
    st.subheader("Document Distribution")
    doc_stats = []
    for doc_id, text in documents.items():
        tokens = tokenize(text)
        doc_stats.append(
            {
                "Document": doc_id,
                "Tokens": len(tokens),
                "Unique Terms": len(set(tokens)),
                "Characters": len(text),
            }
        )
    stats_df = pd.DataFrame(doc_stats)
    st.bar_chart(stats_df.set_index("Document")["Tokens"], horizontal=True)
    st.subheader("Document Statistics")
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

elif page == "Document Explorer":
    st.header("Healthcare Document Explorer")
    if documents:
        selected_doc = st.selectbox("Select Document", list(documents.keys()))
        text = documents[selected_doc]
        tokens = tokenize(text)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens", len(tokens))
        c2.metric("Vocabulary", len(set(tokens)))
        c3.metric("Characters", len(text))
        st.subheader("Document Content")
        st.text_area("Text", text, height=350)
        st.subheader("Most Frequent Terms")
        freq_df = pd.DataFrame(Counter(tokens).most_common(15), columns=["Term", "Frequency"])
        st.bar_chart(freq_df.set_index("Term"))
    else:
        st.warning("No documents found.")

elif page == "Information Retrieval":
    st.header("Information Retrieval")
    st.write("Search the healthcare corpus using the assignment's retrieval methods.")
    query = st.text_input("Enter your query", placeholder="Example: COVID vaccination")
    method = st.selectbox("Retrieval Method", ["KEYWORD", "PHRASE", "AND", "OR", "NOT"])

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            results = retrieve(query, method, documents, inverted_index)
            st.success(f"Retrieved {len(results)} documents using {method}.")
            if results:
                for rank, doc_id in enumerate(results, 1):
                    with st.expander(f"Result {rank}: {doc_id}"):
                        st.write(documents[doc_id])
                        st.download_button(
                            "Download Document",
                            data=documents[doc_id],
                            file_name=f"{doc_id}.txt",
                            mime="text/plain",
                            key=f"download_{rank}_{doc_id}",
                        )
            else:
                st.info("No matching documents found.")

elif page == "Evaluation Results":
    st.header("Five-Query Baseline vs Adaptive Evaluation")
    st.caption(
        "Official protocol: the same five queries, multi-document relevance "
        "labels, and macro set-based Precision, Recall, and F1 over all retrieved "
        "documents."
    )
    summary = read_json("same_query_evaluation_summary.json", MAIN_RESULTS_DIR)
    comparison_path = MAIN_RESULTS_DIR / "same_query_comparison.csv"
    if not summary or not comparison_path.exists():
        st.info(
            "The shared comparison report is not available. Regenerate it from "
            "MAIN_ADAPTIVE_PIPELINE with experiments\\run_baseline.py, "
            "src\\adaptive_pipeline.py, and experiments\\baseline_vs_adaptive.py."
        )
    else:
        comparison = pd.read_csv(comparison_path)
        summary_rows = [
            {
                "Pipeline": pipeline,
                "Precision": summary[pipeline]["Precision"],
                "Recall": summary[pipeline]["Recall"],
                "F1": summary[pipeline]["F1_Score"],
            }
            for pipeline in ("Baseline", "Adaptive")
        ]
        st.dataframe(
            pd.DataFrame(summary_rows).round(3),
            hide_index=True,
            use_container_width=True,
        )
        columns = [
            "Query_ID",
            "Query",
            "Expected_Relevant_Documents_Baseline",
            "Relevant_Count",
            "Precision_Baseline",
            "Recall_Baseline",
            "F1_Score_Baseline",
            "Precision_Adaptive",
            "Recall_Adaptive",
            "F1_Score_Adaptive",
        ]
        st.subheader("Per-query results")
        st.dataframe(
            comparison[columns].round(3),
            hide_index=True,
            use_container_width=True,
        )

elif page == "Query Dataset":
    st.header("Evaluation Query Dataset")
    if QUERY_FILE.exists():
        query_df = pd.read_csv(QUERY_FILE)
        c1, c2 = st.columns(2)
        c1.metric("Total Queries", len(query_df))
        if "topic" in query_df.columns:
            c2.metric("Unique Topics", query_df["topic"].nunique())
        st.dataframe(query_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Query Dataset",
            data=query_df.to_csv(index=False),
            file_name="queries_400.csv",
            mime="text/csv",
        )
    else:
        st.warning("Query dataset not found.")

elif page == "NLP Analysis Artifacts":
    st.header("Existing NLP Analysis Outputs")
    artifact_groups = {
        "Corpus statistics": ["corpus_summary.csv", "document_statistics.csv"],
        "Preprocessing": ["preprocessing_summary.csv", "preprocessing_comparison.csv"],
        "Tokenization": ["tokenization_summary.csv", "tokenization_comparison.csv"],
        "N-grams": ["ngram_summary.csv", "ngram_document_statistics.csv", "ngram_frequencies.csv"],
        "POS tagging": ["pos_summary.csv", "pos_tagging_comparison.csv"],
        "Named entity recognition": ["ner_summary.csv", "ner_entities.csv", "ner_domain_terms.csv"],
        "BPE": ["bpe_summary.csv", "bpe_document_comparison.csv", "bpe_subword_frequencies.csv"],
    }
    for group, filenames in artifact_groups.items():
        with st.expander(group):
            for filename in filenames:
                frame = read_csv(filename)
                st.markdown(f"**{filename}**")
                if frame.empty:
                    st.caption("Not found or empty in results/")
                else:
                    st.dataframe(frame, hide_index=True, use_container_width=True)
                    st.download_button(
                        f"Download {filename}",
                        frame.to_csv(index=False),
                        filename,
                        "text/csv",
                        key=f"artifact_{filename}",
                    )

elif page == "Pipeline and Methodology":
    st.header("Baseline Processing Pipeline")
    pipeline_steps = [
        "Healthcare Document Collection",
        "Text Extraction and Cleaning",
        "Tokenization",
        "Stopword Analysis",
        "Stemming and Lemmatization",
        "POS Tagging",
        "Named Entity Recognition",
        "N-gram Analysis",
        "BPE Tokenization",
        "Inverted Index Construction",
        "Boolean and Keyword Retrieval",
        "Evaluation",
    ]
    for i, step in enumerate(pipeline_steps, 1):
        st.markdown(f"**{i}.** {step}")

    st.divider()
    st.subheader("Retrieval Methods")
    st.markdown(
        """
        - **Keyword:** Retrieves documents containing at least one query term.
        - **Phrase:** Retrieves documents containing the exact normalized phrase.
        - **AND:** Retrieves documents containing every query term.
        - **OR:** Retrieves documents containing at least one query term.
        - **NOT:** Retrieves documents that do not contain any query term.
        """
    )

    st.info(
        "This project implements a static baseline healthcare retrieval system. The Q01–Q05 evaluation and corpus artifacts are the authoritative project outputs."
    )

    st.subheader("Official evaluation")
    st.write(
        "The official result is the shared five-query baseline/adaptive "
        "comparison, using multi-document relevance labels and macro "
        "set-based Precision, Recall, and F1. Other saved evaluation artifacts "
        "remain in the results folder but are not presented as project metrics."
    )

st.divider()
st.caption("Healthcare NLP | Baseline Pipeline | Fixed evaluation: Q01–Q05")
