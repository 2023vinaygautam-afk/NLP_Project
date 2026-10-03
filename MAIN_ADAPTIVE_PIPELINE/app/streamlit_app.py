from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# Healthcare NLP — Main Adaptive Pipeline
# Retains the baseline dashboard's modules and corpus-analysis features,
# while routing interactive and formal retrieval through the adaptive pipeline.
# Formal evaluation uses only the fixed five queries Q01–Q05.

st.set_page_config(
    page_title="Healthcare NLP — Main Adaptive Pipeline",
    page_icon="📚",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
DOC_DIR = DATA_DIR / "documents"
RESULTS_DIR = PROJECT_DIR / "results"
BASELINE_RESULTS_DIR = PROJECT_DIR.parent / "BASELINE_PIPELINE" / "results"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from adaptive_pipeline import (  # noqa: E402
    FIXED_QUERIES,
    load_resources,
    run_adaptive_pipeline,
)
from feature_extraction import (  # noqa: E402
    build_terms_dictionary,
    document_token_counts,
    remove_stop_words,
    stem_documents,
    stem_mapping,
    tokenize_documents,
)

STOPWORDS = set("""
a an and are as at be been being by can could did do does for from had has have
how in into is it its may might of on or should that the their there these this
those to was were what when where which who why will with would
""".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def doc_id_from_path(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return f"D{int(match.group(1)):02d}" if match else path.stem


def load_docs() -> dict[str, str]:
    docs = {}
    if DOC_DIR.exists():
        for path in sorted(DOC_DIR.glob("*.txt")):
            docs[doc_id_from_path(path)] = path.read_text(
                encoding="utf-8", errors="ignore"
            )
    return docs


def title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:160]
    return fallback


def resolve_result_path(name: str, aliases: tuple[str, ...] = ()) -> Path | None:
    for results_dir in (RESULTS_DIR, BASELINE_RESULTS_DIR):
        for filename in (name, *aliases):
            candidate = results_dir / filename
            if candidate.is_file():
                return candidate
    return None


@st.cache_data(ttl=30, max_entries=32)
def read_csv_path(path_string: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path_string)


def read_csv(name: str, aliases: tuple[str, ...] = ()) -> pd.DataFrame:
    path = resolve_result_path(name, aliases)
    if path is None:
        return pd.DataFrame()
    try:
        return read_csv_path(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def read_json(name: str):
    path = resolve_result_path(name)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {name}: {exc}")
        return None


def show_artifact(label: str, filename: str):
    aliases = {
        "pos_summary.csv": ("pos_tagging_summary.csv",),
    }.get(filename, ())
    path = resolve_result_path(filename, aliases)
    st.subheader(label)
    if path is None:
        st.info(
            f"{filename} is not available in the adaptive or matching-corpus "
            "baseline results."
        )
        return
    frame = read_csv(filename, aliases)
    if frame.empty:
        st.warning(f"{path} is present but contains no readable rows.")
        return

    frame = normalize_document_ids(frame)
    source = (
        "adaptive results"
        if path.parent.resolve() == RESULTS_DIR.resolve()
        else "baseline analysis snapshot (same 20 source documents)"
    )
    st.caption(f"Source: {path.parent.name}/{path.name} · {source} · {len(frame):,} rows")
    with st.expander(f"Filter and sort {label}", expanded=False):
        search = st.text_input(
            f"Search {label.lower()}",
            key=f"artifact_search_{filename}",
            placeholder="Search across all columns",
        )
        selected_column = st.selectbox(
            "Sort by",
            ["No sorting", *map(str, frame.columns)],
            key=f"artifact_sort_{filename}",
        )
        descending = st.checkbox(
            "Descending",
            value=selected_column != "No sorting",
            key=f"artifact_descending_{filename}",
        )
        if search.strip():
            search_mask = frame.astype(str).apply(
                lambda column: column.str.contains(
                    search.strip(), case=False, regex=False, na=False
                )
            ).any(axis=1)
            frame = frame.loc[search_mask]
        if selected_column != "No sorting":
            frame = frame.sort_values(
                selected_column,
                ascending=not descending,
                kind="stable",
                na_position="last",
            )

    if frame.empty:
        st.info("No rows match the current search.")
        return

    st.caption(f"{len(frame):,} matching rows")
    preview_rows = min(len(frame), 500)
    st.dataframe(
        frame.head(preview_rows),
        hide_index=True,
        width="stretch",
        height=440,
    )
    if len(frame) > preview_rows:
        st.caption(
            f"Showing the first {preview_rows:,} rows. Download the CSV to inspect "
            "the complete result."
        )
    st.download_button(
        f"Download full {filename}",
        path.read_bytes(),
        path.name,
        "text/csv",
        key=f"download_full_{filename}",
    )
    if len(frame) != len(read_csv(filename, aliases)):
        st.download_button(
            f"Download filtered {label}",
            frame.to_csv(index=False),
            f"filtered_{filename}",
            "text/csv",
            key=f"download_filtered_{filename}",
        )


def normalize_document_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("Document_ID", "document_id", "Document ID", "doc_id"):
        if column not in result.columns:
            continue

        def normalize(value):
            match = re.match(r"\s*(?:D)?(\d+)", str(value), flags=re.IGNORECASE)
            return f"D{int(match.group(1)):02d}" if match else value

        result[column] = result[column].map(normalize)
    return result


@st.cache_resource
def get_resources():
    return load_resources()


def retrieve_ids(output: dict) -> list[str]:
    retrieval = output.get("retrieval", {})
    rows = retrieval.get("results", []) if isinstance(retrieval, dict) else []
    return [
        str(row.get("document_id"))
        for row in rows
        if row.get("document_id") is not None
    ]


def evaluate_fixed_queries(index, document_mapping, corpus, k: int = 5):
    rows = []
    outputs = []

    for item in FIXED_QUERIES:
        output = run_adaptive_pipeline(
            item["query"], index, document_mapping, corpus
        )
        retrieved_ids = retrieve_ids(output)
        expected_value = item["expected_document"]
        expected_documents = (
            {str(value) for value in expected_value}
            if isinstance(expected_value, (list, tuple, set))
            else {str(expected_value)}
        )
        relevant_count = len(expected_documents)
        matches_at_k = expected_documents.intersection(retrieved_ids[:k])
        matches_all = expected_documents.intersection(retrieved_ids)
        rank = next(
            (
                position
                for position, document_id in enumerate(retrieved_ids, start=1)
                if document_id in expected_documents
            ),
            None,
        )
        precision_at_1 = len(expected_documents.intersection(retrieved_ids[:1]))
        precision_at_k = len(matches_at_k) / k
        recall_at_k = len(matches_at_k) / relevant_count if relevant_count else 0.0
        f1_at_k = (
            2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
            if precision_at_k + recall_at_k
            else 0.0
        )
        reciprocal_rank = (
            1 / rank if rank is not None and rank <= k else 0.0
        )
        full_precision = len(matches_all) / len(retrieved_ids) if retrieved_ids else 0.0
        full_recall = len(matches_all) / relevant_count if relevant_count else 0.0
        full_f1 = (
            2 * full_precision * full_recall / (full_precision + full_recall)
            if full_precision + full_recall
            else 0.0
        )
        decision = output.get("adaptive_decision", {})

        rows.append({
            "Query_ID": item["query_id"],
            "Query": item["query"],
            "Relevant_Document(s)": ", ".join(sorted(expected_documents)),
            "Relevant_Count": relevant_count,
            "Selected_Method": decision.get(
                "method", decision.get("selected_method", "Unknown")
            ),
            "Retrieved_Documents": ", ".join(retrieved_ids),
            "First_Relevant_Rank": rank,
            "Precision": full_precision,
            "Recall": full_recall,
            "F1": full_f1,
            "Precision@1": precision_at_1,
            f"Precision@{k}": precision_at_k,
            f"Recall@{k}": recall_at_k,
            f"F1@{k}": f1_at_k,
            "Relevant_Found@K": len(matches_at_k),
            "Relevant_Found": len(matches_all),
            "Reciprocal_Rank": reciprocal_rank,
            f"Hit@{k}": int(bool(matches_at_k)),
            "Status": output.get("status", "UNKNOWN"),
        })
        outputs.append(output)

    frame = pd.DataFrame(rows)
    frame["First_Relevant_Rank"] = pd.array(
        frame["First_Relevant_Rank"],
        dtype="Int64",
    )
    metric_columns = [
        "Precision", "Recall", "F1", "Precision@1",
        f"Precision@{k}", f"Recall@{k}", f"F1@{k}",
        "Reciprocal_Rank", f"Hit@{k}",
    ]
    metrics = {key: float(frame[key].mean()) for key in metric_columns}
    total_relevant = sum(
        len(
            item["expected_document"]
            if isinstance(item["expected_document"], (list, tuple, set))
            else [item["expected_document"]]
        )
        for item in FIXED_QUERIES
    )
    total_retrieved = sum(len(retrieve_ids(output)) for output in outputs)
    total_relevant_retrieved = int(frame["Relevant_Found@K"].sum())
    total_relevant_found = int(frame["Relevant_Found"].sum())
    metrics.update({
        "Micro_Precision": (
            total_relevant_found / total_retrieved if total_retrieved else 0.0
        ),
        "Micro_Recall": (
            total_relevant_found / total_relevant if total_relevant else 0.0
        ),
        "K": k,
        "Total_Queries": len(frame),
        "Total_Documents": len(document_mapping),
        "Expected_Documents_Retrieved": total_relevant_retrieved,
    })
    metrics["Micro_F1"] = (
        2 * metrics["Micro_Precision"] * metrics["Micro_Recall"]
        / (metrics["Micro_Precision"] + metrics["Micro_Recall"])
        if metrics["Micro_Precision"] + metrics["Micro_Recall"]
        else 0.0
    )
    return frame, metrics, outputs


@st.cache_data(ttl=600, max_entries=4)
def cached_feature_tokens(
    documents: tuple[tuple[str, str], ...],
) -> dict[str, list[str]]:
    return tokenize_documents(documents)


def feature_token_key(
    tokens: dict[str, list[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple((document_id, tuple(values)) for document_id, values in tokens.items())


@st.cache_data(ttl=600, max_entries=4)
def cached_feature_stopwords(
    token_key: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    return remove_stop_words(
        {document_id: list(tokens) for document_id, tokens in token_key}
    )


@st.cache_data(ttl=600, max_entries=4)
def cached_feature_stems(
    token_key: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[str, list[str]]:
    return stem_documents(
        {document_id: list(tokens) for document_id, tokens in token_key}
    )


docs = load_docs()

st.sidebar.title("📚 Healthcare NLP")
page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Corpus Overview",
        "Document Explorer",
        "Preprocessing Analysis",
        "Tokenization Analysis",
        "N-gram Analysis",
        "POS Tagging",
        "Named Entity Recognition",
        "BPE Analysis",
        "Information Retrieval",
        "Evaluation Queries",
        "Evaluation Results",
        "Pipeline and Methodology",
        "Feature Extraction & Terms Dictionary",
    ],
    key="main_navigation",
)
st.sidebar.divider()
st.sidebar.write("**Domain:** Healthcare and Public Health")
st.sidebar.write(f"**Documents loaded:** {len(docs)}")
st.sidebar.write("**Formal evaluation:** Q01–Q05 only")

if not docs:
    st.error(f"No .txt documents found in {DOC_DIR}.")
    st.stop()

feature_documents = tuple(docs.items())
for key in (
    "feature_tokens",
    "feature_extracted_terms",
    "feature_stemmed_tokens",
    "feature_stem_mapping",
    "feature_filtered_tokens",
    "feature_removed_tokens",
    "feature_dictionary",
    "feature_dictionary_source_used",
    "feature_last_action",
    "feature_dictionary_sort",
):
    st.session_state.setdefault(key, None)

feature_source = "Tokenized terms"
if page == "Feature Extraction & Terms Dictionary":
    st.sidebar.divider()
    st.sidebar.subheader("Feature Extraction Panel")
    feature_source = st.sidebar.selectbox(
        "Dictionary input",
        [
            "Tokenized terms",
            "Stop-word filtered",
            "Porter stems",
            "Stop-word filtered + Porter stems",
        ],
        key="feature_dictionary_source",
    )
    if st.sidebar.button("Tokenize Documents"):
        st.session_state["feature_tokens"] = cached_feature_tokens(feature_documents)
        st.session_state["feature_last_action"] = "Tokenize Documents"
    if st.sidebar.button("Extract All Terms"):
        tokenized = cached_feature_tokens(feature_documents)
        st.session_state["feature_tokens"] = tokenized
        st.session_state["feature_extracted_terms"] = build_terms_dictionary(tokenized)
        st.session_state["feature_last_action"] = "Extract All Terms"
    if st.sidebar.button("Apply Stemming"):
        tokenized = cached_feature_tokens(feature_documents)
        st.session_state["feature_tokens"] = tokenized
        stems = cached_feature_stems(feature_token_key(tokenized))
        st.session_state["feature_stemmed_tokens"] = stems
        st.session_state["feature_stem_mapping"] = stem_mapping(
            token for tokens in tokenized.values() for token in tokens
        )
        st.session_state["feature_last_action"] = "Apply Stemming"
    if st.sidebar.button("Remove Stop Words"):
        tokenized = cached_feature_tokens(feature_documents)
        st.session_state["feature_tokens"] = tokenized
        try:
            retained, removed = cached_feature_stopwords(feature_token_key(tokenized))
        except LookupError as exc:
            st.sidebar.error(str(exc))
        else:
            st.session_state["feature_filtered_tokens"] = retained
            st.session_state["feature_removed_tokens"] = removed
            st.session_state["feature_last_action"] = "Remove Stop Words"
    if st.sidebar.button("Create Dictionary"):
        tokenized = cached_feature_tokens(feature_documents)
        try:
            if feature_source == "Tokenized terms":
                source_tokens = tokenized
            elif feature_source == "Stop-word filtered":
                source_tokens = cached_feature_stopwords(
                    feature_token_key(tokenized)
                )[0]
            elif feature_source == "Porter stems":
                source_tokens = cached_feature_stems(feature_token_key(tokenized))
            else:
                retained = cached_feature_stopwords(
                    feature_token_key(tokenized)
                )[0]
                source_tokens = cached_feature_stems(feature_token_key(retained))
        except LookupError as exc:
            st.sidebar.error(str(exc))
        else:
            st.session_state["feature_tokens"] = tokenized
            st.session_state["feature_dictionary"] = build_terms_dictionary(
                source_tokens
            )
            st.session_state["feature_dictionary_source_used"] = feature_source
            st.session_state["feature_last_action"] = "Create Dictionary"

    if st.session_state["feature_last_action"]:
        st.sidebar.caption(
            f"Last operation: {st.session_state['feature_last_action']}"
        )

st.title("Healthcare NLP — Main Adaptive Pipeline")
st.caption("Corpus exploration, adaptive lexical retrieval, and fixed-query evaluation")
st.divider()

if page == "Dashboard":
    st.header("Corpus and Adaptive Experiment Overview")
    all_tokens = [token for text in docs.values() for token in tokenize(text)]
    chars = sum(len(text) for text in docs.values())
    sentences = sum(
        len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])
        for text in docs.values()
    )
    derived = {
        "Total Tokens": len(all_tokens),
        "Vocabulary": len(set(all_tokens)),
        "Sentences": sentences,
        "Characters": chars,
    }
    token_total = derived["Total Tokens"]
    vocab_total = derived["Vocabulary"]
    char_total = derived["Characters"]
    sentence_total = derived["Sentences"]
    stats = [
        ("Documents", len(docs)),
        ("Total Tokens", int(token_total)),
        ("Vocabulary", int(vocab_total)),
        ("Sentences", int(sentence_total)),
        ("Characters", int(char_total)),
        ("Average Tokens / Document", round(token_total / len(docs), 1) if docs else 0),
    ]
    for col, (label, value) in zip(st.columns(3), stats[:3]):
        col.metric(label, f"{value:,}")
    for col, (label, value) in zip(st.columns(3), stats[3:]):
        col.metric(label, f"{value:,}" if isinstance(value, int) else value)

    st.caption(
        "Corpus figures are recalculated from the currently loaded source documents "
        "so stale summary files cannot override live corpus data."
    )
    st.divider()
    st.header("Adaptive Pipeline")
    st.markdown(
        "Query → Query Understanding → Domain Check → Adaptive Decision Engine "
        "→ Lexical Retrieval → Ranked Documents"
    )
    st.info(
        "The baseline dashboard's corpus and NLP analysis sections are retained. "
        "Retrieval and formal evaluation now call the Main Adaptive Pipeline."
    )
    if docs:
        lengths = pd.DataFrame({
            "Document": list(docs),
            "Token Count": [len(tokenize(text)) for text in docs.values()],
        })
        st.subheader("Corpus document lengths")
        st.bar_chart(lengths.set_index("Document"), horizontal=True)
        st.download_button(
            "Download live corpus lengths",
            lengths.to_csv(index=False),
            "live_corpus_document_lengths.csv",
            "text/csv",
        )

elif page == "Corpus Overview":
    st.header("Corpus Overview")
    st.write(
        "Healthcare and Public Health; infectious diseases, COVID-19, vaccination, "
        "epidemiology, and prevention."
    )
    st.metric("Text documents loaded", len(docs))
    show_artifact("Corpus summary", "corpus_summary.csv")
    show_artifact("Per-document statistics", "document_statistics.csv")
    if docs:
        rows = []
        for did, content in docs.items():
            tokens = tokenize(content)
            rows.append({
                "Document_ID": did,
                "Title": title_of(content, did),
                "Characters": len(content),
                "Tokens": len(tokens),
                "Unique_Tokens": len(set(tokens)),
            })
        st.subheader("Loaded document inventory")
        inventory = pd.DataFrame(rows)
        search = st.text_input("Filter documents", key="corpus_inventory_search")
        if search.strip():
            matches = inventory.astype(str).apply(
                lambda column: column.str.contains(
                    search.strip(), case=False, regex=False, na=False
                )
            ).any(axis=1)
            inventory = inventory.loc[matches]
        st.dataframe(inventory, hide_index=True, width="stretch")
        st.download_button(
            "Download loaded document inventory",
            inventory.to_csv(index=False),
            "loaded_document_inventory.csv",
            "text/csv",
        )

elif page == "Document Explorer":
    st.header("Document Explorer")
    if not docs:
        st.error(f"No .txt documents found in {DOC_DIR}")
    else:
        selected = st.selectbox(
            "Choose a document",
            list(docs),
            format_func=lambda did: f"{did} — {title_of(docs[did], did)}",
        )
        content = docs[selected]
        words = tokenize(content)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tokens", len(words))
        c2.metric("Unique terms", len(set(words)))
        c3.metric("Characters", len(content))
        document_search = st.text_input(
            "Find text in this document",
            key="document_explorer_search",
            placeholder="Enter a word or phrase",
        )
        if document_search.strip():
            matches = [
                line for line in content.splitlines()
                if document_search.casefold() in line.casefold()
            ]
            st.caption(f"{len(matches):,} matching line(s)")
            if matches:
                st.dataframe(
                    pd.DataFrame({"Matching text": matches}),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("No matching text in this document.")
        st.text_area("Full document text", content, height=420)
        st.download_button(
            "Download document", content, f"{selected}.txt", "text/plain"
        )

elif page == "Preprocessing Analysis":
    st.header("Text Preprocessing")
    show_artifact("Preprocessing summary", "preprocessing_summary.csv")
    show_artifact("Preprocessing comparison", "preprocessing_comparison.csv")
    st.caption("Displays existing results artifacts; does not silently rerun preprocessing.")

elif page == "Tokenization Analysis":
    st.header("Tokenization Analysis")
    show_artifact("Tokenization summary", "tokenization_summary.csv")
    show_artifact("Tokenization comparison", "tokenization_comparison.csv")

elif page == "N-gram Analysis":
    st.header("N-gram Analysis")
    show_artifact("N-gram summary", "ngram_summary.csv")
    show_artifact("N-gram document statistics", "ngram_document_statistics.csv")
    show_artifact("N-gram frequencies", "ngram_frequencies.csv")

elif page == "POS Tagging":
    st.header("Part-of-Speech Tagging")
    show_artifact("POS tag frequencies", "pos_tagging_summary.csv")
    show_artifact("POS tagging comparison", "pos_tagging_comparison.csv")

elif page == "Named Entity Recognition":
    st.header("Named Entity Recognition")
    show_artifact("NER summary", "ner_summary.csv")
    show_artifact("Detected entities", "ner_entities.csv")
    show_artifact("Domain terms", "ner_domain_terms.csv")

elif page == "BPE Analysis":
    st.header("Byte Pair Encoding Analysis")
    show_artifact("BPE summary", "bpe_summary.csv")
    show_artifact("BPE document comparison", "bpe_document_comparison.csv")
    show_artifact("BPE subword frequencies", "bpe_subword_frequencies.csv")

elif page == "Information Retrieval":
    st.header("Interactive Adaptive Information Retrieval")
    st.caption("Exploratory query search is separate from the formal Q01–Q05 test.")
    st.caption(
        "The adaptive engine chooses the lexical method automatically. "
        "Result display limits and coverage filters below do not alter retrieval."
    )
    top_n = st.slider("Number of results to display", min_value=1, max_value=20, value=5)
    minimum_coverage = st.slider(
        "Minimum query-term coverage",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Filters the displayed results only; it does not change the retrieval run.",
    )
    with st.form("adaptive_search_form"):
        query = st.text_area(
            "Enter a healthcare query",
            height=90,
            placeholder='Examples: COVID-19 prevention · "vaccine effectiveness" · '
            "COVID AND vaccination",
        )
        submitted = st.form_submit_button(
            "Search with Adaptive Pipeline",
            type="primary",
        )
    if submitted:
        if not query.strip():
            st.warning("Enter a query before searching.")
        else:
            try:
                with st.spinner("Running adaptive query processing and retrieval..."):
                    index, mapping, corpus = get_resources()
                    result = run_adaptive_pipeline(query.strip(), index, mapping, corpus)
                st.session_state["adaptive_interactive_output"] = result
                st.session_state["adaptive_interactive_query"] = query.strip()
            except Exception as exc:
                st.exception(exc)

    output = st.session_state.get("adaptive_interactive_output")
    if output:
        decision = output.get("adaptive_decision", {})
        retrieval = output.get("retrieval", {})
        st.subheader("Query Analysis")
        st.json(output.get("query_understanding", {}))
        d1, d2, d3 = st.columns(3)
        d1.metric(
            "Selected Method",
            str(decision.get("method", decision.get("selected_method", "N/A"))),
        )
        d2.metric(
            "Decision Confidence",
            (
                f"{float(decision.get('confidence', decision.get('decision_confidence'))):.0%}"
                if isinstance(
                    decision.get("confidence", decision.get("decision_confidence")),
                    (int, float),
                )
                else str(decision.get("confidence", decision.get("decision_confidence", "N/A")))
            ),
        )
        d3.metric("Status", str(output.get("status", "N/A")))
        st.write(
            "**Decision reason:**",
            decision.get(
                "reason",
                decision.get("decision_reason", "Not provided"),
            ),
        )
        results = retrieval.get("results", []) if isinstance(retrieval, dict) else []
        saved_query = st.session_state.get("adaptive_interactive_query", "")
        if output.get("status") == "OUT_OF_DOMAIN":
            st.warning(
                "The domain check abstained from retrieval for this query. "
                "Try a healthcare or public-health query."
            )
        else:
            st.success(
                f"Retrieved {len(results)} documents for: {saved_query}"
            )
        filtered_results = [
            row
            for row in results
            if float(row.get("coverage", 0) or 0) >= minimum_coverage
        ]
        if results and not filtered_results:
            st.info("No retrieved documents meet the current coverage filter.")
        if filtered_results:
            visible = []
            for rank, row in enumerate(filtered_results[:top_n], 1):
                visible.append({
                    "Rank": rank,
                    "Document_ID": row.get("document_id"),
                    "Document_Name": row.get("document_name"),
                    "Score": row.get("score"),
                    "Matched_Terms": ", ".join(map(str, row.get("matched_terms", []))),
                    "Coverage": row.get("coverage"),
                })
            st.dataframe(
                pd.DataFrame(visible),
                hide_index=True,
                width="stretch",
            )
            for rank, row in enumerate(filtered_results[:top_n], 1):
                did = str(row.get("document_id", ""))
                with st.expander(
                    f'Rank {rank}: {did} — {row.get("document_name", did)}'
                ):
                    st.write("Matched terms:", row.get("matched_terms", []))
                    st.write("Score:", row.get("score", "N/A"))
                    if did in docs:
                        st.text(docs[did])
        else:
            st.info("No documents were retrieved by the adaptive pipeline.")

elif page == "Evaluation Queries":
    st.header("Fixed Evaluation Queries — Q01 to Q05")
    st.caption(
        "Canonical five-query test set. Each query currently has one expected "
        "relevant document, validated against the loaded D01–D20 corpus."
    )
    selected_query_id = st.selectbox(
        "Inspect an evaluation query",
        [item["query_id"] for item in FIXED_QUERIES],
    )
    selected_query = next(
        item for item in FIXED_QUERIES if item["query_id"] == selected_query_id
    )
    expected_id = selected_query["expected_document"]
    expected_text = docs.get(expected_id)
    if expected_text is None:
        st.error(f"{selected_query_id} refers to missing corpus document {expected_id}.")
    else:
        with st.container(border=True):
            st.markdown(f"**Query:** {selected_query['query']}")
            st.markdown(
                f"**Expected document:** {expected_id} — "
                f"{title_of(expected_text, expected_id)}"
            )
            st.caption(
                "The label is the project's current ground truth; inspect the "
                "source document to validate whether it remains relevant."
            )
            with st.expander("View expected document text"):
                st.text(expected_text)
    for item in FIXED_QUERIES:
        with st.expander(
            f'{item["query_id"]} — Expected: {item["expected_document"]}'
        ):
            st.write(item["query"])
            st.write("Relevant document:", item["expected_document"])

elif page == "Evaluation Results":
    st.header("Main Adaptive Pipeline Evaluation Results")
    st.caption(
        "Metrics are recalculated from the current retrieval output and the fixed "
        "Q01–Q05 relevance labels."
    )
    k = st.select_slider(
        "K for top-K metrics",
        options=[1, 3, 5, 10, 20],
        value=5,
        help=(
            "Precision@K uses K as its fixed denominator, matching this project's "
            "saved evaluation convention. Recall@K divides relevant hits by the "
            "number of relevant labels for that query."
        ),
    )
    if st.button(
        f"Run Adaptive Evaluation (Q01–Q05, K={k})",
        type="primary",
    ):
        try:
            with st.spinner("Evaluating the five fixed queries..."):
                index, mapping, corpus = get_resources()
                frame, metrics, outputs = evaluate_fixed_queries(
                    index, mapping, corpus, k
                )
            st.session_state["adaptive_eval_frame"] = frame
            st.session_state["adaptive_eval_metrics"] = metrics
            st.session_state["adaptive_eval_outputs"] = outputs
        except Exception as exc:
            st.exception(exc)

    result_df = st.session_state.get("adaptive_eval_frame")
    metrics = st.session_state.get("adaptive_eval_metrics")
    if result_df is None or metrics is None:
        st.info("Click the evaluation button to calculate current adaptive results.")
    else:
        evaluated_k = int(metrics.get("K", 5))
        if evaluated_k != k:
            st.info(
                f"Displayed results use K={evaluated_k}. Run the evaluation again "
                f"to calculate metrics for K={k}."
            )
        metric_keys = [
            ("Precision", "Precision"),
            ("Recall", "Recall"),
            ("F1", "F1"),
            ("Precision@1", "Precision@1"),
            (f"Precision@{evaluated_k}", f"Precision@{evaluated_k}"),
            (f"Recall@{evaluated_k}", f"Recall@{evaluated_k}"),
            (f"F1@{evaluated_k}", f"F1@{evaluated_k}"),
            ("MRR@K", "Reciprocal_Rank"),
            (f"Hit@{evaluated_k}", f"Hit@{evaluated_k}"),
        ]
        for start in range(0, len(metric_keys), 3):
            metric_cols = st.columns(3)
            for column, (label, key) in zip(
                metric_cols,
                metric_keys[start : start + 3],
            ):
                column.metric(label, f"{metrics[key]:.3f}")

        st.caption(
            f"Macro averages over {metrics['Total_Queries']} queries. "
            f"{metrics['Expected_Documents_Retrieved']} relevant document(s) "
            f"were retrieved within K={evaluated_k}."
        )
        st.subheader("Micro-average over all retrieved documents")
        micro_columns = st.columns(3)
        micro_columns[0].metric("Micro precision", f"{metrics['Micro_Precision']:.3f}")
        micro_columns[1].metric("Micro recall", f"{metrics['Micro_Recall']:.3f}")
        micro_columns[2].metric("Micro F1", f"{metrics['Micro_F1']:.3f}")
        st.subheader("Per-query results")
        query_filter = st.selectbox(
            "Filter query results",
            ["All queries", *result_df["Query_ID"].tolist()],
        )
        filtered_results = result_df
        if query_filter != "All queries":
            filtered_results = result_df.loc[
                result_df["Query_ID"] == query_filter
            ]
        st.dataframe(filtered_results, hide_index=True, width="stretch")
        chart_columns = [
            f"Precision@{evaluated_k}",
            f"Recall@{evaluated_k}",
            f"F1@{evaluated_k}",
        ]
        st.subheader(f"Query-level top-{evaluated_k} metrics")
        st.bar_chart(
            result_df.set_index("Query_ID")[chart_columns],
            width="stretch",
        )
        st.download_button(
            "Download adaptive fixed-query evaluation CSV",
            result_df.to_csv(index=False),
            f"adaptive_q01_q05_evaluation_k{evaluated_k}.csv",
            "text/csv",
        )
        st.download_button(
            "Download adaptive metrics JSON",
            json.dumps(metrics, indent=2),
            f"adaptive_q01_q05_metrics_k{evaluated_k}.json",
            "application/json",
        )
        st.warning(
            "Precision@K uses a fixed K denominator even when fewer than K "
            "documents are returned. MRR is zero when the first relevant result "
            "appears below the selected K. The test set has one labeled relevant "
            "document per query; labels should be reviewed when the corpus changes."
        )

elif page == "Pipeline and Methodology":
    st.header("Main Adaptive Pipeline and Evaluation Method")
    st.markdown("""
    **Corpus:** healthcare/public-health text documents.

    **Formal test set:** the five fixed queries Q01–Q05, each with its predefined
    relevant document label.

    **Pipeline:**
    ```text
    User Query
       ↓
    Query Understanding
       ↓
    Domain Relevance Check
       ├── Out of domain → Abstain
       └── In domain
             ↓
       Adaptive Decision Engine
             ↓
       Select lexical retrieval method
             ↓
       Retrieve and rank relevant documents
             ↓
       Display retrieved evidence documents
    ```

    **Metrics:** Precision, Recall, F1, Precision@K, Recall@K, F1@K, MRR@K,
    and Hit@K. Precision@K uses a fixed K denominator even when fewer than K
    documents are returned; recall divides relevant results by the number of
    relevant labels for that query.

    **Scope:** adaptive lexical retrieval. Embeddings, vector search, graph or
    multi-hop retrieval, and LLM answer generation are not added by this app.
    """)
    st.subheader("Available experiment artifacts")
    for filename in [
        "corpus_summary.csv", "document_statistics.csv",
        "preprocessing_summary.csv", "preprocessing_comparison.csv",
        "tokenization_summary.csv", "tokenization_comparison.csv",
        "ngram_summary.csv", "ngram_document_statistics.csv",
        "ngram_frequencies.csv", "pos_tagging_summary.csv",
        "pos_tagging_comparison.csv", "ner_summary.csv",
        "ner_entities.csv", "ner_domain_terms.csv", "bpe_summary.csv",
        "bpe_document_comparison.csv", "bpe_subword_frequencies.csv",
        "adaptive_pipeline_results.json",
        "adaptive_evaluation_metrics.csv",
        "adaptive_evaluation_metrics.json",
    ]:
        available_path = resolve_result_path(filename)
        st.write(
            ("✓ " if available_path else "— ")
            + filename
            + (
                f" — {available_path.parent.name}/"
                if available_path is not None
                else ""
            )
        )

elif page == "Feature Extraction & Terms Dictionary":
    st.header("Feature Extraction & Terms Dictionary")
    st.caption(
        "Interactive lexical preprocessing over the existing adaptive corpus. "
        "Source document text and retrieval artifacts are never overwritten."
    )

    tokenized = st.session_state["feature_tokens"]
    if tokenized is None:
        tokenized = cached_feature_tokens(feature_documents)
        st.session_state["feature_tokens"] = tokenized

    token_key = feature_token_key(tokenized)
    filtered_tokens = st.session_state["feature_filtered_tokens"]
    removed_tokens = st.session_state["feature_removed_tokens"]
    stopwords_error = None
    if filtered_tokens is None or removed_tokens is None:
        try:
            filtered_tokens, removed_tokens = cached_feature_stopwords(token_key)
        except LookupError as exc:
            stopwords_error = str(exc)

    if feature_source == "Tokenized terms":
        processed_tokens = tokenized
    elif feature_source == "Stop-word filtered":
        processed_tokens = filtered_tokens
    elif feature_source == "Porter stems":
        processed_tokens = cached_feature_stems(token_key)
    else:
        processed_tokens = (
            cached_feature_stems(feature_token_key(filtered_tokens))
            if filtered_tokens is not None
            else None
        )

    raw_total = sum(len(tokens) for tokens in tokenized.values())
    raw_vocabulary = {
        term for document_tokens in tokenized.values() for term in document_tokens
    }
    if processed_tokens is None:
        processed_total = 0
        processed_vocabulary: set[str] = set()
        st.warning(stopwords_error or "Stop-word filtering could not be calculated.")
    else:
        processed_total = sum(len(tokens) for tokens in processed_tokens.values())
        processed_vocabulary = {
            term for document_tokens in processed_tokens.values() for term in document_tokens
        }
    vocabulary_reduction = len(raw_vocabulary) - len(processed_vocabulary)
    reduction_percent = (
        vocabulary_reduction / len(raw_vocabulary) * 100 if raw_vocabulary else 0.0
    )

    with st.container(horizontal=True):
        st.metric("Total documents", f"{len(docs):,}", border=True)
        st.metric("Total tokens", f"{processed_total:,}", border=True)
        st.metric("Unique terms", f"{len(processed_vocabulary):,}", border=True)
        st.metric("Total term occurrences", f"{processed_total:,}", border=True)
        st.metric(
            "Average document length",
            f"{processed_total / len(docs):,.1f}" if docs else "0",
            border=True,
        )
        st.metric(
            "Vocabulary reduction",
            f"{vocabulary_reduction:,} ({reduction_percent:.1f}%)",
            border=True,
        )
    st.caption(
        f"Statistics for **{feature_source}**. Original tokenized corpus: "
        f"{raw_total:,} tokens and {len(raw_vocabulary):,} terms. "
        "Term occurrences are total processed tokens; document frequency counts "
        "documents with at least one occurrence."
    )

    dictionary_tabs = st.tabs(["Terms Dictionary", "Document ID Names"])
    with dictionary_tabs[0]:
        st.subheader("Terms Dictionary")
        st.caption(
            "The Documents column contains document IDs and within-document "
            "term counts. Use the sidebar’s Create Dictionary action to retain "
            "a generated copy for this session."
        )

        dictionary = st.session_state["feature_dictionary"]
        dictionary_source_used = (
            st.session_state["feature_dictionary_source_used"] or feature_source
        )
        if dictionary is None:
            dictionary = (
                build_terms_dictionary(processed_tokens)
                if processed_tokens is not None
                else pd.DataFrame(
                    columns=[
                        "S No.",
                        "Term",
                        "Documents",
                        "Term Occurrences",
                        "Document Frequency",
                    ]
                )
            )
        if dictionary.empty:
            st.info(
                "Create a dictionary from the sidebar after the stop-word corpus "
                "is available, or choose an available token representation."
            )
        else:
            search_column, sort_term, sort_doc = st.columns([2, 1, 1])
            with search_column:
                term_search = st.text_input(
                    "Search terms",
                    placeholder="Filter by vocabulary term or document posting",
                    key="feature_term_search",
                )
            with sort_term:
                if st.button(
                    "Sort By Term Frequency",
                    key="feature_sort_term",
                ):
                    st.session_state["feature_dictionary_sort"] = (
                        "Term Occurrences"
                    )
            with sort_doc:
                if st.button(
                    "Sort By Doc Frequency",
                    key="feature_sort_doc",
                ):
                    st.session_state["feature_dictionary_sort"] = (
                        "Document Frequency"
                    )
            displayed = dictionary.copy()
            if term_search.strip():
                term_match = displayed["Term"].str.contains(
                    term_search.strip(),
                    case=False,
                    regex=False,
                )
                document_match = displayed["Documents"].str.contains(
                    term_search.strip(),
                    case=False,
                    regex=False,
                )
                displayed = displayed.loc[term_match | document_match]

            sort_key = st.session_state["feature_dictionary_sort"]
            if sort_key in {"Term Occurrences", "Document Frequency"}:
                displayed = displayed.sort_values(
                    [sort_key, "Term"],
                    ascending=[False, True],
                    kind="stable",
                )
            else:
                displayed = displayed.sort_values("Term", kind="stable")
            displayed = displayed.reset_index(drop=True)
            displayed["S No."] = range(1, len(displayed) + 1)
            st.caption(
                f"{len(displayed):,} of {len(dictionary):,} terms · "
                f"dictionary source: {dictionary_source_used}"
            )
            if dictionary_source_used != feature_source:
                st.info(
                    f"The displayed dictionary was generated from "
                    f"{dictionary_source_used}. Click Create Dictionary to "
                    f"regenerate it from {feature_source}."
                )
            st.dataframe(
                displayed,
                hide_index=True,
                width="stretch",
                height=520,
            )
            st.download_button(
                "Download Terms Dictionary CSV",
                displayed.to_csv(index=False),
                "terms_dictionary.csv",
                "text/csv",
                key="feature_dictionary_download",
            )

    with dictionary_tabs[1]:
        st.subheader("Document ID Names")
        names = pd.DataFrame(
            [
                {
                    "Document ID": document_id,
                    "Document Name": title_of(text, document_id),
                }
                for document_id, text in docs.items()
            ]
        )
        st.dataframe(names, hide_index=True, width="stretch")
        st.download_button(
            "Download document ID names",
            names.to_csv(index=False),
            "document_id_names.csv",
            "text/csv",
            key="feature_document_names_download",
        )

    operations = st.tabs(
        [
            "Tokenize Documents",
            "Extract All Terms",
            "Apply Stemming",
            "Remove Stop Words",
        ]
    )
    with operations[0]:
        st.subheader("Tokenized documents")
        token_counts = pd.DataFrame(document_token_counts(tokenized))
        st.dataframe(token_counts, hide_index=True, width="stretch")
        selected_document = st.selectbox(
            "Inspect tokenized content",
            list(tokenized),
            format_func=lambda did: f"{did} — {title_of(docs[did], did)}",
            key="feature_token_document",
        )
        with st.expander("Token sequence"):
            st.code(" ".join(tokenized[selected_document]), language=None)

    with operations[1]:
        st.subheader("Extracted terms")
        extracted = st.session_state["feature_extracted_terms"]
        if extracted is None:
            extracted = build_terms_dictionary(tokenized)
        st.metric("Original tokenized vocabulary", f"{len(extracted):,}")
        st.dataframe(extracted, hide_index=True, width="stretch", height=420)
        st.caption(
            "This vocabulary is calculated from all original tokenized documents; "
            "term frequency is the corpus-wide occurrence count."
        )

    with operations[2]:
        st.subheader("Porter stemming")
        stemmed_tokens = st.session_state["feature_stemmed_tokens"]
        if stemmed_tokens is None:
            stemmed_tokens = cached_feature_stems(token_key)
        stemmed_vocabulary = {
            term for values in stemmed_tokens.values() for term in values
        }
        stem_columns = st.columns(2)
        stem_columns[0].metric("Vocabulary before", f"{len(raw_vocabulary):,}")
        stem_columns[1].metric("Vocabulary after", f"{len(stemmed_vocabulary):,}")
        mapping = st.session_state["feature_stem_mapping"]
        if mapping is None:
            mapping = stem_mapping(
                term for values in tokenized.values() for term in values
            )
        st.dataframe(
            pd.DataFrame(mapping),
            hide_index=True,
            width="stretch",
            height=420,
        )
        st.caption(
            "NLTK Porter stemming is applied to analysis tokens only. The original "
            "document text remains unchanged."
        )

    with operations[3]:
        st.subheader("English stop-word removal")
        if filtered_tokens is None or removed_tokens is None:
            st.error(stopwords_error or "The NLTK stop-word list is unavailable.")
        else:
            before_count = sum(len(values) for values in tokenized.values())
            after_count = sum(len(values) for values in filtered_tokens.values())
            before_vocab = {
                term for values in tokenized.values() for term in values
            }
            after_vocab = {
                term for values in filtered_tokens.values() for term in values
            }
            stopword_metrics = st.columns(4)
            stopword_metrics[0].metric("Tokens before", f"{before_count:,}")
            stopword_metrics[1].metric("Tokens after", f"{after_count:,}")
            stopword_metrics[2].metric(
                "Removed occurrences",
                f"{before_count - after_count:,}",
            )
            stopword_metrics[3].metric(
                "Vocabulary before / after",
                f"{len(before_vocab):,} / {len(after_vocab):,}",
            )
            removed_counts = Counter(
                term for values in removed_tokens.values() for term in values
            )
            retained_counts = Counter(
                term for values in filtered_tokens.values() for term in values
            )
            removed_frame = pd.DataFrame(
                [
                    {"Term": term, "Occurrences": count}
                    for term, count in sorted(removed_counts.items())
                ]
            )
            retained_frame = pd.DataFrame(
                [
                    {"Term": term, "Occurrences": count}
                    for term, count in sorted(retained_counts.items())
                ]
            )
            removed_column, retained_column = st.columns(2)
            with removed_column:
                st.markdown("**Removed stop words**")
                st.dataframe(
                    removed_frame,
                    hide_index=True,
                    width="stretch",
                    height=380,
                )
            with retained_column:
                st.markdown("**Retained terms**")
                st.dataframe(
                    retained_frame,
                    hide_index=True,
                    width="stretch",
                    height=380,
                )

st.divider()
st.caption(
    "Healthcare NLP | Main Adaptive Pipeline | Formal evaluation: Q01–Q05"
)
