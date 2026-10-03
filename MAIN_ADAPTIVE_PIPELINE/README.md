# Healthcare NLP — Main Adaptive Pipeline

The Streamlit application retains its radio-navigation layout and provides:

- Dashboard and corpus overview calculated from the current adaptive corpus.
- Document explorer with document search and download.
- Preprocessing, tokenization, N-gram, POS, NER, and BPE analysis sections.
- Interactive adaptive lexical retrieval, fixed evaluation queries, and
  selectable-K evaluation.
- Feature Extraction & Terms Dictionary with NLTK tokenization, Porter
  stemming, stop-word removal, term/document frequencies, sorting, and CSV
  downloads.

When an analysis CSV is not present in `MAIN_ADAPTIVE_PIPELINE\results`, the
app looks for the corresponding artifact in the sibling
`BASELINE_PIPELINE\results`. Those analysis snapshots are labeled in the UI;
the two project document folders were checked to contain the same 20 source
documents. The baseline application and its result files are not modified.

Analysis result tables support searching, sorting, and full-source downloads.
Large tables preview up to 500 rows to keep the app responsive.

Evaluation supports K values 1, 3, 5, 10, and 20. Precision@K uses K as a fixed
denominator (including when fewer than K documents are returned), and recall
divides relevant hits in the top K by the number of relevant labels. The
current fixed query set has one expected document for each query. The live
evaluation can also handle multiple relevant-document labels if configured.

## Run

From the `MAIN_ADAPTIVE_PIPELINE` directory:

```powershell
..\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

The app reads corpus documents from `data\documents`, adaptive results from
`results`, and missing analysis snapshots from the sibling
`BASELINE_PIPELINE\results` directory. It does not rerun or overwrite analysis
artifacts or the baseline project.

## Feature extraction and terms dictionary

The feature module uses the same `D01`–`D20` document IDs and source texts as
the retrieval pipeline. Its sidebar actions tokenize the 20 documents, extract
term counts, stem tokens with NLTK's Porter stemmer, remove standard NLTK
English stop words, or create a dictionary from the selected representation.
Each document posting is counted from that document's processed tokens; term
occurrences are the sum of those counts, and document frequency is the number
of distinct document postings. Term-frequency and document-frequency sorts
are descending, with alphabetical term ordering for ties.

NLTK is already listed in `requirements.txt`. If its English stop-word corpus
is unavailable in the active environment, install it from PowerShell with:

```powershell
..\.venv\Scripts\python.exe -m nltk.downloader stopwords
```

New or updated files:

- `app\streamlit_app.py` — adds the sidebar actions and the feature-extraction
  display while retaining the existing dashboard tabs.
- `src\feature_extraction.py` — reusable NLTK tokenization, stemming,
  stop-word removal, and term dictionary calculations.
- `README.md` — run and feature documentation.