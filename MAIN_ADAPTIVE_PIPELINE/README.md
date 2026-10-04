# Healthcare NLP — Main Adaptive Pipeline

The Streamlit application retains its radio-navigation layout and provides:

- Dashboard and corpus overview calculated from the current adaptive corpus.
- Document explorer with document search and download.
- Preprocessing, tokenization, N-gram, POS, NER, and BPE analysis sections.
- Interactive adaptive lexical retrieval, fixed evaluation queries, and the
  official five-query baseline/adaptive comparison.
- Feature Extraction & Terms Dictionary with NLTK tokenization, Porter
  stemming, stop-word removal, term/document frequencies, sorting, and CSV
  downloads.

When an analysis CSV is not present in `MAIN_ADAPTIVE_PIPELINE\results`, the
app looks for the corresponding artifact in the sibling
`BASELINE_PIPELINE\results`. Those analysis snapshots are labeled in the UI;
the two project document folders were checked to contain the same 20 source
documents. Both applications display the same baseline/adaptive comparison
report; the baseline app's other saved evaluation artifacts are retained but
not presented as official results.

Analysis result tables support searching, sorting, and full-source downloads.
Large tables preview up to 500 rows to keep the app responsive.

## Official evaluation protocol

The official comparison uses the same five fixed queries for both pipelines
and multi-document relevance labels:

| Query | Relevant documents |
| --- | --- |
| Q01 | D01 |
| Q02 | D02, D03 |
| Q03 | D01, D02 |
| Q04 | D02, D03, D16 |
| Q05 | D06, D09 |

Only macro set-based Precision, Recall, and F1 over all retrieved documents
are reported. Precision is relevant retrieved documents divided by all
retrieved documents; recall is relevant retrieved documents divided by all
relevant labels; F1 is their harmonic mean. The app displays the paired
comparison saved by `experiments\baseline_vs_adaptive.py`.

Current results (rounded to three decimals):

| Pipeline | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Baseline | 0.123 | 1.000 | 0.218 |
| Adaptive | 0.833 | 0.733 | 0.767 |

Regenerate the shared report from this directory, in order:

```powershell
python experiments\run_baseline.py
python src\adaptive_pipeline.py
python experiments\baseline_vs_adaptive.py
```

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