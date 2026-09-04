# Tweet Sentiment Assessment Project

A modular, reproducible Python project for the supplied tweet sentiment assessment. It trains leakage-safe classical baselines, saves evaluation tables and plots, and provides a Streamlit dashboard for exploring results.

## Project layout

```text
assessment 7/
├── app.py                    # Streamlit dashboard
├── run_pipeline.py           # Cross-platform training entry point
├── requirements.txt
├── data/                     # Supplied CSV files
├── src/sentiment_app/
│   ├── config.py             # Settings and output paths
│   ├── data.py               # Loaders, normalization fields, frozen splits
│   ├── modeling.py           # Dummy, NB, Logistic Regression, LinearSVC
│   ├── evaluation.py         # CV, locked-test metrics, plots
│   └── pipeline.py           # End-to-end orchestration
└── artifacts/
    ├── results/              # CSV/JSON result tables
    ├── figures/               # PNG plots
    └── models/                # Reloadable joblib pipelines
```

## What is included

- Twitter US Airline Sentiment: 3-class product/service entity analysis.
- TweetEval Sentiment: official train/validation/test files preserved.
- Sentiment140: balanced, seed-locked binary sample; no fabricated neutral class.
- Minimal normalization: URLs become `<URL>`, mentions become `<USER>`, while negation, punctuation, hashtags, and emojis are preserved.
- Training-only stratified cross-validation and a locked test evaluation.
- Dummy, Multinomial Naive Bayes, Logistic Regression, and LinearSVC models.
- Macro F1 as the primary selection metric, with accuracy and weighted F1 reported.
- Saved predictions, class metrics, quality summary, confusion matrices, and model comparison plot.
- Streamlit dashboard with dataset/model filters and an interactive prediction box.
- Optional BiLSTM training for all three datasets from the CLI or Streamlit sidebar.

## macOS setup

Open Terminal in the project folder:

```bash
cd "/Users/your-name/Desktop/assessment 7"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

To include the optional BiLSTM model in the training run:

```bash
pip install -r requirements-advanced.txt
python3 run_pipeline.py --sample-size 60000 --advanced
```

The Streamlit sidebar includes a **Train advanced BiLSTM** checkbox. Optional Transformer dependencies are listed in `requirements-advanced.txt` for a later Transformer run.

The dashboard opens at `http://localhost:8501`. If training has not been run, use the **Train all datasets** button in the sidebar. It trains Airline Sentiment, TweetEval, and Sentiment140 and then refreshes the dashboard with the generated tables, plots, and saved models. The same control becomes **Retrain all datasets** when artifacts already exist.

## Windows PowerShell setup

```powershell
cd "C:\Users\your-name\Desktop\assessment 7"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```


```powershell
pip install -r requirements-advanced.txt
python run_pipeline.py --advanced
```


If PowerShell blocks activation, run PowerShell as the current user and execute:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the environment again.

## Training options

The default command samples 60,000 Sentiment140 rows, balanced across its two labels:

```bash
python run_pipeline.py
```

For a quick smoke run:

```bash
python run_pipeline.py --sample-size 12000
```

To run with another project location:

```bash
python run_pipeline.py --project-root "/path/to/assessment 7" --sample-size 60000
```

Run the pipeline again whenever the data or configuration changes. Existing artifacts are replaced.

## Results

After training, inspect:

- `artifacts/results/cv_results.csv`
- `artifacts/results/test_results.csv`
- `artifacts/results/class_metrics.csv`
- `artifacts/results/predictions.csv`
- `artifacts/results/quality_summary.csv`
- `artifacts/results/training_results.ipynb` (generated from the current run, with tables and embedded figures)
- `artifacts/figures/`
- `artifacts/models/`

The Streamlit app can generate these artifacts itself through its sidebar training control. The training command also creates `training_results.ipynb`; it contains the live run's data-quality table, CV table, locked-test table, class metrics, full test predictions, and embedded generated figures.

## Notes for the assessment

The advanced BiLSTM is available through `--advanced` or the Streamlit sidebar and produces real results for all three datasets. Transformer execution is kept optional because it requires downloading pretrained checkpoints; install `requirements-advanced.txt` before adding that extension. The core pipeline remains usable without a GPU. Sentiment140 is not directly comparable to the 3-class datasets because it contains only positive and negative labels and uses distant supervision.
