from pathlib import Path
import sys
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from sentiment_app.data import AIRLINE_NAME, S140_NAME, TWEETEVAL_NAME
from sentiment_app.pipeline import run as run_pipeline
from sentiment_app.text import normalize_tweet, safe_filename

ARTIFACTS = ROOT / "artifacts"
RESULTS = ARTIFACTS / "results"
FIGURES = ARTIFACTS / "figures"
MODELS = ARTIFACTS / "models"

st.set_page_config(page_title="Tweet Sentiment Lab", page_icon="", layout="wide")
st.markdown("""
<style>
:root { --ink:#17202a; --muted:#68737d; --accent:#d05a3a; --paper:#f6f2ea; }
.stApp { background:var(--paper); color:var(--ink); }
.block-container { max-width:1200px; padding-top:2rem; }
[data-testid="stMetricValue"] { color:var(--accent); }
</style>
""", unsafe_allow_html=True)

st.title("Tweet Sentiment Lab")
st.caption("Leakage-safe training results across Airline Sentiment, TweetEval, and Sentiment140")

artifacts_ready = (RESULTS / "cv_results.csv").exists()

with st.sidebar:
    st.header("Pipeline")
    sample_size = st.number_input(
        "Sentiment140 sample size",
        min_value=2000,
        max_value=60000,
        value=60000,
        step=2000,
        help="Balanced sample size used when training all three datasets.",
    )
    train_advanced = st.checkbox(
        "Train advanced BiLSTM",
        value=False,
        help="Runs the optional neural model for all three datasets and takes longer.",
    )
    train_label = "Retrain all datasets" if artifacts_ready else "Train all datasets"
    if st.button(train_label, type="primary", use_container_width=True):
        with st.spinner("Training Airline Sentiment, TweetEval, and Sentiment140..."):
            try:
                run_pipeline(sample_size=int(sample_size), project_root=ROOT, advanced=train_advanced)
            except Exception as error:
                st.error(f"Training failed: {type(error).__name__}: {error}")
                st.stop()
        st.success("All three datasets trained and results saved.")
        st.rerun()

if not artifacts_ready:
    st.info("The pipeline has not been trained yet. Choose a Sentiment140 sample size and click **Train all datasets** in the sidebar.")
    st.stop()

cv = pd.read_csv(RESULTS / "cv_results.csv")
test = pd.read_csv(RESULTS / "test_results.csv")
classes = pd.read_csv(RESULTS / "class_metrics.csv")
quality = pd.read_csv(RESULTS / "quality_summary.csv")
manifest = json.loads((RESULTS / "run_manifest.json").read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Explore")
    dataset = st.selectbox("Dataset", sorted(cv["Dataset"].unique()))
    model = st.selectbox("Model", sorted(cv.loc[cv["Dataset"] == dataset, "Model"].unique()))
    st.divider()
    st.write("Selected models")
    st.json(manifest.get("selected_models", {}))

filtered_cv = cv[cv["Dataset"] == dataset]
filtered_test = test[test["Dataset"] == dataset]
selected_test = filtered_test[filtered_test["Model"] == model].iloc[0]

metric_cols = st.columns(3)
metric_cols[0].metric("Test Macro F1", f"{selected_test['Macro F1']:.3f}")
metric_cols[1].metric("Test Accuracy", f"{selected_test['Accuracy']:.3f}")
metric_cols[2].metric("Test Weighted F1", f"{selected_test['Weighted F1']:.3f}")

st.subheader("Model comparison")
st.line_chart(filtered_cv.set_index("Model")[["CV Macro F1 Mean", "Validation Macro F1"]])

left, right = st.columns(2)
with left:
    st.subheader("Test results")
    st.dataframe(filtered_test, use_container_width=True, hide_index=True)
with right:
    st.subheader("Class metrics")
    st.dataframe(classes[(classes["Dataset"] == dataset) & (classes["Model"] == model)], use_container_width=True, hide_index=True)

st.subheader("Confusion matrix")
confusion = FIGURES / f"{safe_filename(dataset)}_{safe_filename(model)}_confusion.png"
if confusion.exists():
    st.image(str(confusion), use_container_width=True)

st.subheader("Dataset quality")
st.dataframe(quality, use_container_width=True, hide_index=True)

advanced_path = RESULTS / "advanced_results.csv"
advanced_status_path = RESULTS / "advanced_status.csv"
if advanced_path.exists() or advanced_status_path.exists():
    st.subheader("Advanced models")
    if advanced_path.exists():
        st.dataframe(pd.read_csv(advanced_path), use_container_width=True, hide_index=True)
    if advanced_status_path.exists():
        st.dataframe(pd.read_csv(advanced_status_path), use_container_width=True, hide_index=True)

st.subheader("Try a prediction")
text = st.text_area("Tweet text", "The service was quick and helpful today!", height=100)
if st.button("Predict sentiment", type="primary"):
    path = MODELS / f"selected_{safe_filename(dataset)}.joblib"
    if not path.exists():
        st.error("The selected model is not the saved pipeline for this dataset. Choose the selected model shown in the manifest.")
    else:
        prediction = joblib.load(path).predict([normalize_tweet(text)])[0]
        st.success(f"Predicted sentiment: {prediction}")

with st.expander("About the data and governance"):
    st.markdown("""
    The Airline and TweetEval tasks use three labels: negative, neutral, and positive. Sentiment140 is binary and is never assigned a fabricated neutral class. Usernames, IDs, locations, annotation confidence, and negative-reason metadata are excluded from model predictors. Tweet share is not market share, and sentiment is not causal evidence of customer satisfaction.
    """)
