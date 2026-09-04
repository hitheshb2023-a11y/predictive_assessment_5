from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import Settings
from .text import safe_filename


def metric_row(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
    }


def evaluate_dataset(name: str, splits: Dict[str, pd.DataFrame], models: Dict[str, object], settings: Settings):
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    cv = StratifiedKFold(settings.cv_folds, shuffle=True, random_state=settings.seed)
    cv_rows, test_rows, class_rows, prediction_rows = [], [], [], []
    fitted = {}
    for model_name, model in models.items():
        cv_result = cross_validate(
            model, train["clean_text"], train["label"], cv=cv,
            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro", "weighted_f1": "f1_weighted"},
            n_jobs=-1,
        )
        model.fit(train["clean_text"], train["label"])
        fitted[model_name] = model
        validation_pred = model.predict(validation["clean_text"])
        test_pred = model.predict(test["clean_text"])
        cv_rows.append({
            "Dataset": name, "Model": model_name,
            "CV Macro F1 Mean": cv_result["test_macro_f1"].mean(),
            "CV Macro F1 SD": cv_result["test_macro_f1"].std(),
            "CV Weighted F1": cv_result["test_weighted_f1"].mean(),
            "CV Accuracy": cv_result["test_accuracy"].mean(),
            "Validation Macro F1": f1_score(validation["label"], validation_pred, average="macro"),
            "Test Accuracy": accuracy_score(test["label"], test_pred),
            "Test Macro F1": f1_score(test["label"], test_pred, average="macro"),
            "Test Weighted F1": f1_score(test["label"], test_pred, average="weighted"),
        })
        test_metrics = metric_row(test["label"], test_pred)
        test_rows.append({"Dataset": name, "Model": model_name, **test_metrics})
        report = classification_report(test["label"], test_pred, output_dict=True, zero_division=0)
        for label, values in report.items():
            if label in ("negative", "neutral", "positive"):
                class_rows.append({"Dataset": name, "Model": model_name, "Class": label,
                                   "Precision": values["precision"], "Recall": values["recall"],
                                   "F1": values["f1-score"], "Support": values["support"]})
        for index, (text, actual, predicted) in enumerate(zip(test["text"], test["label"], test_pred)):
            prediction_rows.append({"Dataset": name, "Model": model_name, "Test index": index,
                                    "Text": text, "True label": actual, "Predicted label": predicted})

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)
    class_df = pd.DataFrame(class_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    selected = cv_df.sort_values(["CV Macro F1 Mean", "CV Macro F1 SD"], ascending=[False, True]).iloc[0]["Model"]
    return cv_df, test_df, class_df, predictions_df, fitted, str(selected)


def save_confusion_plots(test_results: pd.DataFrame, predictions: pd.DataFrame, settings: Settings) -> None:
    for _, row in test_results.iterrows():
        subset = predictions[(predictions["Dataset"] == row["Dataset"]) & (predictions["Model"] == row["Model"])]
        labels = [label for label in ("negative", "neutral", "positive") if label in set(subset["True label"]) | set(subset["Predicted label"])]
        matrix = confusion_matrix(subset["True label"], subset["Predicted label"], labels=labels, normalize="true")
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        fig.colorbar(image, ax=axis)
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
                 yticklabels=labels, xlabel="Predicted", ylabel="Actual",
                 title=f"{row['Dataset']} - {row['Model']}\nNormalized confusion matrix")
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(settings.figures_dir / f"{safe_filename(row['Dataset'])}_{safe_filename(row['Model'])}_confusion.png", dpi=160)
        plt.close(fig)


def save_summary_plot(cv_results: pd.DataFrame, settings: Settings) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for dataset, group in cv_results.groupby("Dataset"):
        axis.plot(group["Model"], group["CV Macro F1 Mean"], marker="o", label=dataset)
    axis.set(xlabel="Model", ylabel="Cross-validation Macro F1", title="Leakage-safe model comparison")
    axis.tick_params(axis="x", rotation=20)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(settings.figures_dir / "model_comparison.png", dpi=160)
    plt.close(fig)
