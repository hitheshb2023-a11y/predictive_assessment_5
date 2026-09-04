import argparse
import json
import time
from datetime import datetime, timezone

import joblib
import pandas as pd

from .advanced import run_bilstm, run_transformers, transformer_status
from .config import default_settings
from .data import build_splits, load_datasets, quality_summary
from .evaluation import evaluate_dataset, save_confusion_plots, save_summary_plot
from .modeling import make_models
from .reporting import write_results_notebook
from .text import safe_filename


def run(sample_size: int = 60000, project_root=None, advanced: bool = False) -> None:
    settings = default_settings(project_root, sample_size=sample_size)
    settings.create_output_dirs()
    started = time.perf_counter()
    datasets = load_datasets(settings)
    splits = build_splits(datasets, settings)
    quality = quality_summary(splits)
    quality.to_csv(settings.results_dir / "quality_summary.csv", index=False)

    cv_parts, test_parts, class_parts, prediction_parts = [], [], [], []
    selected = {}
    for name, split_map in splits.items():
        cv, test, classes, predictions, fitted, best = evaluate_dataset(
            name, split_map, make_models(settings), settings
        )
        cv_parts.append(cv)
        test_parts.append(test)
        class_parts.append(classes)
        prediction_parts.append(predictions)
        selected[name] = best
        for model_name, model in fitted.items():
            if model_name == best:
                joblib.dump(model, settings.models_dir / f"selected_{safe_filename(name)}.joblib")

    cv_results = pd.concat(cv_parts, ignore_index=True)
    test_results = pd.concat(test_parts, ignore_index=True)
    class_metrics = pd.concat(class_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    cv_results.to_csv(settings.results_dir / "cv_results.csv", index=False)
    test_results.to_csv(settings.results_dir / "test_results.csv", index=False)
    class_metrics.to_csv(settings.results_dir / "class_metrics.csv", index=False)
    predictions.to_csv(settings.results_dir / "predictions.csv", index=False)
    save_confusion_plots(test_results, predictions, settings)
    save_summary_plot(cv_results, settings)

    advanced_results = pd.DataFrame()
    advanced_status = transformer_status(splits)
    if advanced:
        advanced_results, bilstm_status = run_bilstm(splits, seed=settings.seed)
        transformer_results, transformer_run_status = run_transformers(splits, seed=settings.seed)
        advanced_results = pd.concat([advanced_results, transformer_results], ignore_index=True, sort=False)
        advanced_status = pd.concat([bilstm_status, transformer_run_status], ignore_index=True)
        advanced_results.to_csv(settings.results_dir / "advanced_results.csv", index=False)
    advanced_status.to_csv(settings.results_dir / "advanced_status.csv", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": settings.seed,
        "sentiment140_sample_size": settings.s140_sample_size,
        "datasets": list(splits),
        "selected_models": selected,
        "advanced_requested": advanced,
        "advanced_models": sorted(advanced_results["Model"].unique()) if not advanced_results.empty else [],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (settings.results_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_results_notebook(
        settings.results_dir / "training_results.ipynb",
        manifest,
        {
            "Data quality": quality,
            "Cross-validation results": cv_results,
            "Locked test results": test_results,
            "Class metrics": class_metrics,
            "Test predictions": predictions,
            "Advanced results": advanced_results,
            "Advanced status": advanced_status,
        },
        settings.figures_dir,
    )
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate all sentiment datasets.")
    parser.add_argument("--sample-size", type=int, default=60000, help="Sentiment140 balanced sample size.")
    parser.add_argument("--project-root", default=None, help="Project root containing data/.")
    parser.add_argument("--advanced", action="store_true", help="Also train the optional BiLSTM model.")
    args = parser.parse_args()
    run(args.sample_size, args.project_root, args.advanced)


if __name__ == "__main__":
    main()
