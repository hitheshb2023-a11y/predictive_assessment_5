import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {"language": "markdown"},
        "source": source.splitlines(keepends=True),
    }


def _table_cell(title: str, frame: pd.DataFrame) -> dict:
    html = frame.to_html(index=False, border=0, justify="left")
    return _markdown_cell(f"## {title}\n\n{html}")


def _image_cell(title: str, image_path: Path) -> dict:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"language": "python"},
        "source": [f"# {title}\n"],
        "outputs": [{
            "output_type": "display_data",
            "data": {"image/png": encoded, "text/plain": [f"{title}\n"]},
            "metadata": {},
        }],
    }


def write_results_notebook(
    output_path: Path,
    manifest: dict,
    tables: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Write a new notebook containing outputs produced by the current training run."""
    cells = [
        _markdown_cell(
            "# Tweet Sentiment Training Results\n\n"
            "This notebook was generated automatically by the training pipeline. "
            "All tables and figures below come from the current run; no existing notebook was copied.\n\n"
            f"Generated: `{manifest['generated_at_utc']}`  \n"
            f"Sentiment140 sample size: `{manifest['sentiment140_sample_size']}`  \n"
            f"Seed: `{manifest['seed']}`"
        ),
        _markdown_cell("## Selected models\n\n" + json.dumps(manifest["selected_models"], indent=2)),
    ]

    for title, frame in tables.items():
        cells.append(_table_cell(title, frame))

    cells.append(_markdown_cell("## Figures\n\nThese images were generated during the same training run and embedded into this notebook."))
    for image_path in sorted(figures_dir.glob("*.png")):
        cells.append(_image_cell(image_path.stem.replace("_", " ").title(), image_path))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "generated_by": "sentiment_app.pipeline",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")