from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    output_dir: Path
    s140_sample_size: int = 60000
    test_size: float = 0.10
    validation_size: float = 0.10
    cv_folds: int = 3
    seed: int = 42
    max_word_features: int = 80000
    max_char_features: int = 60000

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    def create_output_dirs(self) -> None:
        for path in (self.output_dir, self.figures_dir, self.models_dir, self.results_dir):
            path.mkdir(parents=True, exist_ok=True)


def default_settings(project_root: Path | None = None, sample_size: int = 60000) -> Settings:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    return Settings(root, root / "data", root / "artifacts", s140_sample_size=sample_size)
