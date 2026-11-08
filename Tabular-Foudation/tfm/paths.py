from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]          # .../Tabular-Foudation
PROJECT_ROOT = CODE_DIR.parent                          # .../advanced-data-mining
DATASETS = PROJECT_ROOT / "datasets"
RESULTS = PROJECT_ROOT / "results"
SPLITS = PROJECT_ROOT / "splits"
