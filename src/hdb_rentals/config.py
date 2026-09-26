"""Project paths and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
AUX_DIR = DATA_DIR / "auxiliary"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

AUX_CSV = {
    "hdb_block": AUX_DIR / "sg-hdb-block.csv",
    "mrt": AUX_DIR / "sg-mrt-stations.csv",
    "schools": AUX_DIR / "sg-schools.csv",
    "malls": AUX_DIR / "sg-shopping-malls.csv",
    "coe": AUX_DIR / "sg-coe-prices.csv",
    "stocks": AUX_DIR / "sg-stock-prices.csv",
}

TARGET = "MONTHLY_RENT"

# Columns that carry no signal in the data as shipped:
#   FEE      is 0.0 for every row of train and test
#   FURNISHED is "yes" for every row of train and test
CONSTANT_COLUMNS = ["FEE", "FURNISHED"]

# The natural join key onto sg-hdb-block.csv. Matches 100% of train and test
# rows once all three parts are lowercased and stripped.
BLOCK_KEY = ["TOWN", "BLOCK", "STREET"]

# The test set begins where the training set ends, so validation must be
# ordered in time rather than shuffled. See validation.py.
TRAIN_PERIOD = ("2021-01", "2025-03")
TEST_PERIOD = ("2025-03", "2026-07")

for _d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
