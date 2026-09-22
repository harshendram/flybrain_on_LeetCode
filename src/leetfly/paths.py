from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"
WEB_DATA = ROOT / "web" / "public" / "data"

MALECNS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
MALECNS_ANNOTATIONS = RAW / "malecns" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
MALECNS_WEIGHTS = RAW / "malecns" / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
LEETCODE = RAW / "leetcode" / "leetcode_dataset.parquet"
