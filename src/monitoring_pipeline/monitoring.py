import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = REPO_ROOT / "evidently_artifacts/json_reports"
OUT = REPO_ROOT / "src/feature_repo/data/monitoring_metrics.parquet"

SKIP = {"CorrelationMatrix"}

# Evidently metric type -> short name (anything not listed is snake_cased)
NAMES = {
    "MinValue": "min",
    "MaxValue": "max",
    "MeanValue": "mean",
    "StdValue": "std",
    "MissingValueCount": "missing",
    "DatasetMissingValueCount": "missing",
    "ValueDrift": "drift",
    "DriftedColumnsCount": "drifted_columns",
}

# Suffix for metrics whose value is a dict
STAT_SUFFIX = {"mean": "_mean", "std": "_std", "count": "_count", "share": "_share"}


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def extract_date(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.stem.removeprefix("report-"))


def metric_key(cfg: dict) -> str:
    kind = cfg["type"].split(":")[-1]
    if kind == "QuantileValue":
        base = f"q{round(cfg['quantile'] * 100)}"
    elif kind == "ColumnCount" and "column_type" in cfg:
        base = f"column_count_{cfg['column_type']}"
    else:
        base = NAMES.get(kind) or snake(kind)
    column = cfg.get("column")
    return f"{column}__{base}" if column else base


def json_to_row(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text())
    row = {"report_date": extract_date(json_path)}
    for metric in data["metrics"]:
        cfg = metric["config"]
        if cfg["type"].split(":")[-1] in SKIP:
            continue
        key, value = metric_key(cfg), metric["value"]
        if isinstance(value, dict):
            for stat, v in value.items():
                row.setdefault(key + STAT_SUFFIX[stat], float(v))  # type: ignore
        else:
            row.setdefault(key, float(value))  # type: ignore
    return pd.DataFrame([row])


def main() -> None:
    old = pd.read_parquet(OUT) if OUT.exists() else None
    done = set(old["report_date"]) if old is not None else set()
    paths = [
        p for p in sorted(JSON_DIR.glob("report-*.json")) if extract_date(p) not in done
    ]
    if not paths:
        print("nothing new to add")
        return
    new = pd.concat([json_to_row(p) for p in paths], ignore_index=True)
    df = (
        pd.concat([old, new], ignore_index=True) if old is not None else new
    )  # the append
    df = (
        df.drop_duplicates("report_date", keep="last")
        .sort_values("report_date")
        .reset_index(drop=True)
    )
    df.to_parquet(OUT)
    print(f"{OUT.name}: {df.shape[0]} reports x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
