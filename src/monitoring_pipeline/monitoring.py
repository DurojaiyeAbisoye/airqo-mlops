import json
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_DIR = REPO_ROOT / "evidently_artifacts/json_reports"
OUT = REPO_ROOT / "src/feature_repo/data/monitoring_metrics.parquet"
SKIP = ("CorrelationMatrix",)  # the only metric we do not keep


def extract_date(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.stem.removeprefix("report-"))


def json_to_row(json_path: Path) -> pd.DataFrame:
    with open(json_path) as f:
        data = json.load(f)
    metrics = pd.DataFrame(data["metrics"])[["metric_name", "value"]]
    metrics = metrics[~metrics.metric_name.str.startswith(SKIP)].reset_index(drop=True)

    values = pd.json_normalize(
        [v if isinstance(v, dict) else {"value": v} for v in metrics["value"]]
    )
    df = pd.concat([metrics["metric_name"], values], axis=1)
    long = df.melt(id_vars="metric_name", var_name="stat", value_name="number").dropna(
        subset=["number"]
    )
    long["key"] = long["metric_name"] + "|" + long["stat"]
    row = pd.DataFrame([long.set_index("key")["number"].astype(float).to_dict()])
    row.insert(0, "report_date", extract_date(json_path))
    return row


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
