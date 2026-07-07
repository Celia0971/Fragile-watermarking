"""
Dataset Preparation — Fragile Watermarking Experiments
=======================================================

Produces four cleaned CSV files in datasets/processed/, each with:
  - First column:  id  (synthetic integer primary key, 1-based, immutable)
  - Remaining cols: dataset-specific attributes (see per-dataset notes below)
  - No index column in the output CSV

Output files:
  FCT.csv    — Forest Cover Type    (numerical)
  Adult.csv  — Adult Income         (categorical)
  AGNews.csv — AG News              (textual)
  Bank.csv   — Bank Marketing       (mixed: numerical + categorical)

──────────────────────────────────────────────────────────────────────────────
Preprocessing summary
──────────────────────────────────────────────────────────────────────────────

FCT  (Forest Cover Type)
  Source : datasets/raw/covtype.data.gz  (UCI, 581,012 rows, 55 columns, no header)
  Columns: First 10 continuous cartographic attributes retained:
             Elevation, Aspect, Slope,
             Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology,
             Horizontal_Distance_To_Roadways,
             Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
             Horizontal_Distance_To_Fire_Points
           Wilderness area (cols 11-14), soil type (cols 15-54), and
           Cover_Type (col 55) are dropped.
  Missing: None in the source file.
  Final  : 581,012 rows × 11 cols (id + 10 numerical).

Adult  (Adult / Census Income)
  Source : datasets/raw/adult.data (32,561) + adult.test (16,281) = 48,842 rows.
           All rows kept; '?' retained as a regular categorical value (same
           convention as Bank's 'unknown'). Test label trailing '.' normalized.
  Columns: Nine categorical attributes retained:
             workclass, education, marital_status, occupation,
             relationship, race, sex, native_country, income
           Dropped: age, fnlwgt, education_num, capital_gain,
                    capital_loss, hours_per_week  (all numerical)
  Missing: Rows containing '?' (missing marker) are removed.
           Affected cols: workclass (1,836), occupation (1,843),
                          native_country (583).
           Overlapping rows removed once: ~30,162 rows retained.
  Final  : ~30,162 rows × 10 cols (id + 9 categorical).

AGNews  (AG News Corpus)
  Source : datasets/raw/ag_news_train.csv  (full training split, 120,000 rows)
           First row is a header (Class Index, Title, Description) and is skipped.
  Columns: class_label (integer 1–4: World/Sports/Business/Sci_Tech),
           title (short text), description (longer text)
  Missing: None; source is complete.
  Final  : 120,000 rows × 4 cols (id + class_label + title + description).

Bank  (UCI Bank Marketing — bank-additional-full.csv)
  Source : datasets/raw/bank-additional.zip →
           bank-additional/bank-additional-full.csv  (41,188 rows, 21 cols)
  Columns: All 20 input attributes + y (subscription outcome) retained:
             Numerical (10): age, duration, campaign, pdays, previous,
                             emp.var.rate, cons.price.idx, cons.conf.idx,
                             euribor3m, nr.employed
             Categorical (11): job, marital, education, default, housing,
                               loan, contact, month, day_of_week, poutcome, y
  Missing: The value "unknown" appears in some categorical columns (job 0.8%,
           marital 0.2%, education 4.2%, default 20.9%, housing 2.4%, loan 2.4%).
           "unknown" is treated as a regular attribute value — NO rows removed,
           NO columns dropped.
  Final  : 41,188 rows × 22 cols (id + 21 attributes).

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────

    python prepare_datasets.py --dataset FCT
    python prepare_datasets.py --dataset Adult
    python prepare_datasets.py --dataset AGNews
    python prepare_datasets.py --dataset Bank
    python prepare_datasets.py --dataset all   # all four

Download instructions are printed when a required raw file is not found.
"""

import argparse
import gzip
import shutil
import zipfile
from pathlib import Path

import pandas as pd

RAW_DIR  = Path(__file__).parent / "raw"
PROC_DIR = Path(__file__).parent / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ── Shared helper ─────────────────────────────────────────────────────────────

def add_pk_and_save(df: pd.DataFrame, name: str) -> Path:
    """Prepend id column (1-based), save to processed/, print summary."""
    df = df.reset_index(drop=True).copy()
    df.insert(0, "id", range(1, len(df) + 1))
    out = PROC_DIR / name
    df.to_csv(out, index=False)
    print(f"  → {out.name}  ({len(df):,} rows × {len(df.columns)} cols)")
    return out


# ── FCT: Forest Cover Type ────────────────────────────────────────────────────

FCT_COLS = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

def prepare_fct() -> Path:
    raw    = RAW_DIR / "covtype.data"
    raw_gz = RAW_DIR / "covtype.data.gz"

    if raw_gz.exists() and not raw.exists():
        print("  Decompressing covtype.data.gz ...")
        with gzip.open(raw_gz, "rb") as f_in, open(raw, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    if not raw.exists():
        print("\n[FCT] Raw file not found. Download:")
        print("  https://archive.ics.uci.edu/dataset/31/covertype")
        print(f"  Place covtype.data.gz in: {RAW_DIR}/")
        return None

    print("  Loading covtype.data (581,012 rows, no header) ...")
    all_cols = [f"c{i}" for i in range(1, 56)]
    df_raw = pd.read_csv(raw, header=None, names=all_cols)

    df = df_raw.iloc[:, :10].copy()
    df.columns = FCT_COLS

    return add_pk_and_save(df, "FCT.csv")


# ── Adult: Census Income ──────────────────────────────────────────────────────

ADULT_ALL_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]
ADULT_KEEP_COLS = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country", "income",
]

def prepare_adult() -> Path:
    # Use BOTH the training (adult.data) and test (adult.test) splits, and keep
    # ALL rows: the '?' missing marker is retained as a regular categorical value
    # (same convention as Bank's 'unknown'). '?' is a plain string to every method
    # that runs on Adult (Proposed/B0 content digest, B1 tuple hash, B7 char
    # frequency), so no rows need to be dropped.
    train = RAW_DIR / "adult.data"
    test  = RAW_DIR / "adult.test"
    if not train.exists():
        print("\n[Adult] Raw file not found. Download:")
        print("  https://archive.ics.uci.edu/dataset/2/adult")
        print(f"  Place adult.data (and adult.test) in: {RAW_DIR}/")
        return None

    parts = []
    print("  Loading adult.data (training split, 32,561 rows) ...")
    parts.append(pd.read_csv(train, header=None, names=ADULT_ALL_COLS,
                             skipinitialspace=True))   # no na_values: keep '?'
    if test.exists():
        # adult.test has a junk first line ('|1x3 Cross validator') -> skiprows=1
        print("  Loading adult.test (test split, 16,281 rows) ...")
        parts.append(pd.read_csv(test, header=None, names=ADULT_ALL_COLS,
                                 skipinitialspace=True, skiprows=1))
    else:
        print("  [warn] adult.test not found — using training split only.")

    df = pd.concat(parts, ignore_index=True)
    df = df[ADULT_KEEP_COLS].copy()
    # Normalize the label: test rows carry a trailing '.' ('<=50K.' vs '<=50K').
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    print(f"  Kept all rows ('?' retained as a value) → {len(df):,} rows")

    return add_pk_and_save(df, "Adult.csv")


# ── AGNews: AG News Corpus ────────────────────────────────────────────────────

def prepare_agnews() -> Path:
    raw = RAW_DIR / "ag_news_train.csv"
    if not raw.exists():
        print("\n[AGNews] Raw file not found. Download:")
        print("  kaggle datasets download -d amananandrai/ag-news-classification-dataset")
        print("  Rename train.csv → ag_news_train.csv")
        print(f"  Place in: {RAW_DIR}/")
        return None

    print("  Loading ag_news_train.csv (120,000 rows) ...")
    df = pd.read_csv(raw, header=None, skiprows=1,
                     names=["class_label", "title", "description"])

    for col in ["title", "description"]:
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    print(f"  Loaded {len(df):,} rows")
    return add_pk_and_save(df, "AGNews.csv")


# ── Bank: UCI Bank Marketing ──────────────────────────────────────────────────

def prepare_bank() -> Path:
    zip_path = RAW_DIR / "bank-additional.zip"
    csv_name = "bank-additional/bank-additional-full.csv"

    if not zip_path.exists():
        print("\n[Bank] Raw file not found. Download:")
        print("  https://archive.ics.uci.edu/dataset/222/bank+marketing")
        print(f"  Place bank-additional.zip in: {RAW_DIR}/")
        return None

    print("  Extracting bank-additional-full.csv from zip ...")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, sep=";")

    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
    print(f"  'unknown' values kept as-is (treated as regular attribute values)")

    return add_pk_and_save(df, "Bank.csv")


# ── Main ──────────────────────────────────────────────────────────────────────

DATASETS = {
    "FCT":    prepare_fct,
    "Adult":  prepare_adult,
    "AGNews": prepare_agnews,
    "Bank":   prepare_bank,
}

def main():
    parser = argparse.ArgumentParser(description="Prepare experiment datasets")
    parser.add_argument(
        "--dataset", default="all",
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to prepare (default: all)"
    )
    args = parser.parse_args()

    todo = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    results = {}
    for name in todo:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        results[name] = DATASETS[name]()

    print(f"\n{'='*60}")
    print("Summary:")
    for name, path in results.items():
        if path:
            print(f"  ✓  {name:8s} → {path}")
        else:
            print(f"  ✗  {name:8s} — raw file missing (see instructions above)")


if __name__ == "__main__":
    main()
