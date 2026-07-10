"""
Build a real cohort from NHANES + linked mortality files.

Produces data/cohort.csv with the same columns the synthetic generator
produces, so the rest of the pipeline (Cox, Bayesian, app) is unchanged.

This is NOT run by default because the NHANES linked-mortality files must be
downloaded manually (they sit behind a click-through data-use agreement and
can't be scraped automatically). For the full download walkthrough, see
"Using real NHANES data" in the project README.

Quick reference once the files are in data/raw/:
    python -m src.build_cohort

Notes
-----
* Variable names drift between NHANES cycles (ALQ especially). The parser
  below is defensive and will warn about anything it cannot find rather than
  crash.
* NHANES is a complex survey with weights (WTMEC2YR). This script keeps the
  weight column so you can weight later, but the default models treat rows as
  an unweighted cohort. That biases absolute rates; it's a documented
  limitation, fine for a learning project.
* The mortality file's fixed-width column positions match the 2019
  public-use Linked Mortality File layout. If you use a newer release,
  double-check MORT_COLSPECS below against its codebook PDF.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw")
OUT_CSV = os.path.join(ROOT, "data", "cohort.csv")

# 2019 public-use LMF fixed-width layout (1-indexed cols in the codebook).
# (name, start, end) using 0-indexed half-open slices for pandas read_fwf colspecs.
MORT_COLSPECS = [
    ("seqn", 0, 6),
    ("eligstat", 14, 15),
    ("mortstat", 15, 16),
    ("permth_int", 42, 45),   # months from exam to death/censor (integer months)
    ("permth_exm", 45, 48),
]


def _read_xpt(pattern: str) -> pd.DataFrame | None:
    import fnmatch
    all_files = os.listdir(RAW) if os.path.isdir(RAW) else []
    files = sorted(
        os.path.join(RAW, f) for f in all_files
        if fnmatch.fnmatch(f.lower(), pattern.lower())
    )
    frames = []
    for fp in files:
        try:
            df = pd.read_sas(fp, format="xport")
            df.columns = [c.upper() for c in df.columns]
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"[nhanes] WARN could not read {fp}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _read_mortality() -> pd.DataFrame | None:
    files = sorted(glob.glob(os.path.join(RAW, "*MORT*PUBLIC*.dat")))
    if not files:
        files = sorted(glob.glob(os.path.join(RAW, "*mort*.dat")))
    frames = []
    colspecs = [(s, e) for _, s, e in MORT_COLSPECS]
    names = [n for n, _, _ in MORT_COLSPECS]
    for fp in files:
        try:
            df = pd.read_fwf(fp, colspecs=colspecs, names=names)
            frames.append(df)
        except Exception as e:  # noqa: BLE001
            print(f"[nhanes] WARN could not read mortality file {fp}: {e}")
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build() -> pd.DataFrame:
    demo = _read_xpt("DEMO*.XPT")
    smq = _read_xpt("SMQ*.XPT")
    bmx = _read_xpt("BMX*.XPT")
    alq = _read_xpt("ALQ*.XPT")
    bpx = _read_xpt("BPX*.XPT")
    mort = _read_mortality()

    missing = [n for n, d in
               [("DEMO", demo), ("SMQ", smq), ("BMX", bmx), ("BPX", bpx), ("MORT", mort)]
               if d is None]
    if missing:
        raise FileNotFoundError(
            f"Missing required inputs in {RAW}: {missing}. "
            "See 'Using real NHANES data' in the project README for download steps."
        )

    df = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].copy()
    df = df.rename(columns={"SEQN": "seqn", "RIDAGEYR": "age", "RIAGENDR": "sex_code"})
    df["sex"] = np.where(df["sex_code"] == 1, "male", "female")

    # smoking: current smoker if SMQ040 in {1,2} (every day / some days)
    if smq is not None and "SMQ040" in smq.columns:
        smk = smq[["SEQN", "SMQ040"]].rename(columns={"SEQN": "seqn"})
        df = df.merge(smk, on="seqn", how="left")
        df["smoking_status"] = np.where(df["SMQ040"].isin([1, 2]), "current", "never")
    else:
        print("[nhanes] WARN no SMQ040; defaulting smoking_status='never'")
        df["smoking_status"] = "never"

    # BMI
    bm = bmx[["SEQN", "BMXBMI"]].rename(columns={"SEQN": "seqn", "BMXBMI": "bmi"})
    df = df.merge(bm, on="seqn", how="left")

    # alcohol: heavy if >2 drinks/day proxy. Column names vary; try a few.
    df["alcohol_use"] = "moderate_or_none"
    if alq is not None:
        for col in ("ALQ130", "ALQ120Q", "ALQ151"):
            if col in alq.columns:
                a = alq[["SEQN", col]].rename(columns={"SEQN": "seqn"})
                df = df.merge(a, on="seqn", how="left")
                df["alcohol_use"] = np.where(df[col].fillna(0) >= 3, "heavy",
                                             "moderate_or_none")
                break
        else:
            print("[nhanes] WARN no recognized ALQ column; alcohol_use defaulted.")

    # systolic BP: average available BPXSY1..4
    sy_cols = [c for c in ("BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4") if c in bpx.columns]
    if sy_cols:
        bp = bpx[["SEQN"] + sy_cols].rename(columns={"SEQN": "seqn"})
        bp["systolic_bp"] = bp[sy_cols].mean(axis=1, skipna=True)
        df = df.merge(bp[["seqn", "systolic_bp"]], on="seqn", how="left")
    else:
        print("[nhanes] WARN no BPXSY columns; systolic_bp missing.")
        df["systolic_bp"] = np.nan

    # mortality
    m = mort[["seqn", "eligstat", "mortstat", "permth_exm"]].copy()
    m["eligstat"] = pd.to_numeric(m["eligstat"], errors="coerce")
    m["mortstat"] = pd.to_numeric(m["mortstat"], errors="coerce")
    df = df.merge(m, on="seqn", how="inner")
    df = df[df["eligstat"] == 1]                     # eligible for linkage
    df = df[df["mortstat"].isin([0, 1])]             # known vital status
    df["event"] = df["mortstat"].astype(int)
    df["duration_months"] = pd.to_numeric(df["permth_exm"], errors="coerce")

    # adults only, drop rows missing essentials
    df = df[df["age"] >= 18]
    keep = ["seqn", "age", "sex", "smoking_status", "bmi", "alcohol_use",
            "systolic_bp", "duration_months", "event"]
    df = df[keep].dropna(subset=["age", "bmi", "systolic_bp", "duration_months"])
    df = df[df["duration_months"] > 0].reset_index(drop=True)
    return df


def main():
    df = build()
    df.to_csv(OUT_CSV, index=False)
    print(f"[nhanes] Wrote {len(df)} rows -> {OUT_CSV}")
    print(f"[nhanes] Deaths: {int(df['event'].sum())} ({100*df['event'].mean():.1f}%)")
    print("[nhanes] Re-run the models: python -m src.cox_model && "
          "python -m src.bayesian_hazard")


if __name__ == "__main__":
    main()
