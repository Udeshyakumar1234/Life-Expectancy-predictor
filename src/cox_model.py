"""
Frequentist Cox proportional-hazards model (lifelines).

This is the validation anchor for the whole project: it is fast, well understood,
and its hazard ratios should come out with sensible signs and magnitudes. If the
Bayesian model later disagrees wildly with these estimates, that is a bug signal,
not a modelling insight.

Reads data/cohort.csv (synthetic by default, or real NHANES if you ran
build_cohort.py) and produces:
    artifacts/cox_summary.csv     hazard ratios + CIs
    artifacts/cox_model.pkl       fitted model (for the app / survival curves)
    artifacts/cox_validation.png  two example survival curves
"""
from __future__ import annotations

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COHORT_CSV = os.path.join(ROOT, "data", "cohort.csv")
ART = os.path.join(ROOT, "artifacts")

# The design-matrix columns the model is trained on. The app must build the
# same columns for a new person, so this list is the single source of truth.
FEATURES = ["age", "male", "smoker", "bmi_over25_per5", "heavy_alcohol", "sbp_per20"]


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw cohort columns into the numeric covariates the model uses."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float)
    out["male"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    out["smoker"] = (df["smoking_status"].astype(str).str.lower() == "current").astype(int)
    out["bmi_over25_per5"] = np.maximum(df["bmi"].astype(float) - 25.0, 0.0) / 5.0
    out["heavy_alcohol"] = (df["alcohol_use"].astype(str).str.lower() == "heavy").astype(int)
    out["sbp_per20"] = np.maximum(df["systolic_bp"].astype(float) - 120.0, 0.0) / 20.0
    return out


def load_cohort() -> pd.DataFrame:
    if not os.path.exists(COHORT_CSV):
        raise FileNotFoundError(
            f"{COHORT_CSV} not found. Run `python -m src.synthetic_cohort` "
            "(or build_cohort.py for real NHANES data) first."
        )
    return pd.read_csv(COHORT_CSV)


def fit_cox(df: pd.DataFrame) -> CoxPHFitter:
    X = build_design_matrix(df)
    X["duration"] = df["duration_months"].astype(float)
    X["event"] = df["event"].astype(int)

    cph = CoxPHFitter()
    cph.fit(X, duration_col="duration", event_col="event")
    return cph


def _person_row(age, sex, smoker, bmi, heavy_alcohol, sbp) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "age": float(age),
            "male": 1 if sex == "male" else 0,
            "smoker": 1 if smoker else 0,
            "bmi_over25_per5": max(bmi - 25.0, 0.0) / 5.0,
            "heavy_alcohol": 1 if heavy_alcohol else 0,
            "sbp_per20": max(sbp - 120.0, 0.0) / 20.0,
        }]
    )[FEATURES]


def validation_plot(cph: CoxPHFitter, path: str):
    """Overlay survival curves for a healthy vs high-risk 50-year-old male."""
    healthy = _person_row(50, "male", smoker=False, bmi=23, heavy_alcohol=False, sbp=115)
    risky = _person_row(50, "male", smoker=True, bmi=33, heavy_alcohol=True, sbp=160)

    fig, ax = plt.subplots(figsize=(8, 5))
    cph.predict_survival_function(healthy).plot(ax=ax, label="50M non-smoker, BMI 23, BP 115")
    cph.predict_survival_function(risky).plot(ax=ax, label="50M smoker, BMI 33, BP 160")
    ax.set_xlabel("Follow-up time (months)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Cox model: example survival curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    os.makedirs(ART, exist_ok=True)
    df = load_cohort()
    cph = fit_cox(df)

    summary = cph.summary[["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    summary = summary.rename(columns={"exp(coef)": "hazard_ratio"})
    summary.to_csv(os.path.join(ART, "cox_summary.csv"))

    print("[cox] Hazard ratios (exp coef) with 95% CIs:")
    print(summary.to_string())

    with open(os.path.join(ART, "cox_model.pkl"), "wb") as f:
        pickle.dump(cph, f)
    validation_plot(cph, os.path.join(ART, "cox_validation.png"))
    print(f"[cox] Saved model, summary, and validation plot to {ART}/")


if __name__ == "__main__":
    main()
