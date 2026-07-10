"""
Synthetic cohort generator.

Produces an NHANES-like survival dataset so the whole modelling pipeline runs
end-to-end without needing the (manually-downloaded) real NHANES linked-mortality
files. The synthetic data is drawn from a *known* ground-truth hazard model, which
is genuinely useful: after fitting, you can check that the Cox / Bayesian models
recover the hazard ratios you put in. That is a real validation step, not a toy.

Risk factors and their (log) hazard ratios are loosely based on published
epidemiological meta-analyses:
    - current smoker vs never:        HR ~ 2.3
    - each +5 kg/m^2 of BMI over 25:  HR ~ 1.10
    - heavy alcohol use:              HR ~ 1.20
    - each +20 mmHg systolic BP:      HR ~ 1.25
Correlations between risk factors are induced deliberately (smokers drink more,
higher BMI tracks higher BP) so that naive additive penalty schemes would
double-count -- exactly the failure mode a proper hazard model avoids.

Columns produced (match build_cohort.py so the two are interchangeable):
    seqn, age, sex, smoking_status, bmi, alcohol_use, systolic_bp,
    duration_months, event
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_CSV = os.path.join(ROOT, "data", "cohort.csv")

# Ground-truth log hazard ratios (kept in one place so tests can import them).
TRUE_BETAS = {
    "smoker": np.log(2.3),          # current smoker vs never/former
    "bmi_over25_per5": np.log(1.10),  # per 5 units of BMI above 25
    "heavy_alcohol": np.log(1.20),
    "sbp_per20": np.log(1.25),      # per 20 mmHg systolic BP above 120
    "male": np.log(1.6),            # male vs female
}


def generate(n: int = 8000, seed: int = 7, follow_up_years: float = 10.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- baseline demographics ------------------------------------------------
    age = rng.uniform(30, 80, n)
    male = rng.integers(0, 2, n)

    # --- correlated risk factors ---------------------------------------------
    # latent "unhealthy lifestyle" factor drives smoking, alcohol, and (weakly) BMI
    lifestyle = rng.normal(0, 1, n)

    p_smoke = 1 / (1 + np.exp(-(-1.2 + 0.9 * lifestyle + 0.15 * male)))
    smoker = (rng.uniform(0, 1, n) < p_smoke).astype(int)

    p_heavy = 1 / (1 + np.exp(-(-1.6 + 0.8 * lifestyle + 0.3 * male)))
    heavy_alcohol = (rng.uniform(0, 1, n) < p_heavy).astype(int)

    bmi = 27 + 0.6 * lifestyle + 0.03 * (age - 55) + rng.normal(0, 4, n)
    bmi = np.clip(bmi, 15, 55)

    # systolic BP rises with age and BMI (so it correlates with the rest)
    sbp = 110 + 0.45 * (age - 40) + 0.6 * (bmi - 25) + rng.normal(0, 12, n)
    sbp = np.clip(sbp, 85, 210)

    # --- construct the true linear predictor ---------------------------------
    lp = (
        TRUE_BETAS["smoker"] * smoker
        + TRUE_BETAS["bmi_over25_per5"] * np.maximum(bmi - 25, 0) / 5.0
        + TRUE_BETAS["heavy_alcohol"] * heavy_alcohol
        + TRUE_BETAS["sbp_per20"] * np.maximum(sbp - 120, 0) / 20.0
        + TRUE_BETAS["male"] * male
    )

    # --- age-dependent baseline hazard (Gompertz) + covariate multiplier -----
    # baseline annual hazard rises with age; scale chosen for a realistic ~7-9%
    # death rate over 10 years in this middle-aged-to-elderly sample.
    baseline_annual = 0.0006 * np.exp(0.085 * (age - 30))
    annual_hazard = baseline_annual * np.exp(lp)

    # time-to-death ~ Exponential(annual_hazard) in years
    t_death = rng.exponential(1.0 / np.clip(annual_hazard, 1e-6, None))

    event = (t_death <= follow_up_years).astype(int)
    duration_years = np.minimum(t_death, follow_up_years)

    df = pd.DataFrame(
        {
            "seqn": np.arange(1, n + 1),
            "age": np.round(age, 1),
            "sex": np.where(male == 1, "male", "female"),
            "smoking_status": np.where(smoker == 1, "current", "never"),
            "bmi": np.round(bmi, 1),
            "alcohol_use": np.where(heavy_alcohol == 1, "heavy", "moderate_or_none"),
            "systolic_bp": np.round(sbp, 0).astype(int),
            "duration_months": np.round(duration_years * 12.0, 1),
            "event": event,
        }
    )
    return df


def main():
    df = generate()
    df.to_csv(OUT_CSV, index=False)
    print(f"[synthetic] Wrote {len(df)} rows -> {OUT_CSV}")
    print(f"[synthetic] Deaths (event=1): {df['event'].sum()} "
          f"({100 * df['event'].mean():.1f}%)")
    print(f"[synthetic] Median follow-up: {df['duration_months'].median():.0f} months")
    print("[synthetic] Ground-truth hazard ratios embedded in the data:")
    for k, v in TRUE_BETAS.items():
        print(f"            {k:18s} HR = {np.exp(v):.2f}")


if __name__ == "__main__":
    main()
