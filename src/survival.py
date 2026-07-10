"""
Individual survival curves with credible bands.

This is where the population table meets the person. The design mirrors the
original project idea: the SSA age-specific baseline hazard is the *prior* for
how mortality rises with age; the Bayesian posterior supplies distributions over
the lifestyle hazard ratios; we combine them and propagate the posterior
uncertainty all the way into the survival curve.

For a person of a given sex, age, and lifestyle covariates:
    baseline hazard at age a : mu0(a)         (SSA Gompertz-Makeham, sex-specific)
    lifestyle multiplier (per posterior draw s):
        m_s = exp( b_smoker*smoker + b_bmi*bmi_over25_per5
                   + b_alcohol*heavy_alcohol + b_sbp*sbp_per20 )
    adjusted hazard         : mu_s(a) = mu0(a) * m_s
    survival                : S_s(a)  = prod_{k<a} exp(-mu_s(k))
Across posterior draws we report the median survival curve, a 5-95% credible
band, a posterior distribution of remaining life expectancy, and P(survive to N).

Notes / honest limitations
--------------------------
* Sex is handled by picking the sex-specific SSA baseline, so the "male" hazard
  ratio from the model is intentionally NOT reapplied here.
* Age is the curve's x-axis, supplied by the richer SSA baseline, so the model's
  own "age" coefficient is not reused for prediction (it was only needed to
  de-confound the other coefficients during fitting).
* The SSA baseline is a population average that already contains smokers, high
  BMI, etc. Multiplying it by a full hazard ratio therefore slightly overstates
  effects versus a clean never-exposed baseline. Fine for this project; flagged
  so it is not mistaken for actuarial-grade pricing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, "artifacts")

# Covariates that multiply the age baseline (age + sex are handled by baseline).
LIFESTYLE = ["smoker", "bmi_over25_per5", "heavy_alcohol", "sbp_per20"]


@dataclass
class Person:
    age: int
    sex: str  # "male" | "female"
    smoker: bool
    bmi: float
    heavy_alcohol: bool
    systolic_bp: float

    def covariates(self) -> dict:
        return {
            "smoker": 1.0 if self.smoker else 0.0,
            "bmi_over25_per5": max(self.bmi - 25.0, 0.0) / 5.0,
            "heavy_alcohol": 1.0 if self.heavy_alcohol else 0.0,
            "sbp_per20": max(self.systolic_bp - 120.0, 0.0) / 20.0,
        }


# --------------------------------------------------------------------------- #
# Loading fitted artifacts
# --------------------------------------------------------------------------- #
def load_baseline_params():
    path = os.path.join(ART, "gm_params.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "artifacts/gm_params.json not found. Run `python -m src.ssa_baseline` first."
        )
    with open(path) as f:
        return json.load(f)


def gm_hazard(params_for_sex: dict, ages: np.ndarray) -> np.ndarray:
    A, B, G = params_for_sex["A"], params_for_sex["B"], params_for_sex["G"]
    return A + B * np.exp(G * ages)


def load_lifestyle_draws() -> np.ndarray:
    """
    Return an array of shape (n_samples, len(LIFESTYLE)) of log-hazard-ratio draws.

    Prefers the Bayesian posterior (artifacts/bayes_posterior.npz). If it is
    missing, falls back to the Cox point estimates as a single 'draw' so the app
    still works without the Bayesian step (just no credible bands).
    """
    npz_path = os.path.join(ART, "bayes_posterior.npz")
    if os.path.exists(npz_path):
        post = np.load(npz_path, allow_pickle=True)
        feats = list(post["features"])
        idx = [feats.index(name) for name in LIFESTYLE]
        return post["beta"][:, idx]  # (n_samples, 4)

    # fallback: Cox point estimates
    import pandas as pd

    cox_path = os.path.join(ART, "cox_summary.csv")
    if not os.path.exists(cox_path):
        raise FileNotFoundError(
            "Neither artifacts/bayes_posterior.npz nor artifacts/cox_summary.csv "
            "found. Run the Bayesian or Cox step first."
        )
    cox = pd.read_csv(cox_path, index_col=0)
    betas = np.array([np.log(cox.loc[name, "hazard_ratio"]) for name in LIFESTYLE])
    return betas.reshape(1, -1)  # single pseudo-draw


# --------------------------------------------------------------------------- #
# Survival computation
# --------------------------------------------------------------------------- #
def survival_matrix(person: Person, baseline_params: dict, lifestyle_draws: np.ndarray,
                    max_age: int = 119):
    """
    Returns (ages, S) where ages is 1-D from person.age..max_age and S has shape
    (n_samples, n_ages): a survival curve conditional on being alive at person.age
    for every posterior draw.
    """
    ages = np.arange(person.age, max_age + 1)
    mu0 = gm_hazard(baseline_params[person.sex], ages.astype(float))  # (n_ages,)

    cov = person.covariates()
    cov_vec = np.array([cov[name] for name in LIFESTYLE])           # (4,)
    log_mult = lifestyle_draws @ cov_vec                            # (n_samples,)
    mult = np.exp(log_mult)[:, None]                               # (n_samples, 1)

    mu = mu0[None, :] * mult                                       # (n_samples, n_ages)
    annual_surv = np.exp(-mu)
    # S at first age = 1; then cumulative product of survival through prior years
    S = np.cumprod(
        np.concatenate([np.ones((mu.shape[0], 1)), annual_surv[:, :-1]], axis=1),
        axis=1,
    )
    return ages, S


@dataclass
class SurvivalResult:
    ages: np.ndarray
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    life_expectancy_draws: np.ndarray  # remaining years per posterior draw

    @property
    def median_le(self) -> float:
        return float(np.median(self.life_expectancy_draws))

    @property
    def le_ci(self):
        return (float(np.percentile(self.life_expectancy_draws, 5)),
                float(np.percentile(self.life_expectancy_draws, 95)))

    def prob_survive_to(self, target_age: int):
        """Posterior distribution summary of P(alive at target_age)."""
        if target_age <= self.ages[0]:
            return 1.0, (1.0, 1.0)
        if target_age > self.ages[-1]:
            return 0.0, (0.0, 0.0)
        return None  # replaced below by compute_result closure


def compute_result(person: Person, baseline_params: dict, lifestyle_draws: np.ndarray,
                   max_age: int = 119) -> SurvivalResult:
    ages, S = survival_matrix(person, baseline_params, lifestyle_draws, max_age)
    median = np.median(S, axis=0)
    lower = np.percentile(S, 5, axis=0)
    upper = np.percentile(S, 95, axis=0)
    # remaining life expectancy per draw = sum of survival over future years
    le_draws = person.age + S.sum(axis=1) - 1.0  # -1 because S[0]=1 is "now"
    remaining = le_draws - person.age
    res = SurvivalResult(ages=ages, median=median, lower=lower, upper=upper,
                         life_expectancy_draws=remaining)
    # attach a closure-based prob_survive_to that uses the full S matrix
    def _p(target_age: int):
        if target_age <= ages[0]:
            return 1.0, (1.0, 1.0)
        if target_age > ages[-1]:
            return 0.0, (0.0, 0.0)
        j = int(target_age - ages[0])
        col = S[:, j]
        return float(np.median(col)), (float(np.percentile(col, 5)),
                                       float(np.percentile(col, 95)))
    res.prob_survive_to = _p  # type: ignore[assignment]
    return res


def estimate(person: Person, max_age: int = 119) -> SurvivalResult:
    """Convenience entry point: load artifacts and compute for one person."""
    baseline = load_baseline_params()
    draws = load_lifestyle_draws()
    return compute_result(person, baseline, draws, max_age)


if __name__ == "__main__":
    # quick self-test
    p = Person(age=50, sex="male", smoker=True, bmi=32, heavy_alcohol=True, systolic_bp=155)
    r = estimate(p)
    med_age = p.age + r.median_le
    lo, hi = r.le_ci
    print(f"50M smoker, BMI 32, heavy drinker, BP 155:")
    print(f"  remaining life expectancy: {r.median_le:.1f} yrs "
          f"(median age at death ~ {med_age:.0f}); 90% CI {lo:.1f}-{hi:.1f} yrs")
    for target in (70, 80, 90):
        m, (clo, chi) = r.prob_survive_to(target)
        print(f"  P(survive to {target}) = {m:.2f}  (90% CI {clo:.2f}-{chi:.2f})")

    p2 = Person(age=50, sex="male", smoker=False, bmi=23, heavy_alcohol=False, systolic_bp=115)
    r2 = estimate(p2)
    print(f"50M non-smoker, BMI 23, BP 115:")
    print(f"  remaining life expectancy: {r2.median_le:.1f} yrs "
          f"(median age at death ~ {p2.age + r2.median_le:.0f})")
