"""
Bayesian survival model (PyMC): Weibull proportional hazards.

Why Bayesian here: the Cox model gives point estimates + CIs and then throws the
uncertainty away. Instead we put priors on each log-hazard-ratio, centred on
published epidemiological estimates, and let the cohort likelihood update them.
The output is a full *posterior distribution* over hazard ratios, which we later
push through to get a posterior over an individual's survival curve -- credible
bands, not a single line.

Model
-----
For subject i with covariates x_i and linear predictor
    eta_i = b0 + x_i . beta            (beta are log hazard ratios)
we use a Weibull proportional-hazards form with shape alpha:
    hazard      h_i(t) = alpha * t^(alpha-1) * exp(eta_i)
    cum. hazard H_i(t) = t^alpha * exp(eta_i)
Right-censored log-likelihood for observed time t_i, event d_i:
    ll_i = d_i * (log alpha + (alpha-1) log t_i + eta_i) - t_i^alpha * exp(eta_i)

Time is handled in YEARS and age is centred at 55 to keep the sampler well-scaled.

Priors (informative, from meta-analyses; wide enough for the data to move them):
    age (per yr)        Normal(0.080, 0.030)   ~ HR 1.08 / year
    male               Normal(log 1.6, 0.30)
    smoker             Normal(log 2.0, 0.30)
    bmi_over25_per5    Normal(log 1.1, 0.15)
    heavy_alcohol      Normal(log 1.2, 0.25)
    sbp_per20          Normal(log 1.2, 0.20)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COHORT_CSV = os.path.join(ROOT, "data", "cohort.csv")
ART = os.path.join(ROOT, "artifacts")

AGE_CENTER = 55.0

# (name, prior mean on log-HR, prior sd). "age" is per-year and centred.
PRIORS = [
    ("age", 0.080, 0.030),
    ("male", np.log(1.6), 0.30),
    ("smoker", np.log(2.0), 0.30),
    ("bmi_over25_per5", np.log(1.1), 0.15),
    ("heavy_alcohol", np.log(1.2), 0.25),
    ("sbp_per20", np.log(1.2), 0.20),
]
FEATURES = [p[0] for p in PRIORS]


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float) - AGE_CENTER
    out["male"] = (df["sex"].astype(str).str.lower() == "male").astype(int)
    out["smoker"] = (df["smoking_status"].astype(str).str.lower() == "current").astype(int)
    out["bmi_over25_per5"] = np.maximum(df["bmi"].astype(float) - 25.0, 0.0) / 5.0
    out["heavy_alcohol"] = (df["alcohol_use"].astype(str).str.lower() == "heavy").astype(int)
    out["sbp_per20"] = np.maximum(df["systolic_bp"].astype(float) - 120.0, 0.0) / 20.0
    return out[FEATURES]


def build_model(df: pd.DataFrame):
    import pymc as pm
    import pytensor.tensor as pt

    X = build_design_matrix(df).to_numpy()
    t = np.clip(df["duration_months"].to_numpy(dtype=float) / 12.0, 1e-3, None)  # years
    d = df["event"].to_numpy(dtype=float)

    prior_mean = np.array([p[1] for p in PRIORS])
    prior_sd = np.array([p[2] for p in PRIORS])

    with pm.Model(coords={"feature": FEATURES}) as model:
        beta = pm.Normal("beta", mu=prior_mean, sigma=prior_sd, dims="feature")
        b0 = pm.Normal("b0", mu=-4.0, sigma=3.0)          # baseline log-scale
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.5)     # Weibull shape (>0)

        eta = b0 + pt.dot(X, beta)
        log_h = pt.log(alpha) + (alpha - 1.0) * np.log(t) + eta
        cum_h = pt.power(t, alpha) * pt.exp(eta)
        loglik = d * log_h - cum_h
        pm.Potential("survival_ll", loglik.sum())

        # track hazard ratios directly
        pm.Deterministic("hazard_ratio", pt.exp(beta), dims="feature")
    return model


def sample(df: pd.DataFrame, draws=1000, tune=1000, chains=2, cores=1, seed=7,
           target_accept=0.9):
    import pymc as pm

    model = build_model(df)
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=cores, random_seed=seed,
            target_accept=target_accept, progressbar=False,
        )
    return idata


def extract_posterior(idata) -> dict:
    """Flatten posterior draws into plain numpy arrays (chain*draw, ...)."""
    post = idata.posterior
    beta = post["beta"].stack(sample=("chain", "draw")).transpose("sample", "feature").values
    b0 = post["b0"].stack(sample=("chain", "draw")).values
    alpha = post["alpha"].stack(sample=("chain", "draw")).values
    return {
        "beta": np.asarray(beta),        # (n_samples, n_features)
        "b0": np.asarray(b0),            # (n_samples,)
        "alpha": np.asarray(alpha),      # (n_samples,)
        "features": np.array(FEATURES),
        "age_center": np.array([AGE_CENTER]),
    }


def summarize_from_posterior(post: dict) -> pd.DataFrame:
    """Build a hazard-ratio summary table directly from posterior draws.

    Done from raw arrays so it does not depend on a specific ArviZ version.
    """
    beta = post["beta"]              # (n_samples, n_features)
    hr = np.exp(beta)
    rows = []
    for j, name in enumerate(FEATURES):
        col = hr[:, j]
        rows.append({
            "feature": name,
            "hr_mean": col.mean(),
            "hr_median": np.median(col),
            "hr_sd": col.std(),
            "hr_ci_5%": np.percentile(col, 5),
            "hr_ci_95%": np.percentile(col, 95),
        })
    rows.append({
        "feature": "alpha (Weibull shape)",
        "hr_mean": post["alpha"].mean(),
        "hr_median": np.median(post["alpha"]),
        "hr_sd": post["alpha"].std(),
        "hr_ci_5%": np.percentile(post["alpha"], 5),
        "hr_ci_95%": np.percentile(post["alpha"], 95),
    })
    return pd.DataFrame(rows).set_index("feature")


def diagnostics(idata) -> str:
    """Best-effort convergence diagnostics; resilient to ArviZ API changes."""
    try:
        import arviz as az

        rhat_max = float(np.nanmax(np.asarray(az.rhat(idata)["hazard_ratio"].values)))
        ess_min = float(np.nanmin(np.asarray(az.ess(idata)["hazard_ratio"].values)))
        return (f"max r_hat = {rhat_max:.3f} (want < 1.01), "
                f"min ESS = {ess_min:.0f} (want > 400)")
    except Exception as e:  # noqa: BLE001
        return f"(diagnostics unavailable in this ArviZ version: {e!r})"


def main(draws=1000, tune=1000, chains=2, cores=1):
    os.makedirs(ART, exist_ok=True)
    df = pd.read_csv(COHORT_CSV)
    print(f"[bayes] Sampling Weibull PH model on {len(df)} rows "
          f"(draws={draws}, tune={tune}, chains={chains}, cores={cores})...")
    idata = sample(df, draws=draws, tune=tune, chains=chains, cores=cores)

    # Portable persistence: save the posterior draws the survival module needs.
    post = extract_posterior(idata)
    np.savez_compressed(os.path.join(ART, "bayes_posterior.npz"), **post)

    summ = summarize_from_posterior(post)
    summ.to_csv(os.path.join(ART, "bayes_summary.csv"))

    print("[bayes] Posterior hazard ratios (5-95% credible interval):")
    print(summ.round(3).to_string())
    print(f"[bayes] {diagnostics(idata)}")
    print(f"[bayes] Saved posterior draws -> {ART}/bayes_posterior.npz")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--cores", type=int, default=1,
                    help="Parallel sampling cores. Default 1 for portability; "
                         "bump to 2-4 on a normal machine for speed.")
    args = ap.parse_args()
    main(draws=args.draws, tune=args.tune, chains=args.chains, cores=args.cores)
