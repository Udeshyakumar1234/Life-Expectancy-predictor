"""
SSA baseline hazard.

Loads the Social Security Administration period life table (probability of death
within one year, qx, by age and sex), then fits a Gompertz-Makeham hazard curve:

    mu(x) = A + B * exp(G * x)

where A is the age-independent ("Makeham") component and B*exp(G*x) is the
exponentially-rising ("Gompertz") component that dominates at older ages.

The qx values in a life table are annual death *probabilities*. We convert them
to a continuous instantaneous hazard via  mu ~= -ln(1 - qx)  before fitting, then
fit in log-hazard space so the exponential tail does not dominate the least
squares objective.

Primary source (fetched at runtime):
    https://www.ssa.gov/oact/STATS/table4c6.html
If the network is unavailable, we fall back to a bundled 2023 snapshot in
data/ssa_life_table_2023_fallback.csv (real values, transcribed from the same URL).
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SSA_URL = "https://www.ssa.gov/oact/STATS/table4c6.html"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FALLBACK_CSV = os.path.join(ROOT, "data", "ssa_life_table_2023_fallback.csv")
CLEAN_CSV = os.path.join(ROOT, "data", "ssa_life_table_clean.csv")


# ----------------------------------------------------------------------------- #
# Loading
# ----------------------------------------------------------------------------- #
def _parse_ssa_html(html: str) -> pd.DataFrame:
    """Parse the SSA actuarial life table HTML into age / qx_male / qx_female."""
    # pandas.read_html finds every <table>; the life table is the one with ~120 rows.
    tables = pd.read_html(io.StringIO(html))
    life = None
    for t in tables:
        if t.shape[0] >= 100 and t.shape[1] >= 6:
            life = t
            break
    if life is None:
        raise ValueError("Could not locate the SSA life table in the page HTML.")

    # The table has a two-level header (Male/Female each with 3 sub-columns).
    # After flattening, columns are positional: age, m_qx, m_lives, m_ex, f_qx, f_lives, f_ex
    life = life.copy()
    life.columns = range(life.shape[1])
    df = pd.DataFrame(
        {
            "age": pd.to_numeric(life[0], errors="coerce"),
            "qx_male": pd.to_numeric(life[1], errors="coerce"),
            "qx_female": pd.to_numeric(life[4], errors="coerce"),
        }
    ).dropna(subset=["age"])
    df["age"] = df["age"].astype(int)
    df = df[(df["qx_male"] > 0) & (df["qx_female"] > 0)].reset_index(drop=True)
    return df


def load_ssa_table(prefer_download: bool = True, timeout: int = 20) -> pd.DataFrame:
    """
    Return a DataFrame with columns [age, qx_male, qx_female].

    Tries to download the current SSA table first; on any failure, uses the
    bundled real snapshot so the pipeline always runs.
    """
    if prefer_download:
        try:
            import requests

            resp = requests.get(SSA_URL, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df = _parse_ssa_html(resp.text)
            if len(df) >= 100:
                df.to_csv(CLEAN_CSV, index=False)
                print(f"[ssa] Downloaded live SSA table ({len(df)} ages) -> {CLEAN_CSV}")
                return df
            raise ValueError("Parsed table too short; using fallback.")
        except Exception as e:  # noqa: BLE001 - we deliberately catch everything
            print(f"[ssa] Live download failed ({e!r}); using bundled snapshot.")

    df = pd.read_csv(FALLBACK_CSV)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"[ssa] Loaded bundled SSA snapshot ({len(df)} ages) -> {CLEAN_CSV}")
    return df


# ----------------------------------------------------------------------------- #
# Gompertz-Makeham fit
# ----------------------------------------------------------------------------- #
def qx_to_hazard(qx: np.ndarray) -> np.ndarray:
    """Convert annual death probability to continuous instantaneous hazard."""
    qx = np.clip(qx, 1e-9, 0.999999)
    return -np.log(1.0 - qx)


def gompertz_makeham(x, A, B, G):
    return A + B * np.exp(G * x)


@dataclass
class GMParams:
    sex: str
    A: float
    B: float
    G: float

    def hazard(self, age):
        age = np.asarray(age, dtype=float)
        return gompertz_makeham(age, self.A, self.B, self.G)

    def survival(self, ages):
        """
        Discrete survival S(age) from the fitted hazard, integrating year by year.
        Returns S at each integer age in `ages` (S at the first age == 1.0).
        """
        ages = np.asarray(ages, dtype=float)
        mu = self.hazard(ages)
        # annual survival prob per year ~ exp(-mu); cumulative product
        annual_surv = np.exp(-mu)
        S = np.cumprod(np.concatenate([[1.0], annual_surv[:-1]]))
        return S

    def life_expectancy(self, from_age: int = 0, max_age: int = 119) -> float:
        """Expected remaining years of life from `from_age` using the fitted curve."""
        ages = np.arange(from_age, max_age + 1)
        S = self.survival(ages)
        S = S / S[0]  # condition on being alive at from_age
        # e_x ~= sum of survival over future years (+0.5 for mid-year deaths)
        return float(S[1:].sum() + 0.5)


def fit_gompertz_makeham(df: pd.DataFrame, sex: str, fit_max_age: int = 95) -> GMParams:
    """
    Fit Gompertz-Makeham to one sex. We fit up to `fit_max_age` because
    extreme-old-age mortality is deliberately extrapolated by SSA and does not
    follow the Gompertz law; including it distorts the fit for typical ages.
    """
    col = "qx_male" if sex == "male" else "qx_female"
    sub = df[df["age"] <= fit_max_age]
    x = sub["age"].to_numpy(dtype=float)
    y = qx_to_hazard(sub[col].to_numpy(dtype=float))
    log_y = np.log(y)

    def log_gm(x, A, B, G):
        return np.log(gompertz_makeham(x, A, B, G))

    p0 = [1e-4, 2e-5, 0.09]  # sensible starting point for human mortality
    bounds = ([0, 0, 0], [1e-1, 1e-1, 1.0])
    popt, _ = curve_fit(log_gm, x, log_y, p0=p0, bounds=bounds, maxfev=100000)
    return GMParams(sex=sex, A=float(popt[0]), B=float(popt[1]), G=float(popt[2]))


def fit_baseline(prefer_download: bool = True) -> dict:
    """Load the SSA table and fit both sexes. Returns dict of GMParams by sex."""
    df = load_ssa_table(prefer_download=prefer_download)
    params = {sex: fit_gompertz_makeham(df, sex) for sex in ("male", "female")}
    for sex, p in params.items():
        e0 = p.life_expectancy(0)
        e65 = p.life_expectancy(65)
        print(f"[ssa] {sex:6s}  A={p.A:.2e} B={p.B:.2e} G={p.G:.4f}  "
              f"e0={e0:.1f}  e65={e65:.1f}")
    return {"table": df, "params": params}


if __name__ == "__main__":
    import json

    out = fit_baseline(prefer_download=True)
    params_json = {sex: asdict(p) for sex, p in out["params"].items()}
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    with open(os.path.join(ROOT, "artifacts", "gm_params.json"), "w") as f:
        json.dump(params_json, f, indent=2)
    print("[ssa] Saved fitted parameters -> artifacts/gm_params.json")
