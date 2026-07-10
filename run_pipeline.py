"""
One-command pipeline runner.

Runs, in order:
    1. SSA baseline        -> artifacts/gm_params.json
    2. Cohort              -> data/cohort.csv (synthetic unless you built real NHANES)
    3. Cox model           -> artifacts/cox_*.{csv,pkl,png}
    4. Bayesian model      -> artifacts/bayes_posterior.npz, bayes_summary.csv

Then launch the app with:  streamlit run app.py

Examples
--------
    python run_pipeline.py                 # full run, sensible defaults
    python run_pipeline.py --quick         # fewer MCMC draws (faster)
    python run_pipeline.py --skip-bayes    # Cox only (no PyMC needed)
    python run_pipeline.py --cores 4       # parallel MCMC on a multi-core box
"""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
COHORT_CSV = os.path.join(ROOT, "data", "cohort.csv")


def step(msg):
    print("\n" + "=" * 70 + f"\n>>> {msg}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Fewer MCMC draws.")
    ap.add_argument("--skip-bayes", action="store_true", help="Skip PyMC step.")
    ap.add_argument("--rebuild-cohort", action="store_true",
                    help="Regenerate synthetic cohort even if data/cohort.csv exists.")
    ap.add_argument("--cores", type=int, default=1, help="MCMC cores (default 1).")
    args = ap.parse_args()

    t0 = time.time()

    step("Step 1/4  SSA baseline (Gompertz-Makeham)")
    from src import ssa_baseline
    out = ssa_baseline.fit_baseline(prefer_download=True)
    import json
    from dataclasses import asdict
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    with open(os.path.join(ROOT, "artifacts", "gm_params.json"), "w") as f:
        json.dump({s: asdict(p) for s, p in out["params"].items()}, f, indent=2)

    step("Step 2/4  Cohort data")
    if args.rebuild_cohort or not os.path.exists(COHORT_CSV):
        from src import synthetic_cohort
        synthetic_cohort.main()
    else:
        print(f"[cohort] Using existing {COHORT_CSV} "
              "(delete it or pass --rebuild-cohort to regenerate).")

    step("Step 3/4  Cox proportional-hazards model (validation anchor)")
    from src import cox_model
    cox_model.main()

    if args.skip_bayes:
        step("Step 4/4  Bayesian model SKIPPED (--skip-bayes)")
    else:
        step("Step 4/4  Bayesian Weibull PH model (PyMC)")
        try:
            from src import bayesian_hazard
            if args.quick:
                bayesian_hazard.main(draws=500, tune=500, chains=2, cores=args.cores)
            else:
                bayesian_hazard.main(draws=1500, tune=1500, chains=4, cores=args.cores)
        except ImportError as e:
            print(f"[bayes] PyMC not available ({e}). Skipping Bayesian step. "
                  "The app will fall back to Cox point estimates.")

    dt = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Pipeline complete in {dt:.0f}s. Artifacts are in ./artifacts/")
    print("Launch the interactive app with:\n    streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
