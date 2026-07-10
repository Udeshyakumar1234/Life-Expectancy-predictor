# Life-Expectancy Estimator

A survival-analysis project that estimates a person's remaining life
expectancy **as a range with uncertainty, not a single guaranteed number** —
combining the official SSA age-mortality baseline with lifestyle hazard
ratios learned from real health-survey data, and propagating Bayesian
posterior uncertainty into a survival curve. Ships with an interactive
"what-if" web app.

This is an educational project — **not medical, actuarial, or financial
advice.**

---

## Getting started

### 1. Get the code

Clone the repo (or download and unzip it), then open a terminal inside the
project folder:

```bash
git clone <this-repo-url>
cd life-expectancy-model
```

### 2. Install the Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>Windows without bash</summary>

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

</details>

### 3. Fit the models

```bash
python run_pipeline.py            # add --quick for a faster MCMC run
                                   # add --skip-bayes if you don't want PyMC
```

This runs, in order: SSA baseline fit → cohort data (synthetic by default) →
Cox model → Bayesian model. Takes a couple of minutes; PyMC sampling is the
slow part (`--quick` cuts that down, `--cores 4` parallelizes it on a
multi-core machine).

### 4. Launch the app

```bash
streamlit run app.py
```

Opens an interactive page where you enter age, sex, smoking, BMI, blood
pressure, and alcohol use, and see an estimated survival curve.

---

## What it actually does

1. **SSA baseline (the prior).** Downloads the current Social Security period
   life table (falls back to a bundled 2023 snapshot if offline) and fits a
   **Gompertz-Makeham** hazard curve, separately for each sex — capturing how
   mortality rises with age. Recovered life expectancies match SSA's
   published figures (e.g. male e₀ ≈ 75.7, female ≈ 81.1).

2. **Cohort (the evidence).** By default a **synthetic** cohort drawn from a
   *known* ground-truth hazard model — genuinely useful, because after
   fitting you can confirm the models recover the hazard ratios that were
   baked in. Swap in real NHANES data any time (see below).

3. **Cox model (the validation anchor).** A fast frequentist Cox
   proportional-hazards fit (`lifelines`). Its hazard ratios should have
   sensible signs and magnitudes; if the Bayesian model later disagrees
   wildly, that's a bug signal.

4. **Bayesian model (the fun part).** A Weibull proportional-hazards model in
   **PyMC**, with informative priors on each log-hazard-ratio centered on
   published meta-analysis values, updated by the cohort likelihood. Output
   is a full **posterior over hazard ratios**.

5. **Individual survival curve.** For a given person, the SSA age baseline is
   multiplied by the posterior lifestyle hazard-ratio draws, giving a
   posterior distribution over the whole survival curve — reported in the
   app as an estimated age, a plausible range, and the chance of reaching a
   target age.

---

## Project structure

```
life-expectancy-model/
├── setup.sh                 one-time environment setup
├── requirements.txt
├── run_pipeline.py          runs the SSA -> cohort -> Cox -> Bayesian steps
├── app.py                   Streamlit what-if interface
├── data/
│   ├── ssa_life_table_2023_fallback.csv   bundled SSA snapshot (offline fallback)
│   ├── cohort.csv           generated (synthetic by default; gitignored)
│   └── raw/                 put real NHANES files here (see below)
├── artifacts/               all fitted outputs land here (gitignored)
└── src/
    ├── ssa_baseline.py      download/parse SSA table + Gompertz-Makeham fit
    ├── synthetic_cohort.py  generate NHANES-like data with known ground truth
    ├── build_cohort.py      build a REAL cohort from NHANES + mortality files
    ├── cox_model.py         frequentist Cox PH (validation anchor)
    ├── bayesian_hazard.py   Bayesian Weibull PH in PyMC
    └── survival.py          person survival curve + credible bands
```

---

## Using real NHANES data

The default cohort is synthetic so the project runs immediately. To use real
data instead:

**a. Download the survey files.** For each NHANES cycle you want
(2011-2012, 2013-2014, 2015-2016 recommended), download these `.XPT` files
from [wwwn.cdc.gov/nchs/nhanes](https://wwwn.cdc.gov/nchs/nhanes/):

| File | Contents | Key columns |
|---|---|---|
| `DEMO_<X>.XPT` | Demographics | age `RIDAGEYR`, sex `RIAGENDR`, id `SEQN` |
| `SMQ_<X>.XPT`  | Smoking | `SMQ020` (ever), `SMQ040` (now) |
| `BMX_<X>.XPT`  | Body measures | BMI `BMXBMI` |
| `ALQ_<X>.XPT`  | Alcohol | varies by cycle |
| `BPX_<X>.XPT`  | Blood pressure | systolic `BPXSY1`–`BPXSY4` |

`<X>` is the cycle letter: 2011–12 = `G`, 2013–14 = `H`, 2015–16 = `I`.

**b. Download the linked mortality file(s)** from the
[NCHS data-linkage page](https://www.cdc.gov/nchs/data-linkage/mortality-public.htm).
These are fixed-width `.dat` files named like
`NHANES_2011_2012_MORT_2019_PUBLIC.dat`.

**c. Put everything in `data/raw/`, then run:**

```bash
python -m src.build_cohort      # writes a real data/cohort.csv (same columns)
python -m src.cox_model
python -m src.bayesian_hazard
streamlit run app.py
```

Nothing else changes — the real cohort has the same schema as the synthetic
one.

> **Note:** don't run `python run_pipeline.py --rebuild-cohort` after this —
> that flag regenerates the *synthetic* cohort and will overwrite your real
> one. Once you've built the real cohort, re-run the individual steps above
> instead of the full pipeline script.

---

## Known limitations (deliberate, documented)

- The SSA baseline is a **population average** (already contains smokers,
  high BMI, etc.), so multiplying it by full hazard ratios slightly
  overstates effects versus a clean never-exposed baseline.
- NHANES **survey weights** are carried through the build but the models
  treat rows as an unweighted cohort, which biases absolute rates.
- Old-age mortality (95+) doesn't follow Gompertz; the baseline fit is
  deliberately capped at 95 and extrapolated beyond.
- The app only models four risk factors (smoking, BMI, blood pressure,
  heavy alcohol use) — it has no information about family history, other
  medical conditions, diet, exercise, or access to healthcare.

None of these break the pipeline; they're the honest caveats that separate a
"done properly" learning project from actuarial-grade pricing.

## Suggested next steps

- Feed the posterior survival curve into a Monte-Carlo retirement simulator
  (sequence-of-returns risk + mortality risk together).
- Add competing-risks / cause-of-death modeling using the mortality file's
  cause codes.
- Replace the population baseline with a proper never-exposed baseline
  hazard.
