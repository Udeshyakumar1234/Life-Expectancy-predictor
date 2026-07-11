# data/raw/

This folder is where real **NHANES** survey files and **linked mortality**
files go if you want to run the model on real data instead of the synthetic
cohort.

It's intentionally empty in the shipped project — the pipeline runs out of
the box on synthetic data without anything here.

For the full download walkthrough (which files, where to get them, naming
convention), see **"Using real NHANES data"** in the main [README](../../README.md).

Once the files are here, just run the pipeline as normal — it detects real
data automatically:

```bash
python run_pipeline.py
```
