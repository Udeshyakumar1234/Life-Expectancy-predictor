"""
Streamlit 'what-if' interface for the life-expectancy model.

Run with:  streamlit run app.py
Requires the pipeline artifacts (run `python run_pipeline.py` first).
"""
from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.survival import Person, load_baseline_params, load_lifestyle_draws, compute_result

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

st.set_page_config(page_title="Life-Expectancy Estimator", layout="wide")
st.title("Life-expectancy estimator")
st.caption(
    "Combines official U.S. mortality tables with a model fitted on real health-survey "
    "data to estimate how age, sex, and a few lifestyle factors relate to lifespan. "
    "Educational project — not medical or financial advice."
)


@st.cache_data(show_spinner=False)
def _load():
    baseline = load_baseline_params()
    draws = load_lifestyle_draws()
    using_bayes = os.path.exists(os.path.join(ART, "bayes_posterior.npz"))
    return baseline, draws, using_bayes


try:
    baseline, draws, using_bayes = _load()
except FileNotFoundError as e:
    st.error(
        "Model artifacts not found. Open a terminal in this folder and run:\n\n"
        "```\npython run_pipeline.py\n```\n\n"
        f"Details: {e}"
    )
    st.stop()

if not using_bayes:
    st.warning(
        "Using single point estimates (no Bayesian posterior found), so uncertainty "
        "ranges won't be shown. Run `python -m src.bayesian_hazard` for the full model."
    )


# --------------------------------------------------------------------------- #
# Helper: a slider with a linked, editable number box next to it
# --------------------------------------------------------------------------- #
def linked_input(label, min_val, max_val, default, step, key, fmt=None, help=None):
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default
    if number_key not in st.session_state:
        st.session_state[number_key] = st.session_state[slider_key]

    # Clamp in case valid bounds shifted since last run (e.g. target age depends
    # on current age).
    st.session_state[slider_key] = min(max(st.session_state[slider_key], min_val), max_val)
    st.session_state[number_key] = min(max(st.session_state[number_key], min_val), max_val)

    def sync_from_slider():
        st.session_state[number_key] = st.session_state[slider_key]

    def sync_from_number():
        st.session_state[slider_key] = st.session_state[number_key]

    c1, c2 = st.columns([4, 1])
    with c1:
        st.slider(label, min_val, max_val, step=step, key=slider_key,
                  on_change=sync_from_slider, help=help)
    with c2:
        st.number_input(" ", min_val, max_val, step=step, key=number_key,
                        on_change=sync_from_number, label_visibility="collapsed",
                        format=fmt)
    return st.session_state[slider_key]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
left, right = st.columns([1, 2])

with left:
    st.subheader("Your inputs")

    age = int(linked_input("Current age", 18, 100, 50, 1, key="age", fmt="%d"))
    sex = st.radio("Sex (selects baseline table)", ["male", "female"], horizontal=True)
    smoker = st.checkbox("Current smoker", value=False)
    bmi = float(linked_input("BMI (kg/m²)", 15.0, 55.0, 26.0, 0.5, key="bmi", fmt="%.1f"))
    heavy_alcohol = st.checkbox("Heavy alcohol use", value=False)
    sbp = int(linked_input("Systolic blood pressure (mmHg)", 90, 200, 120, 1, key="sbp", fmt="%d"))
    target_age = int(linked_input(
        "Age you'd like to check your odds of reaching",
        age + 1, 105, min(max(age + 20, 65), 105), 1, key="target_age", fmt="%d",
    ))

person = Person(age=age, sex=sex, smoker=smoker, bmi=bmi,
                heavy_alcohol=heavy_alcohol, systolic_bp=sbp)
res = compute_result(person, baseline, draws)

median_le = res.median_le
lo, hi = res.le_ci
p_target, (p_lo, p_hi) = res.prob_survive_to(target_age)

est_death_age = age + median_le
lo_age, hi_age = age + lo, age + hi

# --------------------------------------------------------------------------- #
# Result — one clear headline number per question, uncertainty kept secondary
# --------------------------------------------------------------------------- #
with left:
    st.subheader("Result")

    st.markdown(
        f"""
        <div style="padding:1.1rem 1.3rem;border-radius:0.75rem;background:#eef3f8;margin-bottom:0.9rem;">
          <div style="font-size:0.9rem;color:#555;">Best estimate: you live to about</div>
          <div style="font-size:2.6rem;font-weight:750;color:#1f77b4;line-height:1.15;">
            {est_death_age:.0f} years old
          </div>
          <div style="font-size:0.9rem;color:#555;">
            (about {median_le:.0f} more years from now)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if using_bayes:
        st.caption(
            f"The model isn't certain — a realistic range is **{lo_age:.0f} to "
            f"{hi_age:.0f} years old**."
        )

    st.markdown(
        f"""
        <div style="padding:1.1rem 1.3rem;border-radius:0.75rem;background:#eef8ef;">
          <div style="font-size:0.9rem;color:#555;">Chance of living to age {target_age}</div>
          <div style="font-size:2.6rem;font-weight:750;color:#2ca02c;line-height:1.15;">
            {p_target*100:.0f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if using_bayes:
        st.caption(
            f"Likely somewhere between **{p_lo*100:.0f}% and {p_hi*100:.0f}%**, "
            "accounting for model uncertainty."
        )

# --------------------------------------------------------------------------- #
# Survival curve
# --------------------------------------------------------------------------- #
with right:
    st.subheader("Survival curve")
    fig = go.Figure()
    if using_bayes:
        fig.add_trace(go.Scatter(
            x=np.concatenate([res.ages, res.ages[::-1]]),
            y=np.concatenate([res.upper, res.lower[::-1]]),
            fill="toself", fillcolor="rgba(31,119,180,0.2)",
            line=dict(color="rgba(0,0,0,0)"), name="Uncertainty range",
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=res.ages, y=res.median, mode="lines",
        line=dict(color="rgb(31,119,180)", width=3), name="Chance of being alive",
    ))
    fig.add_vline(x=target_age, line_dash="dash", line_color="gray")
    fig.add_annotation(
        x=target_age, y=p_target, text=f"{p_target*100:.0f}% at age {target_age}",
        showarrow=True, arrowhead=2, ax=40, ay=-40,
        bgcolor="white", bordercolor="gray",
    )
    fig.update_layout(
        xaxis_title="Age", yaxis_title="Chance of being alive",
        yaxis_range=[0, 1.02], yaxis_tickformat=".0%",
        height=520, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
with st.expander("How this works & limitations"):
    st.markdown(
        "**The logic, in plain terms**\n\n"
        "1. We start with a baseline: how mortality rises with age for someone of "
        "your sex, based on official U.S. Social Security Administration life "
        "tables — the same tables actuaries use.\n"
        "2. We nudge that baseline up or down based on how your smoking status, "
        "BMI, blood pressure, and alcohol use compare to average. Those "
        "adjustments come from a statistical model trained on real health-survey "
        "data (NHANES) linked to actual death records, so they reflect observed "
        "outcomes rather than assumptions.\n"
        "3. We repeat the whole calculation thousands of times using slightly "
        "different, equally-plausible versions of those adjustments. That spread "
        "of outcomes is where the uncertainty ranges come from — it reflects "
        "genuine statistical uncertainty, not indecision.\n\n"
        "**Limitations & warnings**\n\n"
        "- **This is an educational project, not medical, actuarial, or financial "
        "advice.** Please don't use it for healthcare, insurance, or life-planning "
        "decisions.\n"
        "- It only considers four factors — smoking, BMI, blood pressure, and "
        "heavy alcohol use. It knows nothing about your family history, other "
        "medical conditions, diet, exercise, sleep, mental health, or access to "
        "healthcare, all of which meaningfully affect real lifespan.\n"
        "- The underlying survey followed roughly 16,000 U.S. adults for under a "
        "decade, so the model has limited information about very old ages or "
        "less common combinations of risk factors — part of why the ranges "
        "exist.\n"
        "- The baseline population already includes smokers, heavy drinkers, "
        "etc., so applying a full risk adjustment on top of it somewhat "
        "overstates the effect of any one factor. Reasonable for a learning "
        "project, not precise enough for professional use.\n"
        "- Every number here is a statistical pattern across a population, not a "
        "personal fact. Two people with identical inputs can have very "
        "different real outcomes."
    )
