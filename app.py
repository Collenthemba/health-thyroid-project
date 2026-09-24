"""
Simple prediction form for the saved Random Forest thyroid model.

Run from this folder:
    python -m streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "best_thyroid_model.joblib"
RESULTS_PATH = BASE_DIR / "outputs" / "model_results.csv"
FEATURE_ORDER = [
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH_measured",
    "TSH",
    "T3_measured",
    "T3",
    "TT4_measured",
    "TT4",
    "T4U_measured",
    "T4U",
    "FTI_measured",
    "FTI",
    "referral_source_other",
    "referral_source_SVI",
    "referral_source_SVHC",
    "referral_source_STMW",
    "referral_source_SVHD",
    "referral_source_WEST",
]
REFERRAL_OPTIONS = {
    "Other": "referral_source_other",
    "SVI": "referral_source_SVI",
    "SVHC": "referral_source_SVHC",
    "STMW": "referral_source_STMW",
    "SVHD": "referral_source_SVHD",
    "WEST": "referral_source_WEST",
}
FLAG_FIELDS = [
    ("on_thyroxine", "On thyroxine"),
    ("query_on_thyroxine", "Query on thyroxine"),
    ("on_antithyroid_medication", "On antithyroid medication"),
    ("sick", "Currently sick"),
    ("thyroid_surgery", "Thyroid surgery"),
    ("I131_treatment", "I-131 treatment"),
    ("query_hypothyroid", "Query hypothyroid"),
    ("query_hyperthyroid", "Query hyperthyroid"),
    ("lithium", "Lithium"),
    ("goitre", "Goitre"),
    ("tumor", "Tumor"),
    ("hypopituitary", "Hypopituitary"),
    ("psych", "Psychiatric referral"),
]


def _patch_sklearn() -> None:
    if "sklearn.ensemble._hist_gradient_boosting.gradient_boosting" in sys.modules:
        return
    hist_pkg = ModuleType("sklearn.ensemble._hist_gradient_boosting")
    hist_mod = ModuleType("sklearn.ensemble._hist_gradient_boosting.gradient_boosting")
    hist_mod.HistGradientBoostingClassifier = object
    hist_mod.HistGradientBoostingRegressor = object
    sys.modules["sklearn.ensemble._hist_gradient_boosting"] = hist_pkg
    sys.modules["sklearn.ensemble._hist_gradient_boosting.gradient_boosting"] = hist_mod


@st.cache_resource
def load_model():
    _patch_sklearn()
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_PATH}. Run train_models.py first."
        )
    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        rf = results[results["Model"] == "Random Forest"]
        if len(rf):
            return float(rf.iloc[0]["Threshold"])
    return 0.34


def build_row(values: dict) -> pd.DataFrame:
    row = {name: 0 for name in FEATURE_ORDER}
    row.update(values)
    return pd.DataFrame([row], columns=FEATURE_ORDER)


st.set_page_config(page_title="Thyroid condition screen", layout="centered")
st.title("Thyroid condition screen")
st.caption(
    "Enter one patient's details. The saved Random Forest model estimates "
    "whether a thyroid condition is likely. This is a class project tool, not a diagnosis."
)

threshold = load_threshold()
model = load_model()

with st.form("patient_form"):
    st.subheader("Patient")
    col_a, col_b, col_c = st.columns(3)
    age = col_a.number_input("Age (years)", min_value=1, max_value=97, value=45, step=1)
    sex_label = col_b.selectbox("Sex", ["Female", "Male"])
    referral = col_c.selectbox("Referral source", list(REFERRAL_OPTIONS.keys()))

    st.subheader("Lab values")
    lab1, lab2, lab3 = st.columns(3)
    tsh = lab1.number_input("TSH", min_value=0.005, max_value=150.0, value=1.4, format="%.3f")
    t3 = lab2.number_input("T3", min_value=0.1, max_value=10.0, value=1.9, format="%.2f")
    tt4 = lab3.number_input("TT4", min_value=5.0, max_value=300.0, value=104.0, format="%.1f")
    t4u = lab1.number_input("T4U", min_value=0.3, max_value=2.0, value=0.96, format="%.2f")
    fti = lab2.number_input("FTI", min_value=10.0, max_value=400.0, value=109.0, format="%.1f")
    pregnant = lab3.checkbox("Pregnant")

    st.subheader("Clinical flags")
    flag_values = {}
    flag_cols = st.columns(3)
    for i, (key, label) in enumerate(FLAG_FIELDS):
        flag_values[key] = int(flag_cols[i % 3].checkbox(label))

    submitted = st.form_submit_button("Predict")

if submitted:
    sex_code = 1 if sex_label == "Male" else 0
    if pregnant and (sex_code == 1 or age < 12 or age > 55):
        st.warning("Pregnancy flag was cleared because it is not realistic for this age or sex.")
        pregnant = False

    values = {
        "age": int(age),
        "sex": sex_code,
        "pregnant": int(pregnant),
        "TSH": float(tsh),
        "T3": float(t3),
        "TT4": float(tt4),
        "T4U": float(t4u),
        "FTI": float(fti),
        "TSH_measured": 1,
        "T3_measured": 1,
        "TT4_measured": 1,
        "T4U_measured": 1,
        "FTI_measured": 1,
        **flag_values,
    }
    for column in REFERRAL_OPTIONS.values():
        values[column] = 0
    values[REFERRAL_OPTIONS[referral]] = 1

    X = build_row(values)
    probability = float(model.predict_proba(X)[0, 1])
    predicted = int(probability >= threshold)

    st.subheader("Result")
    if predicted == 1:
        st.error(
            f"Thyroid condition likely. Model score {probability:.1%} "
            f"(screening threshold {threshold:.0%})."
        )
    else:
        st.success(
            f"No thyroid condition likely. Model score {probability:.1%} "
            f"(screening threshold {threshold:.0%})."
        )
    st.progress(min(max(probability, 0.0), 1.0))
    st.caption(
        "A score at or above the threshold is treated as positive. "
        "Confirm with a clinician and laboratory follow-up."
    )
