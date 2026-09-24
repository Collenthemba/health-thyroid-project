"""
Browser form for the saved Random Forest thyroid model.

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
REFERRAL_CHOICES = [
    ("Other / not specified", "referral_source_other"),
    ("SVI — sick / inpatient", "referral_source_SVI"),
    ("SVHC — health check", "referral_source_SVHC"),
    ("STMW — antenatal clinic", "referral_source_STMW"),
    ("SVHD", "referral_source_SVHD"),
    ("WEST", "referral_source_WEST"),
]
REFERRAL_LABELS = [label for label, _ in REFERRAL_CHOICES]
REFERRAL_COLUMNS = {label: column for label, column in REFERRAL_CHOICES}

FLAG_GROUPS = [
    (
        "Treatment and medicines",
        [
            ("on_thyroxine", "On thyroxine", "Already taking thyroid hormone replacement"),
            ("query_on_thyroxine", "Query on thyroxine", "Asked whether they should be on thyroxine"),
            ("on_antithyroid_medication", "On antithyroid medication", "Medicine that lowers thyroid hormone"),
            ("I131_treatment", "I-131 treatment", "Radioactive iodine treatment"),
            ("lithium", "On lithium", "Lithium can affect thyroid function"),
            ("thyroid_surgery", "Thyroid surgery", "Had an operation on the thyroid"),
        ],
    ),
    (
        "Findings and queries",
        [
            ("sick", "Currently sick", "Unwell at the time of the blood test"),
            ("goitre", "Goitre", "Enlarged thyroid"),
            ("tumor", "Tumor", "Known tumour"),
            ("hypopituitary", "Hypopituitary", "Pituitary underactivity"),
            ("psych", "Psychiatric referral", "Referred from a psychiatric service"),
            ("query_hypothyroid", "Query hypothyroid", "Doctor suspects an underactive thyroid"),
            ("query_hyperthyroid", "Query hyperthyroid", "Doctor suspects an overactive thyroid"),
        ],
    ),
]
FLAG_KEYS = [key for _, fields in FLAG_GROUPS for key, _, _ in fields]
LABS = [
    {
        "key": "tsh",
        "label": "TSH",
        "unit": "mIU/L",
        "min": 0.005,
        "max": 150.0,
        "format": "%.3f",
        "step": 0.1,
        "decimals": 3,
        "low": 0.4,
        "high": 4.0,
        "help": "High TSH often points to an underactive thyroid. Typical adult range is about 0.4–4.0.",
    },
    {
        "key": "t3",
        "label": "T3",
        "unit": "nmol/L",
        "min": 0.1,
        "max": 10.0,
        "format": "%.2f",
        "step": 0.1,
        "decimals": 2,
        "low": 0.8,
        "high": 2.5,
        "help": "Typical adult range is about 0.8–2.5.",
    },
    {
        "key": "tt4",
        "label": "TT4",
        "unit": "nmol/L",
        "min": 5.0,
        "max": 300.0,
        "format": "%.1f",
        "step": 1.0,
        "decimals": 1,
        "low": 60.0,
        "high": 160.0,
        "help": "Total T4. Typical adult range is about 60–160.",
    },
    {
        "key": "t4u",
        "label": "T4U",
        "unit": "ratio",
        "min": 0.3,
        "max": 2.0,
        "format": "%.2f",
        "step": 0.01,
        "decimals": 2,
        "low": 0.8,
        "high": 1.2,
        "help": "T4 uptake. Typical adult range is about 0.8–1.2.",
    },
    {
        "key": "fti",
        "label": "FTI",
        "unit": "index",
        "min": 10.0,
        "max": 400.0,
        "format": "%.1f",
        "step": 1.0,
        "decimals": 1,
        "low": 70.0,
        "high": 150.0,
        "help": "Free thyroxine index. Typical adult range is about 70–150.",
    },
]
DEFAULTS = {
    "age": 45,
    "sex_label": "Female",
    "referral": "Other / not specified",
    "tsh": 1.4,
    "t3": 1.9,
    "tt4": 104.0,
    "t4u": 0.96,
    "fti": 109.0,
    "pregnant": False,
    **{key: False for key in FLAG_KEYS},
}
EXAMPLES = {
    "Typical adult": {
        **DEFAULTS,
        "caption": "Average adult lab values. Use this as a clean starting point.",
    },
    "Woman — condition": {
        **DEFAULTS,
        "age": 31,
        "sex_label": "Female",
        "referral": "SVI — sick / inpatient",
        "tsh": 150.0,
        "t3": 0.6,
        "tt4": 32.0,
        "t4u": 1.04,
        "fti": 31.0,
        "on_thyroxine": True,
        "caption": "Real test patient: very high TSH and low thyroid hormones.",
    },
    "Man — condition": {
        **DEFAULTS,
        "age": 60,
        "sex_label": "Male",
        "referral": "SVI — sick / inpatient",
        "tsh": 98.0,
        "t3": 0.4,
        "tt4": 5.8,
        "t4u": 0.80,
        "fti": 10.0,
        "caption": "Real test patient: high TSH with very low TT4 and FTI.",
    },
    "Woman — no condition": {
        **DEFAULTS,
        "age": 62,
        "sex_label": "Female",
        "referral": "Other / not specified",
        "tsh": 0.005,
        "t3": 2.1,
        "tt4": 130.0,
        "t4u": 0.99,
        "fti": 131.0,
        "caption": "Real test patient: thyroid hormones in a typical range.",
    },
    "Man — no condition": {
        **DEFAULTS,
        "age": 45,
        "sex_label": "Male",
        "referral": "Other / not specified",
        "tsh": 1.4,
        "t3": 1.9,
        "tt4": 104.0,
        "t4u": 0.96,
        "fti": 109.0,
        "caption": "Typical male labs with no thyroid condition.",
    },
}


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


@st.cache_data
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


ENTRY_KEYS = ["age", "sex_label", "referral", "tsh", "t3", "tt4", "t4u", "fti"]
NUMERIC_FIELDS = {
    "age": {"kind": "int", "decimals": 0, "min": 1, "max": 97, "step": 1},
    "tsh": {"kind": "float", "decimals": 3, "min": 0.005, "max": 150.0, "step": 0.1},
    "t3": {"kind": "float", "decimals": 2, "min": 0.1, "max": 10.0, "step": 0.1},
    "tt4": {"kind": "float", "decimals": 1, "min": 5.0, "max": 300.0, "step": 1.0},
    "t4u": {"kind": "float", "decimals": 2, "min": 0.3, "max": 2.0, "step": 0.01},
    "fti": {"kind": "float", "decimals": 1, "min": 10.0, "max": 400.0, "step": 1.0},
}


def raw_key(key: str) -> str:
    return f"{key}_raw"


def format_number(value: float, kind: str, decimals: int) -> str:
    if kind == "int":
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def parse_number(text: str, kind: str) -> float | None:
    cleaned = (text or "").strip().replace(",", ".")
    if cleaned == "":
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if kind == "int":
        return float(int(round(value)))
    return value


def set_numeric(key: str, value: float | None) -> None:
    spec = NUMERIC_FIELDS[key]
    if value is None:
        st.session_state[key] = None
        st.session_state[raw_key(key)] = ""
        return
    number = int(round(value)) if spec["kind"] == "int" else round(float(value), spec["decimals"])
    st.session_state[key] = number
    st.session_state[raw_key(key)] = format_number(number, spec["kind"], spec["decimals"])


def step_numeric(key: str, direction: int) -> None:
    spec = NUMERIC_FIELDS[key]
    current = st.session_state.get(key)
    if current is None:
        next_value = spec["min"]
    else:
        next_value = float(current) + direction * spec["step"]
        next_value = min(spec["max"], max(spec["min"], next_value))
    if spec["kind"] == "int":
        next_value = int(round(next_value))
    else:
        next_value = round(next_value + 1e-12, spec["decimals"])
    set_numeric(key, next_value)


def numeric_stepper(
    label: str,
    *,
    key: str,
    help_text: str | None = None,
    placeholder: str = "Type a number",
    container=None,
) -> float | None:
    ui = container or st
    spec = NUMERIC_FIELDS[key]
    if raw_key(key) not in st.session_state:
        existing = st.session_state.get(key)
        st.session_state[raw_key(key)] = (
            "" if existing is None else format_number(existing, spec["kind"], spec["decimals"])
        )

    minus, field, plus = ui.columns([1, 4, 1], vertical_alignment="bottom")
    go_down = minus.button("−", key=f"{key}_minus", use_container_width=True, help=f"Decrease by {spec['step']:g}")
    go_up = plus.button("+", key=f"{key}_plus", use_container_width=True, help=f"Increase by {spec['step']:g}")
    if go_down:
        step_numeric(key, -1)
    if go_up:
        step_numeric(key, 1)
    field.text_input(label, key=raw_key(key), placeholder=placeholder, help=help_text)

    raw = (st.session_state.get(raw_key(key)) or "").strip()
    if raw == "":
        st.session_state[key] = None
        return None

    typed = parse_number(raw, spec["kind"])
    if typed is None:
        ui.caption("Type a number, or use + and −.")
        st.session_state[key] = None
        return None
    if typed < spec["min"] or typed > spec["max"]:
        ui.caption(f"Enter a value from {spec['min']:g} to {spec['max']:g}.")
        st.session_state[key] = None
        return None

    number = int(round(typed)) if spec["kind"] == "int" else float(typed)
    st.session_state[key] = number
    return number


def apply_example(name: str) -> None:
    example = EXAMPLES[name]
    for key, value in DEFAULTS.items():
        chosen = example.get(key, value)
        if key in NUMERIC_FIELDS:
            set_numeric(key, chosen)
        else:
            st.session_state[key] = chosen
    st.session_state["example_note"] = example["caption"]
    st.session_state["prediction"] = None
    st.session_state["prediction_inputs"] = None
    st.session_state["predict_error"] = None


def clear_form() -> None:
    for key in ENTRY_KEYS:
        st.session_state.pop(key, None)
        st.session_state.pop(raw_key(key), None)
    st.session_state["pregnant"] = False
    for key in FLAG_KEYS:
        st.session_state[key] = False
    st.session_state["prediction"] = None
    st.session_state["prediction_inputs"] = None
    st.session_state["predict_error"] = None
    st.session_state["example_note"] = (
        "Enter the patient yourself, then click Predict."
    )


def input_snapshot() -> dict:
    return {
        "age": st.session_state.get("age"),
        "sex_label": st.session_state.get("sex_label"),
        "referral": st.session_state.get("referral"),
        "tsh": st.session_state.get("tsh"),
        "t3": st.session_state.get("t3"),
        "tt4": st.session_state.get("tt4"),
        "t4u": st.session_state.get("t4u"),
        "fti": st.session_state.get("fti"),
        "pregnant": bool(st.session_state.get("pregnant")),
        "flags": {key: bool(st.session_state.get(key)) for key in FLAG_KEYS},
    }


def lab_band(value: float, low: float, high: float) -> str:
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Typical"


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.15rem; max-width: 980px; }
            h1 { letter-spacing: -0.03em; }
            [data-testid="stToolbar"] { display: none; }
            #MainMenu { visibility: hidden; }
            footer { visibility: hidden; }
            .result-card, .hint-card {
                background: var(--secondary-background-color);
                border: 1px solid rgba(128, 148, 170, 0.28);
                border-radius: 16px;
                padding: 1.1rem 1.2rem 1rem;
            }
            .result-score { font-size: 2.5rem; font-weight: 700; line-height: 1.05; margin: 0.1rem 0 0.35rem; }
            .result-label { font-size: 1.08rem; font-weight: 650; margin-bottom: 0.2rem; }
            .muted { opacity: 0.78; font-size: 0.92rem; }
            .chip {
                display: inline-block;
                padding: 0.12rem 0.5rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 650;
                margin-left: 0.35rem;
            }
            .chip-high { background: #7f1d1d; color: #fee2e2; }
            .chip-low { background: #1e3a5f; color: #dbeafe; }
            .chip-ok { background: #14532d; color: #dcfce7; }
            div[data-testid="stTextInput"] input { text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Thyroid condition screen",
    layout="centered",
    initial_sidebar_state="collapsed",
)
inject_css()

if st.session_state.get("form_version") != 5:
    clear_form()
    st.session_state["form_version"] = 5
for key in FLAG_KEYS:
    st.session_state.setdefault(key, False)
st.session_state.setdefault("pregnant", False)
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("prediction_inputs", None)
st.session_state.setdefault("predict_error", None)

st.title("Thyroid condition screen")
st.caption(
    "Class demo for one patient at a time. The saved Random Forest model estimates "
    "whether a thyroid condition is likely. This is not a diagnosis."
)

try:
    with st.spinner("Loading the saved model…"):
        model = load_model()
    threshold = load_threshold()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if st.button("Clear form"):
    clear_form()
    st.rerun()

with st.expander("Use a demo patient instead", expanded=False):
    st.caption("Optional. These only fill the form if you want a ready-made example.")
    example_names = list(EXAMPLES)
    row_one = st.columns(3)
    row_two = st.columns(2)
    for column, name in zip(row_one, example_names[:3]):
        if column.button(name, use_container_width=True):
            apply_example(name)
            st.rerun()
    for column, name in zip(row_two, example_names[3:]):
        if column.button(name, use_container_width=True):
            apply_example(name)
            st.rerun()

st.subheader("Patient")
st.caption("Start blank. Type a number or use + and −. Press Enter after typing.")
demo1, demo2, demo3 = st.columns(3)
numeric_stepper("Age (years)", key="age", placeholder="e.g. 45", container=demo1)
demo2.selectbox("Sex", ["Female", "Male"], index=None, placeholder="Select sex", key="sex_label")
demo3.selectbox("Referral source", REFERRAL_LABELS, index=None, placeholder="Select referral", key="referral")

age_value = st.session_state.get("age")
sex_label = st.session_state.get("sex_label")
can_be_pregnant = sex_label == "Female" and age_value is not None and 12 <= int(age_value) <= 55
if can_be_pregnant:
    st.checkbox("Pregnant", key="pregnant")
else:
    st.session_state["pregnant"] = False
    if sex_label == "Male":
        st.caption("Pregnancy is hidden for male patients.")
    elif sex_label is None or age_value is None:
        st.caption("Pregnancy appears after you enter a female patient aged 12–55.")
    else:
        st.caption("Pregnancy is shown only for women aged 12–55.")

st.subheader("Lab values")
st.caption("Type a result or tap + / −. Typical adult ranges are shown under each box.")
lab_cols = st.columns(3)
for index, lab in enumerate(LABS):
    box = lab_cols[index % 3]
    numeric_stepper(
        f"{lab['label']} ({lab['unit']})",
        key=lab["key"],
        help_text=lab["help"],
        placeholder="Type a number",
        container=box,
    )
    entered = st.session_state.get(lab["key"])
    if entered is None:
        box.caption(f"Typical {lab['low']:g}–{lab['high']:g}")
    else:
        band = lab_band(float(entered), lab["low"], lab["high"])
        box.caption(f"Typical {lab['low']:g}–{lab['high']:g} · currently **{band}**")

with st.expander("Clinical history (optional)", expanded=False):
    st.caption("Leave these unchecked unless they apply. They change the score only a little.")
    for group_name, fields in FLAG_GROUPS:
        st.markdown(f"**{group_name}**")
        flag_cols = st.columns(2)
        for index, (key, label, help_text) in enumerate(fields):
            flag_cols[index % 2].checkbox(label, help=help_text, key=key)

missing = [
    label
    for key, label in [
        ("age", "age"),
        ("sex_label", "sex"),
        ("referral", "referral source"),
        ("tsh", "TSH"),
        ("t3", "T3"),
        ("tt4", "TT4"),
        ("t4u", "T4U"),
        ("fti", "FTI"),
    ]
    if st.session_state.get(key) is None
]
snapshot = input_snapshot()

st.subheader("Predict")
st.caption("Fill in the patient and lab values, then click Predict.")
if st.button("Predict", type="primary", use_container_width=True):
    if missing:
        st.session_state["prediction"] = None
        st.session_state["prediction_inputs"] = None
        st.session_state["predict_error"] = "Still needed: " + ", ".join(missing) + "."
    else:
        pregnancy_cleared = False
        pregnant = bool(st.session_state["pregnant"])
        sex_code = 1 if st.session_state["sex_label"] == "Male" else 0
        age = int(st.session_state["age"])
        if pregnant and (sex_code == 1 or age < 12 or age > 55):
            pregnant = False
            pregnancy_cleared = True

        flag_values = {key: int(bool(st.session_state[key])) for key in FLAG_KEYS}
        values = {
            "age": age,
            "sex": sex_code,
            "pregnant": int(pregnant),
            "TSH": float(st.session_state["tsh"]),
            "T3": float(st.session_state["t3"]),
            "TT4": float(st.session_state["tt4"]),
            "T4U": float(st.session_state["t4u"]),
            "FTI": float(st.session_state["fti"]),
            "TSH_measured": 1,
            "T3_measured": 1,
            "TT4_measured": 1,
            "T4U_measured": 1,
            "FTI_measured": 1,
            **flag_values,
        }
        for column in REFERRAL_COLUMNS.values():
            values[column] = 0
        values[REFERRAL_COLUMNS[st.session_state["referral"]]] = 1

        probability = float(model.predict_proba(build_row(values))[0, 1])
        predicted = int(probability >= threshold)
        active_flags = [label for _, fields in FLAG_GROUPS for key, label, _ in fields if st.session_state[key]]
        if pregnant:
            active_flags.append("Pregnant")

        if predicted:
            tone = "#dc2626"
            heading = "Thyroid condition likely"
            detail = "The model score is at or above the screening threshold."
        elif probability >= 0.15:
            tone = "#ca8a04"
            heading = "Below the screening threshold"
            detail = "The score is raised, but it is still treated as negative at this cutoff."
        else:
            tone = "#16a34a"
            heading = "No thyroid condition likely"
            detail = "The model score is well below the screening threshold."

        lab_rows = []
        for lab in LABS:
            value = float(st.session_state[lab["key"]])
            band = lab_band(value, lab["low"], lab["high"])
            chip = {"High": "chip-high", "Low": "chip-low", "Typical": "chip-ok"}[band]
            lab_rows.append(
                f"<div style='display:flex;justify-content:space-between;padding:0.28rem 0;border-bottom:1px solid rgba(128,148,170,0.18);'>"
                f"<span>{lab['label']}</span>"
                f"<span>{value:g} {lab['unit']} <span class='chip {chip}'>{band}</span></span>"
                f"</div>"
            )

        st.session_state["prediction"] = {
            "probability": probability,
            "tone": tone,
            "heading": heading,
            "detail": detail,
            "pregnancy_cleared": pregnancy_cleared,
            "lab_rows": lab_rows,
            "summary": (
                f"{st.session_state['sex_label']}, age {age}. "
                f"Referral: {st.session_state['referral']}. "
                + (f"Flags on: {', '.join(active_flags)}." if active_flags else "No clinical flags selected.")
            ),
        }
        st.session_state["prediction_inputs"] = snapshot
        st.session_state["predict_error"] = None

st.subheader("Result")
prediction = st.session_state.get("prediction")
if st.session_state.get("predict_error"):
    st.warning(st.session_state["predict_error"])
elif prediction and st.session_state.get("prediction_inputs") == snapshot:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="muted">Saved Random Forest score</div>
            <div class="result-label" style="color:{prediction['tone']};">{prediction['heading']}</div>
            <div class="result-score" style="color:{prediction['tone']};">{prediction['probability']:.1%}</div>
            <div class="muted">{prediction['detail']} Screening threshold is {threshold:.0%}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(prediction["probability"], 0.0), 1.0))
    st.caption(st.session_state["example_note"])
    if prediction["pregnancy_cleared"]:
        st.warning("Pregnancy was cleared because it is not realistic for this age or sex.")
    st.markdown("<div class='hint-card'><b>Lab snapshot</b>" + "".join(prediction["lab_rows"]) + "</div>", unsafe_allow_html=True)
    st.caption(prediction["summary"])
    st.caption("Confirm unusual results with a clinician and laboratory follow-up. This form is a class project tool.")
elif prediction:
    st.info("The form changed after the last score. Click Predict again.")
else:
    st.info("Fill in the patient and lab values, then click Predict.")
    st.caption(st.session_state["example_note"])
