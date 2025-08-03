from pathlib import Path
import time, importlib, random
import streamlit as st
import pandas as pd
import numpy as np
import joblib, matplotlib.pyplot as plt
from PIL import Image
from simulate_stats import simulate_match_stats

def _patch_sklearn_internals():
    import sklearn, packaging.version as _pkg
    if not hasattr(sklearn.utils, "_print_elapsed_time"):
        sklearn.utils._print_elapsed_time = lambda *a, **k: None
    if not hasattr(sklearn.utils, "_get_column_indices"):
        def _get_column_indices(X, key):
            if isinstance(key, slice):
                return np.arange(X.shape[1])[key]
            if isinstance(key, (list, tuple, np.ndarray)):
                return np.array(key)
            return np.array([key])
        sklearn.utils._get_column_indices = _get_column_indices
    if not hasattr(sklearn.utils, "parse_version"):
        sklearn.utils.parse_version = _pkg.parse
    try:
        scorer_mod = importlib.import_module("sklearn.metrics._scorer")
        if not hasattr(scorer_mod, "_Scorer"):
            class _Scorer: ...
            scorer_mod._Scorer = _Scorer
    except ModuleNotFoundError:
        pass
    if not hasattr(sklearn.utils, "_metadata_requests"):
        sklearn.utils._metadata_requests = None

_patch_sklearn_internals()

MODEL_PATH = Path("soccer_winprob_xgb.pkl")
DATA_PATH  = Path("cwc10matches.csv")
LOGO_DIR   = Path("logos")
CLASS_MAP  = {2: "Home Win", 1: "Draw", 0: "Away Win"}

st.set_page_config(page_title="Simulation Showcase", layout="centered")
st.title("⚽ Simulation Showcase")

@st.cache_resource(show_spinner=True)
def load_model(p: Path):
    return joblib.load(p)

@st.cache_data(show_spinner=True)
def load_matches(p: Path):
    df = pd.read_csv(p)
    if "MatchID" not in df.columns:
        df.insert(0, "MatchID", range(1, len(df) + 1))
    return df

try:
    model   = load_model(MODEL_PATH)
    matches = load_matches(DATA_PATH)
except Exception as e:
    st.exception(e)
    st.stop()

for c in matches.columns:
    if "date" in c.lower() and not np.issubdtype(matches[c].dtype, np.datetime64):
        matches[c] = pd.to_datetime(matches[c], errors="coerce")

model_feats  = getattr(model, "feature_names_in_", None) or getattr(model, "feature_names", None)
feature_cols = [c for c in matches.columns if c in model_feats]

st.sidebar.header("Choose Fixture")
labels = [f"{r['HomeTeam_clean']} vs {r['AwayTeam_clean']}" for _, r in matches.iterrows()]
idx    = st.sidebar.selectbox("Match", range(len(labels)), format_func=lambda i: labels[i])

if st.sidebar.button("Simulate & Predict", type="primary"):
    row = matches.iloc[idx]
    X   = row[feature_cols].to_frame().T.copy()

    with st.spinner("Running simulation… please wait"):
        time.sleep(random.uniform(3, 5))
        for c in X.columns:
            if "date" in c.lower() and not np.issubdtype(X[c].dtype, np.datetime64):
                X[c] = pd.to_datetime(X[c], errors="coerce")
        probas = model.predict_proba(X)[0]

    probas = np.clip(probas + np.random.normal(0, 0.01, probas.shape), 0, None)
    probas = probas / probas.sum()

    pred_idx   = int(np.argmax(probas))
    winner_txt = {2: row["HomeTeam_clean"], 1: "Draw", 0: row["AwayTeam_clean"]}[pred_idx]
    st.subheader(f"🏆 Predicted Winner: **{winner_txt}**")

    left, right = st.columns(2)
    for side, col in zip(["HomeTeam_clean", "AwayTeam_clean"], [left, right]):
        logo_file = LOGO_DIR / f"{row[side]}.png"
        if logo_file.exists():
            img = Image.open(logo_file)
            h   = 120
            w   = int(img.width * (h / img.height))
            col.image(img.resize((w, h)), caption=row[side])
        else:
            col.markdown(f"**{row[side]}**")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(list(CLASS_MAP.values()), probas)
    ax.set_xlim(0, 1)
    for bar, p in zip(bars, probas):
        ax.text(p + 0.02, bar.get_y() + bar.get_height() / 2, f"{p:.1%}", va="center")
    st.pyplot(fig)

    st.subheader("🔍 SHAP Feature Impact")
    import shap
    try:
        final_model = model.named_steps["actual_estimator"]
        from sklearn.pipeline import Pipeline
        preprocessor = Pipeline(model.steps[:-1])
        X_pre = preprocessor.transform(X)
        explainer = shap.TreeExplainer(final_model)
        shap_vals = explainer.shap_values(X_pre)
        if isinstance(shap_vals, list):
            imp = np.vstack([np.abs(s) for s in shap_vals]).mean(axis=0)
        else:
            imp = np.abs(shap_vals).mean(axis=0)
        feature_names = getattr(final_model, "feature_names_", None) or [
            f"f{i}" for i in range(len(imp))
        ]
        imp_series = pd.Series(imp, index=feature_names).sort_values()
        fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(imp_series))))
        ax.barh(imp_series.index, imp_series.values)
        ax.set_xlabel("Mean |SHAP value|")
        st.pyplot(fig)
    except Exception as e:
        st.info(f"SHAP not supported for this model: {e}")

    st.subheader("🎮 Simulated Match Stats")
    sim_stats = simulate_match_stats(row)
    stats_df  = pd.DataFrame([sim_stats])
    st.dataframe(stats_df, use_container_width=True)

    full_row = {
        "Winner"      : winner_txt,
        "HomeWinProb" : probas[2],
        "DrawProb"    : probas[1],
        "AwayWinProb" : probas[0],
        **sim_stats
    }
    csv_bytes = pd.DataFrame([full_row]).to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV Report", csv_bytes,
                       "prediction_report.csv", "text/csv")
else:
    st.info("Select a fixture and click **Simulate & Predict**.")
