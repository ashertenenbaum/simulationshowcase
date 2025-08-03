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
_patch_sklearn_internals()

st.set_page_config("Club World Cup Predictor", page_icon="🌍", layout="wide")
st.title("FIFA Club World Cup Predictor")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict a Match", "How it Works"])

model = joblib.load("/mnt/data/xgb_model.joblib")
data = pd.read_csv("/mnt/data/cwc10matches.csv")

# Clean and prepare feature matrix for prediction
X = data.drop(columns=["MatchID", "Home", "Away", "Result"])

# Try to get feature importances
try:
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })
    feature_importance_df.sort_values("Importance", ascending=True, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="skyblue")
    ax.set_title("Feature Importance (Model-Based)")
    ax.set_xlabel("Relative Importance")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Feature importance not supported for this model: {e}")

st.markdown("---")

if page == "Predict a Match":
    match = st.selectbox("Choose a match to simulate:", data[["Home", "Away"]].apply(lambda x: f"{x[0]} vs {x[1]}", axis=1))
    match_index = data[["Home", "Away"]].apply(lambda x: f"{x[0]} vs {x[1]}", axis=1).tolist().index(match)
    row = data.iloc[match_index]
    X_row = X.iloc[match_index:match_index+1]

    st.subheader("Predicted Outcome")
    proba = model.predict_proba(X_row)[0]
    outcome = model.predict(X_row)[0]
    labels = model.classes_
    results = dict(zip(labels, proba))

    st.write(f"**Prediction:** {outcome}")

    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values(), color=["green", "gray", "red"])
    ax.set_ylabel("Probability")
    ax.set_title("Win/Draw/Loss Probabilities")
    st.pyplot(fig)

    st.subheader("Simulated Match Statistics")
    sim_stats = simulate_match_stats(row)
    st.json(sim_stats)

    if st.button("Download Match Report as CSV"):
        out_df = pd.DataFrame([{
            **row.to_dict(),
            **sim_stats,
            "Predicted Outcome": outcome,
            **{f"P({k})": v for k, v in results.items()}
        }])
        out_path = "/mnt/data/match_prediction_report.csv"
        out_df.to_csv(out_path, index=False)
        st.success("CSV report ready for download.")
        st.download_button("Download CSV", out_path, file_name="match_prediction_report.csv", mime="text/csv")

elif page == "How it Works":
    st.subheader("How the Prediction Works")
    st.markdown("""
        This app uses a machine learning model trained on Club World Cup-style fixtures with Elo ratings and team stats.

        **Steps involved:**
        - Load match data (team names, Elo ratings, past performance stats).
        - Feed features into a trained XGBoost model.
        - Show predicted win/draw/loss probabilities.
        - Simulate realistic match statistics (shots, cards, etc.).
        - Let you download the full prediction as a CSV.

        **About the Feature Importance Chart**
        The bar chart above shows which features the model found most useful when learning to predict outcomes. Longer bars mean more influence.
    """)
