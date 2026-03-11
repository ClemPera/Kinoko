import streamlit as st
import requests
import pandas as pd
import numpy as np
# import pydeck as pdk
import time
import json
import random
import keras

# from streamlit_lottie import st_lottie
# from streamlit_elements import Elements

import tensorflow as tf
from pathlib import Path

# ----------------------------------------------------------
# ⚙️ STREAMLIT CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(
    page_title="🍄 Kinoko (きのこ) 🍄 - Online Batch #2207",
    page_icon="🍄🍄‍🟫",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = [
    ":material/assignment: Project",
    ":material/search: Data Exploration",
    ":material/scatter_plot: Data Analysis",
    ":material/psychology: Modeling",
    ":material/rocket_launch: Prediction"
]
page = st.sidebar.radio("Go to:", pages)

# ----------------------------------------------------------
# 🎨 STYLING — Hover, Highlights, Transitions
# ----------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(90, 132, 255, 0.15);
  border-radius: 10px;
  padding: 6px 8px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
  font-weight: 700;
}
.stButton > button {
  border-radius: 10px;
  transition: transform .06s ease, box-shadow .12s ease, background-color .12s ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(0,0,0,.18);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 📥 DATA LOADING
# ----------------------------------------------------------

# ░░ IMG ░░
DATA_DIR = Path("/Users/geeksterlab/code/ClemPera/Kinoko/data/image_dataset")

# Without aug img
# edible_paths = [p for p in (DATA_DIR / "edible").rglob("*.png") if not p.name.startswith("aug_")]
# poisonous_paths = [p for p in (DATA_DIR / "poisonous").rglob("*.png") if not p.name.startswith("aug_")]

# with aug img
edible_paths = [p for p in (DATA_DIR / "edible").rglob("*.png")]
poisonous_paths = [p for p in (DATA_DIR / "poisonous").rglob("*.png")]

# ░░ MODEL BASELINE + AUGMENTED ░░
results_path = Path("logs/.csv")

# ░░ MODEL ░░

# ░░ LOGS  ░░
LOG_DIR = Path("/Users/geeksterlab/code/ClemPera/Kinoko/logs")

# ░░ RESULTS  ░░
RESULTS_DIR = Path("/Users/geeksterlab/code/ClemPera/Kinoko/results")

# ----------------------------------------------------------
# 📄 PAGE 1 — PROJECT OVERVIEW
# ----------------------------------------------------------

if page == pages[0]:
    st.divider()
    st.header("🍄 :rainbow[Kinoko Lab] 🍄")
    st.divider()
    st.markdown("""
    ### Project Overview
    This project is a full **CNN-ready Streamlit app** including:
    - Data exploration
    - Visual analytics
    - Modeling
    - Real-time prediction

    """)

# ----------------------------------------------------------
# 🔍 PAGE 2 — DATA EXPLORATION
# ----------------------------------------------------------

elif page == pages[1]:
    st.divider()
    st.header("🍄 :rainbow[Data Exploration] 🍄")
    st.divider()

    # ░░ Nombre total par classe ░░
    st.subheader("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("🍄 Edible", len(edible_paths))
    col2.metric("☠️ Poisonous", len(poisonous_paths))
    col3.metric("📁 Total", len(edible_paths) + len(poisonous_paths))

    st.divider()

    # ░░ Random images ░░
    st.subheader("🖼️ Random Images")
    col_e, col_p = st.columns(2)

    with col_e:
        st.markdown("**🍄 Edible**")
        for img_path in random.sample(edible_paths, 3):
            st.image(str(img_path), caption=img_path.parent.name, width=200)

    with col_p:
        st.markdown("**☠️ Poisonous**")
        for img_path in random.sample(poisonous_paths, 3):
            st.image(str(img_path), caption=img_path.parent.name, width=200)

    st.divider()

    # ░░ Distribution by species ░░
    st.subheader("📈 Distribution by Species")

    edible_species = pd.Series([p.parent.name for p in edible_paths]).value_counts().reset_index()
    edible_species.columns = ["species", "count"]
    edible_species["class"] = "edible"

    poisonous_species = pd.Series([p.parent.name for p in poisonous_paths]).value_counts().reset_index()
    poisonous_species.columns = ["species", "count"]
    poisonous_species["class"] = "poisonous"

    df_species = pd.concat([edible_species, poisonous_species])

    tab1, tab2 = st.tabs(["🍄 Edible", "☠️ Poisonous"])

    with tab1:
        st.bar_chart(edible_species.set_index("species")["count"])
    with tab2:
        st.bar_chart(poisonous_species.set_index("species")["count"])

# ----------------------------------------------------------
# 📄 PAGE 3 — DATA ANALYSIS
# ----------------------------------------------------------

elif page == pages[2]:
    st.divider()
    st.header("📊 :rainbow[Data Analysis]")
    st.divider()

    st.caption("**PLACEHOLDER**")

# ----------------------------------------------------------
# 📄 PAGE 4 — MODELING  (BASELINE vs FINE TUNING)
# ----------------------------------------------------------

elif page == pages[3]:
    st.divider()
    st.header("📊 :rainbow[Modeling Results]")
    st.divider()

    if results_path.exists():
        df_results = pd.read_csv(results_path)
        st.subheader("📊 Final Metrics Comparison")

    tab1, tab2 = st.tabs(["📊 Probability", "📈 Validation"])

    with tab1:

        st.caption("Latest Run")
        probability_files = sorted(
            RESULTS_DIR.glob("baselines_probability_*.csv"),
            reverse=True
        )

        if probability_files:
            df_prob = pd.read_csv(probability_files[0])
            st.subheader("Baseline Probability Results")
            st.dataframe(df_prob)

        else:
            st.warning("No probability file found.")

        st.caption("Select the probability results you want to view based on the date 📅")
        selected_file = st.selectbox("Select run", probability_files)
        df_prob = pd.read_csv(selected_file)

    with tab2:

        baseline_logs = sorted(
            LOG_DIR.glob("baseline_history_*.csv"),
            reverse=True
        )

        augmented_logs = sorted(
            LOG_DIR.glob("augmented_history_*.csv"),
            reverse=True
        )

        if baseline_logs and augmented_logs:

            df_baseline = pd.read_csv(baseline_logs[0])
            df_augmented = pd.read_csv(augmented_logs[0])

            st.subheader("📈 Validation Loss")
            st.line_chart(pd.DataFrame({
                "Baseline": df_baseline["val_loss"],
                "Augmented": df_augmented["val_loss"]
            }))

            st.subheader("📈 Validation Accuracy")
            st.line_chart(pd.DataFrame({
                "Baseline": df_baseline["val_accuracy"],
                "Augmented": df_augmented["val_accuracy"]
            }))

            st.subheader("📈 Validation Recall")
            st.line_chart(pd.DataFrame({
                "Baseline": df_baseline["val_recall"],
                "Augmented": df_augmented["val_recall"]
            }))

            st.subheader("📈 Validation AUC")
            st.line_chart(pd.DataFrame({
                "Baseline": df_baseline["val_auc"],
                "Augmented": df_augmented["val_auc"]
            }))

        else:
            st.warning("No training logs found yet.")

# ----------------------------------------------------------
# 📄 PAGE 5 — PREDICTION
# ----------------------------------------------------------
#  upload image →  predict edible/poisonous + confidence score

elif page == pages[4]:
    st.divider()
    st.header("🚀 Prediction")
    st.divider()

    st.caption("**Upload your 🍄‍🟫 mushroom 🍄‍🟫 and see what will happen... 😉**")

    st.caption("**PLACEHOLDER**")
