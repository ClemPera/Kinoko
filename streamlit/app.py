# ----------------------------------------------------------
# 🚛 IMPORTS
# ----------------------------------------------------------

# --- Python / Path setup ---
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

# --- Streamlit / UI / Data visualization ---
import streamlit as st
from streamlit_option_menu import option_menu
import random
import pandas as pd
import plotly.graph_objects as go

# --- ML / Deep Learning ---
import tensorflow as tf

# --- API ---
import requests

# ----------------------------------------------------------
# ⚙️ STREAMLIT CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(
    page_title="🍄 Kinoko (きのこ) 🍄 - Online Batch #2207",
    page_icon="🍄🍄‍🟫",
    layout="wide"
)

# ░░ Navigation ░░
pages = ["Project", "Data Exploration", "Data Analysis", "Prediction"]
icons = ["book", "search", "bar-chart", "rocket"]


styles = {
    "container": {
        "padding": "0!important",
        "background-color": "#2B2D42",
    },
    "icon": {
        "color": "#8ECAE6",
        "font-size": "20px",
    },
    "nav-link": {
        "font-size": "18px",
        "text-align": "center",
        "margin": "0px",
        "color": "#EDF2F4",
        "--hover-color": "#3A3D5C",
    },
    "nav-link-selected": {
        "background-color": "#6BAA75",
        "color": "white",
        "font-weight": "600",
    },
}

# ░░ Horizontalbar Navigation ░░
selected = option_menu(
    menu_title=None,
    options=pages,
    icons=icons,
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles=styles
)

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
            """, unsafe_allow_html=True
)

# ----------------------------------------------------------
# 📥 DATA LOADING
# ----------------------------------------------------------

# ░░ IMG ░░
DATA_DIR = Path("../data/image_dataset")

# with aug img
edible_paths = [p for p in (DATA_DIR / "edible").rglob("*.png")]
poisonous_paths = [p for p in (DATA_DIR / "poisonous").rglob("*.png")]

# ░░ MODEL BASELINE + AUGMENTED ░░
results_path = Path("../models/images/baseline/logs/.csv")

# ░░ LOGS  ░░
LOG_DIR = Path("../models/images/baseline/logs")

# ░░ RESULTS  ░░
RESULTS_DIR = Path("../models/images/baseline/results")

# ----------------------------------------------------------
# 📄 PAGE 1 — PROJECT OVERVIEW
# ----------------------------------------------------------

if selected == "Project":
    st.divider()
    st.header("🍄 :rainbow[Kinoko Lab] 🍄")
    st.divider()
    st.markdown("""
                **Kinoko Lab** is an end-to-end machine learning application designed to classify mushrooms as edible or poisonous.

                The project combines:
                - 📊 Exploratory Data Analysis
                - 🧠 Tabular Machine Learning models
                - 📷 Image-based Deep Learning (CNN)
                - ⚡ Real-time predictions via API integration

                Our objective was to build a complete Machine Learning & Deep Learning pipeline — from raw data exploration to production-ready deployment.
    """)

    st.divider()
    st.subheader("👥 Meet the Team")
    st.caption("Click on a card to learn more ✨")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander("👨‍💻 Clément"):
            st.image("https://github.com/ClemPera.png", width=150)
            st.write("🛠️ Fine-Tuning • ⚙️ Backend API • 🧹Code cleaning • 📈 Baseline Optimization")

    with col2:
        with st.expander("👨‍💻 Bastien"):
            st.image("https://github.com/basspeif.png", width=150)
            st.write("📊 Tabular ML Modeling • 📊 Tabular Model Fine-Tuning • 🦾 AutoML Improvements")

    with col3:
        with st.expander("🧑‍💻 Marie-Ange"):
            st.image("https://github.com/Seiiferu.png", width=150)
            st.write("🧠 CNN Baseline & Augmentation • 💻 Frontend Architecture • 🔗 End-to-end Integration")

    st.divider()
    st.markdown("""
                ### 🙏 Acknowledgements

                We would like to sincerely thank all the instructors, teaching assistants,
                and mentors who guided us throughout the entire bootcamp.

                From foundational concepts to advanced machine learning practices,
                their support, feedback, and technical guidance enabled us to build this
                end-to-end application with confidence and rigor.

                This project reflects the collective learning journey we experienced
                over the course of the program.
    """)

    st.markdown("""
                <div style="text-align:center; padding-top: 1.5rem;">
                    <p style="color:#8ECAE6; font-weight:600; font-size:18px;">
                        🍄 Thanks for exploring Kinoko Lab!
                    </p>
                    <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExejZ2bjJ2eHlveTB6c2o1cWhwbGVrM3FxdndrYXFyajZ1Z2djemlwMSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT1R9JE4hYhm8JmYAU/giphy.gif" width="320">
                </div>
                """, unsafe_allow_html=True
    )

# ----------------------------------------------------------
# 🔍 PAGE 2 — DATA EXPLORATION
# ----------------------------------------------------------

elif selected == "Data Exploration":
    st.divider()

    # ░░ Nombre total par classe ░░
    st.subheader("📊 Dataset Overview")
    st.markdown("""
                This section explores the structure of the dataset.

                First, we examine the overall class distribution
                (edible vs poisonous and total number of images).

                Then, we visualize random samples from each class
                to observe visual differences.

                Finally, we analyze the distribution of images per species
                to detect potential imbalance within each class.
    """)
    st.divider()

    # ░░ Class Distribution ░░
    st.subheader("Summarize class distribution")
    st.info("""
            *These metrics summarize the number of images available in each class and random samples.*
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("🍄 Edible", len(edible_paths))
    col2.metric("☠️ Poisonous", len(poisonous_paths))
    col3.metric("📁 Total", len(edible_paths) + len(poisonous_paths))

    # ░░ Random images ░░
    st.subheader("🖼️ Random Images")
    st.info("""
            *Display random samples to visually inspect differences between edible and poisonous mushrooms.*
    """)
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
    st.info("""
            *These plots show how many images are available per species.*
    """)

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

elif selected == "Data Analysis":
    st.divider()

    if results_path.exists():
        df_results = pd.read_csv(results_path)
        st.subheader("📊 Final Metrics Comparison")

    tab1, tab2, tab3 = st.tabs(["📊 Probability", "📈 Validation", "📉 Metric"])

    with tab1:
        probability_files = sorted(
            RESULTS_DIR.glob("baselines_probability_*.csv"),
            reverse=True
        )

        if probability_files:
            df_prob = pd.read_csv(probability_files[0])
            st.subheader("Prediction Comparison for Each Sample")
            st.info(
                "Use this table to identify disagreements between the two models. "
                "A difference in probability may reveal uncertainty, while a difference "
                "in label or class shows that the models made different final decisions."
            )
            with st.expander("🧠 How to read this table?"):
                st.markdown("""
                            Each row corresponds to one image.

                            - *Baseline_proba* and *Augmented_proba* are the predicted probabilities.
                            - If the probability is greater than 0.5 → label = 1 (poisonous).
                            - If the probability is below 0.5 → label = 0 (edible).
                            - The final class column shows the human-readable prediction.

                            Example → → if both probabilities are below 0.5, both models classify the mushroom as edible.
                """)
            st.dataframe(df_prob)
            st.warning("Latest Run")

        else:
            st.warning("No probability file found.")

        # st.caption("Select the probability results you want to view based on the date 📅")
        # selected_file = st.selectbox("Select run", probability_files)
        # df_prob = pd.read_csv(selected_file)

    with tab2:
        st.subheader("⚖️ Val Comparison: Baseline model vs Augmented model")
        st.info("""
                Validation metrics measure how well the model generalizes
                to unseen data.
        """)
        with st.expander("🧠 How to interpret validation curves?"):
            st.markdown("""
                        - Decreasing validation loss → better generalization.
                        - Validation loss increasing while training improves → overfitting.
                        - Stable and high AUC → strong discrimination ability.
                        - Recall is critical when detecting poisonous mushrooms.
                        - A lower validation loss and stable AUC
                        indicate better robustness.
            """)

        baseline_logs = sorted(LOG_DIR.glob("baseline_history_*.csv"), reverse=True)
        augmented_logs = sorted(LOG_DIR.glob("augmented_history_*.csv"), reverse=True)

        COLORS = {"Baseline": "#4C78A8", "Augmented": "#F58518"}

        if baseline_logs and augmented_logs:
            df_baseline = pd.read_csv(baseline_logs[0])
            df_augmented = pd.read_csv(augmented_logs[0])

            def plot_metric(metric):
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    y=df_baseline[metric],
                    name="Baseline",
                    line=dict(color=COLORS["Baseline"], width=2)
                ))

                fig.add_trace(go.Scatter(
                    y=df_augmented[metric],
                    name="Augmented",
                    line=dict(color=COLORS["Augmented"], width=2)
                ))

                fig.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title=metric.replace("val_", "").capitalize(),
                    height=300,
                    margin=dict(t=10, b=20),
                )

                st.plotly_chart(fig, use_container_width=True)

            val_metric_help = {
                "val_loss": "Validation loss: measures generalization error.",
                "val_recall": "Validation recall: ability to detect positives on unseen data.",
                "val_precision": "Validation precision: correctness of positive predictions.",
                "val_auc": "Validation AUC: discrimination power on unseen data.",
                "val_accuracy": "Validation accuracy: overall correctness on unseen data."
            }

            for metric, label in [
                ("val_loss",      "📈 Validation Loss"),
                ("val_recall",    "📈 Validation Recall"),
                ("val_precision", "📈 Validation Precision"),
                ("val_auc",       "📈 Validation AUC"),
                ("val_accuracy",  "📈 Validation Accuracy"),
            ]:
                st.subheader(label, help=val_metric_help[metric])
                plot_metric(metric)

        else:
            st.warning("No training logs found yet.")

    with tab3:
        st.subheader("⚖️ Metrics Comparison: Baseline model vs Augmented model")
        st.info("""
                Training metrics show how the model learns during training.
        """)
        with st.expander("🧠 How to interpret training curves?"):
            st.markdown("""
                        - Rapid improvement then plateau → stable learning.
                        - Extremely high training performance but lower validation → overfitting.
                        - Slow improvement → possible underfitting or insufficient capacity.
            """)

        baseline_logs = sorted(LOG_DIR.glob("baseline_history_*.csv"), reverse=True)
        augmented_logs = sorted(LOG_DIR.glob("augmented_history_*.csv"), reverse=True)

        COLORS = {"Baseline": "#4C78A8", "Augmented": "#F58518"}

        if baseline_logs and augmented_logs:
            df_baseline = pd.read_csv(baseline_logs[0])
            df_augmented = pd.read_csv(augmented_logs[0])

            def plot_metric_train(metric):
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    y=df_baseline[metric],
                    name="Baseline",
                    line=dict(color=COLORS["Baseline"], width=2)
                ))

                fig.add_trace(go.Scatter(
                    y=df_augmented[metric],
                    name="Augmented",
                    line=dict(color=COLORS["Augmented"], width=2)
                ))

                fig.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title=metric.capitalize(),
                    height=300,
                    margin=dict(t=10, b=20),
                )

                st.plotly_chart(fig, use_container_width=True)

            train_metric_help = {
                "loss": "Training loss: measures how wrong the model is on training data. Should decrease steadily.",
                "recall": "Training recall: ability to detect positive cases in the training set.",
                "precision": "Training precision: proportion of predicted positives that are correct on training data.",
                "auc": "Training AUC: model’s ability to separate classes on training data.",
                "accuracy": "Training accuracy: percentage of correct predictions on training data."
            }

            for metric, label in [
                ("loss",      "📉 Train Loss"),
                ("recall",    "📉 Train Recall"),
                ("precision", "📉 Train Precision"),
                ("auc",       "📉 Train AUC"),
                ("accuracy",  "📉 Train Accuracy"),
            ]:
                st.subheader(label, help=train_metric_help[metric])
                plot_metric_train(metric)

        else:
            st.warning("No training logs found yet.")

# ----------------------------------------------------------
# 📄 PAGE 5 — PREDICTION
# ----------------------------------------------------------

# ═════════════════════ SURVEY ═════════════════════

elif selected == "Prediction":
    st.divider()
    st.caption("**Upload your 🍄 mushroom 🍄 and see what will happen... 😉**")
    with st.expander("📘 How to use this page"):
        st.markdown("""
                    #### 🧾 Required fields
                    You must fill in all fields marked with 🚨 before making a prediction.

                    #### 📊 Tabular prediction
                    Uses morphological features only (shape, color, habitat…).

                    #### 📷 Image prediction
                    Uses the uploaded photo and image-based deep learning models.

                    #### 🔮 Predict All
                    Runs both tabular and image models and compares their outputs.

                    #### ⚠️ Important
                    A high confidence score does not guarantee safety.
                    Never eat a wild mushroom based only on AI prediction 🙈.
    """)

    API_URL = "https://kinokoapi-930685077136.europe-west4.run.app"

    # ░░ DESIGN ░░
    st.markdown(
    """
    <style>

    /* Style only expanders that contain an image */
    div[data-testid="stExpander"]:has(img) {
        background-color: #2B2D42;
        border-radius: 15px;
        border: 1px solid rgba(120, 255, 120, 0.3);
    }

    /* Header styling */
    div[data-testid="stExpander"]:has(img) > details > summary {
        background-color: #2B2D42;
        border-radius: 15px;
        padding: 0.5rem 1rem;
        color: #8ECAE6;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    # ░░ MAPPING ░░
    cap_shape_map = {"bell":"b", "conical":"c", "convex":"x", "flat":"f",
                    "sunken":"s", "spherical":"p", "others":"o"}
    cap_color_map = {"brown":"n","buff":"b","gray":"g","green":"r","pink":"p",
                    "purple":"u","red":"e","white":"w","yellow":"y","blue":"l",
                    "orange":"o","black":"k"}
    gill_color_map = {**cap_color_map, "none":"f"}
    gill_attachment_map = {"adnate":"a","adnexed":"x","decurrent":"d","free":"e",
                        "sinuate":"s","pores":"p","none":"f"}
    stem_color_map = {**cap_color_map, "none":"f"}
    habitat_map = {"grasses":"g","leaves":"l","meadows":"m","paths":"p",
                "heaths":"h","urban":"u","waste":"w","woods":"d"}
    season_map = {"spring":"s","summer":"u","autumn":"a","winter":"w"}
    has_ring_map = {"yes":"t", "none":"f"}
    does_bruise_or_bleed_map = {"yes":"t","no":"f"}

    default_option = "--- select ---"

    col_a1, col_a2 = st.columns(2)

    color_options = [default_option] + [
        "🟤 Brown"       if name=="brown"      else
        "📔 Buff"        if name=="buff"       else
        "🩶 Gray"        if name=="gray"       else
        "🟢 Green"       if name=="green"      else
        "🩷 Pink"        if name=="pink"       else
        "🟣 Purple"      if name=="purple"     else
        "🔴 Red"         if name=="red"        else
        "⚪ White"       if name=="white"      else
        "🟡 Yellow"      if name=="yellow"     else
        "🔵 Blue"        if name=="blue"       else
        "🟠 Orange"      if name=="orange"     else
        "⚫ Black"       if name=="black"      else
        name.capitalize()
        for name in cap_color_map.keys()
    ]

    other_color_options = [default_option] + [
        "🟤 Brown"       if name=="brown"      else
        "📔 Buff"        if name=="buff"       else
        "🩶 Gray"        if name=="gray"       else
        "🟢 Green"       if name=="green"      else
        "🩷 Pink"        if name=="pink"       else
        "🟣 Purple"      if name=="purple"     else
        "🔴 Red"         if name=="red"        else
        "⚪ White"       if name=="white"      else
        "🟡 Yellow"      if name=="yellow"     else
        "🔵 Blue"        if name=="blue"       else
        "🟠 Orange"      if name=="orange"     else
        "⚫ Black"       if name=="black"      else
        "❌ None"        if name=="none"       else
        name.capitalize()
        for name in stem_color_map.keys()
    ]

    season_options = [default_option] + [
        "🌼 spring"      if name=="spring"      else
        "☀️ summer"      if name=="summer"      else
        "🍂 autumn"      if name=="autumn"      else
        "❄️ winter"      if name=="winter"      else
        name.capitalize()
        for name in season_map.keys()
    ]

    habitat_options = [default_option] +[
        "🌱 Grasses"     if name=="grasses"     else
        "🍂 Leaves"      if name=="leaves"      else
        "🌾 Meadows"     if name=="meadows"     else
        "🛤️ Paths"       if name=="paths"       else
        "🌿 Heaths"      if name=="heaths"      else
        "🏙️ Urban"       if name=="urban"       else
        "🗑️ Waste"       if name=="waste"       else
        "🌳 Woods"       if name=="woods"       else
        name.capitalize()
        for name in habitat_map.keys()
    ]

    habitat_help = (
    "**Grasses 🌱:** open grassy areas or lawns.\n"
    "**Leaves 🍂:** forest floor, under leaf litter.\n"
    "**Meadows 🌾:** large meadows or fields.\n"
    "**Heaths 🌿:** nutrient-poor lands with low shrubs.\n"
    "**Urban 🏙️:** city, town areas.\n"
    "**Waste 🗑️:** disturbed areas, garbage, roadsides.\n"
    "**Woods 🌳:** forested areas."
    )

    bruise_bleed_help = (
        "**Bruising 🔵:** color change after being pressed or cut (no liquid). \n"
        "**Bleeding 🩸:** colored liquid comes out when cut. \n"
    )

    # ░░ INPUTS ░░
    with col_a1:
        cap_shape = st.selectbox("🔘 Cap Shape (🚨required)", options=[default_option] + list(cap_shape_map.keys()))
        with st.expander("(⁰▿⁰)✋ Need a hint? Check out the cap shapes!"):
            st.image("assets/cap_shapes.png", width="stretch")
        cap_color = st.selectbox("🎨 Cap Color (🚨required)", options=color_options)
        stem_color = st.selectbox("🖍️ Stem Color (🚨required)", options=other_color_options)
        gill_color = st.selectbox("🌈 Gill Color (🚨required)", options=other_color_options)
        gill_attachment = st.selectbox("🔗 Gill Attachment (🚨required)", options=[default_option] + list(gill_attachment_map.keys()),)
        with st.expander("(⁰▿⁰)✋ Need a hint? Check out the gill attachments!"):
            st.image("assets/gills_attachments.png", width="stretch")

    with col_a2:
        habitat = st.selectbox("🌿 Habitat", options=habitat_options, help=habitat_help)
        season = st.selectbox("🍂 Season", options=season_options)
        has_ring = st.selectbox("💍 Ring ", options= [default_option] + list(has_ring_map.keys()))
        does_bruise_or_bleed = st.selectbox("🩸 Bruises or Bleeds?", options= [default_option] + list(does_bruise_or_bleed_map.keys()), help=bruise_bleed_help)

    # ░░ IMAGE UPLOAD ░░
        st.divider()
        st.markdown("📸 **Upload a photo** *(optional for tabular predict — enables image models)*")
        uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png", "heic"])
        if uploaded_image:
            st.image(uploaded_image, width=200)

    # ░░ REQUIRED FIELDS CHECK ░░
    required_fields = {
        "Cap Shape": cap_shape,
        "Cap Color": cap_color,
        "Stem Color": stem_color,
        "Gill Color": gill_color,
        "Gill Attachment": gill_attachment,
    }

    missing = [name for name, value in required_fields.items() if value == default_option or value.lower() == "none"]

    # ░░ Display the results ░░
    def show_result(poisonous, probability=None):
        """
        Display prediction result with normalized confidence score.
        """
        if probability is not None:
            confidence = float(probability)
            if confidence <= 1:
                confidence *= 100
            prob_text = f" with {confidence:.1f}% confidence"
        else:
            prob_text = ""

        if poisonous:
            st.error(f"☠️ **Poisonous!** Do not eat this mushroom{prob_text}.")
        else:
            st.success(f"✅ **Edible!** This mushroom looks safe to eat{prob_text}.")

    # ░░  Setup for emoji label ░░
    def parse_label(label, mapping):
        """
        Extract raw key from emoji label
        """

        for key in mapping.keys():
            if key.lower() in label.lower():
                return key
        return label

    # ░░   ░░
    def encode_optional(value, mapping):
        """
        Encode an optional categorical feature selected from a Streamlit selectbox.
        """
        if value == default_option:
            return None
        return mapping.get(parse_label(value, mapping))

    # ═════════════════════ API ═════════════════════
    col_b1, col_b2, col_b3 = st.columns(3)

    tab_params = {
        "cap_shape":             cap_shape_map.get(parse_label(cap_shape, cap_shape_map), cap_shape),
        "cap_color":             cap_color_map.get(parse_label(cap_color, cap_color_map), cap_color),
        "gill_attachment":       gill_attachment_map.get(parse_label(gill_attachment, gill_attachment_map), gill_attachment),
        "gill_color":            gill_color_map.get(parse_label(gill_color, gill_color_map), gill_color),
        "stem_color":            stem_color_map.get(parse_label(stem_color, stem_color_map), stem_color),
        "does_bruise_or_bleed":  encode_optional(does_bruise_or_bleed, does_bruise_or_bleed_map),
        "has_ring":              encode_optional(has_ring, has_ring_map),
        "habitat":               encode_optional(habitat, habitat_map),
        "season":                encode_optional(season, season_map),
    }

    tab_params = {k: v for k, v in tab_params.items() if v is not None}

    # ░░ PREDICT TABULAR ░░
    if missing:
        st.warning(f"👮‍♀️✋ Please fill in 🚨 required fields: {', '.join(missing)}")

    with col_b1:
        if st.button("📊 Tabular predict", disabled = bool(missing)):
            placeholder = st.empty()
            placeholder.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExYXZmZmZkbThobXNiMG92NG54YjI5dnU5M2o4Ymxrc2xmc2dzM2IyeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/vdL0YzchVNlxc4ikZh/giphy.gif")
            try:

                # corresponds to the part 'def predict_tab' & @app.get in fast.py
                response = requests.get(
                    f"{API_URL}/predict_tab",
                    params=tab_params
                )
                # placeholder.empty()
                if response.status_code == 200:
                    res = response.json()
                    placeholder.empty()
                    show_result(res.get('poisonous'), res.get('probability'))
                else:
                    st.error(f"🚨 API Error {response.status_code}")
            except Exception as e:
                placeholder.empty()
                st.error(f"❌🕵 Cannot reach API: {e}")

    # ░░ PREDICT IMAGE ░░
    if missing:
        st.warning(f"👮‍♀️✋ Please fill in 🚨 required fields: {', '.join(missing)}")

    with col_b2:
        if st.button("📷 Image predict", disabled=not uploaded_image or bool(missing)):
            placeholder = st.empty()
            placeholder.image("https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMHBibWM3ejVubTN6ZGNiNnV1OWYwaGx1ZzZhNDdpa3F2YTh2amNkdSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/lwPZlz1u7s3hnY69q3/giphy.gif")
            try:
                response = requests.post(
                    f"{API_URL}/predict_img?model=dinov2_baseline",
                    files={"file": (uploaded_image.name,
                                    uploaded_image.getvalue(),
                                    uploaded_image.type)}
                )
                if response.status_code == 200:
                    res = response.json()
                    placeholder.empty()
                    show_result(res.get("poisonous"), res.get("probability"))
                else:
                    placeholder.empty()
                    st.error(f"🚨 API Error: {response.status_code}")
                    placeholder.empty()
            except Exception as e:
                placeholder.empty()
                st.error(f"❌🕵 Cannot reach API: {e}")

    # ░░ PREDICT ALL ░░
    if missing:
        st.warning(f"👮‍♀️✋ Please fill in 🚨 required fields: {', '.join(missing)}")

    with col_b3:
        if st.button(f"🔮 Predict All", disabled=not uploaded_image or bool(missing)):
            placeholder = st.empty()
            placeholder.image("https://64.media.tumblr.com/tumblr_m69168RumW1qmpg90o1_500.gifv")
            try:

                # corresponds to the part 'predict_all' in fast.py
                response = requests.post(
                    f'{API_URL}/predict_all',
                    params=tab_params,
                    files={"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                )

                if response.status_code == 200:
                    data = response.json()
                    placeholder.empty()

                    st.subheader("📊 Tabular models")
                    for model_name, res in data.get("tab_models", {}).items():
                        st.markdown(f"**{model_name}**")
                        show_result(
                            res.get("poisonous"),
                            res.get("probability")
                        )

                    st.subheader("📷 Image models")
                    for model_name, res in data.get("img_models", {}).items():
                        st.markdown(f"**{model_name}**")
                        show_result(
                            res.get("poisonous"),
                            res.get("probability")
                        )
                else:
                    placeholder.empty()
                    st.error(f"🚨 API Error: {response.status_code}")
                    placeholder.empty()
            except Exception as e:
                placeholder.empty()
                st.error(f'❌🕵 Cannot reach API: {e}')
