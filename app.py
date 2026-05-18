import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Engagement Score Predictor 💘",
    page_icon="💘",
    layout="wide",
)


FEATURES_LIST = [
    "income_bracket",
    "education_level",
    "app_usage_time_min",
    "swipe_right_ratio",
    "likes_received",
    "mutual_matches",
    "profile_pics_count",
    "bio_length",
    "message_sent_count",
    "emoji_usage_rate",
    "last_active_hour",
    "swipe_time_of_day",
    "age",
    "bmi",
    "interest_anime",
    "interest_art",
    "interest_astrology",
    "interest_binge_watching",
    "interest_board_games",
    "interest_cars",
    "interest_clubbing",
    "interest_coding",
    "interest_cooking",
    "interest_crafting",
    "interest_diy",
    "interest_dancing",
    "interest_fashion",
    "interest_fitness",
    "interest_foodie",
    "interest_gaming",
    "interest_gardening",
    "interest_hiking",
    "interest_history",
    "interest_investing",
    "interest_k_pop",
    "interest_languages",
    "interest_mma",
    "interest_makeup",
    "interest_meditation",
    "interest_memes",
    "interest_motorcycling",
    "interest_movies",
    "interest_music",
    "interest_painting",
    "interest_parenting",
    "interest_pets",
    "interest_photography",
    "interest_podcasts",
    "interest_poetry",
    "interest_politics",
    "interest_reading",
    "interest_running",
    "interest_skating",
    "interest_sneaker_culture",
    "interest_social_activism",
    "interest_spirituality",
    "interest_stand_up_comedy",
    "interest_startups",
    "interest_tattoos",
    "interest_tech",
    "interest_traveling",
    "interest_writing",
    "interest_yoga",
    "gender_grouped_Female",
    "gender_grouped_Male",
    "gender_grouped_Prefer Not to Say",
    "sexual_orientation_Bisexual",
    "sexual_orientation_Demisexual",
    "sexual_orientation_Gay",
    "sexual_orientation_Lesbian",
    "sexual_orientation_Pansexual",
    "sexual_orientation_Queer",
    "sexual_orientation_Straight",
    "body_vibe_Baseline",
    "body_vibe_Curvy_Plus",
    "location_density_Low_Density",
    "location_density_Med_Density",
    "relationship_intent_Exploring",
    "relationship_intent_Friends Only",
    "relationship_intent_Hookups",
    "relationship_intent_Networking",
    "relationship_intent_Serious Relationship",
]


INCOME_MAP = {
    "Very Low": 0,
    "Low": 1,
    "Lower-Middle": 2,
    "Middle": 3,
    "Upper-Middle": 4,
    "High": 5,
    "Very High": 6,
}

EDUCATION_MAP = {
    "No Formal Education": 0,
    "High School": 1,
    "Diploma": 2,
    "Associate": 3,
    "Bachelor": 4,
    "Master": 5,
    "PhD": 6,
    "Postdoc": 7,
}

SWIPE_TIME_OF_DAY_MAP = {
    0: "Morning",
    1: "Late Morning",
    2: "Afternoon",
    3: "Evening",
    4: "Night",
}

MODEL_PATH_PRIMARY = Path(__file__).with_name("model.pkl")
MODEL_PATH_FALLBACK = Path(__file__).with_name("model (1).pkl")
SCALER_PATH = Path(__file__).with_name("scaler.pkl")


@st.cache_resource
def load_resources():
    model_path = MODEL_PATH_PRIMARY if MODEL_PATH_PRIMARY.exists() else MODEL_PATH_FALLBACK
    with model_path.open("rb") as fh:
        model_obj = pickle.load(fh)

    scaler_obj = None
    if SCALER_PATH.exists():
        with SCALER_PATH.open("rb") as fh:
            scaler_obj = pickle.load(fh)

    return model_obj, scaler_obj


def build_input_dataframe(
    age: int,
    income_bracket: str,
    education_level: str,
    swipe_right_ratio: float,
    likes_received: int,
    mutual_matches: int,
    message_sent_count: int,
    app_usage_time_min: int,
    emoji_usage_rate: float,
    profile_pics_count: int,
    bio_length: int,
) -> pd.DataFrame:
    row = {feature: 0.0 for feature in FEATURES_LIST}
    row["bmi"] = 22.0
    row["last_active_hour"] = 12.0
    row["swipe_time_of_day"] = 2.0
    row["income_bracket"] = float(INCOME_MAP[income_bracket])
    row["education_level"] = float(EDUCATION_MAP[education_level])
    row["age"] = float(age)
    row["swipe_right_ratio"] = float(swipe_right_ratio)
    row["likes_received"] = float(likes_received)
    row["mutual_matches"] = float(mutual_matches)
    row["message_sent_count"] = float(message_sent_count)
    row["app_usage_time_min"] = float(app_usage_time_min)
    row["emoji_usage_rate"] = float(emoji_usage_rate)
    row["profile_pics_count"] = float(profile_pics_count)
    row["bio_length"] = float(bio_length)
    input_df = pd.DataFrame([[row[feature] for feature in FEATURES_LIST]], columns=FEATURES_LIST)
    return input_df


def engagement_label(prediction: int) -> str:
    if prediction == 0:
        return "Low Engagement 💔"
    if prediction == 1:
        return "Medium Engagement ⚠️"
    return "High Engagement 🔥💘"


def engagement_status(prediction: int) -> str:
    if prediction == 0:
        return "error"
    if prediction == 1:
        return "warning"
    return "success"


def color_for_prediction(prediction: int) -> str:
    if prediction == 0:
        return "#ff4d4f"
    if prediction == 1:
        return "#facc15"
    return "#22c55e"


model, scaler = load_resources()


st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #fff7fb 0%, #ffffff 45%, #f7fbff 100%);
        }
        .hero-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(255, 105, 180, 0.14);
            border-radius: 24px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 18px 45px rgba(255, 105, 180, 0.08);
        }
        .section-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 20px;
            padding: 1.25rem 1.3rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
        }
        .small-muted {
            color: rgba(71, 85, 105, 0.9);
            font-size: 0.95rem;
        }
        .result-box {
            border-radius: 20px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(0, 0, 0, 0.06);
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.76));
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin:0; font-size:2.2rem;">AI Engagement Score Predictor 💘</h1>
        <p class="small-muted" style="margin:0.4rem 0 0 0;">Predict dating app engagement tier.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Input Panel 💘")
st.sidebar.caption("Enter the 11 visible user inputs. The remaining model features are filled with neutral defaults.")

with st.sidebar.form("prediction_inputs"):
    age = st.slider("Age", 18, 60, 28)
    income_bracket = st.selectbox(
        "Income bracket",
        list(INCOME_MAP.keys()),
        index=3,
    )
    education_level = st.selectbox(
        "Education level",
        list(EDUCATION_MAP.keys()),
        index=4,
    )
    swipe_right_ratio = st.slider("Swipe right ratio", 0.0, 1.0, 0.42, 0.01)
    likes_received = st.number_input("Likes received", min_value=0, value=12, step=1)
    mutual_matches = st.number_input("Mutual matches", min_value=0, value=4, step=1)
    message_sent_count = st.number_input("Messages sent", min_value=0, value=9, step=1)
    app_usage_time_min = st.slider("App usage time (min)", 0, 300, 75)
    emoji_usage_rate = st.slider("Emoji usage rate", 0.0, 1.0, 0.35, 0.01)
    profile_pics_count = st.slider("Profile pics count", 1, 10, 5)
    bio_length = st.slider("Bio length", 0, 500, 140)
    submitted = st.form_submit_button("Rerun prediction 🔄")

st.sidebar.caption("Adjust values, then click the button to refresh the prediction.")

input_df = build_input_dataframe(
    age=age,
    income_bracket=income_bracket,
    education_level=education_level,
    swipe_right_ratio=swipe_right_ratio,
    likes_received=likes_received,
    mutual_matches=mutual_matches,
    message_sent_count=message_sent_count,
    app_usage_time_min=app_usage_time_min,
    emoji_usage_rate=emoji_usage_rate,
    profile_pics_count=profile_pics_count,
    bio_length=bio_length,
)

# Apply scaler to numerical columns to match training preprocessing
numerical_columns = [
    'income_bracket', 'education_level', 'app_usage_time_min', 'swipe_right_ratio', 'likes_received',
    'mutual_matches', 'profile_pics_count', 'bio_length', 'message_sent_count', 'emoji_usage_rate',
    'last_active_hour', 'swipe_time_of_day', 'age', 'bmi'
]
if scaler is not None:
    try:
        # If the scaler was fitted on a DataFrame with named features, respect that ordering
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(scaler.feature_names_in_)
            temp = input_df.reindex(columns=scaler_cols, fill_value=0.0)
            scaled_all = scaler.transform(temp)
            scaled_all_df = pd.DataFrame(scaled_all, columns=scaler_cols)
            # Map back only the numerical columns we need to scale
            for col in numerical_columns:
                if col in scaled_all_df.columns:
                    input_df[col] = scaled_all_df[col]
                else:
                    # fallback: leave original value
                    st.warning(f"Scaled output missing column {col}; leaving original value.")
        else:
            input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    except Exception as e:
        st.warning(f"Scaler transform failed: {e}")
else:
    st.warning("Scaler not found: skipping numerical scaling (place scaler.pkl next to app.py).")

prediction = int(model.predict(input_df)[0])
prediction_label = engagement_label(prediction)
prediction_status = engagement_status(prediction)

probabilities = None
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_df)[0]
    probability_df = pd.DataFrame(
        {
            "Engagement Tier": ["Low", "Medium", "High"],
            "Probability": probabilities,
        }
    ).set_index("Engagement Tier")
else:
    probability_df = pd.DataFrame(
        {"Engagement Tier": ["Low", "Medium", "High"], "Probability": [np.nan, np.nan, np.nan]}
    ).set_index("Engagement Tier")

col_left, col_right = st.columns([1.05, 0.95], gap="large")

with col_left:
    st.subheader("Prediction Output 🔮")
    st.write("Live prediction from the loaded pickle model.")
    color = color_for_prediction(prediction)
    # Use Streamlit alert boxes for interpretation style
    if prediction_status == "success":
        st.success(prediction_label)
    elif prediction_status == "warning":
        st.warning(prediction_label)
    else:
        st.error(prediction_label)
    st.markdown(
        f"""
        <div class="result-box">
            <div style="font-size:0.95rem; color:#475569; margin-bottom:0.35rem;">Predicted engagement tier</div>
            <div style="font-size:1.8rem; font-weight:800; color:{color};">{prediction_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Top factors influencing engagement: mutual matches, message activity, likes received.")

    if probabilities is not None:
        confidence = float(np.max(probabilities))
        st.metric("Model confidence", f"{confidence:.1%}")
        st.bar_chart(probability_df)
    else:
        st.warning("Probability output is not available for this model.")

with col_right:
    st.subheader("Insights 📈")
    st.metric("Predicted class", f"{prediction} • {prediction_label}")
    st.metric("User age", age)
    st.metric("Likes received", likes_received)
    st.metric("Mutual matches", mutual_matches)
    st.metric("Messages sent", message_sent_count)
    st.caption(
        f"Baseline values used for hidden features: bmi=22.0, last_active_hour=12, swipe_time_of_day=2 ({SWIPE_TIME_OF_DAY_MAP[2]})."
    )
st.markdown("</div>", unsafe_allow_html=True)
with st.expander("Show model input dataframe"):
    st.dataframe(input_df, use_container_width=True)
