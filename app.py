import streamlit as st
import pandas as pd
import numpy as np
import pickle
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    go = None
    PLOTLY_AVAILABLE = False


st.set_page_config(
    page_title="AI Engagement Predictor 💘",
    page_icon="💘",
    layout="wide",
    initial_sidebar_state="expanded",
)


INCOME_MAP = {"Low": 0, "Medium": 1, "High": 2}
EDUCATION_MAP = {"High School": 0, "Diploma": 1, "Bachelor": 2, "Master": 3, "PhD": 4}


@st.cache_resource
def load_resources():
    with open("model.pkl", "rb") as model_file:
        model_obj = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler_obj = pickle.load(scaler_file)
    return model_obj, scaler_obj


model = None
scaler = None
feature_order = []

try:
    model, scaler = load_resources()
    feature_order = list(getattr(scaler, "feature_names_in_", []))
except Exception:
    model = None
    scaler = None
    feature_order = []

if not feature_order:
    feature_order = [f"feature_{idx}" for idx in range(82)]


def build_feature_vector(inputs):
    vector = pd.Series(0.0, index=feature_order, dtype=float)
    if "bmi" in vector.index:
        vector.loc["bmi"] = 22.0
    if "last_active_hour" in vector.index:
        vector.loc["last_active_hour"] = 12.0
    if "swipe_time_of_day" in vector.index:
        vector.loc["swipe_time_of_day"] = 2.0

    mapping = {
        "age": float(inputs["age"]),
        "bio_length": float(inputs["bio_length"]),
        "profile_pics_count": float(inputs["profile_pics_count"]),
        "income_bracket": float(INCOME_MAP[inputs["income_bracket"]]),
        "education_level": float(EDUCATION_MAP[inputs["education_level"]]),
        "app_usage_time_min": float(inputs["app_usage_time_min"]),
        "swipe_right_ratio": float(inputs["swipe_right_ratio"]),
        "likes_received": float(inputs["likes_received"]),
        "mutual_matches": float(inputs["mutual_matches"]),
        "message_sent_count": float(inputs["message_sent_count"]),
        "emoji_usage_rate": float(inputs["emoji_usage_rate"]),
    }

    for feature_name, feature_value in mapping.items():
        if feature_name in vector.index:
            vector.loc[feature_name] = feature_value

    return pd.DataFrame([vector.values], columns=feature_order)


def scale_features(frame):
    if scaler is None:
        return frame
    try:
        scaled = scaler.transform(frame.reindex(columns=feature_order, fill_value=0.0))
        return pd.DataFrame(scaled, columns=feature_order)
    except Exception:
        return frame.reindex(columns=feature_order, fill_value=0.0)


def get_probability_vector(scaled_frame):
    if model is None:
        return np.array([0.0, 0.0, 0.0])
    try:
        return np.asarray(model.predict_proba(scaled_frame)[0], dtype=float)
    except Exception:
        try:
            scores = np.asarray(model.decision_function(scaled_frame)[0], dtype=float)
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            return exp_scores / exp_scores.sum()
        except Exception:
            return np.array([0.0, 0.0, 0.0])


def prediction_message(label):
    if label == 0:
        return "💔 Low Engagement — High Ghosting / Churn Risk", st.error
    if label == 1:
        return "⚠️ Medium Engagement — Passive Interaction Pattern", st.warning
    return "🔥 High Engagement — Strong Active Connection", st.success


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Playfair Display', serif;
            letter-spacing: -0.02em;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 207, 220, 0.60), transparent 30%),
                radial-gradient(circle at top right, rgba(255, 184, 200, 0.40), transparent 28%),
                radial-gradient(circle at bottom right, rgba(194, 159, 255, 0.18), transparent 30%),
                linear-gradient(180deg, #fff8fb 0%, #fff4f7 45%, #fffdfd 100%);
        }
        .main .block-container {
            padding-top: 2.8rem;
            padding-bottom: 2.8rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255, 244, 247, 0.98), rgba(251, 245, 255, 0.95));
            border-right: 1px solid rgba(170, 88, 120, 0.10);
            position: relative;
            overflow: hidden !important;
        }
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"],
        section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
            padding-bottom: 0 !important;
        }
        section[data-testid="stSidebar"] *::-webkit-scrollbar {
            width: 0 !important;
            height: 0 !important;
        }
        .sidebar-footer-wrap {
            width: calc(100% - 1.6rem);
            margin: 52vh 0 0 0.8rem !important;
            padding: 0;
        }
        .sidebar-footer {
            width: 100%;
            font-size: 0.84rem;
            line-height: 1.45;
            color: rgba(68, 31, 50, 0.88);
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(170, 88, 120, 0.14);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            box-shadow: 0 8px 20px rgba(130, 54, 79, 0.08);
            box-sizing: border-box;
        }
        .sidebar-footer-title {
            font-weight: 700;
            font-size: 0.80rem;
            letter-spacing: 0.02em;
            margin-bottom: 0.2rem;
            color: rgba(107, 41, 72, 0.96);
        }
        .hero-shell {
            padding: 1.4rem 1.5rem;
            border-radius: 24px;
            border: 1px solid rgba(153, 61, 94, 0.12);
            background:
                linear-gradient(135deg, rgba(255,255,255,0.93), rgba(255,240,245,0.88)),
                radial-gradient(circle at top right, rgba(255, 171, 192, 0.18), transparent 42%);
            box-shadow: 0 18px 36px rgba(130, 54, 79, 0.10);
        }
        .subtle-note {
            color: rgba(68, 31, 50, 0.78);
            font-size: 0.98rem;
        }
        .insight-banner {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(90deg, rgba(255, 236, 241, 0.96), rgba(255, 247, 250, 0.96), rgba(247, 239, 255, 0.92));
            border: 1px solid rgba(180, 76, 113, 0.14);
            box-shadow: 0 10px 24px rgba(130, 54, 79, 0.06);
        }
        .section-chip {
            display: inline-block;
            padding: 0.34rem 0.7rem;
            margin: 1.05rem 0 0.95rem 0;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(255, 127, 143, 0.16), rgba(125, 183, 255, 0.16));
            color: #6b2948;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .glow-card {
            padding: 1rem 1.05rem;
            margin: 0.5rem 0 1rem 0;
            border-radius: 16px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(255, 242, 247, 0.92));
            border: 1px solid rgba(180, 76, 113, 0.10);
            box-shadow: 0 8px 22px rgba(130, 54, 79, 0.06);
        }
        .page-gap {
            height: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Live Predictor",
        "🧠 Behavioral Insights",
        "🏆 Model Benchmarks",
    ],
)

st.sidebar.markdown(
    """
    <div class="sidebar-footer-wrap">
        <div class="sidebar-footer">
            <div class="sidebar-footer-title">Group 4 OCC 7 WIA1006</div>
            Chua Pei Ying<br>
            Josephine Ding Jie Yu<br>
            Lim Hui Yun<br>
            Ng Geok Liu<br>
            Ng Shao Ern
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if page == "🔮 Live Predictor":
    st.markdown(
        """
        <div class="hero-shell">
            <div style="font-size:0.88rem;letter-spacing:0.12em;text-transform:uppercase;color:#a24c68;font-weight:700;">✨ Live Predictor 🔮 💘</div>
            <h1 style="margin:0.35rem 0 0.35rem 0;color:#3f1830;">Real-Time Engagement Prediction</h1>
            <div class="subtle-note">Enter a user profile and analyze the engagement tier with the production model. 💫</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="page-gap"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.05, 1.1])
    with col1:
        st.markdown('<div class="section-chip">Profile Inputs 🧬</div>', unsafe_allow_html=True)
        age = st.slider("Age", 18, 60, 28)
        bio_length = st.slider("Bio Length", 0, 500, 150)
        profile_pics_count = st.slider("Profile Pics Count", 1, 10, 4)

        st.markdown('<div class="section-chip">Demographics 🌍</div>', unsafe_allow_html=True)
        income_bracket = st.selectbox("Income Bracket", ["Low", "Medium", "High"], index=1)
        education_level = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master", "PhD"], index=2)

        st.markdown('<div class="section-chip">Behavior 📲</div>', unsafe_allow_html=True)
        app_usage_time_min = st.slider("App Usage Time (min)", 0, 300, 72)
        swipe_right_ratio = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.42, 0.01)

        st.markdown('<div class="section-chip">Social Activity 💬</div>', unsafe_allow_html=True)
        likes_received = st.number_input("Likes Received", min_value=0, value=14, step=1)
        mutual_matches = st.number_input("Mutual Matches", min_value=0, value=5, step=1)
        message_sent_count = st.number_input("Messages Sent", min_value=0, value=28, step=1)
        emoji_usage_rate = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.31, 0.01)

        analyze = st.button("Analyze Engagement Profile 🔥", use_container_width=True)

    with col2:
        st.markdown('<div class="section-chip">Prediction Output 🌈</div>', unsafe_allow_html=True)
        if analyze:
            input_frame = build_feature_vector(
                {
                    "age": age,
                    "bio_length": bio_length,
                    "profile_pics_count": profile_pics_count,
                    "income_bracket": income_bracket,
                    "education_level": education_level,
                    "app_usage_time_min": app_usage_time_min,
                    "swipe_right_ratio": swipe_right_ratio,
                    "likes_received": likes_received,
                    "mutual_matches": mutual_matches,
                    "message_sent_count": message_sent_count,
                    "emoji_usage_rate": emoji_usage_rate,
                }
            )
            scaled_frame = scale_features(input_frame)

            if model is None:
                st.error("Model could not be loaded from model.pkl.")
            else:
                try:
                    prediction = int(model.predict(scaled_frame)[0])
                    proba_vector = get_probability_vector(scaled_frame)
                    label_text, render_fn = prediction_message(prediction)
                    render_fn(label_text)

                    # Probability visualization with emoji labels
                    engagement_labels = ["Low 💔", "Medium ⚠️", "High 🔥"]
                    probability_df = pd.DataFrame({
                        "Engagement Level": engagement_labels,
                        "Probability": proba_vector[:3] * 100.0
                    })
                    probability_df["Label"] = probability_df["Probability"].map(lambda value: f"{value:.1f}%")
                    
                    if PLOTLY_AVAILABLE:
                        fig_prob = px.bar(
                            probability_df.sort_values("Probability"),
                            x="Probability",
                            y="Engagement Level",
                            orientation="h",
                            text="Label",
                            color="Engagement Level",
                            color_discrete_map={"Low 💔": "#ff7f8f", "Medium ⚠️": "#f4b860", "High 🔥": "#7db7ff"},
                            title="Engagement Probability Distribution 🔮"
                        )
                        fig_prob.update_traces(textposition="inside")
                        fig_prob.update_layout(
                            height=360,
                            margin=dict(l=10, r=10, t=35, b=10),
                            showlegend=False,
                            xaxis_title="Probability (%)",
                            yaxis_title="",
                            xaxis=dict(range=[0, 100]),
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                    else:
                        st.bar_chart(probability_df.set_index("Engagement Level")["Probability"])
                    
                    # Confidence indicator
                    confidence = max(proba_vector)
                    if confidence < 0.6:
                        st.info("⚠️ Low confidence — mixed behavioral signals detected")
                    else:
                        st.success("🔥 High confidence prediction")
                    
                    st.caption("Model confidence reflects behavioral consistency in interaction patterns.")

                    st.markdown("---")
                    st.dataframe(input_frame[feature_order[:14]], use_container_width=True, height=220)
                except Exception:
                    st.error("Prediction failed for the provided profile.")
        else:
            st.info("Configure the profile inputs and click Analyze Engagement Profile 🔥")
            st.caption("The model uses an 82-feature vector with defaults for non-entered fields.")


elif page == "🧠 Behavioral Insights":
    st.title("🧠 Behavioral Insights Dashboard")

    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is required for this page. Please install `plotly` to view Behavioral Insights charts.")
    else:
        import os
        from pathlib import Path
        
        images_dir = Path("images")
        
        # 0. Top 10 Feature Importance
        with st.container():
            st.markdown("<h3 style='font-size: 24px;'>🏆 Top 10 Feature Importance (Random Forest)</h3>", unsafe_allow_html=True)
            img_path = images_dir / "top10features.png"
            if img_path.exists():
                st.image(str(img_path), width=700)
            else:
                st.error("Image not found: top10features.png")
            st.write("This chart displays the top 10 most important features from the Random Forest model. Likes received emerges as the most significant predictor of engagement, followed by message sent count and bio length. These features reveal that user popularity, communication activity, and profile completeness are the primary drivers of engagement on the platform.")
            st.write("")
        
        # 1. Distribution of Engagement Tiers
        with st.container():
            st.markdown("<h3 style='font-size: 24px;'>📊 Distribution of Engagement Tiers</h3>", unsafe_allow_html=True)
            img_path = images_dir / "distribution.png"
            if img_path.exists():
                st.image(str(img_path), width=700)
            else:
                st.error("Image not found: distribution.png")
            st.write("This plot shows the count of users in each engagement tier (Low, Medium, High). It appears the tiers are roughly balanced, which is expected since pd.qcut was used to create them.")
            st.write("")

        # 2. Messages Sent Count vs. Engagement Tier
        with st.container():
            st.markdown("<h3 style='font-size: 24px;'>💬 Messages Sent Count vs. Engagement Tier</h3>", unsafe_allow_html=True)
            img_path = images_dir / "messagecount.png"
            if img_path.exists():
                st.image(str(img_path), width=700)
            else:
                st.error("Image not found: messagecount.png")
            st.write("This box plot illustrates the relationship between the number of messages sent and engagement. Generally, users in higher engagement tiers (Medium and High) tend to send more messages, with the highest engagement tier showing the highest median and range of messages sent. This indicates that active communication is a strong indicator of engagement.")
            st.write("")

        # 3. Swipe Right Ratio vs. Engagement Tier
        with st.container():
            st.markdown("<h3 style='font-size: 24px;'>👍 Swipe Right Ratio vs. Engagement Tier</h3>", unsafe_allow_html=True)
            img_path = images_dir / "swiperightratio.png"
            if img_path.exists():
                st.image(str(img_path), width=700)
            else:
                st.error("Image not found: swiperightratio.png")
            st.write("This plot displays how the ratio of right swipes relates to engagement. We can see a slight upward trend, suggesting that users who swipe right more often (indicating more interest in profiles) tend to be in higher engagement tiers. This aligns with the idea that active participation in the core app functionality drives engagement.")
            st.write("")

        # 4. App Usage Time vs. Engagement Tier
        with st.container():
            st.markdown("<h3 style='font-size: 24px;'>⏱️ App Usage Time (Minutes) vs. Engagement Tier</h3>", unsafe_allow_html=True)
            img_path = images_dir / "appusagetime.png"
            if img_path.exists():
                st.image(str(img_path), width=700)
            else:
                st.error("Image not found: appusagetime.png")
            st.write("This box plot shows that users in higher engagement tiers (Medium and High) spend significantly more time on the app. This is an expected and strong correlation, as higher app usage directly translates to higher engagement.")
            st.write("")

        # 5. Emoji Usage Rate vs. Engagement Tier
        with st.container():
            st.markdown("<h3 style='font-size: 24px;'>😀 Emoji Usage Rate vs. Engagement Tier</h3>", unsafe_allow_html=True)
            img_path = images_dir / "emojiusagerate.png"
            if img_path.exists():
                st.image(str(img_path), width=700)
            else:
                st.error("Image not found: emojiusagerate.png")
            st.write("This plot reveals that users in higher engagement tiers, especially 'High', tend to have a higher emoji usage rate. This suggests that more expressive communicators are often more engaged with the platform.")
            st.write("")

        # 6. Relationship Intent Impact on Engagement Tier
        with st.container():
            st.markdown("<h3 style='font-size: 24px;'>💑 Relationship Intent Impact on Engagement Tier</h3>", unsafe_allow_html=True)
            img_path = images_dir / "relationship intent.png"
            if img_path.exists():
                st.image(str(img_path), width=700)
            else:
                st.error("Image not found: relationship intent.png")
            st.write("This bar chart breaks down the distribution of engagement tiers across different relationship intents. You can observe which relationship intents are more prevalent in 'Low', 'Medium', or 'High' engagement groups. For instance, 'Serious Relationship' intent seems to have a substantial presence across all tiers, while other intents might be more skewed towards certain engagement levels. This provides insight into how user intentions influence their engagement.")


else:
    st.markdown(
        """
        <div class="hero-shell">
            <div style="font-size:0.88rem;letter-spacing:0.12em;text-transform:uppercase;color:#a24c68;font-weight:700;">🏆 Model Benchmarks 📊</div>
            <h1 style="margin:0.35rem 0 0.35rem 0;color:#3f1830;">Production Model Benchmarking</h1>
            <div class="subtle-note">A clean comparison of tuned models and Auto-Sklearn. 🔍</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="page-gap"></div>', unsafe_allow_html=True)

    benchmarks = pd.DataFrame(
        [
            {"Model": "Logistic Regression (Tuned)", "Accuracy": 0.9965, "Precision": 0.9965, "Recall": 0.9965, "F1-score": 0.9965},
            {"Model": "Linear SVM (Tuned)", "Accuracy": 0.9923, "Precision": 0.9924, "Recall": 0.9923, "F1-score": 0.9923},
            {"Model": "Gradient Boosting (Tuned)", "Accuracy": 0.9752, "Precision": 0.9757, "Recall": 0.9752, "F1-score": 0.9753},
            {"Model": "Random Forest (Tuned)", "Accuracy": 0.9197, "Precision": 0.9209, "Recall": 0.9197, "F1-score": 0.9201},
            {"Model": "KNN (Tuned)", "Accuracy": 0.7111, "Precision": 0.7123, "Recall": 0.7111, "F1-score": 0.7116},
            {"Model": "Auto-Sklearn", "Accuracy": 0.9858, "Precision": 0.985926, "Recall": 0.9858, "F1-score": 0.985760},
        ]
    )

    ranking_df = benchmarks.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    st.dataframe(ranking_df, use_container_width=True, height=260)

    melted = ranking_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"], var_name="Metric", value_name="Score")
    if PLOTLY_AVAILABLE:
        fig_bench = px.bar(
            melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_map={"Accuracy": "#a855f7", "Precision": "#fb7185", "Recall": "#f59e0b", "F1-score": "#10b981"},
        )
        fig_bench.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10), xaxis_tickangle=-18, yaxis_title="Score")
        st.plotly_chart(fig_bench, use_container_width=True)
    else:
        bench_pivot = melted.pivot(index="Model", columns="Metric", values="Score").fillna(0.0)
        st.bar_chart(bench_pivot)

    st.markdown(
        """
        <div class="glow-card" style="font-weight:700;color:#4a2238;">
            Key Drivers of Sustained Engagement: Reciprocal Matching Behavior, Messaging Frequency, and Consistent App Interaction Patterns. 💘
        </div>
        """,
        unsafe_allow_html=True,
    )
