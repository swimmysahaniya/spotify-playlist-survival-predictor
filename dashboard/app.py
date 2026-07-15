import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="Spotify Playlist Survival Predictor",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color:#0E1117;
}

h1,h2,h3 {
    color:#1DB954;
}

div[data-testid="metric-container"]{
    background-color:#262730;
    border-radius:10px;
    padding:15px;
}

.stButton>button{
    width:100%;
    background:#1DB954;
    color:white;
    border-radius:8px;
    height:3em;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("models/final_gradient_boosting.pkl")


model = load_model()

st.title("🎵 Spotify Playlist Survival Predictor")

st.markdown("""
Predict how long a song is likely to remain on the
**Spotify Spain Top 50 Playlist**
using Machine Learning.
""")

st.divider()

# Feature importance
st.subheader("📈 Model Performance")

c1,c2,c3=st.columns(3)

c1.metric("R²","0.943")

c2.metric("RMSE","27.69")

c3.metric("MAE","10.03")

st.divider()

st.subheader("🎯 Prediction")

with st.form("prediction_form"):

    st.subheader("🎵 Song Information")

    song_name = st.text_input(
        "Song Name",
        placeholder="e.g. Espresso"
    )

    artist_name = st.text_input(
        "Artist Name",
        placeholder="e.g. Sabrina Carpenter"
    )

    # -----------------------------
    # Chart Performance
    # -----------------------------
    with st.expander("📈 Chart Performance", expanded=True):
        best_rank = st.number_input(
            "Best Rank", 1, 50, 10
        )

        worst_rank = st.number_input(
            "Worst Rank", 1, 50, 30
        )

        avg_rank = st.number_input(
            "Average Rank", 1, 50, 20
        )

        initial_rank = st.number_input(
            "Initial Rank", 1, 50, 15
        )

        rank_improvement = st.number_input(
            "Rank Improvement", 0, 50, 12
        )

        rank_range = st.number_input(
            "Rank Range", 0, 50, 20
        )

        rank_volatility = st.number_input(
            "Rank Volatility",
            0.0,
            30.0,
            6.5
        )

        avg_daily_rank_change = st.number_input(
            "Average Daily Rank Change",
            0.0,
            10.0,
            1.5
        )

    # -----------------------------
    # Popularity
    # -----------------------------
    with st.expander("🔥 Popularity"):
        peak_popularity = st.slider(
            "Peak Spotify Popularity",
            0,
            100,
            90
        )

        avg_popularity = st.slider(
            "Average Spotify Popularity",
            0,
            100,
            82
        )

        popularity_gap = st.number_input(
            "Popularity Gap",
            0,
            100,
            8
        )

    # -----------------------------
    # Playlist Behaviour
    # -----------------------------
    with st.expander("🎵 Playlist Behaviour"):
        playlist_entries = st.number_input(
            "Playlist Entries",
            1,
            500,
            70
        )

        days_to_peak = st.number_input(
            "Days to Peak",
            0,
            200,
            15
        )

        stability_score = st.number_input(
            "Stability Score",
            0,
            300,
            80
        )

    predict = st.form_submit_button(
        "🎯 Predict Playlist Survival",
        use_container_width=True,
        type="primary"
    )

sample = pd.DataFrame({
    "best_rank": [best_rank],
    "worst_rank": [worst_rank],
    "avg_rank": [avg_rank],
    "initial_rank": [initial_rank],
    "rank_improvement": [rank_improvement],
    "rank_range": [rank_range],
    "rank_volatility": [rank_volatility],
    "avg_daily_rank_change": [avg_daily_rank_change],
    "peak_popularity": [peak_popularity],
    "avg_popularity": [avg_popularity],
    "popularity_gap": [popularity_gap],
    "playlist_entries": [playlist_entries],
    "days_to_peak": [days_to_peak],
    "stability_score": [stability_score]
})

if predict:
    prediction = model.predict(sample)

    pred = prediction[0]

    st.metric(
        "Estimated Playlist Survival",
        f"🎵 {pred:.0f} Days"
    )

    progress = min(int(pred / 365 * 100), 100)

    st.write("### Playlist Longevity")

    if pred >= 250:
        st.success("🟢 Excellent Longevity")
        st.progress(progress)
        st.caption(f"{progress}% of one year")

    elif pred >= 150:
        st.warning("🟡 Moderate Longevity")
        st.progress(progress)
        st.caption(f"{progress}% of one year")

    else:
        st.error("🔴 Short Playlist Lifespan")
        st.progress(progress)
        st.caption(f"{progress}% of one year")

    st.divider()

    summary = pd.DataFrame({
        "Feature": [
            "Best Rank",
            "Worst Rank",
            "Average Rank",
            "Initial Rank",
            "Peak Popularity",
            "Average Popularity",
            "Playlist Entries",
            "Days to Peak",
            "Stability Score"
        ],
        "Value": [
            best_rank,
            worst_rank,
            avg_rank,
            initial_rank,
            peak_popularity,
            avg_popularity,
            playlist_entries,
            days_to_peak,
            stability_score
        ]
    })

    st.subheader("📋 Prediction Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption(
        f"Prediction generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}"
    )

if predict:
    if song_name:
        st.success(
            f"Prediction completed for **{song_name}** by **{artist_name}**"
        )

st.divider()

st.subheader("📊 Global Feature Importance")


@st.cache_data
def load_feature_importance():
    return pd.read_csv("data/feature_importance.csv")


importance = load_feature_importance()

top10 = (
    importance
    .sort_values("Importance", ascending=False)
    .head(10)
)

top10 = top10.sort_values("Importance")

fig, ax = plt.subplots(figsize=(9,5))

fig.patch.set_facecolor("#0E1117")
ax.set_facecolor("#0E1117")

ax.tick_params(colors="white")

ax.xaxis.label.set_color("white")

ax.title.set_color("white")

for spine in ax.spines.values():
    spine.set_color("white")

ax.barh(
    top10["Feature"],
    top10["Importance"],
    color="#1DB954"
)

ax.set_xlabel("Importance")
ax.set_ylabel("")
ax.set_title("Top Predictive Features")

st.pyplot(fig)

st.info("""
**Key Insight**

The model relies most heavily on:

• Playlist Entries
• Stability Score
• Rank Volatility

These features have the strongest influence on predicting how long a song remains in the Spotify Top 50 playlist.
""")

st.divider()

st.subheader("📊 Dataset Overview")

d1,d2,d3,d4 = st.columns(4)

d1.metric("Songs", "575")

d2.metric("Features", "14")

d3.metric("Algorithm","GBR")

d4.metric("Train/Test","460 / 115")

# About project
st.markdown("---")

st.subheader("📖 About Project")

st.markdown("""
### 🎯 Project Objective

This application uses a **Gradient Boosting Regressor** to estimate how long a song is expected to remain on the Spotify Spain Top 50 playlist.

The prediction is based on engineered chart-performance, popularity, and playlist-behaviour features derived from historical Spotify ranking data.
""")

st.markdown("""
### 🛠 Built With

- Python
- Scikit-Learn
- Pandas
- Streamlit
- Matplotlib

**Model:** Gradient Boosting Regressor

**Dataset:** Spotify Spain Top 50

**Target:** Playlist Survival (Days)
""")

st.divider()

with st.expander("ℹ Model Details"):
    st.markdown("""
    ### Model Details

    - **Algorithm:** Gradient Boosting Regressor
    - **Training Samples:** 460
    - **Testing Samples:** 115
    - **Features:** 14
    - **Hyperparameter Tuning:** GridSearchCV
    - **Best Parameters:**
        - **n_estimators** = 300
        - **learning_rate** = 0.1
        - **max_depth** = 2
        - **subsample** = 1.0
    - **Target:** Playlist Survival (Days)
    - **R² Score:** 0.943
    - **RMSE:** 27.69
    - **MAE:** 10.03
    """)

st.divider()

st.info(
    "This prediction is generated using a trained machine learning model and should be interpreted as an estimate based on historical Spotify playlist data."
)

st.divider()

st.subheader("🔗 GitHub / Live Demo")

st.markdown("""
GitHub: https://github.com/swimmysahaniya/spotify-playlist-survival-predictor.git
""")

st.markdown("""
**Live Demo:** https://spotify-playlist-survival-predictor.streamlit.app
""")

st.divider()

st.markdown(
"""
<div style="text-align:center; padding:25px 0;">

<h4 style="color:#BDBDBD; margin-bottom:8px;">
Developed by
</h4>

<h2 style="color:#1DB954; margin:0;">
Swimmy Sahaniya
</h2>

<p style="color:#D0D0D0; margin-top:10px;">
Python • Scikit-Learn • Pandas • Streamlit
</p>

<p style="color:#AFAFAF;">
Machine Learning Portfolio Project
</p>

<p style="color:#7F7F7F;">
Spotify Playlist Survival Prediction • 2026
</p>

</div>
""",
unsafe_allow_html=True
)