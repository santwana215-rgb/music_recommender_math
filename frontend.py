import streamlit as st
import pandas as pd
from backend import GENRES, calculate_linear_algebra_pipeline

st.set_page_config(
    page_title="Sound Match Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Sound Match Arcade · Linear Algebra Recommender")
st.caption("Implementation of Project 7: 'Norms, angles, and your movie choices'")

st.info(
    "Pipeline: 1. Rating Matrix (m x n)  →  "
    "2. Euclidean Metric ||u - v||  →  "
    "3. Mean-Centering (u - mean)  →  "
    "4. Cosine Correlation cos(θ)"
)

st.sidebar.header("Your Rating Vector (u)")
st.sidebar.write("Rate each feature from 1 (dislike) to 5 (strongly like):")

default_ratings = [5, 3, 1, 4, 3, 2, 5, 2]
active_ratings = []

for idx, genre in enumerate(GENRES):
    val = st.sidebar.slider(
        label=genre,
        min_value=1,
        max_value=5,
        value=default_ratings[idx],
        step=1
    )
    active_ratings.append(float(val))

output = calculate_linear_algebra_pipeline(active_ratings)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Active User Vector")
    st.markdown(f"**Vector u:** `{output['user_vector']}`")
    st.markdown(f"**Euclidean Norm ||u||:** `{output['user_norm']}` | **Mean Rating:** `{output['user_mean']}`")

    table_data = []
    for cand in output["candidates"]:
        table_data.append({
            "Candidate User": cand["name"],
            "Euclidean Dist": round(cand["euclidean_distance"], 3),
            "Correlation (cos θ)": round(cand["pearson_corr"], 4),
            "Angle (θ)": f"{round(cand['angle_deg'], 1)}°"
        })

    df_results = pd.DataFrame(table_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("Model Predictions")

    best_d = output["best_by_dist"]
    st.markdown("### Closest by Euclidean Distance")
    st.write(f"**Matched:** {best_d['name']} (Distance = `{round(best_d['euclidean_distance'], 3)}`)")
    st.write("Recommended Songs:")
    for song in best_d["recommendations"]:
        st.write(f"- 🎵 {song}")

    st.divider()

    best_p = output["best_by_pearson"]
    st.markdown("### Closest by Pearson / Cosine")
    st.write(
        f"**Matched:** {best_p['name']} "
        f"(cos θ = `{round(best_p['pearson_corr'], 4)}`, θ = `{round(best_p['angle_deg'], 1)}°`)"
    )
    st.write("Recommended Songs:")
    for song in best_p["recommendations"]:
        st.write(f"- 🎵 {song}")