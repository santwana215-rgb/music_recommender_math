# music_recommender_math
# Sound Match Recommender

A Linear Algebra music recommender system built with Python, NumPy, and Streamlit. Based on **Project 7: "Norms, angles, and your movie choices"** from Anna Zemlyanova's *Applied Projects for an Introductory Linear Algebra Class*.

The app models listener preferences as vectors in an 8-dimensional genre space ($\mathbb{R}^8$). It demonstrates why simple Euclidean distance fails due to user rating bias and shows how vector centralization, orthogonal projections, and cosine similarity yield accurate recommendations.

---

## Features

- **Interactive Vector Input:** Adjust ratings across 8 genres using live sliders.
- **Euclidean Metric ($L_2$ Norm):** Calculates straight-line distance between preference vectors.
- **Mean-Centering (De-biasing):** Subtracts personal average ratings to eliminate strict vs. lenient rater bias.
- **Cosine Similarity & Angles:** Computes $\cos(\theta)$ and angular difference via Cauchy-Schwarz to find true taste alignment[cite: 86, 90].
- **Latent Space & Projections:** Uses Truncated SVD and Gram-Schmidt QR projections ($P = QQ^T$) to filter noise and verify idempotency ($P^2 = P$)[cite: 107, 108, 110, 171, 176, 196].
- **Real-Time Predictions:** Compares Euclidean matches against angular matches side by side[cite: 90, 92].

---

## File Structure

```text
├── backend.py       # NumPy math engine (Projections, Distances, Cosine)
├── frontend.py      # Streamlit user interface and dashboards
├── requirements.txt # Python package dependencies
└── README.md        # Documentation
