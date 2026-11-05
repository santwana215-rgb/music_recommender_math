import numpy as np

GENRES = [
    "Synthwave", "Indie Rock", "Classical", "Hip-Hop",
    "Pop Ballad", "Ambient / Lo-Fi", "EDM / Dance", "Acoustic Folk"
]

DATABASE_USERS = [
    {
        "name": "User 101 (Pop/Electronic)",
        "ratings": [5.0, 3.0, 1.0, 4.0, 3.0, 2.0, 5.0, 2.0],
        "recommendations": ["Midnight City (M83)", "Blinding Lights (The Weeknd)"]
    },
    {
        "name": "User 102 (Acoustic/Chill)",
        "ratings": [2.0, 4.0, 5.0, 1.0, 4.0, 5.0, 1.0, 5.0],
        "recommendations": ["Skinny Love (Bon Iver)", "Clair de Lune (Debussy)"]
    },
    {
        "name": "User 103 (Strict Rater)",
        "ratings": [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0], 
        "recommendations": ["Nightcall (Kavinsky)", "Techno Bunker 01"]
    },
    {
        "name": "User 104 (Lenient Rater)",
        "ratings": [5.0, 4.0, 4.0, 5.0, 4.0, 4.0, 5.0, 4.0], 
        "recommendations": ["Good 4 U (Olivia Rodrigo)", "24K Magic (Bruno Mars)"]
    },
    {
        "name": "User 105 (Hip-Hop Core)",
        "ratings": [3.0, 2.0, 1.0, 5.0, 2.0, 3.0, 4.0, 1.0],
        "recommendations": ["HUMBLE. (Kendrick Lamar)", "Lose Yourself (Eminem)"]
    },
    {
        "name": "User 106 (Classical Hall)",
        "ratings": [1.0, 2.0, 5.0, 1.0, 3.0, 5.0, 1.0, 4.0],
        "recommendations": ["Gymnopédie No. 1 (Erik Satie)", "Weightless (Marconi Union)"]
    },
    {
        "name": "User 107 (Rock & Folk)",
        "ratings": [3.0, 5.0, 2.0, 2.0, 2.0, 3.0, 2.0, 5.0],
        "recommendations": ["Riptide (Vance Joy)", "Little Talks (Of Monsters & Men)"]
    },
    {
        "name": "User 108 (Club Nightlife)",
        "ratings": [5.0, 2.0, 1.0, 5.0, 2.0, 1.0, 5.0, 1.0],
        "recommendations": ["Titanium (David Guetta)", "Animals (Martin Garrix)"]
    }
]

R_MATRIX = np.array([u["ratings"] for u in DATABASE_USERS], dtype=float)

def calculate_linear_algebra_pipeline(trial_ratings: list[float]) -> dict:
    u = np.array(trial_ratings, dtype=float)

    u_mean = float(np.mean(u))
    u_norm = float(np.linalg.norm(u))

    diff = R_MATRIX - u
    eucl_distances = np.linalg.norm(diff, axis=1)

    R_mean = np.mean(R_MATRIX, axis=1, keepdims=True)
    R_cent = R_MATRIX - R_mean
    u_cent = u - u_mean

    numerators = np.dot(R_cent, u_cent)
    denominators = np.linalg.norm(R_cent, axis=1) * np.linalg.norm(u_cent)
    denominators = np.where(denominators == 0, 1e-12, denominators)
    pearson_corrs = np.clip(numerators / denominators, -1.0, 1.0)

    angles_deg = np.arccos(pearson_corrs) * (180.0 / np.pi)

    results = []
    for idx, cand in enumerate(DATABASE_USERS):
        results.append({
            "name": cand["name"],
            "vector": cand["ratings"],
            "euclidean_distance": float(eucl_distances[idx]),
            "pearson_corr": float(pearson_corrs[idx]),
            "angle_deg": float(angles_deg[idx]),
            "recommendations": cand["recommendations"]
        })

    sorted_by_dist = sorted(results, key=lambda x: x["euclidean_distance"])
    sorted_by_pearson = sorted(results, key=lambda x: x["pearson_corr"], reverse=True)

    return {
        "user_vector": u.tolist(),
        "user_norm": round(u_norm, 4),
        "user_mean": round(u_mean, 2),
        "candidates": sorted_by_pearson,
        "best_by_dist": sorted_by_dist[0],
        "best_by_pearson": sorted_by_pearson[0]
    }