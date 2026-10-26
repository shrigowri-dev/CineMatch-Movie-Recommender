from flask import Flask, render_template, request, jsonify
from ai_engine import integrated_recommendation_signals
from pathlib import Path
import csv, math, time

BASE = Path(__file__).resolve().parent
app = Flask(__name__)

def load_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

MOVIES = load_csv(BASE / "data" / "movies.csv")
RATINGS = load_csv(BASE / "data" / "ratings.csv")
MOVIE_BY_ID = {m["movie_id"]: m for m in MOVIES}

# Poster images are loaded from TMDB's public image CDN. The app still works
# if a remote poster is unavailable because each card has a styled fallback.
POSTER_BASE = "https://image.tmdb.org/t/p/w500"
POSTERS = {
    "M01": "/kXfqcdQKsToO0OUXHcrrNCHDBzO.jpg", # The Shawshank Redemption
    "M02": "/q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg", # The Godfather
    "M03": "/8Gxv8gSFCU0XGDykEGv7zR1n2ua.jpg",
    "M04": "/1E5baUa86eM3X9iB2l9q5Y8aJm.jpg",
    "M05": "/wKiOkZTN9lUUUNZLmtnwubZYONg.jpg",
    "M06": "/q719jXXEzOoYaps6babgKnONONX.jpg",
    "M07": "/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
    "M08": "/5KCVkau1HEl7ZzfPsKAPM0sMiKc.jpg",
    "M09": "/f89U3ADr1oiB1s9GkdPOEpXUk5H.jpg",
    "M10": "/xBHvZcj6UsBzcNx8wHZ6zATQA5.jpg",
    "M11": "/d5NXSklXo0qyIYkgV94XAgMIckC.jpg",
    "M12": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",
    "M13": "/uDO8zWDhfWwoFdKS4fzkUJt0Rf0.jpg",
    "M14": "/8Y43POKjjKDGI9MH89NW0NAzzp8.jpg",
    "M15": "/5a7lMDn3nAj2ByO0X1fg6BhUphR.jpg",
    "M16": "/t6HIqrRAclMCA60NsSmeqe9RmNV.jpg",
    "M17": "/d4KNaTrltq6bpkFS01pYtyXa09m.jpg",
    "M18": "/qF2LJ0kG4YbH0Q0w9Jv6s4qZ3Q.jpg",
    "M19": "/gOnmaxHo045T2C6p0wzn1w5ltkT.jpg",
    "M20": "/x2LSRK2Cm7MZhjluni1msVJ3wDF.jpg"
}
for movie in MOVIES:
    movie["year"] = int(movie["year"])
    movie["rating"] = float(movie["rating"])
    movie["genres_list"] = movie["genres"].split("|")
    movie["poster"] = POSTER_BASE + POSTERS.get(movie["movie_id"], "")
    movie["poster"] = movie["poster"] if POSTERS.get(movie["movie_id"]) else ""

def cosine_similarity(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[u] * b[u] for u in common)
    na = math.sqrt(sum(a[u] ** 2 for u in common))
    nb = math.sqrt(sum(b[u] ** 2 for u in common))
    return dot / (na * nb) if na and nb else 0.0

def recommend(selected_ids, selected_ratings=None, genre_filter=None, top_n=8):
    """Integrated hybrid AI recommender: all syllabus methods feed one result list."""
    selected_ratings = selected_ratings or {}
    user_movie_ratings = {}
    movie_user_ratings = {}
    for row in RATINGS:
        uid, mid, rating = row["user_id"], row["movie_id"], float(row["rating"])
        user_movie_ratings.setdefault(uid, {})[mid] = rating
        movie_user_ratings.setdefault(mid, {})[uid] = rating

    known = {mid: float(selected_ratings.get(mid, 4)) for mid in selected_ids if mid in MOVIE_BY_ID}
    liked_ids = [mid for mid, rating in known.items() if rating >= 4]
    global_mean = sum(float(r["rating"]) for r in RATINGS) / max(1, len(RATINGS))
    # Run the integrated reasoning/learning pipeline once for this recommendation request.
    ai_signals = integrated_recommendation_signals(MOVIES, list(known), known)
    output = []

    for movie in MOVIES:
        mid = movie["movie_id"]
        if mid in known:
            continue
        if genre_filter and genre_filter != "All" and genre_filter not in movie["genres_list"]:
            continue

        weighted, weights = 0.0, 0.0
        for rated_mid, user_rating in known.items():
            sim = cosine_similarity(movie_user_ratings.get(mid, {}), movie_user_ratings.get(rated_mid, {}))
            if sim > 0:
                weighted += sim * user_rating
                weights += sim
        cf = weighted / weights if weights else global_mean

        signal = ai_signals.get(mid, {})
        liked_genres = set(g for liked_id in liked_ids if liked_id in MOVIE_BY_ID for g in MOVIE_BY_ID[liked_id]["genres_list"])
        genres = set(movie["genres_list"])
        overlap = len(genres & liked_genres) / len(genres) if genres else 0.0
        heuristic = 1 + 4 * overlap
        popularity = 1 + 4 * (movie["rating"] / 5)
        # Each Unit contributes to the same final recommendation ranking, not a separate demo.
        bayes_score = 1 + 4 * signal.get("bayes_score", 0.5)
        induction_score = 1 + 4 * signal.get("induction_score", 0.5)
        analogy_score = 1 + 4 * signal.get("analogy_score", 0.0)
        search_score = 1 + 4 * signal.get("search_score", 0.0)
        expert_score = 1 + 4 * signal.get("expert_rule_score", 0.0)
        neural_score = max(1.0, min(5.0, signal.get("neural_rating", movie["rating"])))
        session_score = 1 + 4 * signal.get("session_learning_score", 0.0)
        # Knowledge representation + predicate logic/resolution and forward/backward rules
        # determine whether the movie is supported by the user's known genre preferences.
        logic_bonus = 0.15 if signal.get("logic_candidate") and signal.get("backward_reasoning_supported") else 0.0
        score = (
            0.32 * cf + 0.12 * heuristic + 0.07 * popularity +
            0.08 * search_score + 0.08 * bayes_score + 0.07 * induction_score +
            0.06 * analogy_score + 0.06 * expert_score + 0.10 * neural_score +
            0.04 * session_score + logic_bonus
        )
        score = max(1.0, min(5.0, score))

        # Recommendation explanations and AI-method labels are intentionally omitted
        # from the user interface; the scoring algorithms above remain active.
        output.append({**movie, "score": round(score, 2)})

    output.sort(key=lambda m: (m["score"], m["rating"]), reverse=True)
    return output[:top_n]

@app.route("/")
def index():
    return render_template("index.html", movies=MOVIES)

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    payload = request.get_json(silent=True) or {}
    selected = payload.get("selected", [])
    selected_ratings = payload.get("ratings", {})
    genre = payload.get("genre", "All")
    started = time.perf_counter()
    results = recommend(selected, selected_ratings, genre)
    elapsed = (time.perf_counter() - started) * 1000
    return jsonify({"recommendations": results, "elapsed_ms": round(elapsed, 2)})

@app.route("/api/evaluate")
def api_evaluate():
    # A transparent, small educational hold-out evaluation over the sample dataset.
    started = time.perf_counter()
    users = {}
    for r in RATINGS:
        users.setdefault(r["user_id"], {})[r["movie_id"]] = float(r["rating"])
    hits, relevant, total_recs, cases = 0, 0, 0, 0
    for uid, values in users.items():
        high = [mid for mid, rating in values.items() if rating >= 4]
        if len(values) < 3 or not high:
            continue
        held = high[-1]
        known = {mid: rating for mid, rating in values.items() if mid != held}
        recs = recommend(list(known), known, top_n=5)
        ids = {m["movie_id"] for m in recs}
        hits += int(held in ids)
        relevant += 1
        total_recs += len(recs)
        cases += 1
    elapsed = (time.perf_counter() - started) * 1000
    return jsonify({
        "precision_at_5": round(hits / total_recs, 3) if total_recs else 0,
        "recall_at_5": round(hits / relevant, 3) if relevant else 0,
        "evaluation_cases": cases,
        "runtime_ms": round(elapsed, 2)
    })

if __name__ == "__main__":
    app.run(debug=False)
