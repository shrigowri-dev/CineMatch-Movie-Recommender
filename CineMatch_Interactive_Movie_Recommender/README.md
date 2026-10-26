# CineMatch — Interactive AI Movie Recommendation System

CineMatch is a Flask web app that integrates concepts from Units 1–5 directly into one movie recommendation workflow. There is no separate AI lab or Run demo section: the methods contribute to the results shown when the user clicks Recommend.

## Main features
- Movie catalogue with genre filtering, search, sorting, dark/light theme, and movie selection.
- Hybrid recommender using item-based collaborative filtering, cosine similarity, genre-overlap heuristic scoring, and catalogue ratings.
- Evaluation endpoint with a small educational hold-out demo for Precision@5, Recall@5, case count, and runtime.
- Integrated AI signals in the main recommendation request: BFS, DFS, Greedy Best-First Search, genre heuristics, predicate-style knowledge and resolution checks, forward/backward reasoning, Bayesian estimates, induction, analogy, a small neural network trained with backpropagation, and expert rules.
- No external ML library is required; the small neural network is implemented using Python's standard library.

## Requirements
- Python 3.9+
- Flask
- Gunicorn (for Render deployment)

## Run on Windows / VS Code
1. Extract the ZIP and open the folder containing `app.py` in VS Code.
2. Open Terminal → New Terminal.
3. Create and activate a virtual environment (recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   python app.py
   ```

6. Open `http://127.0.0.1:5000`.

## How to use
1. Browse the catalogue and select movies you have watched.
2. Click **Recommend** to get a personalized list.
3. Click **Evaluate AI model** to see the educational metrics.
4. Read the AI reasoning summary shown on each recommendation card. All integrated signals are used automatically when you click Recommend.

## Project structure
- `app.py` — Flask app, recommendation algorithm and API routes.
- `ai_engine.py` — integrated search, logic, probability and learning signals used by the recommender.
- `templates/index.html` — interface.
- `static/style.css` — responsive styling.
- `static/app.js` — interactions, API calls and output rendering.
- `data/movies.csv` — small movie catalogue.
- `data/ratings.csv` — sample ratings.
- `REPORT.md` — project report.
- `AI_SYLLABUS_MAPPING.md` — unit-by-unit mapping and implementation notes.
- `VIVA_QUESTIONS.md` — viva preparation.

## Deployment on Render
1. Push the project to a GitHub repository.
2. Create a Render Web Service from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Leave Root Directory blank if `app.py` is in the repository root.

## Important academic limitations
- The catalogue and ratings are a small educational dataset, not a production MovieLens-scale dataset.
- The recommender combines separate educational signals with manually selected weights; these methods are not trained together as one model.
- The Unit 2 graph links movies sharing genres; search signals are lightweight ranking signals.
- Predicate-resolution and rule reasoning are simplified implementations, not a complete first-order theorem prover.
- Bayesian estimates use smoothed counts from the small catalogue and are not calibrated real-world probabilities.
- Induction and neural-network learning use a tiny dataset and are not evidence of production predictive performance.
- User selections and learned genre weights are not persisted between sessions.
- External poster images require internet access; fallback poster styling is provided.
