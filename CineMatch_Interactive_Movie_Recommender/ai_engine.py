"""Educational AI demonstrations used by CineMatch Units 1-5.
All algorithms use Python's standard library; the neural network is implemented from scratch.
"""
from collections import defaultdict, deque
import math
import random

GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "History", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def movie_graph(movies):
    """Graph search state space: movies are nodes; shared genres create edges."""
    graph = {m['movie_id']: [] for m in movies}
    by_id = {m['movie_id']: m for m in movies}
    for i, a in enumerate(movies):
        ga = set(a['genres_list'])
        for b in movies[i + 1:]:
            if ga & set(b['genres_list']):
                graph[a['movie_id']].append(b['movie_id'])
                graph[b['movie_id']].append(a['movie_id'])
    return graph, by_id


def bfs(graph, start, goal):
    q = deque([(start, [start])]); seen = {start}; expanded = []
    while q:
        node, path = q.popleft(); expanded.append(node)
        if node == goal: return {"path": path, "expanded": expanded}
        for nxt in graph.get(node, []):
            if nxt not in seen:
                seen.add(nxt); q.append((nxt, path + [nxt]))
    return {"path": [], "expanded": expanded}


def dfs(graph, start, goal):
    stack = [(start, [start])]; seen = set(); expanded = []
    while stack:
        node, path = stack.pop()
        if node in seen: continue
        seen.add(node); expanded.append(node)
        if node == goal: return {"path": path, "expanded": expanded}
        for nxt in reversed(graph.get(node, [])):
            if nxt not in seen: stack.append((nxt, path + [nxt]))
    return {"path": [], "expanded": expanded}


def best_first(graph, by_id, start, goal):
    """Greedy best-first search, using genre overlap with goal as its heuristic."""
    target = set(by_id[goal]['genres_list'])
    def h(mid):
        genres = set(by_id[mid]['genres_list'])
        return 1 - len(genres & target) / max(1, len(genres | target))
    frontier = [(h(start), [start])]; seen = set(); expanded = []
    while frontier:
        frontier.sort(key=lambda item: item[0], reverse=True)
        _, path = frontier.pop(); node = path[-1]
        if node in seen: continue
        seen.add(node); expanded.append(node)
        if node == goal: return {"path": path, "expanded": expanded, "heuristic": "Jaccard genre distance to target"}
        for nxt in graph.get(node, []):
            if nxt not in seen: frontier.append((h(nxt), path + [nxt]))
    return {"path": [], "expanded": expanded, "heuristic": "Jaccard genre distance to target"}


def movie_facts(movies):
    facts = []
    for m in movies:
        facts.append(f"Movie({m['movie_id']})")
        facts.append(f"Title({m['movie_id']}, '{m['title']}')")
        facts.append(f"Year({m['movie_id']}, {m['year']})")
        facts.append(f"Rating({m['movie_id']}, {m['rating']:.1f})")
        for g in m['genres_list']:
            facts.append(f"Genre({m['movie_id']}, '{g}')")
    return facts


def forward_chain(liked_genres, movies):
    """Forward chaining: known preference facts trigger recommendation rules."""
    facts = {f"LikesGenre('{g}')" for g in liked_genres}
    fired = []
    recommendations = []
    for m in movies:
        overlap = sorted(set(m['genres_list']) & set(liked_genres))
        if overlap:
            fact = f"Candidate('{m['movie_id']}')"
            if fact not in facts:
                facts.add(fact)
                fired.append({"rule": "LikesGenre(g) AND Genre(movie,g) -> Candidate(movie)", "inferred": fact, "because": overlap})
                recommendations.append(m)
    recommendations.sort(key=lambda m: (len(set(m['genres_list']) & set(liked_genres)), m['rating']), reverse=True)
    return {"initial_facts": sorted(f for f in facts if f.startswith("LikesGenre")), "rules_fired": fired[:10], "recommendations": [{"title":m['title'],"movie_id":m['movie_id'],"genres":m['genres_list']} for m in recommendations[:8]]}


def backward_chain(target_id, liked_genres, by_id):
    movie = by_id.get(target_id)
    if not movie: return {"goal": f"Recommend({target_id})", "proved": False, "steps": ["Movie identifier was not found."]}
    overlap = sorted(set(movie['genres_list']) & set(liked_genres))
    steps = [f"Goal: Recommend({target_id})", f"Find rule: LikesGenre(g) AND Genre({target_id},g) -> Recommend({target_id})", f"Known preference facts: {', '.join(liked_genres) if liked_genres else 'none'}", f"Movie genre facts: {', '.join(movie['genres_list'])}"]
    if overlap: steps.append(f"Both premises hold for {', '.join(overlap)}; goal is proved.")
    else: steps.append("No shared liked genre was found; this goal cannot be proved from the current facts.")
    return {"goal": f"Recommend({target_id})", "proved": bool(overlap), "steps": steps}


def bayes_demo(movies, liked_genres):
    """Estimate P(high rating | genre match) using Laplace-smoothed counts."""
    liked = set(liked_genres)
    matched = [m for m in movies if liked and liked & set(m['genres_list'])]
    unmatched = [m for m in movies if not (liked & set(m['genres_list']))]
    high_match = sum(m['rating'] >= 4 for m in matched)
    high_unmatch = sum(m['rating'] >= 4 for m in unmatched)
    # Beta/Laplace smoothing keeps probabilities finite with a small sample.
    p_high_match = (high_match + 1) / (len(matched) + 2) if matched else 0.5
    p_high_unmatch = (high_unmatch + 1) / (len(unmatched) + 2) if unmatched else 0.5
    prior = (sum(m['rating'] >= 4 for m in movies) + 1) / (len(movies) + 2)
    denom = p_high_match * prior + p_high_unmatch * (1-prior)
    posterior = p_high_match * prior / denom if denom else prior
    return {"network": "GenreMatch -> HighRating -> Recommend", "prior_high_rating": round(prior,3), "p_high_given_genre_match": round(p_high_match,3), "p_high_given_no_match": round(p_high_unmatch,3), "posterior_demo": round(posterior,3), "sample_counts": {"genre_match_movies":len(matched),"nonmatching_movies":len(unmatched),"high_rated_matches":high_match}, "note":"Illustrative Bayesian update from the small supplied movie catalogue; not a production-calibrated probability."}


def resolution_demo(liked_genres, target_id, by_id):
    """A tiny resolution proof over propositionalized predicate facts."""
    movie = by_id.get(target_id)
    if not movie: return {"goal":f"Recommend({target_id})", "proved":False, "clauses":[], "steps":["Unknown movie identifier."]}
    shared = sorted(set(movie['genres_list']) & set(liked_genres))
    # Clauses: (not LikesGenre(g) OR Recommend(m)); LikesGenre(g); resolve -> Recommend(m).
    clauses = []
    steps = []
    if shared:
        g = shared[0]
        clauses = [f"¬LikesGenre('{g}') ∨ Recommend('{target_id}')", f"LikesGenre('{g}')", f"Recommend('{target_id}') (resolvent)"]
        steps = [f"Clause 1 is the implication LikesGenre('{g}') -> Recommend('{target_id}') in disjunctive form.", f"Clause 2 states LikesGenre('{g}') is a known fact.", "Resolve the complementary literals ¬LikesGenre(g) and LikesGenre(g); the remaining literal proves the goal."]
    else:
        clauses = [f"¬LikesGenre(g) ∨ Recommend('{target_id}')", "No matching LikesGenre(g) fact", "No resolvent for this goal"]
        steps = ["No complementary positive LikesGenre fact exists, so this simple resolution proof cannot derive the goal."]
    return {"goal":f"Recommend({target_id})", "proved":bool(shared), "clauses":clauses, "steps":steps}


def induction_demo(movies):
    """Induce an interpretable one-level decision rule from labeled examples."""
    labels = [(m, m['rating'] >= 4.0) for m in movies]
    groups = defaultdict(list)
    for m, high in labels:
        groups['High rating' if high else 'Below 4.0'].append(m)
    high_genres = defaultdict(lambda:[0,0])
    for m, high in labels:
        for g in m['genres_list']:
            high_genres[g][1] += 1
            high_genres[g][0] += int(high)
    ranked = sorted(((yes/total, g, yes, total) for g,(yes,total) in high_genres.items() if total), reverse=True)
    return {"training_examples":len(movies), "label_definition":"high-rated = stored audience rating >= 4.0", "class_counts":{k:len(v) for k,v in groups.items()}, "induced_patterns":[{"genre":g,"high_rating_fraction":round(frac,3),"high_rated_examples":yes,"examples_with_genre":total} for frac,g,yes,total in ranked[:6]], "learned_rule":"Estimate a movie's high-rating tendency from genre frequencies in labeled examples.", "note":"This is a transparent induction demo, not a validated general-purpose decision tree."}


def analogy_demo(selected_ids, movies):
    by_id = {m['movie_id']:m for m in movies}
    selected = [by_id[mid] for mid in selected_ids if mid in by_id]
    liked_genres = set(g for m in selected for g in m['genres_list'])
    candidates=[]
    for m in movies:
        if m['movie_id'] in selected_ids: continue
        gs=set(m['genres_list'])
        score=len(gs & liked_genres)/max(1,len(gs | liked_genres))
        candidates.append({"title":m['title'],"movie_id":m['movie_id'],"shared_genres":sorted(gs & liked_genres),"analogy_score":round(score,3)})
    candidates.sort(key=lambda x:(x['analogy_score'],x['title']), reverse=True)
    return {"source_movies":[m['title'] for m in selected], "analogy":"Recommend a new movie by analogy with genre properties shared by selected movies.", "results":candidates[:6]}


def expert_system(movies, liked_genres):
    rules = [
        ("Drama preference", bool(set(liked_genres)&{"Drama"}), "If the user likes Drama, prefer high-rated drama titles."),
        ("Sci-Fi preference", bool(set(liked_genres)&{"Sci-Fi"}), "If the user likes Sci-Fi, include science-fiction candidates."),
        ("Adventure preference", bool(set(liked_genres)&{"Adventure"}), "If the user likes Adventure, include adventure candidates."),
        ("High-rating preference", True, "Rank movies with stored ratings >= 4.5 as quality candidates."),
    ]
    fired=[{"rule":name,"condition":description} for name,ok,description in rules if ok]
    candidates=[]
    for m in movies:
        overlap=sorted(set(m['genres_list']) & set(liked_genres))
        score=2*len(overlap)+m['rating']
        if m['rating']>=4.5: score+=1
        if overlap or (not liked_genres and m['rating']>=4.5):
            candidates.append({"title":m['title'],"score":round(score,2),"reason":overlap or ["high audience rating"]})
    candidates.sort(key=lambda x:x['score'],reverse=True)
    return {"system":"Rule-based movie-selection expert system", "rules_fired":fired, "recommendations":candidates[:6], "explanation":"Uses explicit IF-THEN rules; this is not a human expert or a medical/financial decision system."}


def neural_net_demo(movies, selected_ids):
    """Tiny 1-hidden-layer neural network trained with backpropagation from scratch."""
    # Features: genre one-hot for common genres, normalized year, normalized audience rating.
    genres = ["Action","Adventure","Animation","Comedy","Crime","Drama","Fantasy","Horror","Romance","Sci-Fi","Thriller"]
    def features(m):
        # Predict catalogue rating from movie attributes; do not include the target rating as an input.
        gs=set(m['genres_list'])
        return [1.0 if g in gs else 0.0 for g in genres] + [(m['year']-1970)/60.0]
    xs=[features(m) for m in movies]
    ys=[m['rating']/5.0 for m in movies]
    rng=random.Random(17); n_in=len(xs[0]); n_hidden=6
    w1=[[rng.uniform(-0.3,0.3) for _ in range(n_in)] for _ in range(n_hidden)]
    b1=[0.0]*n_hidden; w2=[rng.uniform(-0.3,0.3) for _ in range(n_hidden)]; b2=0.0
    def sigmoid(x): return 1/(1+math.exp(-max(-30,min(30,x))))
    # deterministic full-batch SGD; small illustrative dataset only
    for epoch in range(220):
        for x,y in zip(xs,ys):
            h=[sigmoid(sum(w*xv for w,xv in zip(row,x))+b) for row,b in zip(w1,b1)]
            out=sigmoid(sum(a*b for a,b in zip(w2,h))+b2)
            delta=(out-y)*out*(1-out)
            oldw2=w2[:]
            for j in range(n_hidden): w2[j]-=0.08*delta*h[j]
            b2-=0.08*delta
            for j in range(n_hidden):
                dh=delta*oldw2[j]*h[j]*(1-h[j])
                for k in range(n_in): w1[j][k]-=0.08*dh*x[k]
                b1[j]-=0.08*dh
    predictions=[]
    for m,x in zip(movies,xs):
        h=[sigmoid(sum(w*xv for w,xv in zip(row,x))+b) for row,b in zip(w1,b1)]
        pred=5*sigmoid(sum(a*b for a,b in zip(w2,h))+b2)
        if m['movie_id'] not in selected_ids:
            predictions.append({"title":m['title'],"predicted_rating":round(pred,2),"actual_catalogue_rating":m['rating']})
    predictions.sort(key=lambda x:x['predicted_rating'],reverse=True)
    mae=sum(abs(5*sigmoid(sum(a*sigmoid(sum(w*xv for w,xv in zip(row,x))+b) for a,row,b in zip(w2,w1,b1))+b2)-m['rating']) for m,x in zip(movies,xs))/max(1,len(movies))
    return {"architecture":f"{n_in}-{n_hidden}-1 feed-forward network with sigmoid activation", "training_examples":len(movies), "epochs":220, "training_mae":round(mae,3), "top_predictions":predictions[:6], "all_predictions":predictions, "note":"A small teaching network trained with backpropagation on catalogue attributes; results are illustrative because the dataset is tiny."}


def integrated_recommendation_signals(movies, selected_ids, selected_ratings=None):
    """Compute the syllabus concepts as signals in the real recommendation workflow.

    The UI does not expose separate demos: these signals are used by the main recommender.
    """
    selected_ratings = selected_ratings or {}
    by_id = {m["movie_id"]: m for m in movies}
    selected = [by_id[mid] for mid in selected_ids if mid in by_id]
    liked = [m for m in selected if float(selected_ratings.get(m["movie_id"], 4)) >= 4]
    liked_genres = set(g for m in liked for g in m["genres_list"])
    # Session-level learning: ratings create a temporary weighted genre preference profile.
    session_genre_weights = defaultdict(float)
    for movie in selected:
        rating = float(selected_ratings.get(movie["movie_id"], 4))
        strength = max(0.0, rating - 2.5)
        for genre in movie["genres_list"]:
            session_genre_weights[genre] += strength
    session_total = sum(session_genre_weights.values())
    if session_total:
        session_genre_weights = {g: v / session_total for g, v in session_genre_weights.items()}

    # Unit 1: the recommendation agent perceives selections and pursues the goal of relevance.
    # Unit 2: BFS/DFS traverse the movie state-space graph; best-first ranking uses genre distance.
    graph, _ = movie_graph(movies)
    start = selected[0]["movie_id"] if selected else (movies[0]["movie_id"] if movies else None)
    bfs_order = bfs(graph, start, "__goal_not_in_graph__")["expanded"] if start else []
    dfs_order = dfs(graph, start, "__goal_not_in_graph__")["expanded"] if start else []
    bfs_rank = {mid: i for i, mid in enumerate(bfs_order)}
    dfs_rank = {mid: i for i, mid in enumerate(dfs_order)}
    # Run greedy best-first search for each movie goal; path length is one ranking signal.
    best_first_lengths = {}
    if start:
        for goal_movie in movies:
            search_result = best_first(graph, by_id, start, goal_movie["movie_id"])
            path = search_result.get("path", [])
            best_first_lengths[goal_movie["movie_id"]] = len(path) if path else len(movies) + 1

    # Unit 3: predicate-style facts and a resolution-style rule establish genre eligibility.
    # Unit 4: forward/backward chaining, explicit expert rules, and Bayesian estimates.
    forward_result = forward_chain(sorted(liked_genres), movies)
    inferred_candidates = {row["movie_id"] for row in forward_result.get("recommendations", [])}
    expert = expert_system(movies, sorted(liked_genres))
    expert_ids = {m["title"] for m in expert.get("recommendations", [])}

    high = [m for m in movies if float(m["rating"]) >= 4.0]
    total_high = len(high)
    prior_high = (total_high + 1) / (len(movies) + 2)
    genre_high_rates = {}
    for genre in {g for m in movies for g in m["genres_list"]}:
        group = [m for m in movies if genre in m["genres_list"]]
        positives = sum(float(m["rating"]) >= 4.0 for m in group)
        genre_high_rates[genre] = (positives + 1) / (len(group) + 2)

    # Unit 5: induction estimates genre-level high-rating tendencies from examples.
    induced = induction_demo(movies)
    induced_weights = {row["genre"]: row["high_rating_fraction"] for row in induced.get("induced_patterns", [])}
    # Learning by analogy uses overlap with the selected movies' genre properties.
    # Neural network learning estimates ratings from movie attributes via backpropagation.
    nn = neural_net_demo(movies, selected_ids)
    nn_scores = {row["title"]: float(row["predicted_rating"]) for row in nn.get("all_predictions", [])}

    signals = {}
    for movie in movies:
        mid = movie["movie_id"]
        genres = set(movie["genres_list"])
        overlap = genres & liked_genres
        # Resolution-style proof: LikesGenre(g) AND Genre(movie,g) entails Candidate(movie).
        resolution = resolution_demo(sorted(liked_genres), mid, by_id)
        backward = backward_chain(mid, sorted(liked_genres), by_id)
        logic_candidate = bool(overlap) and bool(resolution.get("proved"))
        backward_supported = bool(backward.get("proved"))
        # Analogy score: Jaccard similarity between candidate genres and user-liked genres.
        analogy = len(overlap) / max(1, len(genres | liked_genres))
        # Bayesian estimate: smoothed P(high rating | each candidate genre), averaged across genres.
        bayes = sum(genre_high_rates.get(g, prior_high) for g in genres) / max(1, len(genres))
        # Inductive score from labelled catalogue examples.
        induction = sum(induced_weights.get(g, prior_high) for g in genres) / max(1, len(genres))
        # Session learning in problem solving: match candidate genres to weighted user examples.
        session_score = sum(session_genre_weights.get(g, 0.0) for g in genres)
        session_score = min(1.0, session_score / max(0.25, max(session_genre_weights.values(), default=0.25)))
        # Rule-based expert signal: genre rules plus high-rating rule.
        rule_score = min(1.0, (0.65 if logic_candidate else 0.0) + (0.35 if float(movie["rating"]) >= 4.0 else 0.0))
        # Best-first heuristic signal uses genre distance; BFS/DFS reachability provides search context.
        search_reachable = mid in bfs_rank
        search_score = 1.0 / (1.0 + len(genres - liked_genres)) if liked_genres else float(movie["rating"]) / 5.0
        traversal_score = 0.4 * (1.0 / (1 + bfs_rank.get(mid, len(movies)))) + 0.3 * (1.0 / (1 + dfs_rank.get(mid, len(movies)))) + 0.3 * (1.0 / best_first_lengths.get(mid, len(movies) + 1))
        nn_rating = nn_scores.get(movie["title"], float(movie["rating"]))
        signals[mid] = {
            "logic_candidate": logic_candidate,
            "forward_chaining_candidate": mid in inferred_candidates,
            "backward_reasoning_supported": backward_supported,
            "analogy_score": analogy,
            "bayes_score": bayes,
            "induction_score": induction,
            "session_learning_score": session_score,
            "expert_rule_score": rule_score,
            "search_score": 0.7 * search_score + 0.3 * traversal_score,
            "neural_rating": nn_rating,
            "search_reachable": search_reachable,
            "expert_rule_matched": movie["title"] in expert_ids,
        }
    return signals


def agent_demo():
    return {"agent":"CineMatch recommendation agent", "percepts":["selected movie IDs", "user ratings", "genre filter", "catalogue ratings"], "actions":["rank candidate movies", "explain recommendation", "return top recommendations"], "performance_measure":["relevance to selected genres", "rating preference fit", "runtime", "Precision@5 / Recall@5 demo"], "environment":"Movie catalogue and rating CSV files", "characteristics":["goal-directed", "reactive to user selections", "uses a simple model of preference", "not autonomous outside this website"], "typical_agent_types":[{"type":"Simple reflex agent","illustration":"IF a selected movie has Drama, consider Drama candidates."},{"type":"Model-based agent","illustration":"Uses catalogue genre and rating facts as a simple environment model."},{"type":"Goal-based agent","illustration":"Attempts to return the requested top-N recommendations."},{"type":"Utility-based agent","illustration":"Ranks candidates using a weighted score."},{"type":"Learning agent","illustration":"The separate neural-network demo learns weights from example ratings; the main recommender does not persist feedback."}], "behavioral_characteristics":["perception from inputs", "decision using rules/scores", "action through displayed recommendations", "performance evaluated with sample metrics"], "problem_solving":"Represent candidates, calculate scores, rank and return the best-scoring movies."}


def learning_problem_demo(selected_ids, movies):
    """Show preference learning from examples for one recommendation session."""
    by_id = {m['movie_id']: m for m in movies}
    examples = [by_id[mid] for mid in selected_ids if mid in by_id]
    genre_counts = defaultdict(int)
    for movie in examples:
        for genre in movie['genres_list']:
            genre_counts[genre] += 1
    total = sum(genre_counts.values())
    weights = {g: round(n / total, 3) for g, n in genre_counts.items()} if total else {}
    ranked = []
    for movie in movies:
        if movie['movie_id'] in selected_ids: continue
        score = sum(weights.get(g, 0) for g in movie['genres_list']) + movie['rating'] / 25
        ranked.append({"title": movie['title'], "learned_preference_score": round(score, 3), "genres": movie['genres_list']})
    ranked.sort(key=lambda x: x['learned_preference_score'], reverse=True)
    return {"examples_used": [m['title'] for m in examples], "learned_genre_weights": weights, "top_candidates": ranked[:6], "explanation":"Genre weights are estimated from the user's selected examples for this request. They are not saved between sessions."}


def run_demo(topic, movies, selected_ids=None, liked_genres=None, target_id=None):
    selected_ids = selected_ids or []
    liked_genres = liked_genres or []
    graph, by_id = movie_graph(movies)
    if not target_id or target_id not in by_id:
        target_id = movies[min(1, len(movies)-1)]['movie_id']
    start_id = next((x for x in selected_ids if x in by_id), movies[0]['movie_id'])
    if not liked_genres:
        liked_genres = by_id[start_id]['genres_list'][:1]
    if topic == 'unit1':
        return agent_demo()
    if topic == 'search':
        return {"state_space":"Movies are nodes; an edge connects two movies sharing at least one genre.", "start":{"id":start_id,"title":by_id[start_id]['title']}, "goal":{"id":target_id,"title":by_id[target_id]['title']}, "BFS":bfs(graph,start_id,target_id), "DFS":dfs(graph,start_id,target_id), "Best-first":best_first(graph,by_id,start_id,target_id), "heuristic":"Jaccard genre distance to the goal; lower is closer.", "note":"BFS finds a shortest path in this unweighted graph. DFS depends on traversal order. Greedy best-first uses a heuristic and is not guaranteed to find the shortest path."}
    if topic == 'knowledge':
        return {"representation":"Predicate-style facts for movies, titles, years, ratings and genres.", "example_facts":movie_facts(movies)[:16], "mapping":"CSV row -> Movie(movie_id), Title(movie_id,title), Year(movie_id,year), Rating(movie_id,rating), Genre(movie_id,genre).", "representation_issues":["Missing values must be handled", "Genre labels must be consistent", "Ratings are subjective and may be sparse", "Facts are only as reliable as the dataset"]}
    if topic == 'logic':
        return {"predicate_logic":"LikesGenre(User, Drama) AND Genre(M01, Drama) -> Recommend(User, M01)", "resolution":resolution_demo(liked_genres,target_id,by_id), "note":"The demo uses a small, explicit resolution proof over a propositionalized rule; it is not a complete first-order theorem prover."}
    if topic == 'rules':
        return {"forward_chaining":forward_chain(liked_genres,movies), "backward_chaining":backward_chain(target_id,liked_genres,by_id), "rule_representation":["IF LikesGenre(user, g) AND Genre(movie, g) THEN Candidate(movie)", "IF Candidate(movie) AND Rating(movie) >= 4 THEN Recommend(movie)"]}
    if topic == 'bayes':
        return bayes_demo(movies,liked_genres)
    if topic == 'induction':
        return induction_demo(movies)
    if topic == 'analogy':
        return analogy_demo(selected_ids,movies)
    if topic == 'neural':
        return neural_net_demo(movies,selected_ids)
    if topic == 'expert':
        return expert_system(movies,liked_genres)
    if topic == 'learning':
        return {"learning_from_examples":induction_demo(movies), "learning_in_problem_solving":learning_problem_demo(selected_ids,movies), "analogy":analogy_demo(selected_ids,movies), "neural_network":neural_net_demo(movies,selected_ids)}
    return {"error":"Unknown AI demo."}
