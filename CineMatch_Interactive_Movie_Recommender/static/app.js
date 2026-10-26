const selected = new Map();
const cards = [...document.querySelectorAll(".movie-card")];
const grid = document.getElementById("movie-grid");
const count = document.getElementById("selected-count");
const genreFilter = document.getElementById("genre-filter");
const sortOrder = document.getElementById("sort-order");
const search = document.getElementById("search");
const toast = document.getElementById("toast");
let toastTimer;

function notify(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2300);
}
function updateCount() { count.textContent = selected.size; }
function selectedPayload() {
  return [...selected.entries()].map(([id, rating]) => ({id, rating}));
}
cards.forEach(card => {
  card.addEventListener("click", () => {
    const id = card.dataset.id;
    if (selected.has(id)) {
      selected.delete(id);
      card.classList.remove("selected");
      card.setAttribute("aria-pressed", "false");
    } else {
      selected.set(id, 4);
      card.classList.add("selected");
      card.setAttribute("aria-pressed", "true");
    }
    updateCount();
  });
});

function applyFilters() {
  const genre = genreFilter.value;
  const query = search.value.trim().toLowerCase();
  cards.forEach(card => {
    const genres = card.dataset.genres.split("|");
    const matchesGenre = genre === "All" || genres.includes(genre);
    const matchesSearch = card.dataset.title.includes(query);
    card.hidden = !(matchesGenre && matchesSearch);
  });
  const visible = cards.filter(c => !c.hidden);
  document.getElementById("empty-state").hidden = visible.length !== 0;
  const mode = sortOrder.value;
  visible.sort((a,b) => {
    if (mode === "year-desc") return Number(b.dataset.year) - Number(a.dataset.year);
    if (mode === "year-asc") return Number(a.dataset.year) - Number(b.dataset.year);
    if (mode === "rating-desc") return Number(b.dataset.rating) - Number(a.dataset.rating);
    return a.dataset.title.localeCompare(b.dataset.title);
  });
  visible.forEach(card => grid.appendChild(card));
}
genreFilter.addEventListener("change", () => {
  document.querySelectorAll(".genre").forEach(b => b.classList.toggle("active", b.dataset.genre === genreFilter.value));
  applyFilters();
});
document.querySelectorAll(".genre").forEach(button => button.addEventListener("click", () => {
  genreFilter.value = button.dataset.genre;
  document.querySelectorAll(".genre").forEach(b => b.classList.toggle("active", b === button));
  applyFilters();
}));
sortOrder.addEventListener("change", applyFilters);
search.addEventListener("input", applyFilters);

document.getElementById("clear-btn").addEventListener("click", () => {
  selected.clear();
  cards.forEach(card => { card.classList.remove("selected"); card.setAttribute("aria-pressed","false"); });
  updateCount();
  document.getElementById("recommendation-grid").innerHTML =
    '<div class="recommend-empty"><div class="sparkle">✦</div><strong>Your next favourite movie is waiting.</strong><span>Select movies above and click Recommend.</span></div>';
  document.getElementById("recommend-subtitle").textContent = "Your personalized movie list will appear here.";
  document.getElementById("metrics").hidden = true;
  notify("Your selection has been cleared.");
});
document.getElementById("recommend-btn").addEventListener("click", async () => {
  if (!selected.size) { notify("Select at least one movie you have watched."); return; }
  const age = Number(document.getElementById("age").value);
  if (!Number.isFinite(age) || age < 13 || age > 100) { notify("Enter an age between 13 and 100."); return; }
  const btn = document.getElementById("recommend-btn");
  btn.disabled = true; btn.textContent = "Finding your movies...";
  try {
    const ratings = Object.fromEntries(selected.entries());
    const response = await fetch("/api/recommend", {
      method: "POST", headers: {"Content-Type":"application/json"},
      body: JSON.stringify({selected:[...selected.keys()], ratings, genre:genreFilter.value, age})
    });
    const data = await response.json();
    renderRecommendations(data.recommendations);
    document.getElementById("recommend-subtitle").textContent =
      `Based on ${selected.size} selected movie${selected.size === 1 ? "" : "s"} · Hybrid AI pipeline · ${data.elapsed_ms} ms`;
    document.getElementById("metrics").hidden = true;
    notify("Your recommendations are ready.");
    document.getElementById("recommendation-section").scrollIntoView({behavior:"smooth", block:"start"});
  } catch (e) {
    notify("Could not reach the recommendation engine. Check that Flask is running.");
  } finally {
    btn.disabled = false; btn.innerHTML = "<span>✦</span> Recommend";
  }
});

function posterMarkup(movie, small=false) {
  const title = escapeHtml(movie.title);
  const fallback = `https://placehold.co/180x260/282c3b/ffffff?text=${encodeURIComponent(movie.title)}`;
  const url = movie.poster || fallback;
  return `<img class="${small ? "recommend-poster" : "poster"}" src="${url}" alt="${title} poster" onerror="this.onerror=null;this.src='${fallback}'">`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}
function renderRecommendations(movies) {
  const target = document.getElementById("recommendation-grid");
  if (!movies.length) {
    target.innerHTML = '<div class="recommend-empty"><strong>No matching recommendations</strong><span>Try another genre or select more movies.</span></div>';
    return;
  }
  target.innerHTML = movies.map(movie => `
    <article class="recommend-card">
      ${posterMarkup(movie, true)}
      <div class="recommend-copy">
        <h3>${escapeHtml(movie.title)}</h3>
        <div class="rating-line"><span class="star">★</span><strong>${movie.rating}</strong><span class="out-of">/ 5 · ${movie.year}</span></div>
        <div class="genre-tags">${movie.genres_list.slice(0,2).map(g=>`<span>${escapeHtml(g)}</span>`).join("")}</div>
        <span class="score-pill">AI score ${movie.score}/5</span>
      </div>
    </article>`).join("");
}
document.getElementById("theme-toggle").addEventListener("change", event => {
  document.body.classList.toggle("light", !event.target.checked);
  document.getElementById("theme-label").textContent = event.target.checked ? "Dark Mode" : "Light Mode";
});
document.getElementById("menu-btn").addEventListener("click", () => document.getElementById("sidebar").classList.toggle("open"));
document.getElementById("evaluate-btn").addEventListener("click", async () => {
  const btn = document.getElementById("evaluate-btn");
  btn.disabled = true; btn.textContent = "Evaluating...";
  try {
    const response = await fetch("/api/evaluate");
    const m = await response.json();
    const box = document.getElementById("metrics");
    box.innerHTML = `
      <div class="metric"><span>Precision@5</span><strong>${m.precision_at_5}</strong></div>
      <div class="metric"><span>Recall@5</span><strong>${m.recall_at_5}</strong></div>
      <div class="metric"><span>Evaluation cases</span><strong>${m.evaluation_cases}</strong></div>
      <div class="metric"><span>Runtime</span><strong>${m.runtime_ms} ms</strong></div>`;
    box.hidden = false;
    notify("Evaluation complete.");
  } catch(e) { notify("Could not evaluate model. Check Flask server."); }
  finally { btn.disabled = false; btn.textContent = "Evaluate AI model"; }
});

