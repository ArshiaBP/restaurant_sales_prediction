// api.js
// All communication with the FastAPI backend lives here.
// Every function either returns data or throws an Error with a human-readable message.

const BASE = "http://localhost:8000";

// ─── Core fetch wrapper ───────────────────────────────────────────────────────
async function request(path, options = {}, token = null) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res  = await fetch(`${BASE}${path}`, { ...options, headers });
  const data = await res.json();

  if (!res.ok) {
    const msg = Array.isArray(data.detail)
      ? data.detail.map((e) => e.msg).join(", ")
      : data.detail || "Request failed";
    throw new Error(msg);
  }
  return data;
}

// ─── Auth ─────────────────────────────────────────────────────────────────────
// Login uses application/x-www-form-urlencoded because FastAPI's
// OAuth2PasswordRequestForm requires that content-type.
export async function login(username, password) {
  const body = new URLSearchParams({ username, password });
  const res  = await fetch(`${BASE}/auth/login`, {
    method : "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Login failed");
  return data; // { access_token, token_type, expires_in }
}

export async function signup(username, password, restaurant_name) {
  return request("/auth/signup", {
    method: "POST",
    body  : JSON.stringify({
      username,
      password,
      ...(restaurant_name ? { restaurant_name } : {}),
    }),
  });
  // returns { access_token, token_type, expires_in }
}

export async function getMe(token) {
  return request("/auth/me", {}, token);
  // returns { id, username, restaurant_name, created_at }
}

// ─── Charts ───────────────────────────────────────────────────────────────────
export async function getChartsList(token) {
  return request("/charts/list", {}, token);
  // returns { eda: {label, files}, diagnostics: {...}, lag: {...}, timeseries: {...} }
}

export function chartUrl(category, filename) {
  return `${BASE}/charts/${category}/${filename}`;
}

export async function getMetrics(token) {
  return request("/metrics", {}, token);
  // returns { metrics, best_model, selected_features, note }
}

// ─── Prediction ───────────────────────────────────────────────────────────────
export async function predict(date, model, token) {
  return request(`/predict?date=${date}&model=${model}`, {}, token);
  /*
  returns {
    date, is_weekend, is_holiday,
    weather: { temperature_mean, precipitation, source },
    predictions: [{ item, category, avg_price, predicted_quantity, predicted_revenue }],
    total_predicted_quantity, total_predicted_revenue,
    selected_features, model_used, prediction_note
  }
  */
}