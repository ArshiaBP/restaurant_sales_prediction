// pages/AuthPage.jsx
import { useState } from "react";
import { login, signup, getMe } from "../api.js";
import Field     from "../components/Field.jsx";
import Button    from "../components/Button.jsx";
import AlertBox  from "../components/AlertBox.jsx";

const HERO_BADGES = [
  "Gradient Boosting",
  "LSTM",
  "Weather-aware",
  "14-day forecast",
];

export default function AuthPage({ onAuth }) {
  const [tab,     setTab]     = useState("login");   // "login" | "signup"
  const [form,    setForm]    = useState({ username: "", password: "", restaurant_name: "" });
  const [error,   setError]   = useState("");
  const [loading, setLoading] = useState(false);

  const set = (key) => (e) =>
    setForm((prev) => ({ ...prev, [key]: e.target.value }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      let tokenData;
      if (tab === "login") {
        tokenData = await login(form.username, form.password);
      } else {
        tokenData = await signup(form.username, form.password, form.restaurant_name);
      }
      const me = await getMe(tokenData.access_token);
      onAuth(tokenData.access_token, me);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrap">
      {/* ── Hero panel ── */}
      <div className="auth-hero">
        <p className="serif auth-hero-title">
          Predict<br />tomorrow's<br />sales today.
        </p>
        <p className="auth-hero-sub">
          AI-powered demand forecasting for restaurants — combining historical
          data, live weather, and holiday signals.
        </p>
        <div className="auth-hero-badges">
          {HERO_BADGES.map((b) => (
            <span key={b} className="badge-pill">{b}</span>
          ))}
        </div>
      </div>

      {/* ── Form panel ── */}
      <div className="auth-form-wrap">
        <div className="auth-card fade-up">
          {/* Tab switcher */}
          <div className="auth-tab-row">
            <button
              className={`auth-tab ${tab === "login" ? "active" : ""}`}
              onClick={() => setTab("login")}
            >
              Sign in
            </button>
            <button
              className={`auth-tab ${tab === "signup" ? "active" : ""}`}
              onClick={() => setTab("signup")}
            >
              Create account
            </button>
          </div>

          <AlertBox message={error} variant="error" />

          <form onSubmit={handleSubmit}>
            <Field label="Username">
              <input
                value={form.username}
                onChange={set("username")}
                placeholder="your_username"
                required
              />
            </Field>

            {tab === "signup" && (
              <Field label="Restaurant name" optional>
                <input
                  value={form.restaurant_name}
                  onChange={set("restaurant_name")}
                  placeholder="Burger Land"
                />
              </Field>
            )}

            <Field label="Password">
              <input
                type="password"
                value={form.password}
                onChange={set("password")}
                placeholder="••••••••"
                required
              />
            </Field>

            <Button type="submit" variant="primary" full loading={loading}>
              {tab === "login" ? "Sign in" : "Create account & sign in"}
            </Button>
          </form>
        </div>
      </div>
    </div>
  );
}
