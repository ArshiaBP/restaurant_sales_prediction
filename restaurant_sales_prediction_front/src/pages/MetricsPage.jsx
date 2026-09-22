// pages/MetricsPage.jsx
import { useState, useEffect } from "react";
import { getMetrics } from "../api.js";
import Spinner   from "../components/Spinner.jsx";
import AlertBox  from "../components/AlertBox.jsx";
import Card      from "../components/Card.jsx";
import Icon      from "../components/Icon.jsx";

// Metrics we want to display — cv rows removed
const METRIC_KEYS = ["test_MAE", "test_RMSE", "test_R2"];

export default function MetricsPage({ token }) {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState("");

  useEffect(() => {
    getMetrics(token)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [token]);

  if (loading) {
    return (
      <div style={{ padding: "48px", textAlign: "center" }}>
        <Spinner size={28} />
      </div>
    );
  }

  const modelEntries = data
    ? Object.entries(data.metrics).filter(([k]) => !k.startsWith("_"))
    : [];

  return (
    <div>
      {/* ── Page header ── */}
      <div className="page-header">
        <div>
          <div className="serif page-title">Model Metrics</div>
          <div className="page-sub">
            Test-set evaluation for all trained models
          </div>
        </div>
      </div>

      <div className="page-body">
        <AlertBox message={error} variant="error" />

        {data && (
          <>
            {/* Top info chips */}
            <div className="metrics-top-row">
              <Card className="metrics-info-chip" style={{ padding: "14px 20px" }}>
                <Icon name="star" size={16} stroke="var(--accent)" />
                <span style={{ fontSize: "13px" }}>
                  Best model:{" "}
                  <strong style={{ color: "var(--accent)" }}>{data.best_model}</strong>
                </span>
              </Card>
            </div>

            {/* Model cards grid */}
            <div className="metrics-grid">
              {modelEntries.map(([name, m]) => (
                <ModelCard
                  key={name}
                  name={name}
                  metrics={m}
                  isBest={name === data.best_model}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── Sub-component ────────────────────────────────────────────────────────────

function ModelCard({ name, metrics, isBest }) {
  return (
    <div className={`metric-card ${isBest ? "is-best" : ""}`}>
      <div className="metric-name">
        {isBest && <Icon name="star" size={14} stroke="var(--accent)" />}
        <span>{name}</span>
        {isBest && <span className="tag tag-best" style={{ marginLeft: "auto" }}>Best</span>}
      </div>

      {METRIC_KEYS.map(
        (key) =>
          metrics[key] !== undefined && (
            <div key={key} className="metric-row">
              <span className="metric-key">{key.replace("_", " ")}</span>
              <span
                className="metric-val"
                style={{
                  color: key.includes("R2")
                    ? metrics[key] > 0.1
                      ? "var(--green)"
                      : "var(--muted)"
                    : "var(--text)",
                }}
              >
                {metrics[key]}
              </span>
            </div>
          )
      )}

      {metrics.note && (
        <p className="metric-note">{metrics.note}</p>
      )}
    </div>
  );
}
