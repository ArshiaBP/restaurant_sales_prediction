// pages/PredictPage.jsx
import { useState } from "react";
import { predict }    from "../api.js";
import { MODELS, categoryColor, tomorrowStr, maxDateStr } from "../constants.js";
import Field    from "../components/Field.jsx";
import Button   from "../components/Button.jsx";
import AlertBox from "../components/AlertBox.jsx";
import Card     from "../components/Card.jsx";
import Icon     from "../components/Icon.jsx";

export default function PredictPage({ token }) {
  const [date,    setDate]    = useState(tomorrowStr());
  const [model,   setModel]   = useState("gradient_boosting");
  const [loading, setLoading] = useState(false);
  const [result,  setResult]  = useState(null);
  const [error,   setError]   = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    setResult(null);
    try {
      const data = await predict(date, model, token);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const maxQty = result
    ? Math.max(...result.predictions.map((p) => p.predicted_quantity), 1)
    : 1;

  return (
    <div>
      {/* ── Page header ── */}
      <div className="page-header">
        <div>
          <div className="serif page-title">Sales Prediction</div>
          <div className="page-sub">
            Select a date (tomorrow → +14 days) and a model to run a forecast
          </div>
        </div>
      </div>

      <div className="page-body">
        {/* ── Predict form ── */}
        <form onSubmit={handleSubmit} className="predict-form">
          <Field label="Date">
            <input
              type="date"
              value={date}
              min={tomorrowStr()}
              max={maxDateStr()}
              onChange={(e) => setDate(e.target.value)}
              required
            />
          </Field>

          <Field label="Model">
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.label}
                  {m.tag ? ` — ${m.tag}` : ""}
                </option>
              ))}
            </select>
          </Field>

          {/* spacer */}
          <div />

          <div style={{ display: "flex", alignItems: "flex-end" }}>
            <Button
              type="submit"
              variant="primary"
              size="sm"
              loading={loading}
              style={{ padding: "11px 28px" }}
            >
              Run prediction
            </Button>
          </div>
        </form>

        <AlertBox message={error} variant="error" />

        {/* ── Results ── */}
        {result && (
          <div className="fade-up">
            {/* Summary stat cards */}
            <div className="stat-grid">
              <div className="stat-card">
                <div className="stat-label">Total predicted revenue</div>
                <div className="stat-value stat-accent">
                  ${result.total_predicted_revenue.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </div>
                <div className="stat-sub">for {result.date}</div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Total predicted quantity</div>
                <div className="stat-value">
                  {result.total_predicted_quantity.toLocaleString()}
                </div>
                <div className="stat-sub">
                  units across {result.predictions.length} items
                </div>
              </div>

              <div className="stat-card">
                <div className="stat-label">Model used</div>
                <div className="stat-value" style={{ fontSize: "18px", marginTop: "10px" }}>
                  {result.model_used}
                </div>
              </div>
            </div>

            {/* Weather strip */}
            <WeatherStrip weather={result.weather} isHoliday={result.is_holiday} isWeekend={result.is_weekend} />

            {/* Predictions table */}
            <Card title="Per-item predictions — sorted by quantity">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Item</th>
                    <th>Category</th>
                    <th>Avg price</th>
                    <th>Predicted qty</th>
                    <th>Predicted revenue</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.map((p, i) => (
                    <PredictionRow key={i} item={p} maxQty={maxQty} />
                  ))}
                </tbody>
              </table>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────────────────────

function WeatherStrip({ weather, isHoliday, isWeekend }) {
  return (
    <div className="weather-strip">
      <div className="weather-item">
        <Icon name="sun" size={16} stroke="#f4b942" />
        <span>
          <strong>{weather.temperature_mean?.toFixed(1)}°C</strong> avg temp
        </span>
      </div>
      <div className="weather-item">
        <Icon name="rain" size={16} stroke="#60a5fa" />
        <span>
          <strong>{weather.precipitation?.toFixed(1)}mm</strong> precip
        </span>
      </div>
      <div className="weather-item">
        <Icon name="info" size={16} stroke="var(--muted)" />
        <span>
          Source: <strong>{weather.source}</strong>
        </span>
      </div>
      <div className="weather-tags">
        {isHoliday  && <span className="tag tag-holiday">Holiday</span>}
        {!isHoliday && isWeekend  && <span className="tag tag-weekend">Weekend</span>}
      </div>
    </div>
  );
}

function PredictionRow({ item, maxQty }) {
  const barWidth = Math.max(4, (item.predicted_quantity / maxQty) * 80);
  return (
    <tr>
      <td style={{ fontWeight: 500 }}>{item.item}</td>
      <td>
        <span
          className="cat-dot"
          style={{ background: categoryColor(item.category) }}
        />
        {item.category}
      </td>
      <td style={{ color: "var(--muted)" }}>${item.avg_price.toFixed(2)}</td>
      <td>
        <div className="qty-bar-wrap">
          <span>{item.predicted_quantity.toFixed(1)}</span>
          <div className="qty-bar" style={{ width: `${barWidth}px` }} />
        </div>
      </td>
      <td style={{ color: "var(--accent)", fontWeight: 600 }}>
        ${item.predicted_revenue.toFixed(2)}
      </td>
    </tr>
  );
}
