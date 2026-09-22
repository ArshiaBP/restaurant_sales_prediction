// constants.js
// Shared static data: model definitions, category color map, SVG icon paths.

// ─── Models ───────────────────────────────────────────────────────────────────
export const MODELS = [
  { key: "gradient_boosting", label: "Gradient Boosting", tag: "Best"  },
  { key: "random_forest",     label: "Random Forest",     tag: null     },
  { key: "ridge",             label: "Ridge Regression",  tag: null     },
  { key: "linear",            label: "Linear Regression", tag: null     },
  { key: "rnn",               label: "RNN",               tag: "Deep"  },
  { key: "lstm",              label: "LSTM",              tag: "Deep"  },
];

// ─── Category colors ─────────────────────────────────────────────────────────
export const CATEGORY_COLORS = {
  "Starters"   : "#f97316",
  "Main Dishes": "#8b5cf6",
  "Desserts"   : "#ec4899",
  "Side Dishes": "#06b6d4",
  "Drinks"     : "#10b981",
};

export const categoryColor = (cat) => CATEGORY_COLORS[cat] ?? "#6b7280";

// ─── Navigation items ─────────────────────────────────────────────────────────
export const NAV_ITEMS = [
  { id: "predict", label: "Predict",  icon: "chart"    },
  { id: "metrics", label: "Metrics",  icon: "star"     },
];

// ─── SVG icon path data ──────────────────────────────────────────────────────
export const ICONS = {
  chart   : "M3 3v18h18M7 16l4-4 4 4 4-8",
  calendar: "M8 2v4m8-4v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z",
  cpu     : "M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 0-2-2V9m0 0h18",
  sun     : "M12 3v2m0 14v2M4.22 4.22l1.42 1.42m12.72 12.72 1.42 1.42M3 12h2m14 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42M12 7a5 5 0 1 0 0 10A5 5 0 0 0 12 7z",
  rain    : "M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242M16 14v6m-8-6v6m4-4v6",
  user    : "M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
  logout  : "M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9",
  spinner : "M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83",
  star    : "M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z",
  check   : "M20 6 9 17l-5-5",
  info    : "M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10zM12 8h.01M11 12h1v4h1",
  building: "M3 21h18M3 7v1a3 3 0 0 0 6 0V7m0 1a3 3 0 0 0 6 0V7m0 1a3 3 0 0 0 6 0V7H3l2-4h14l2 4M5 21V10.85",
  image   : "M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zM8.5 10a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM21 15l-5-5L5 21",
};

// ─── Date helpers ─────────────────────────────────────────────────────────────
export const tomorrowStr = () => {
  const d = new Date();
  d.setDate(d.getDate() + 1);
  return d.toISOString().split("T")[0];
};

export const maxDateStr = () => {
  const d = new Date();
  d.setDate(d.getDate() + 14);
  return d.toISOString().split("T")[0];
};
