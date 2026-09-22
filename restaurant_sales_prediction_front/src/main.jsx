// main.jsx — Vite entry point
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";

// Global styles (imported in order: resets → components → layout → pages)
import "./styles/global.css";
import "./styles/components.css";
import "./styles/auth.css";
import "./styles/layout.css";
import "./styles/pages.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
