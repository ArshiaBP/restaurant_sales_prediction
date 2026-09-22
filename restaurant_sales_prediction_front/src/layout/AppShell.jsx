// layout/AppShell.jsx
import { useState, useEffect } from "react";
import Sidebar     from "./Sidebar.jsx";
import PredictPage from "../pages/PredictPage.jsx";
import MetricsPage from "../pages/MetricsPage.jsx";

const PAGES = {
  predict: PredictPage,
  metrics: MetricsPage,
};

function getPageFromUrl() {
  const hash = window.location.hash.replace("#", "") || "predict";
  return PAGES[hash] ? hash : "predict";
}

export default function AppShell({ token, user, onLogout }) {
  const [page, setPage] = useState(getPageFromUrl);

  // Keep URL in sync with active page
  useEffect(() => {
    window.location.hash = page;
  }, [page]);

  // Handle browser back/forward
  useEffect(() => {
    const onHash = () => setPage(getPageFromUrl());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const PageComponent = PAGES[page] ?? PredictPage;

  return (
    <div className="app-shell">
      <Sidebar
        page={page}
        onNavigate={setPage}
        user={user}
        onLogout={onLogout}
      />
      <main className="main">
        <PageComponent token={token} />
      </main>
    </div>
  );
}
