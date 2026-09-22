// App.jsx
// Always starts on the login page.
// After successful login/signup, switches to the main app.
// Logging out returns to login.

import { useState } from "react";
import AuthPage from "./pages/AuthPage.jsx";
import AppShell from "./layout/AppShell.jsx";

export default function App() {
  const [token, setToken] = useState("");
  const [user,  setUser]  = useState(null);

  const handleAuth = (tok, me) => {
    setToken(tok);
    setUser(me);
    // Set initial URL to predict after login
    window.location.hash = "predict";
  };

  const handleLogout = () => {
    setToken("");
    setUser(null);
    window.location.hash = "";
  };

  if (token && user) {
    return <AppShell token={token} user={user} onLogout={handleLogout} />;
  }

  return <AuthPage onAuth={handleAuth} />;
}
