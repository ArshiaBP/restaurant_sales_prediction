// layout/Sidebar.jsx
import Icon from "../components/Icon.jsx";
import { NAV_ITEMS } from "../constants.js";

export default function Sidebar({ page, onNavigate, user, onLogout }) {
  const initial = user?.username?.[0]?.toUpperCase() ?? "U";

  return (
    <nav className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <h1 className="serif">SalesAI</h1>
        <p>Restaurant forecasting</p>
      </div>

      {/* Navigation */}
      {NAV_ITEMS.map((item) => (
        <button
          key={item.id}
          className={`nav-item ${page === item.id ? "active" : ""}`}
          onClick={() => onNavigate(item.id)}
        >
          <Icon name={item.icon} size={16} />
          {item.label}
        </button>
      ))}

      {/* User info + logout */}
      <div className="sidebar-bottom">
        <div className="user-chip">
          <div className="user-avatar">{initial}</div>
          <div className="user-info">
            <div className="user-name">{user?.username}</div>
            {user?.restaurant_name && (
              <div className="user-rest">{user.restaurant_name}</div>
            )}
          </div>
        </div>

        <button className="nav-item" onClick={onLogout}>
          <Icon name="logout" size={16} />
          Sign out
        </button>
      </div>
    </nav>
  );
}
