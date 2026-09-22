// components/Card.jsx
// Simple surface card with optional title.

export default function Card({ title, children, style, className = "" }) {
  return (
    <div className={`card ${className}`} style={style}>
      {title && <div className="card-title">{title}</div>}
      {children}
    </div>
  );
}
