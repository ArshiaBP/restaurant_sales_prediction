// components/Button.jsx
import Spinner from "./Spinner.jsx";

// variant : "primary" | "ghost"
// size    : "default" | "sm"
// full    : fills container width
export default function Button({
  children,
  variant  = "primary",
  size     = "default",
  full     = false,
  loading  = false,
  disabled = false,
  onClick,
  type     = "button",
  style,
}) {
  const classes = [
    "btn",
    `btn-${variant}`,
    size === "sm" ? "btn-sm" : "",
    full ? "btn-full" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      type={type}
      className={classes}
      onClick={onClick}
      disabled={disabled || loading}
      style={style}
    >
      {loading ? <Spinner size={15} /> : null}
      {children}
    </button>
  );
}
