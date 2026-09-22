// components/Spinner.jsx
import Icon from "./Icon.jsx";

export default function Spinner({ size = 18 }) {
  return (
    <span className="spin" aria-label="Loading">
      <Icon name="spinner" size={size} />
    </span>
  );
}
