// components/Icon.jsx
// Renders a single SVG icon from the ICONS map in constants.js.
// Usage: <Icon name="chart" size={18} stroke="#f4b942" />

import { ICONS } from "../constants.js";

export default function Icon({ name, size = 18, stroke = "currentColor" }) {
  const d = ICONS[name];
  if (!d) return null;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={stroke}
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={d} />
    </svg>
  );
}
