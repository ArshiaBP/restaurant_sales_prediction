// components/AlertBox.jsx
// Displays an error or success message.
// variant: "error" | "success"

export default function AlertBox({ message, variant = "error" }) {
  if (!message) return null;
  return (
    <div className={`alert alert-${variant}`} role="alert">
      {message}
    </div>
  );
}
