// components/Field.jsx
// Labelled input or select wrapper.
// Usage:
//   <Field label="Date" optional>
//     <input type="date" ... />
//   </Field>

export default function Field({ label, optional = false, children, style }) {
  return (
    <div className="field" style={style}>
      <label>
        {label}
        {optional && <span className="optional"> (optional)</span>}
      </label>
      {children}
    </div>
  );
}
