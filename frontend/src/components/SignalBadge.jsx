
export default function SignalBadge({ signal }) {
  if (!signal) return null;
  const s = signal.toLowerCase();

  let bg, text;
  if (s.includes("buy")) {
    bg = "bg-green-100";
    text = "text-green-700";
  } else if (s.includes("sell")) {
    bg = "bg-red-100";
    text = "text-red-700";
  } else {
    bg = "bg-yellow-100";
    text = "text-yellow-700";
  }

  return (
    <span
      className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold ${bg} ${text}`}
    >
      {signal}
    </span>
  );
}
