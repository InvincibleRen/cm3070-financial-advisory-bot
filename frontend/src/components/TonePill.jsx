
export default function TonePill({ tone }) {
  if (!tone) return null;
  const t = tone.toLowerCase();

  let bg, text;
  if (t.includes("positive")) {
    bg = "bg-green-100";
    text = "text-green-700";
  } else if (t.includes("negative")) {
    bg = "bg-red-100";
    text = "text-red-700";
  } else {
    bg = "bg-gray-100";
    text = "text-gray-600";
  }

  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${bg} ${text}`}
    >
      {tone}
    </span>
  );
}
