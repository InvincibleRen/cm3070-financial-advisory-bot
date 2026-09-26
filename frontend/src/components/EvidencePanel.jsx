
export default function EvidencePanel({ title, intro, lines = [], reference }) {
  if (!lines.length && !reference) return null;
  return (
    <section className="bg-white border border-gray-200 rounded-xl p-5 mb-8">
      <h2 className="text-lg font-semibold border-l-4 border-brand-500 pl-3 mb-3">{title}</h2>
      {intro && <p className="text-sm text-gray-500 mb-3">{intro}</p>}
      <div className="text-sm text-gray-700 space-y-2">
        {lines.map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
      {reference && (
        <p className="text-sm bg-blue-50 border border-blue-100 text-blue-900 rounded-lg p-3 mt-4">
          {reference}
        </p>
      )}
    </section>
  );
}
