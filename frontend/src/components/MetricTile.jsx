
export default function MetricTile({
  label,
  value,
  delta,
  accent = "border-brand-500",
  tooltip,
}) {
  return (
    <div
      className={`bg-white rounded-lg border-l-4 ${accent} px-4 py-3 shadow-sm card-hover relative group`}
      title={tooltip}
    >
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
        {label}
      </p>
      <p className="text-xl font-bold mt-1">{value ?? "-"}</p>
      {delta && (
        <p className="text-sm text-gray-500 mt-0.5">{delta}</p>
      )}
      {tooltip && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
          {tooltip}
        </div>
      )}
    </div>
  );
}
