import { useState } from "react";
import { Link } from "react-router-dom";
import SignalBadge from "./SignalBadge";

export default function PickCard({ pick }) {
  const { ticker, quote, recommendation } = pick;
  const modelReasons = pick.model_reasons ?? [];
  const [showIndicators, setShowIndicators] = useState(false);

  const price = quote?.latest_price;
  const change = quote?.change_percent;
  const signal = recommendation?.signal;
  const reasons = recommendation?.reasons ?? [];
  const trendStrength = recommendation?.trend_strength;

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 card-hover">

      <div className="flex items-start justify-between mb-3">
        <div>
          <Link
            to={`/stock/${ticker}`}
            className="text-lg font-bold text-brand-700 hover:underline"
          >
            {ticker}
          </Link>
          {price != null && (
            <span className="ml-2 text-sm text-gray-500">
              ${price.toFixed(2)}
              {change != null && (
                <span
                  className={`ml-1 ${
                    change >= 0 ? "text-green-600" : "text-red-600"
                  }`}
                >
                  ({change >= 0 ? "+" : ""}
                  {change.toFixed(2)}%)
                </span>
              )}
            </span>
          )}
        </div>
        <SignalBadge signal={signal} />
      </div>

      {modelReasons.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
            Why the model picked it
          </p>
          <ul className="text-sm text-gray-600 space-y-1">
            {modelReasons.map((r, i) => (
              <li key={i} className="flex items-start gap-1.5">
                <span className={`mt-0.5 ${r.contribution > 0 ? "text-green-600" : "text-red-500"}`}>
                  {r.contribution > 0 ? "\u2191" : "\u2193"}
                </span>
                <span>{r.text}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {reasons.length > 0 && (
        <div className="mb-3">
          <button
            onClick={() => setShowIndicators(!showIndicators)}
            className="flex items-center gap-1 text-xs font-semibold text-gray-500 uppercase tracking-wide hover:text-gray-700"
          >
            <span className={`transition-transform ${showIndicators ? "rotate-90" : ""}`}>
              &#9656;
            </span>
            What the indicators say today
          </button>
          {showIndicators && (
            <>
              <ul className="text-sm text-gray-600 space-y-1 mt-2">
                {reasons.slice(0, 3).map((r, i) => (
                  <li key={i} className="flex items-start gap-1.5">
                    <span className="text-brand-500 mt-0.5">&#8226;</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
              {trendStrength && trendStrength !== "unknown" && (
                <p className="text-xs text-gray-400 mt-2">
                  Trend strength (ADX):{" "}
                  <span className="font-medium text-gray-600 capitalize">{trendStrength}</span>
                </p>
              )}
            </>
          )}
        </div>
      )}

      <div className="mt-4 text-sm">
        <Link
          to={`/stock/${ticker}`}
          className="text-brand-600 hover:text-brand-800 font-medium"
        >
          View {ticker} in detail &rarr;
        </Link>
      </div>
    </div>
  );
}
