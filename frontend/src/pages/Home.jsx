import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

const EXAMPLES = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM", "JNJ"];

export default function Home() {
  const navigate = useNavigate();
  const [ticker, setTicker] = useState("");

  function go() {
    const t = ticker.trim().toUpperCase();
    if (t) navigate(`/stock/${encodeURIComponent(t)}`);
  }

  return (
    <div className="page-enter max-w-2xl mx-auto">
      <div className="flex items-center gap-2 mb-2">
        <h1 className="text-3xl font-bold">Should you buy this stock?</h1>
        <span className="inline-flex items-center gap-1 text-xs font-medium text-green-700 bg-green-100 rounded-full px-2 py-0.5">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
          Live
        </span>
      </div>
      <p className="text-gray-500 mb-6 leading-relaxed">
        Type a ticker. This bot gathers the usual buy/sell signals for it, then does
        what most tools won&apos;t: it shows you, honestly, whether any of those
        signals have ever actually predicted anything, and whether the stock is worth
        it over a plain index fund.
      </p>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          go();
        }}
        className="flex gap-2 mb-4"
      >
        <input
          autoFocus
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="e.g. AAPL"
          className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-lg focus:outline-none focus:ring-2 focus:ring-brand-500"
        />
        <button
          type="submit"
          className="bg-brand-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-brand-700 transition-colors disabled:opacity-40"
          disabled={!ticker.trim()}
        >
          Investigate &rarr;
        </button>
      </form>

      <div className="flex flex-wrap items-center gap-2 mb-8">
        <span className="text-sm text-gray-400">Try:</span>
        {EXAMPLES.map((s) => (
          <button
            key={s}
            onClick={() => setTicker(s)}
            className={`text-sm border rounded-lg px-3 py-1 transition-colors ${
              ticker.trim().toUpperCase() === s
                ? "border-brand-500 bg-brand-50 text-brand-700"
                : "border-gray-300 hover:bg-gray-50"
            }`}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="mb-8">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
          What you&apos;ll see for each stock
        </p>
        <ol className="space-y-3">
          {[
            ["The signals", "What technical analysis, a machine-learning model and the news each say about it right now."],
            ["The honest test", "Whether those signals have ever predicted anything, tested out-of-sample across the whole S&P 500, not curve-fitted to look good."],
            ["Stock vs index fund", "How it compares, on return and risk, with simply holding a low-cost S&P 500 index fund (SPY)."],
          ].map(([title, desc], i) => (
            <li key={title} className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-brand-600 text-white text-xs font-bold flex items-center justify-center mt-0.5">
                {i + 1}
              </span>
              <span className="text-sm text-gray-600">
                <strong className="text-gray-900">{title}</strong>: {desc}
              </span>
            </li>
          ))}
        </ol>
      </div>

      <Link
        to="/evidence"
        className="flex items-center justify-between gap-4 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 hover:bg-amber-100 transition-colors"
      >
        <span className="text-sm text-amber-900">
          <strong>Before you trust any of this:</strong> in our own out-of-sample
          tests, this kind of stock advice did not beat simply holding a low-cost
          S&amp;P 500 index fund.
        </span>
        <span className="text-sm font-medium text-amber-900 whitespace-nowrap">
          See the track record &rarr;
        </span>
      </Link>
    </div>
  );
}
