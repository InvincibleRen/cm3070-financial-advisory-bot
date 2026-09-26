import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  fetchStock,
  fetchSentiment,
  fetchEvidence,
  fetchEvaluation,
  fetchDirection,
  fetchResearch,
  fetchCompare,
} from "../lib/api";
import TrustStrip from "../components/TrustStrip";
import MetricTile from "../components/MetricTile";
import SignalBadge from "../components/SignalBadge";
import TonePill from "../components/TonePill";
import Spinner from "../components/Spinner";
import ErrorBox from "../components/ErrorBox";

function Details({ summary, children }) {
  return (
    <details className="group mt-3 border-t border-gray-100 pt-3">
      <summary className="cursor-pointer list-none text-sm font-medium text-brand-600 hover:text-brand-700 flex items-center gap-1">
        <span className="transition-transform group-open:rotate-90">&#9656;</span>
        {summary}
      </summary>
      <div className="mt-3">{children}</div>
    </details>
  );
}

export default function StockDetail() {
  const { ticker } = useParams();

  const [stockData, setStockData] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [direction, setDirection] = useState(null);
  const [directionState, setDirectionState] = useState("loading");
  const [research, setResearch] = useState(null);
  const [compare, setCompare] = useState(null);
  const [compareState, setCompareState] = useState("loading");
  const [backtest, setBacktest] = useState(null);
  const [backtestState, setBacktestState] = useState("idle");

  function runBacktest() {
    setBacktestState("running");
    fetchEvaluation(ticker, { period: "5 years" })
      .then((r) => {
        setBacktest(r);
        setBacktestState("done");
      })
      .catch((e) => {
        setBacktest({ error: e.message });
        setBacktestState("done");
      });
  }

  function loadStock() {
    setLoading(true);
    setError(null);
    Promise.all([fetchStock(ticker, { period: "1 year" }), fetchSentiment(ticker)])
      .then(([s, sent]) => {
        setStockData(s);
        setSentiment(sent);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    loadStock();
  }, [ticker]);

  useEffect(() => {
    setDirection(null);
    setDirectionState("loading");
    fetchDirection(ticker)
      .then((r) => {
        setDirection(r);
        setDirectionState("done");
      })
      .catch((e) => {
        setDirection({ error: e.message });
        setDirectionState("done");
      });
  }, [ticker]);

  useEffect(() => {
    fetchResearch().then(setResearch).catch(() => setResearch(null));
  }, []);

  useEffect(() => {
    setCompare(null);
    setCompareState("loading");
    fetchCompare(ticker, { period: "5y" })
      .then((r) => {
        setCompare(r);
        setCompareState("done");
      })
      .catch((e) => {
        setCompare({ error: e.message });
        setCompareState("done");
      });
  }, [ticker]);

  const d = stockData;
  const q = d?.quote;
  const rec = d?.recommendation;
  const ind = d?.indicators ?? {};

  const [evidence, setEvidence] = useState(null);
  useEffect(() => {
    if (!rec?.signal) return;
    fetchEvidence({ signal: rec.signal }).then(setEvidence).catch(() => setEvidence(null));
  }, [rec?.signal]);

  function pct(v) {
    return v == null ? "-" : `${(v * 100).toFixed(2)}%`;
  }
  function fmt(v, dp = 2) {
    return v == null ? "-" : Number(v).toFixed(dp);
  }
  function fmtPrice(v) {
    return v != null ? `$${Number(v).toFixed(2)}` : "-";
  }
  function pctd(v, dp = 1) {
    return v == null ? "-" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(dp)}%`;
  }

  const selectorRows = research?.selectors?.rows ?? [];
  const factorRows = selectorRows.filter((r) => r.model?.startsWith("Single factor"));
  const fittedRows = selectorRows.filter((r) => !r.model?.startsWith("Single factor"));
  const beatsRandom = (r) => r.p_random != null && r.p_random < 0.05;
  const winningFactor = factorRows
    .filter(beatsRandom)
    .sort((a, b) => (b.cagr ?? -Infinity) - (a.cagr ?? -Infinity))[0];
  const fmtP = (p) => (p == null ? "-" : p < 0.001 ? "< 0.001" : `= ${p.toFixed(3)}`);
  const examinedFactors = research?.shap?.rows ?? [];

  const dir = direction?.forecast;
  const ev = direction?.evaluation;

  if (loading && !d) return <Spinner message={`Loading ${ticker}...`} />;
  if (error && !d) return <ErrorBox message={error} onRetry={loadStock} />;
  if (!d) return null;

  return (
    <div className="page-enter max-w-3xl mx-auto">

      <nav className="text-sm text-gray-400 mb-4">
        <Link to="/" className="hover:text-gray-600">Analyse a stock</Link>
        <span className="mx-2">/</span>
        <span className="text-gray-700 font-medium">{d.ticker}</span>
      </nav>

      <div className="flex items-baseline gap-3 mb-5">
        <h1 className="text-2xl font-bold">{d.ticker}</h1>
        {q && (
          <span className="text-lg text-gray-500">
            {fmtPrice(q.latest_price)}
            {q.change_percent != null && (
              <span className={`ml-1 text-sm font-medium ${q.change_percent >= 0 ? "text-green-600" : "text-red-600"}`}>
                ({q.change_percent >= 0 ? "+" : ""}{q.change_percent.toFixed(2)}%)
              </span>
            )}
          </span>
        )}
      </div>

      <TrustStrip />

      <section className="bg-white border border-gray-200 rounded-xl p-5 mb-6">
        <h2 className="text-lg font-semibold mb-1">What the usual signals say about {d.ticker}</h2>
        <p className="text-sm text-gray-500 mb-4">
          The three sources retail investors are told to watch. Then we test whether any of them
          actually predicts anything.
        </p>

        <div className="divide-y divide-gray-100">

          <div className="flex items-center justify-between py-2.5">
            <span className="text-sm text-gray-500">Technical analysis</span>
            <span className="flex items-center gap-2">
              <SignalBadge signal={rec?.signal} />
              {rec?.risk_level && <span className="text-xs text-gray-400">Risk: {rec.risk_level}</span>}
            </span>
          </div>

          <div className="flex items-center justify-between py-2.5">
            <span className="text-sm text-gray-500">Machine learning{dir ? ` (${dir.horizon_days}-day)` : ""}</span>
            <span className="text-sm">
              {directionState === "loading" && <span className="text-gray-400">running…</span>}
              {dir && (
                <>
                  <span className="font-semibold">{dir.prob_up_pct}% </span>
                  <span className={dir.direction === "up" ? "text-green-600" : "text-red-600"}>
                    {dir.direction === "up" ? "▲ up" : "▼ down"}
                  </span>
                  <span className="text-xs text-amber-600"> · ≈ coin flip (see below)</span>
                </>
              )}
              {directionState === "done" && direction?.error && <span className="text-gray-400">n/a</span>}
            </span>
          </div>

          {sentiment && !sentiment.error && (
            <div className="flex items-center justify-between py-2.5">
              <span className="text-sm text-gray-500">News sentiment (FinBERT)</span>
              <span className="flex items-center gap-2">
                <TonePill tone={sentiment.label} />
                <span className="text-xs text-gray-400">{sentiment.n_total} headlines</span>
              </span>
            </div>
          )}
        </div>

        <Details summary="Price chart & why the rules said that">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={d.price_chart}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} tickFormatter={(v) => v.slice(5)} minTickGap={40} />
              <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} tickFormatter={(v) => `$${v}`} />
              <Tooltip contentStyle={{ backgroundColor: "#fff", border: "1px solid #e5e7eb", borderRadius: 8, fontSize: 12 }} formatter={(v) => [`$${Number(v).toFixed(2)}`]} />
              <Line type="monotone" dataKey="Close" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
              {d.price_chart?.[0]?.SMA50 != null && (
                <Line type="monotone" dataKey="SMA50" stroke="#8b5cf6" strokeWidth={1} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
              )}
              {d.price_chart?.[0]?.SMA200 != null && (
                <Line type="monotone" dataKey="SMA200" stroke="#ef4444" strokeWidth={1} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
              )}
            </LineChart>
          </ResponsiveContainer>
          {ind.ADX != null && (
            <p className="text-xs text-gray-400 mt-2">
              ADX {ind.ADX.toFixed(1)} ({ind.ADX >= 25 ? "trending" : "weak trend"})
              {ind.cross_event && ind.cross_event !== "none" ? ` · ${ind.cross_event}` : ""}
            </p>
          )}
          {rec?.reasons?.length > 0 && (
            <ul className="list-disc pl-5 space-y-1 text-sm text-gray-600 mt-3">
              {rec.reasons.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          )}
          {rec?.warnings?.length > 0 && (
            <ul className="list-disc pl-5 space-y-1 text-sm text-amber-700 mt-2">
              {rec.warnings.map((w, i) => <li key={i}>{w}</li>)}
            </ul>
          )}
        </Details>

        {sentiment && !sentiment.error && sentiment.headlines?.length > 0 && (
          <Details summary={`The ${sentiment.n_total} headlines FinBERT read`}>
            <p className="text-xs text-gray-400 mb-2">An extra signal only, not validated as a core predictor.</p>
            <div className="divide-y divide-gray-100">
              {sentiment.headlines.map((h, i) => (
                <div key={i} className="py-2 flex items-start gap-2">
                  <TonePill tone={h.tone} />
                  <div className="flex-1 min-w-0">
                    {h.url ? (
                      <a href={h.url} target="_blank" rel="noopener noreferrer" className="text-sm text-gray-700 hover:text-brand-600 hover:underline line-clamp-2">{h.text}</a>
                    ) : (
                      <span className="text-sm text-gray-700 line-clamp-2">{h.text}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </Details>
        )}
      </section>

      <section className="bg-white border-2 border-amber-200 rounded-xl p-5 mb-6">
        <h2 className="text-lg font-semibold mb-3">Does any of it actually predict {d.ticker}? Tested honestly: no.</h2>

        {directionState === "loading" && <Spinner message="Running the machine-learning test…" />}

        {dir && ev && (
          <p className="text-sm text-gray-700 mb-3">
            The machine-learning model gives {d.ticker} a confident-looking{" "}
            <strong>{dir.prob_up_pct}%</strong> chance of rising, but out of sample it has been{" "}
            <strong>{ev.accuracy != null ? `${(ev.accuracy * 100).toFixed(1)}%` : "-"}</strong> accurate
            versus a <strong>{ev.base_rate != null ? `${(ev.base_rate * 100).toFixed(1)}%` : "-"}</strong>{" "}
            coin-flip base rate. No better than chance.
          </p>
        )}

        {evidence?.available && (
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2">
              How this kind of signal has actually done, across the whole S&amp;P 500
            </h3>
            <div className="text-sm text-gray-700 space-y-2">
              {evidence.signal_lines.map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
            {evidence.index_reference && (
              <p className="text-sm bg-blue-50 border border-blue-100 text-blue-900 rounded-lg p-3 mt-3">
                {evidence.index_reference}
              </p>
            )}
          </div>
        )}

        <div className="bg-amber-50 rounded-lg px-4 py-3 text-sm text-amber-900">
          <p>
            And this is not one model having a bad day. Across the whole S&amp;P 500
            {fittedRows.length ? ` we tested ${fittedRows.length} models` : " we tested a range of models"},
            from a <strong>simple linear</strong> one to a <strong>complex gradient booster</strong>,
            and <strong>none beat picking stocks at random</strong>. When both the simple and the complex
            model fail on the same data, the signal isn&apos;t there to find.{" "}
            <strong>Treat confident technical-analysis calls with caution</strong>: the burden of proof
            is on whoever claims these signals work.
          </p>
        </div>

        <Details summary={`Test the signals on ${d.ticker} itself`}>
          <p className="text-sm text-gray-500 mb-3">
            Replays the indicator signals on this stock over five years, always training before the
            period tested, against simply holding it.
          </p>
          {backtestState === "idle" && (
            <button onClick={runBacktest} className="bg-brand-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-brand-700 transition-colors">
              Run the test
            </button>
          )}
          {backtestState === "running" && <Spinner message="Running walk-forward test…" />}
          {backtestState === "done" && backtest?.error && <p className="text-sm text-amber-600">{backtest.error}</p>}
          {backtestState === "done" && !backtest?.error && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <MetricTile label="Signals" value={pct(backtest.strategy?.annualised_return)} accent="border-brand-500" />
              <MetricTile label="Buy & hold" value={pct(backtest.benchmark?.annualised_return)} accent="border-gray-400" />
              <MetricTile label="Sharpe" value={fmt(backtest.strategy?.sharpe_ratio, 2)} accent="border-purple-500" />
              <MetricTile label="Worst drop" value={pct(backtest.strategy?.max_drawdown)} accent="border-red-500" />
            </div>
          )}
        </Details>

      </section>

      <section className="bg-white border border-gray-200 rounded-xl p-5 mb-6">
        <h2 className="text-lg font-semibold mb-3">What actually separated winners from losers: risk</h2>
        <p className="text-sm text-gray-700 mb-4">
          The one thing that beat random picking across the S&amp;P 500 was ranking on a{" "}
          <strong>risk factor</strong> (how volatile a stock is, and its beta to the market) not any
          chart pattern.{" "}
          {winningFactor && (
            <>Ranking on the strongest of them returned <strong>{pctd(winningFactor.cagr)}</strong> a year (p {fmtP(winningFactor.p_random)}).</>
          )}
        </p>

        {compare?.stock && (
          <p className="text-sm text-gray-600 mb-4">
            {d.ticker}&apos;s own risk (5y): beta{" "}
            <strong>{compare.stock.beta != null ? compare.stock.beta.toFixed(2) : "-"}</strong>, volatility{" "}
            <strong>{compare.stock.volatility != null ? `${(compare.stock.volatility * 100).toFixed(0)}%` : "-"}</strong>.
            That risk level, not its chart, is what the evidence ties its return to.
          </p>
        )}

        {compareState === "loading" && <Spinner message="Comparing with SPY…" />}
        {compareState === "done" && !compare?.error && compare?.stock && compare?.spy && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-200">
                    <th className="py-2 pr-4 font-medium">Buy & hold, 5y</th>
                    <th className="py-2 pr-4 font-medium text-right">{d.ticker}</th>
                    <th className="py-2 font-medium text-right">SPY (S&amp;P 500)</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {[
                    ["Return / year", (m) => pctd(m.ann_return)],
                    ["Volatility", (m) => (m.volatility != null ? `${(m.volatility * 100).toFixed(1)}%` : "-")],
                    ["Sharpe (return per risk)", (m) => (m.sharpe != null ? m.sharpe.toFixed(2) : "-")],
                    ["Beta (vs market)", (m) => (m.beta != null ? m.beta.toFixed(2) : "-")],
                  ].map(([label, f]) => (
                    <tr key={label} className="border-b border-gray-100">
                      <td className="py-2 pr-4 text-gray-500">{label}</td>
                      <td className="py-2 pr-4 text-right font-medium">{f(compare.stock)}</td>
                      <td className="py-2 text-right">{f(compare.spy)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-sm text-gray-500 mt-3">{compare.note}</p>
          </>
        )}

        <div className="bg-amber-50 rounded-lg px-4 py-3 mt-4 text-sm text-amber-900">
          A higher-risk stock earning more is not skill: it is payment for the risk, in a market that
          mostly rose. In a falling market the high-beta names fall hardest.
        </div>

        {research && (
          <Details summary="How we know: the whole-market factor test">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-400 border-b border-gray-200">
                  <th className="py-2 pr-4 font-medium">Strategy</th>
                  <th className="py-2 pr-4 font-medium">Return/yr</th>
                  <th className="py-2 font-medium">Beat random?</th>
                </tr>
              </thead>
              <tbody>
                {fittedRows.map((r) => (
                  <tr key={r.model} className="border-b border-gray-100">
                    <td className="py-2 pr-4">{r.model}</td>
                    <td className="py-2 pr-4 tabular-nums">{pctd(r.cagr)}</td>
                    <td className="py-2 text-gray-500">No (p = {r.p_random != null ? r.p_random.toFixed(3) : "-"})</td>
                  </tr>
                ))}
                {factorRows.map((r) => (
                  <tr key={r.model} className="border-b border-gray-100 bg-green-50">
                    <td className="py-2 pr-4 font-medium">{r.model.replace("Single factor: ", "Rank on ")}</td>
                    <td className="py-2 pr-4 tabular-nums font-semibold">{pctd(r.cagr)}</td>
                    <td className="py-2">
                      {beatsRandom(r)
                        ? <span className="text-green-700 font-medium">Yes (p {r.p_random < 0.001 ? "< 0.001" : `= ${r.p_random.toFixed(3)}`}) ✓</span>
                        : <span className="text-gray-500">No (p = {r.p_random != null ? r.p_random.toFixed(3) : "-"})</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {examinedFactors.length > 0 && (
              <p className="text-xs text-gray-400 mt-3">
                The model was given {examinedFactors.length} candidate factors; risk exposure is what
                survived. Full breakdown on the{" "}
                <Link to="/evidence" className="text-brand-600 hover:underline">evaluation</Link> page.
              </p>
            )}
          </Details>
        )}
      </section>
    </div>
  );
}
