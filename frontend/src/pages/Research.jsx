import { useEffect, useState } from "react";
import { fetchResearch } from "../lib/api";
import Spinner from "../components/Spinner";
import ErrorBox from "../components/ErrorBox";

const pct = (v, signed = true, digits = 1) =>
  v == null ? "-" : `${signed && v > 0 ? "+" : ""}${(v * 100).toFixed(digits)}%`;
const num = (v, digits = 2) => (v == null ? "-" : Number(v).toFixed(digits));
const withT = (v, t, digits = 1) => (v == null ? "-" : `${pct(v, true, digits)} (t ${num(t)})`);

function Section({ id, title, verdict, intro, method, children }) {
  return (
    <section id={id} className="bg-white border border-gray-200 rounded-xl p-5 mb-6 scroll-mt-20">
      <h2 className="text-base font-semibold border-l-4 border-brand-500 pl-3 mb-2">{title}</h2>
      {verdict && <p className="text-sm text-gray-700">{verdict}</p>}
      {(children || intro || method) && (
        <details className="group mt-3">
          <summary className="cursor-pointer list-none text-sm font-medium text-brand-600 hover:text-brand-700 flex items-center gap-1">
            <span className="transition-transform group-open:rotate-90">&#9656;</span>
            Show the numbers
          </summary>
          <div className="mt-3">
            {intro && <p className="text-sm text-gray-600 mb-3">{intro}</p>}
            {children}
            {method && (
              <p className="text-xs text-gray-500 mt-3">
                <span className="font-semibold uppercase tracking-wide">How tested: </span>
                {method}
              </p>
            )}
          </div>
        </details>
      )}
    </section>
  );
}

const CONTENTS = [
  ["build", "1. How the models were built"],
  ["survivorship", "2. Survivorship bias in the universe"],
  ["honest-test", "3. Out-of-sample portfolio vs the index"],
  ["forecaster", "4. Predictive accuracy (out-of-sample R²)"],
  ["learned", "5. What the model relied on"],
  ["signals", "6. Event study of the rule-based signals"],
  ["events", "7. Index-membership events"],
  ["fundamentals", "8. Fundamental features (2023 onwards)"],
  ["limits", "9. Limitations and statistical power"],
];

function Table({ head, rows, highlight }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-gray-500">
            {head.map((h, i) => (
              <th key={i} className={`py-2 pr-4 font-medium ${i === 0 ? "text-left" : "text-right"}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              className={`border-b border-gray-100 ${highlight?.(i) ? "bg-blue-50 font-medium" : ""}`}
            >
              {r.map((c, j) => (
                <td key={j} className={`py-2 pr-4 tabular-nums ${j === 0 ? "text-left" : "text-right"}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Research() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchResearch().then(setData).catch((e) => setError(e.message));
  }, []);

  if (error) return <ErrorBox message={error} />;
  if (!data) return <Spinner message="Loading the evidence..." />;
  if (!data.available)
    return <ErrorBox message="No evidence file yet. Run: python -m src.cli.evidence_cli" />;

  const { survivorship, selectors, shap, signals, index_events: events, fundamentals_2023: fund, model_eval: me } = data;
  const spy = selectors.spy;
  const r2f = (v, digits = 3) => (v == null ? "-" : `${v >= 0 ? "+" : ""}${Number(v).toFixed(digits)}`);
  const fitted = selectors.rows.filter((r) => !r.model.startsWith("Single factor"));
  const factors = selectors.rows.filter((r) => r.model.startsWith("Single factor"));
  const beatsRandom = (r) => r.p_random != null && r.p_random < 0.05 && (r.expectancy ?? 0) > 0;
  const skilledFitted = fitted.filter(beatsRandom);
  const winningFactors = factors.filter(beatsRandom);
  const fittedP = fitted.filter((r) => r.p_random != null).map((r) => r.p_random);
  const noisy = shap && shap.rows.every((r) => r.own_ic_t == null || Math.abs(r.own_ic_t) < 2);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-1">How we built and evaluated the ML models behind this bot</h1>
      <p className="text-sm text-gray-500 mb-4">
        The data, the models, the training, and every test we ran on them.
      </p>

      <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-3">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">In brief</p>
        <p className="text-sm text-gray-700 leading-relaxed">
          We built cross-sectional selection models on the S&amp;P 500 reconstructed as it stood each
          month (survivorship-free), 2015&ndash;present, using seventeen leakage-safe features and a
          deliberate ladder of model complexity: a linear baseline, a random forest and
          gradient-boosted trees: all trained walk-forward so no future data leaks. We then evaluated
          them out of sample from several angles: predictive accuracy, portfolio performance against a
          low-cost index fund with a random-selection null, feature attribution, an event study of the
          rule-based signals, and the statistical power of the tests themselves.{" "}
          {skilledFitted.length === 0 ? (
            <>
              The result was consistent across the whole ladder: no model showed a predictive edge out
              of sample
              {fittedP.length ? ` (p from ${Math.min(...fittedP).toFixed(3)} to ${Math.max(...fittedP).toFixed(3)})` : ""}
              , and every one trailed the index fund.
              {winningFactors.length
                ? ` The only factor that beat random selection was risk exposure: ranking on ${winningFactors
                    .map((r) => r.model.replace("Single factor: ", ""))
                    .join(" and ")}, which is compensation for bearing risk, not evidence of forecasting skill.`
                : ""}{" "}
              We therefore report no evidence of a predictive edge, established across the whole method
              space, rather than proof that none could exist.
            </>
          ) : (
            <>
              {skilledFitted.map((r) => r.model).join(", ")} beat random selection on return; the
              caveats below qualify how far that can be relied on.
            </>
          )}
        </p>
      </div>
      <p className="text-xs text-gray-400 mb-6">
        Historical research, not personal financial advice. Evidence generated {data.generated_at}.
        Each section opens to its full numbers and method.
      </p>

      <nav className="flex flex-wrap gap-x-4 gap-y-1 mb-6 text-sm text-brand-700">
        {CONTENTS.map(([id, label]) => (
          <a key={id} href={`#${id}`} className="hover:underline">
            {label}
          </a>
        ))}
      </nav>

      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3">
        How we built the models
      </h2>

      <section id="build" className="bg-white border border-gray-200 rounded-xl p-5 mb-6 scroll-mt-20">
        <h3 className="text-base font-semibold border-l-4 border-brand-500 pl-3 mb-2">
          1. How the models were built
        </h3>
        <p className="text-sm text-gray-700">
          Each month the bot ranks the stocks that were really in the S&amp;P 500 that month and holds
          the top five equally. To keep the evaluation honest, we built the same task on a deliberate
          ladder of model complexity: a linear baseline, a random forest and gradient-boosted trees,
          on one shared, leakage-safe feature set, so any result could be attributed to the data
          rather than to a single architecture.
        </p>
        <details className="group mt-3">
          <summary className="cursor-pointer list-none text-sm font-medium text-brand-600 hover:text-brand-700 flex items-center gap-1">
            <span className="transition-transform group-open:rotate-90">&#9656;</span>
            The data, features and training in full
          </summary>
          <div className="mt-3 text-sm text-gray-600 space-y-3">
            <p>
              <strong>Data &amp; universe.</strong> The S&amp;P 500 reconstructed month by month from a
              historical membership list (survivorship-free), with dividend-adjusted prices. Company
              fundamentals enter a feature only after a 60-day reporting lag, so nothing uses a filing
              before it could have been read.
            </p>
            <p>
              <strong>Features (17).</strong> Ten technical factors: momentum at 1, 3, 6 and 12
              months, risk-adjusted momentum, short-term reversal, distance from the 52-week high,
              volatility, MACD and ADX, and six fundamentals: P/E, P/B, ROE, earnings growth, net
              margin and debt/equity. A FinBERT news-sentiment feature is a removable extension.
            </p>
            <p>
              <strong>Task &amp; label.</strong> Cross-sectional selection: each stock is labelled by
              whether its next-month return beats the cross-section, and the model ranks the whole
              universe rather than forecasting one price.
            </p>
            <p>
              <strong>Model ladder.</strong> A logistic / ridge linear baseline, a random forest
              (Breiman, 2001) and gradient-boosted trees (XGBoost; Chen &amp; Guestrin, 2016), compared
              on identical folds so complexity is the only thing that changes.
            </p>
            <p>
              <strong>Training: walk-forward.</strong> Each month the model trains only on earlier
              months (time-series cross-validation); because the label horizon equals the rebalance
              spacing, the most recent training label resolves exactly at the rebalance date, so no
              future information leaks. Returns are net of a 10-basis-point per-turnover cost.
            </p>
            <p>
              <strong>Extensions.</strong> A FinBERT news-sentiment feature, and a short-horizon
              direction forecast pooled across stocks and probability-calibrated (Platt, 1999),
              evaluated in sections below. 297 automated tests cover the indicator maths, the leakage
              guards and the statistics.
            </p>
          </div>
        </details>
      </section>

      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-3 mt-8">
        How we evaluated them
      </h2>

      {survivorship && (
        <Section
          id="survivorship"
          method="The same models, features, folds and costs were run twice, changing only the universe: today's constituent list against the index reconstructed month by month from a historical membership file, with 30 verified ticker changes mapped and monthly coverage reported."
          title="2. Survivorship bias: the universe drives the result"
          verdict="The same models look far better on today's S&P 500 list than on the index as it really was each month. Today's list only contains the companies that survived, so most of a typical backtest's shine is survivorship bias, not skill."
          intro="The same models tested two ways. Today's S&P 500 list only contains companies that went on to succeed, which nobody could have known in advance; the point-in-time list contains the stocks actually in the index each month. Even a rule with no model at all looks brilliant on today's list."
          source={survivorship.sources}
        >
          <Table
            head={["Model", "Today's list: return / yr", "alpha (t)", "Point-in-time: return / yr", "alpha (t)"]}
            rows={[
              ...survivorship.rows.map((r) => [
                r.model,
                pct(r.current.cagr),
                withT(r.current.alpha, r.current.alpha_t),
                pct(r.historical.cagr),
                withT(r.historical.alpha, r.historical.alpha_t),
              ]),
              ["SPY (index fund)", pct(survivorship.spy?.cagr), "", pct(survivorship.spy?.cagr), ""],
            ]}
            highlight={(i) => i === survivorship.rows.length}
          />
        </Section>
      )}

      <Section
        id="honest-test"
        title="3. Out-of-sample portfolio performance against the index"
        verdict={
          skilledFitted.length === 0
            ? `None of the fitted machine-learning models beat picking stocks at random, and every one trailed a low-cost index fund. Only ranking on a single risk factor (${winningFactors.map((r) => r.model.replace("Single factor: ", "")).join(" or ") || "beta / volatility"}) beat random, and that is payment for extra risk, not forecasting.`
            : `${skilledFitted.map((r) => r.model).join(", ")} beat random picking on return: see the caveats before relying on it.`
        }
        method="Monthly walk-forward (time-series cross-validation): each month the model is trained only on earlier months, buys its top five names net of 10 basis points of costs, and holds SPY when nothing qualifies. Four model families are compared on identical folds (a linear/logistic baseline, a random forest and gradient-boosted trees (XGBoost)) each given the same seventeen features: ten technical factors (momentum at 1, 3, 6 and 12 months, risk-adjusted momentum, short-term reversal, distance from the 52-week high, volatility, MACD and ADX) plus six fundamentals (P/E, P/B, ROE, earnings growth, net margin, debt/equity). Only model complexity varies across the ladder, not the inputs. Alpha comes from a CAPM regression net of the Treasury-bill rate with Newey-West standard errors, so a portfolio cannot earn alpha simply by carrying more market risk."
        intro={`${selectors.period}. Each month the model buys its top five stocks (holding SPY if none qualify), after 0.1% trading costs. The primary test ("beats random") compares the picks with the same number of stocks drawn at random from the whole cross-section that month; p below 0.05 means they earned more than chance. "vs similar-risk random" is a stricter, secondary check that also holds volatility constant, which by construction strips out any return earned simply by choosing riskier stocks, so a factor rule that works by taking risk is not expected to pass it. IC measures whether the whole ranking lines up with what happened next.`}
        source={selectors.source}
      >
        <Table
          head={["Model", "Return / yr", "Sharpe", "Worst drawdown", "alpha (t)", "Return / pick, beats random (p)", "vs similar-risk random (p)", "IC t"]}
          rows={[
            ...selectors.rows.map((r) => [
              r.model,
              pct(r.cagr),
              num(r.sharpe),
              pct(r.max_dd, false),
              withT(r.alpha, r.alpha_t),
              r.p_random == null ? "-" : `${pct(r.expectancy, true, 2)}, p=${num(r.p_random, 3)}`,
              num(r.p_matched, 3),
              num(r.ic_t),
            ]),
            ["SPY (index fund)", pct(spy?.cagr), num(spy?.sharpe), pct(spy?.max_dd, false), "", "", "", ""],
          ]}
          highlight={(i) => i === selectors.rows.length}
        />
      </Section>

      {me && (
        <Section
          id="forecaster"
          title="4. Predictive accuracy: out-of-sample R²"
          verdict={`No. Out of sample every model scores a negative R² (worse than predicting a constant) while its in-sample R² is strongly positive. That gap is the signature of memorising the training window, not learning; and more training data never turns it positive.`}
          method="Every figure is out of sample: each month's model trains only on earlier rebalances and scores that month's cross-section, pooled across all 132 months (walk-forward, i.e. time-series cross-validation). The same simple-to-complex ladder is used (a ridge linear baseline, a random forest and gradient-boosted trees) on identical features and folds, so a negative result cannot be blamed on a single model being under- or over-powered. The baseline is the mean of the model's own training window: the only constant a forecaster could use without seeing the future (Campbell and Thompson, 2008). R² above zero beats that constant; below zero is worse. Gap is in-sample minus out-of-sample R²."
          intro="A forecaster is only useful if it beats a constant. It does not: out of sample the R² is negative for every model, even as the in-sample fit looks strong: the classic memorisation signature."
          source={me.source}
        >
          <h3 className="text-sm font-semibold mb-2">Out-of-sample R²: does it beat a constant?</h3>
          <Table
            head={["Model", "R² out-of-sample", "R² in-sample", "Gap", "RMSE vs baseline", "Mean IC (t)", "Verdict"]}
            rows={me.r2.map((r) => [
              r.model,
              r2f(r.r2_oos),
              r2f(r.r2_is),
              r2f(r.gap),
              `${num(r.rmse)} vs ${num(r.baseline_rmse)}`,
              `${num(r.ic, 3)} (${num(r.ic_t)})`,
              r.verdict,
            ])}
          />

          <h3 className="text-sm font-semibold mt-5 mb-2">Does more training data help? (learning curve)</h3>
          <p className="text-sm text-gray-600 mb-2">
            A model learning a real pattern improves, or at least stops degrading, as its training
            window lengthens. Here the out-of-sample R² rises toward zero but never crosses it: more
            data buys less over-fitting, not prediction. The product&apos;s fixed 12-month window is
            one row of this table.
          </p>
          <Table
            head={["Training window", "R² out-of-sample", "R² in-sample", "Gap"]}
            rows={me.learning_curve.map((r) => [r.window, r2f(r.r2_oos), r2f(r.r2_is), r2f(r.gap)])}
            highlight={(i) => me.learning_curve[i].window === "12m"}
          />

          {me.stability && (
            <>
              <h3 className="text-sm font-semibold mt-5 mb-2">Is the ranking stable, or just repeating itself?</h3>
              <p className="text-sm text-gray-600">
                Month to month, the model&apos;s predictions for the same stocks correlate{" "}
                <strong>{r2f(me.stability.pred_autocorr, 2)}</strong> with last month&apos;s, while the
                thing it is trying to predict correlates only{" "}
                <strong>{r2f(me.stability.target_autocorr, 2)}</strong>. A prediction far more
                persistent than its target means the model is largely repeating itself, not tracking
                anything real.
              </p>
            </>
          )}

          {(me.by_year || me.by_vol) && (
            <div className="grid md:grid-cols-2 gap-6 mt-5">
              {me.by_year && (
                <div>
                  <h3 className="text-sm font-semibold mb-2">Out-of-sample R² by year</h3>
                  <Table
                    head={["Year", "R² OOS", "IC", "Rows"]}
                    rows={me.by_year.map((r) => [r.year, r2f(r.r2_oos), num(r.ic, 3), Number(r.rows).toLocaleString()])}
                  />
                </div>
              )}
              {me.by_vol && (
                <div>
                  <h3 className="text-sm font-semibold mb-2">By volatility bucket (0 = calmest, 4 = wildest)</h3>
                  <Table
                    head={["Bucket", "R² OOS", "IC", "Rows"]}
                    rows={me.by_vol.map((r) => [r.bucket, r2f(r.r2_oos), num(r.ic, 3), Number(r.rows).toLocaleString()])}
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    The top bucket matters most: those are the high-volatility names a risk-taking
                    selector actually buys, and its error is largest there.
                  </p>
                </div>
              )}
            </div>
          )}
        </Section>
      )}

      {shap && (
        <Section
          id="learned"
          title="5. What the model relied on"
          verdict={
            noisy
              ? "No input predicts returns on its own: the model spreads its reliance evenly across inputs that each carry no signal, so what it learned is noise, not a reliable pattern."
              : "Reliance is spread across the inputs; only a few carry any signal of their own."
          }
          method="TreeSHAP values were taken for every out-of-sample prediction, so the contributions sum exactly to the prediction the model made that month. Each feature's own information coefficient is the monthly rank correlation between that feature alone and what happened next."
          intro={`How much each input moved the model's out-of-sample predictions (reliance), which way it pushed them, and whether that input predicted returns on its own (own IC).${
            noisy ? " No input has a significant IC of its own, so what the model learned is noise rather than a reliable pattern." : ""
          }`}
          source={shap.source}
        >
          <Table
            head={["Input", "Reliance", "Direction learned", "Own IC (t)", "Backed by data?"]}
            rows={shap.rows.map((r) => [
              `${r.meaning}`,
              pct(r.reliance, false),
              r.direction,
              `${num(r.own_ic, 3)} (${num(r.own_ic_t)})`,
              r.backed,
            ])}
          />
        </Section>
      )}

      <Section
        id="signals"
        title="6. Event study of the rule-based signals"
        verdict="A Buy signal gave no advantage over a typical S&P 500 stock; the only difference clearing significance ran the wrong way (Sell slightly out-performed). Every gap is smaller than a tenth of a percent a week: one round-trip in costs erases it."
        method="Every stock was replayed day by day across the days it was an index member, 1.25 million stock-days in all. Each signal is entered at that day's close and held 5, 20 or 60 trading days. Because signals cluster on the same days and holding periods overlap, each day is averaged first and the t-statistic uses Newey-West standard errors."
        intro={`The Buy / Sell / Hold label is produced by the rule-based advisor from moving-average crossovers, MACD momentum and trend strength (ADX), using only prices up to each day. It is a fixed rule, so there is nothing here to fit or cross-validate: this section asks a simpler question: when the rule fired historically, what happened next? Every stock on the days it was in the S&P 500, ${signals.period}. "Beat universe" is how often the stock did better than the average S&P 500 member over the same days; a useful Buy signal would sit clearly above the typical-stock row.`}
        source={signals.source}
      >
        <Table
          head={["Signal", "Held (days)", "Events", "Beat universe", "vs universe (t)", "Beat SPY", "vs SPY (t)"]}
          rows={signals.rows.map((r) => [
            r.signal,
            r.horizon,
            Number(r.n).toLocaleString(),
            pct(r.beat_uni, false),
            r.t_uni == null ? pct(r.x_uni, true, 2) : `${pct(r.x_uni, true, 2)} (${num(r.t_uni)})`,
            pct(r.beat_spy, false),
            `${pct(r.x_spy, true, 2)} (${num(r.t_spy)})`,
          ])}
          highlight={(i) => signals.rows[i].signal.startsWith("Every")}
        />
        <p className="text-sm text-gray-600 mt-4">
          These signals are fixed rules, so nothing is trained or cross-validated here. The{" "}
          <em>trained-model</em> version of the same question: models fitted on the seventeen
          features with walk-forward (time-series) cross-validation, run from a{" "}
          <strong>simple linear baseline through a random forest to gradient-boosted trees</strong>{" "}
         : is in sections{" "}
          <a href="#honest-test" className="text-brand-600 hover:underline">3</a> and{" "}
          <a href="#forecaster" className="text-brand-600 hover:underline">4</a>, and reaches the same
          verdict. Neither the hand-built rules nor models of increasing complexity found reliable
          information about what a stock does next, so the conclusion is <em>no evidence of a
          predictive edge</em>, established across the whole method ladder, not one model&apos;s
          failure.
        </p>
      </Section>

      {events && (
        <Section
          id="events"
          title="7. Returns around index-membership events"
          verdict="The big move happens before the change takes effect: the index adds stocks that already rose and drops ones that already fell. After the effective date there is no reliable pattern to trade."
          method="Day 0 is the first trading day on or after the change takes effect. Returns are measured against SPY over the same days, and each date's events are averaged before the t-statistic, since many changes share an effective date."
          intro="Day 0 is the date the change takes effect. The big moves happen before it: the index adds stocks that have risen and removes ones that have fallen. Afterwards there is no reliable pattern."
          source={events.source}
        >
          <div className="grid md:grid-cols-2 gap-6">
            {[["Added", events.added], ["Removed (still trading)", events.demoted]].map(([label, rows]) => (
              <div key={label}>
                <h3 className="text-sm font-semibold mb-2">{label}</h3>
                <Table
                  head={["Window", "Beat SPY", "Mean", "t"]}
                  rows={rows.map((r) => [r.window, pct(r.beat_spy, false), pct(r.mean), num(r.t)])}
                />
              </div>
            ))}
          </div>
        </Section>
      )}

      {fund && (
        <Section
          id="fundamentals"
          title="8. Fundamental features, 2023 onwards (preliminary)"
          verdict="The one encouraging signal in the whole study, but a lead, not a result: the gain concentrates in 2025-26, the ranking IC is still about zero, Sharpe stays below SPY's, and look-ahead cannot be ruled out."
          method="Identical universe, folds and rules, run with and without the six filing-based features over the only window in which the provider supplies them. The p-value compares each pick with random stocks of similar volatility from the same month."
          intro={`Company financials are only available from 2023, so this is the one window where they can be tested: the same models with and without them. The gain is concentrated in 2025-2026, the ranking IC is still about zero, Sharpe stays below SPY's, and the data may carry look-ahead (current share counts, a fixed reporting lag). Treat it as a lead to check, not a result. Walk-forward starts ${fund.walk_forward_start}.`}
          source={fund.sources}
        >
          <Table
            head={["Features", "Model", "Return / yr", "Sharpe", "Worst drawdown", "alpha (t)", "Pick p"]}
            rows={[
              ...fund.rows.map((r) => [
                r.variant,
                r.model,
                pct(r.cagr),
                num(r.sharpe),
                pct(r.max_dd, false),
                withT(r.alpha, r.alpha_t),
                num(r.p_matched, 3),
              ]),
              ["SPY (index fund)", "", pct(fund.spy?.cagr), num(fund.spy?.sharpe), pct(fund.spy?.max_dd, false), "", ""],
            ]}
            highlight={(i) => i === fund.rows.length}
          />
        </Section>
      )}

      <Section
        id="limits"
        title="9. Limitations and statistical power"
        verdict="This detects large effects, not small ones, with 140 months, only an edge near ten points a year would show. The claim is “no evidence of an edge”, not “an edge is impossible”."
        intro="A test that finds nothing is only as strong as its ability to find something, so its limits belong beside its results."
      >
        <ul className="text-sm text-gray-600 space-y-2 list-disc pl-5">
          <li>
            <strong>It detects large effects, not small ones.</strong> With 140 monthly
            observations, an edge worth roughly ten percentage points a year would be needed
            before it could be called significant, so a smaller genuine edge would not show up
            here.
          </li>
          <li>
            <strong>Some companies are missing.</strong> Constituents whose prices the data
            provider no longer serves, mostly takeovers and bankruptcies, cannot be included, so
            the reconstructed index is better than today&apos;s list but not complete.
          </li>
          <li>
            <strong>One market, one era.</strong> The evidence covers large US companies from
            2015 to 2026, a period dominated by a rising market. It says nothing about smaller
            companies, other countries, or trading within the day.
          </li>
          <li>
            <strong>Many questions were asked of one dataset.</strong> Where a single number here
            just clears the usual threshold, it is reported as suggestive rather than settled.
          </li>
        </ul>
        <p className="text-sm text-gray-600 mt-4">
          &ldquo;No evidence of an edge&rdquo; is therefore the claim, not &ldquo;an edge is
          impossible&rdquo;.
        </p>
      </Section>
    </div>
  );
}
