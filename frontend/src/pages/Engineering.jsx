

const SNAPSHOT = "2026-09-24";

function Card({ title, children, tone = "default" }) {
  const border =
    tone === "gap" ? "border-amber-200 bg-amber-50" : "border-gray-200 bg-white";
  return (
    <section className={`border ${border} rounded-xl p-5`}>
      <h2 className="text-base font-semibold border-l-4 border-brand-500 pl-3 mb-3">{title}</h2>
      <div className="text-sm text-gray-600 space-y-2">{children}</div>
    </section>
  );
}

function Stat({ value, label }) {
  return (
    <div className="border border-gray-200 rounded-xl p-4 text-center bg-white">
      <div className="text-2xl font-bold tabular-nums">{value}</div>
      <div className="text-xs text-gray-500 mt-1">{label}</div>
    </div>
  );
}

export default function Engineering() {
  return (
    <div className="page-enter">
      <h1 className="text-2xl font-bold mb-2">How it&apos;s built</h1>
      <p className="text-gray-600 mb-6 leading-relaxed">
        The advice on this site is only as trustworthy as the apparatus behind it. This page shows
        the engineering that makes the results reproducible, tracked and stable.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <Stat value="297" label="automated tests" />
        <Stat value="0" label="explanation faithfulness violations (59,952 sentences)" />
        <Stat value="4,959" label="fundamental rows audited for look-ahead" />
        <Stat value={SNAPSHOT} label="frozen data snapshot" />
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        <Card title="Data pipeline">
          <p>
            One directed path with an injectable data fetcher at its head. Prices and fundamentals
            are pulled through the provider into <strong>validated local CSV files</strong>: an audit
            rejects duplicate, missing or non-positive rows and verifies a 60-day fundamentals
            reporting lag on all 4,959 rows.
          </p>
          <p>
            The feature builder reads <strong>only that cache</strong>, so every run is deterministic
            and independent of the network, and every feature uses information available up to its
            date only (leakage-safe by construction).
          </p>
        </Card>

        <Card title="Run tracking &amp; provenance">
          <p>
            Every experiment writes a <strong>timestamped report</strong> whose filename encodes its
            configuration, and <strong>git is the run history</strong>. Every figure on this site is
            drawn from a single committed snapshot.
          </p>
          <p>
            One chain keeps the product and the study in step: the same reports are parsed once into
            <code className="mx-1 px-1 bg-gray-100 rounded">evidence.json</code>, which the API and
            the interface both read, so the numbers a user sees and the numbers in the evaluation
            <strong> cannot drift apart, and none is hand-typed</strong>.
          </p>
        </Card>

        <Card title="Testing &amp; stability">
          <p>
            Stability is guarded by a <strong>297-test regression harness</strong> rather than by
            live instrumentation. Beyond the indicator mathematics it includes:
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li>a direct <strong>look-ahead test</strong>: injecting a future price spike and a future filing leaves every past feature row unchanged;</li>
            <li>a <strong>per-extension isolation test</strong>: the core imports neither extension;</li>
            <li><strong>synthetic-data checks</strong> on the statistics whose answer is known in advance;</li>
            <li>two <strong>honesty tests</strong> that pin the reporting, so a failed check reports itself as failing.</li>
          </ul>
        </Card>

        <Card title="Reproducibility &amp; data-drift">
          <p>
            The environment is pinned to <strong>exact dependency versions</strong> and the data to a
            frozen snapshot, so the whole evaluation rebuilds from committed files offline.
          </p>
          <p>
            This matters because the price provider <strong>silently revises history</strong>: a
            model&apos;s significance once shifted between two runs for that reason alone. The response
            is to pin every figure to the frozen batch and read from committed runs rather than
            re-fetch.
          </p>
        </Card>
      </div>
    </div>
  );
}
