const QUESTIONS = [
  {
    key: "risk",
    label: "How much swing can you live with?",
    options: [
      ["low", "Not much"],
      ["medium", "Some"],
      ["high", "A lot"],
    ],
  },
  {
    key: "horizon",
    label: "How long is this money invested for?",
    options: [
      ["under_1y", "Under a year"],
      ["1_to_5y", "1-5 years"],
      ["over_5y", "5 years or more"],
    ],
  },
  {
    key: "single_stocks",
    label: "Are you willing to hold individual shares?",
    options: [
      [true, "Yes"],
      [false, "No, funds only"],
    ],
  },
];

export default function ProfileCard({ profile, guidance, onChange, onReset }) {
  return (
    <section className="bg-white border border-gray-200 rounded-xl p-5 mb-8">
      <h2 className="text-lg font-semibold border-l-4 border-brand-500 pl-3 mb-1">
        Three questions before you read the advice
      </h2>
      <p className="text-sm text-gray-500 mb-4">
        Your answers stay in this browser. They change how many names are shown and what
        this page leads with, and nothing else.
      </p>

      <div className="space-y-4">
        {QUESTIONS.map((q) => (
          <div key={q.key}>
            <p className="text-sm font-medium text-gray-700 mb-2">{q.label}</p>
            <div className="flex flex-wrap gap-2">
              {q.options.map(([value, label]) => {
                const selected = profile?.[q.key] === value;
                return (
                  <button
                    key={String(value)}
                    onClick={() => onChange({ ...(profile ?? {}), [q.key]: value })}
                    className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
                      selected
                        ? "bg-brand-600 text-white border-brand-600"
                        : "bg-white text-gray-600 border-gray-300 hover:border-brand-400"
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {guidance && (
        <div className="mt-4 bg-gray-50 border border-gray-200 rounded-lg p-3">
          <ul className="text-sm text-gray-700 space-y-1">
            {guidance.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
          <p className="text-xs text-gray-400 mt-2">{guidance.disclaimer}</p>
        </div>
      )}

      {profile && (
        <div className="mt-4 text-sm">
          <button onClick={onReset} className="text-gray-500 hover:text-gray-700">
            Clear my answers
          </button>
        </div>
      )}
    </section>
  );
}
