import { Link } from "react-router-dom";

export default function TrustStrip() {
  return (
    <Link
      to="/evidence"
      className="flex items-center justify-between gap-4 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 mb-8 hover:bg-amber-100 transition-colors"
    >
      <span className="text-sm text-amber-900">
        <strong>Before you act on this:</strong> in our own out-of-sample tests, this kind of
        advice did not beat simply holding a low-cost S&amp;P 500 index fund.
      </span>
      <span className="text-sm font-medium text-amber-900 whitespace-nowrap">
        See the track record &rarr;
      </span>
    </Link>
  );
}
