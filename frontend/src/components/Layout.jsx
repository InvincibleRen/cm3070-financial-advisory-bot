import { Link, Outlet, useLocation } from "react-router-dom";

export default function Layout() {
  const { pathname } = useLocation();

  return (
    <div className="min-h-screen flex flex-col">

      <header className="bg-white border-b border-gray-200 sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 h-14 flex items-center gap-6">
          <Link to="/" className="text-lg font-bold text-brand-700 tracking-tight">
            Financial Advisor Bot
          </Link>
          <nav className="flex gap-4 text-sm font-medium">
            <Link
              to="/"
              className={`transition-colors ${
                pathname === "/" ? "text-brand-600" : "text-gray-500 hover:text-gray-800"
              }`}
            >
              Analyse a stock
            </Link>
            <Link
              to="/evidence"
              className={`transition-colors ${
                pathname === "/evidence" ? "text-brand-600" : "text-gray-500 hover:text-gray-800"
              }`}
            >
              The evaluation
            </Link>
            <Link
              to="/engineering"
              className={`transition-colors ${
                pathname === "/engineering" ? "text-brand-600" : "text-gray-500 hover:text-gray-800"
              }`}
            >
              How it&apos;s built
            </Link>
          </nav>
        </div>
      </header>

      <main className="flex-1">
        <div className="max-w-6xl mx-auto px-4 py-8">
          <Outlet />
        </div>
      </main>

      <footer className="border-t border-gray-200 py-4 text-center text-xs text-gray-400">
        CM3070 Final-Year Project &middot; Ren Haowen (230726235)
      </footer>
    </div>
  );
}
