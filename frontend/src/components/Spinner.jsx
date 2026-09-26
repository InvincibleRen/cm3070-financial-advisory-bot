
export default function Spinner({ message }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-gray-400">
      <svg
        className="animate-spin h-8 w-8 mb-3"
        viewBox="0 0 24 24"
        fill="none"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
        />
      </svg>
      {message && <p className="text-sm">{message}</p>}
    </div>
  );
}
