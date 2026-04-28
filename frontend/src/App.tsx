export default function App() {
  return (
    <main className="min-h-full flex flex-col items-center justify-center p-12 gap-8">
      <h1 className="font-display text-6xl font-medium leading-none tracking-tight text-fg">
        RecruitAI
      </h1>
      <p className="font-sans text-lg text-fg-muted max-w-md text-center">
        Editorial intelligence meets enterprise scale. Foundation phase smoke test.
      </p>
      <div className="flex gap-3 items-center">
        <span className="hairline rounded-md px-3 py-1 font-mono text-xs text-fg-subtle">
          accent: #1F3A2E
        </span>
        <span className="hairline rounded-md px-3 py-1 font-mono text-xs text-fg-subtle">
          bg: #FAFAF7
        </span>
        <span className="hairline rounded-md px-3 py-1 font-mono text-xs text-fg-subtle">
          font: Fraunces · Geist · Geist Mono
        </span>
      </div>
      <button
        type="button"
        className="bg-accent text-accent-fg px-5 py-2 rounded-md font-sans text-sm font-medium transition-colors duration-200 hover:bg-accent-hover"
      >
        Foundation OK
      </button>
    </main>
  );
}
