interface RoutePlaceholderProps {
  screen: string;
  description: string;
  phase?: string;
  requirements?: string[];
}

export function RoutePlaceholder({
  screen,
  description,
  phase,
  requirements,
}: RoutePlaceholderProps) {
  return (
    <main className="min-h-full p-12 flex flex-col gap-6 max-w-3xl mx-auto">
      <div className="hairline-b pb-4">
        <p className="font-mono text-xs text-fg-subtle uppercase tracking-wider">
          Placeholder · {phase ?? "TBD"}
        </p>
        <h1 className="font-display text-4xl font-medium leading-tight tracking-tight text-fg mt-2">
          {screen}
        </h1>
        <p className="font-sans text-base text-fg-muted mt-2">{description}</p>
      </div>
      {requirements && requirements.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {requirements.map((r) => (
            <span
              key={r}
              className="hairline rounded-md px-2 py-1 font-mono text-xs text-fg-subtle"
            >
              {r}
            </span>
          ))}
        </div>
      )}
    </main>
  );
}
