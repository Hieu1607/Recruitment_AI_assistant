import { Link } from "react-router";
import { routes } from "@/routes";

export default function NotFoundRoute() {
  return (
    <main className="min-h-full flex flex-col items-center justify-center gap-6 p-12">
      <p className="font-mono text-xs text-fg-subtle uppercase tracking-wider">404</p>
      <h1 className="font-display text-5xl font-medium tracking-tight text-fg">
        Page not found.
      </h1>
      <p className="font-sans text-base text-fg-muted max-w-md text-center">
        The page you&apos;re looking for moved, never existed, or you mistyped the URL.
      </p>
      <Link
        to={routes.dashboard}
        className="bg-accent text-accent-fg px-5 py-2 rounded-md font-sans text-sm font-medium hover:bg-accent-hover transition-colors"
      >
        Back to dashboard
      </Link>
    </main>
  );
}
