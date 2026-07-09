import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import {
  Badge,
  Button,
} from "@/components/ui";
import {
  CheckCircle2,
  ChevronRight,
  FileText,
  Sparkles,
  Users,
  Zap,
} from "lucide-react";

type StoryboardFrame = {
  id: string;
  title: string;
  detail: string;
  kicker: string;
  accentClass: string;
};

const storyboardFrames: StoryboardFrame[] = [
  {
    id: "job",
    title: "Tạo job",
    detail: "Role brief, hiring targets, and must-have signals are defined in one workspace.",
    kicker: "01",
    accentClass: "from-sky-100 via-white to-cyan-50",
  },
  {
    id: "upload",
    title: "Upload CV",
    detail: "Recruiters drop a batch of resumes and watch the queue organize itself instantly.",
    kicker: "02",
    accentClass: "from-amber-100 via-white to-orange-50",
  },
  {
    id: "parse",
    title: "AI parse CV",
    detail: "Skills, years, seniority, and role evidence are extracted into structured profile cards.",
    kicker: "03",
    accentClass: "from-violet-100 via-white to-fuchsia-50",
  },
  {
    id: "shortlist",
    title: "Shortlist",
    detail: "Top candidates surface with scores, rationale, and recruiter-ready next actions.",
    kicker: "04",
    accentClass: "from-emerald-100 via-white to-teal-50",
  },
];

function usePrefersReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }

    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updatePreference = () => setPrefersReducedMotion(mediaQuery.matches);

    updatePreference();
    mediaQuery.addEventListener("change", updatePreference);

    return () => mediaQuery.removeEventListener("change", updatePreference);
  }, []);

  return prefersReducedMotion;
}

function useRevealMotion(prefersReducedMotion: boolean) {
  const ref = useRef<HTMLElement | null>(null);
  const [isVisible, setIsVisible] = useState(prefersReducedMotion);

  useEffect(() => {
    if (prefersReducedMotion) {
      setIsVisible(true);
      return undefined;
    }

    const node = ref.current;
    if (!node) {
      return undefined;
    }

    const rect = node.getBoundingClientRect();
    if (rect.top < window.innerHeight * 0.92 && rect.bottom > 0) {
      setIsVisible(true);
      return undefined;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      {
        threshold: 0.18,
        rootMargin: "0px 0px -8% 0px",
      },
    );

    observer.observe(node);

    return () => observer.disconnect();
  }, [prefersReducedMotion]);

  return { ref, isVisible };
}

function BrowserFrame({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-sand-200 bg-white shadow-2xl overflow-hidden shadow-forest-900/10">
      <div className="h-12 bg-sand-100 border-b border-sand-200 flex items-center px-4 gap-2">
        <div className="w-3 h-3 rounded-full bg-red-400"></div>
        <div className="w-3 h-3 rounded-full bg-amber-400"></div>
        <div className="w-3 h-3 rounded-full bg-green-400"></div>
        <div className="mx-auto bg-white border border-sand-200 text-xs text-forest-400 px-4 py-1 rounded-md w-64 text-center">
          app.easyhr.com
        </div>
      </div>
      {children}
    </div>
  );
}

function OverviewStory() {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [isLoaded, setIsLoaded] = useState(prefersReducedMotion);
  const { ref, isVisible } = useRevealMotion(prefersReducedMotion);

  useEffect(() => {
    if (prefersReducedMotion) {
      setIsLoaded(true);
      return undefined;
    }

    const timeoutId = window.setTimeout(() => setIsLoaded(true), 80);
    return () => window.clearTimeout(timeoutId);
  }, [prefersReducedMotion]);

  return (
    <BrowserFrame>
      <div
        data-testid="landing-story-overview"
        data-loaded={isLoaded ? "true" : "false"}
        data-motion="storyboard-shell"
        data-visible={isVisible ? "true" : "false"}
        className="grid gap-6 bg-[radial-gradient(circle_at_top_left,_rgba(37,99,235,0.12),_transparent_28%),radial-gradient(circle_at_bottom_right,_rgba(16,185,129,0.1),_transparent_28%),linear-gradient(180deg,_#fffdf8,_#f5efe4)] p-5 md:p-8"
        ref={ref as React.RefObject<HTMLDivElement>}
      >
        <div className="landing-fade-up max-w-3xl" style={{ animationDelay: "0ms" }}>
          <p className="text-xs font-semibold uppercase tracking-[0.24em] text-accent-700">
            Product storyboard
          </p>
          <h3 className="mt-3 font-serif text-3xl text-forest-900 md:text-5xl">
            From job setup to a recruiter-ready shortlist.
          </h3>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-forest-600 md:text-base">
            Four static snapshots show how EasyHR actually moves a recruiter from role setup,
            to batch intake, to AI profile parsing, to a clean shortlist.
          </p>
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          {storyboardFrames.map((frame, index) => (
            <StoryboardSnapshot
              key={frame.id}
              frame={frame}
              prefersReducedMotion={prefersReducedMotion}
              delayMs={140 + index * 90}
            />
          ))}
        </div>
      </div>
    </BrowserFrame>
  );
}

function StoryboardSnapshot({
  frame,
  delayMs,
  prefersReducedMotion,
}: {
  frame: StoryboardFrame;
  delayMs: number;
  prefersReducedMotion: boolean;
}) {
  return (
    <article
      className="landing-fade-up rounded-[28px] border border-sand-200 bg-white/92 p-4 shadow-lg shadow-forest-900/10 transition-transform duration-500 hover:-translate-y-1 hover:shadow-xl hover:shadow-forest-900/12"
      style={prefersReducedMotion ? undefined : { animationDelay: `${delayMs}ms` }}
    >
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-forest-400">{frame.kicker}</p>
          <h4 className="mt-2 font-serif text-2xl text-forest-900">{frame.title}</h4>
        </div>
        <span className="rounded-full bg-forest-900 px-3 py-1 text-xs font-medium text-white">
          Snapshot
        </span>
      </div>
      <p className="mt-3 text-sm leading-6 text-forest-600">{frame.detail}</p>

      <div className={`mt-4 rounded-[24px] bg-gradient-to-br ${frame.accentClass} p-4`}>
        <StoryboardPanel frameId={frame.id} prefersReducedMotion={prefersReducedMotion} />
      </div>
    </article>
  );
}

function StoryboardPanel({
  frameId,
  prefersReducedMotion,
}: {
  frameId: StoryboardFrame["id"];
  prefersReducedMotion: boolean;
}) {
  if (frameId === "job") {
    return (
      <div className="grid gap-3">
        <div className="rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
          <p className="text-xs uppercase tracking-[0.16em] text-forest-400">Role setup</p>
          <p className="mt-2 text-sm font-medium text-forest-900">Senior Frontend Engineer</p>
          <p className="mt-1 text-sm text-forest-600">
            React, TypeScript, design systems, remote-first hiring.
          </p>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          {[
            ["Hiring manager", "Linh Tran"],
            ["Priority signals", "Leadership, system design"],
            ["Applicants target", "30 profiles"],
            ["Interview loop", "4 stages"],
          ].map(([label, value]) => (
            <div key={label} className="rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
              <p className="text-xs uppercase tracking-[0.16em] text-forest-400">{label}</p>
              <p className="mt-2 text-sm font-medium text-forest-900">{value}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (frameId === "upload") {
    return (
      <div className="grid gap-3">
        <div className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
          <div>
            <p className="text-xs uppercase tracking-[0.16em] text-forest-400">Batch intake</p>
            <p className="mt-1 text-sm font-medium text-forest-900">12 resumes added</p>
          </div>
          <span
            className="landing-soft-pulse rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-800"
            data-motion={prefersReducedMotion ? "static" : "pulse"}
            data-testid="landing-uploading-badge"
          >
            Uploading
          </span>
        </div>
        {["FrontendLead.pdf", "StaffReact.pdf", "ProductUI.pdf"].map((fileName, index) => (
          <div
            key={fileName}
            className="flex items-center justify-between rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5"
          >
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-sand-50 p-2 text-forest-700">
                <FileText className="h-4 w-4" />
              </div>
              <span className="text-sm font-medium text-forest-900">{fileName}</span>
            </div>
            <div className="w-24 rounded-full bg-sand-100">
              <div
                className="h-2 rounded-full bg-amber-500"
                style={{ width: `${72 + index * 9}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (frameId === "parse") {
    return (
      <div className="grid gap-3 md:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-[24px] bg-white p-4 shadow-sm shadow-forest-900/5">
          <p className="text-sm font-medium text-forest-900">Avery Chen</p>
          <p className="mt-1 text-xs uppercase tracking-[0.16em] text-forest-400">
            Structured profile
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {["React", "TypeScript", "7 years", "Staff+", "Mentoring"].map((chip) => (
              <span key={chip} className="rounded-full bg-sand-50 px-3 py-1 text-xs text-forest-700">
                {chip}
              </span>
            ))}
          </div>
        </div>
        <div className="space-y-3">
          {[
            ["Signals extracted", "18"],
            ["Seniority confidence", "High"],
            ["Role match", "92%"],
          ].map(([label, value]) => (
            <div key={label} className="rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
              <p className="text-xs uppercase tracking-[0.18em] text-forest-400">{label}</p>
              <p className="mt-2 font-serif text-2xl text-forest-900">{value}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-3">
      {[
        ["Avery Chen", "94", "Strong architecture depth and hiring loop ownership"],
        ["Noah Martins", "91", "Reliable React platform and mentoring experience"],
        ["Mia Tran", "88", "Product-minded frontend delivery for growth teams"],
      ].map(([name, score, reason], index) => (
        <div
          key={name}
          data-motion={index === 0 && !prefersReducedMotion ? "hero-highlight" : "static"}
          data-testid={index === 0 ? "landing-shortlist-top" : undefined}
          className={`rounded-[24px] px-4 py-4 shadow-sm shadow-forest-900/5 ${
            index === 0
              ? "landing-breathe bg-[linear-gradient(135deg,_#134e4a,_#1f766e)] text-white"
              : "bg-white text-forest-900"
          }`}
        >
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="font-medium">{name}</p>
              <p className={`mt-1 text-sm ${index === 0 ? "text-sand-100" : "text-forest-600"}`}>
                {reason}
              </p>
            </div>
            <span
              className={`rounded-2xl px-3 py-2 text-lg font-semibold ${
                index === 0 ? "bg-white text-forest-900" : "bg-sand-50 text-forest-900"
              }`}
            >
              {score}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

type SupportingSnapshotProps = {
  eyebrow: string;
  title: string;
  description: string;
  testId: string;
  variant: "assistant" | "scoring";
};

function SupportingSnapshot({
  eyebrow,
  title,
  description,
  testId,
  variant,
}: SupportingSnapshotProps) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const { ref, isVisible } = useRevealMotion(prefersReducedMotion);

  return (
    <div
      data-testid={testId}
      data-motion="reveal"
      data-visible={isVisible ? "true" : "false"}
      className="relative aspect-square overflow-hidden rounded-3xl border border-sand-300 bg-[radial-gradient(circle_at_top_right,_rgba(27,55,39,0.08),_transparent_36%),linear-gradient(180deg,_#fffcf4,_#f4efe2)] p-4 md:p-5"
      ref={ref as React.RefObject<HTMLDivElement>}
    >
      <div className="landing-fade-up rounded-[28px] border border-white/70 bg-white/92 p-4 shadow-xl shadow-forest-900/10" style={prefersReducedMotion ? undefined : { animationDelay: "40ms" }}>
        <p className="text-xs uppercase tracking-[0.2em] text-forest-400">{eyebrow}</p>
        <h3 className="mt-2 font-serif text-2xl text-forest-900">{title}</h3>
        <p className="mt-3 text-sm leading-6 text-forest-600">{description}</p>
      </div>

      <div className="landing-fade-up mt-4 rounded-[26px] border border-sand-200 bg-white/88 p-4 shadow-lg shadow-forest-900/5" style={prefersReducedMotion ? undefined : { animationDelay: "120ms" }}>
        {variant === "scoring" ? <ScoringSnapshotPanel /> : <AssistantSnapshotPanel />}
      </div>
    </div>
  );
}

function ScoringSnapshotPanel() {
  return (
    <div className="grid gap-3">
      <div className="flex items-center justify-between rounded-2xl bg-sand-50 px-4 py-3">
        <div>
          <p className="text-sm font-medium text-forest-900">Candidate scorecard</p>
          <p className="text-xs uppercase tracking-[0.16em] text-forest-400">
            Technical · Communication · Domain
          </p>
        </div>
        <span className="landing-soft-pulse rounded-full bg-accent-50 px-3 py-1 text-xs font-medium text-accent-700">
          Live ranking
        </span>
      </div>

      {[
        ["Avery Chen", "92", "Technical fit 40%"],
        ["Jordan Lee", "88", "Communication 30%"],
        ["Priya Raman", "84", "Domain fit 30%"],
      ].map(([name, score, note]) => (
        <div key={name} className="transition-transform duration-300 hover:-translate-y-0.5 flex items-center justify-between gap-4 rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
          <div>
            <p className="text-sm font-medium text-forest-900">{name}</p>
            <p className="text-xs text-forest-500">{note}</p>
          </div>
          <span className="min-w-12 rounded-xl bg-forest-900 px-3 py-2 text-center text-sm font-semibold text-white">
            {score}
          </span>
        </div>
      ))}
    </div>
  );
}

function AssistantSnapshotPanel() {
  return (
    <div className="grid gap-3">
      <div className="rounded-[20px] bg-sand-50 p-4">
        <p className="text-xs uppercase tracking-[0.16em] text-forest-400">Talent search</p>
        <p className="mt-2 text-sm text-forest-700">
          Who has React platform depth, 5+ years, and experience mentoring teams?
        </p>
      </div>
      {[
        ["Avery Chen", "Staff Frontend Engineer"],
        ["Noah Martins", "Senior React Engineer"],
      ].map(([name, role]) => (
        <div key={name} className="transition-transform duration-300 hover:-translate-y-0.5 flex items-center justify-between rounded-2xl bg-white px-4 py-3 shadow-sm shadow-forest-900/5">
          <div>
            <p className="text-sm font-medium text-forest-900">{name}</p>
            <p className="text-xs text-forest-500">{role}</p>
          </div>
          <span className="landing-soft-pulse rounded-full bg-emerald-50 px-3 py-1 text-xs font-medium text-emerald-700">
            Match
          </span>
        </div>
      ))}
    </div>
  );
}

export default function LandingRoute() {
  return (
    <div className="min-h-screen bg-sand-50 selection:bg-accent-200">
      <style>{`
        @keyframes landingFadeUp {
          from {
            opacity: 0;
            transform: translateY(18px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        @keyframes landingSoftPulse {
          0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 0 rgba(245, 158, 11, 0);
          }
          50% {
            transform: scale(1.03);
            box-shadow: 0 0 0 8px rgba(245, 158, 11, 0.08);
          }
        }

        @keyframes landingBreathe {
          0%, 100% {
            transform: translateY(0);
            box-shadow: 0 10px 24px rgba(19, 78, 74, 0.14);
          }
          50% {
            transform: translateY(-2px);
            box-shadow: 0 16px 34px rgba(19, 78, 74, 0.2);
          }
        }

        .landing-fade-up {
          opacity: 0;
          animation: landingFadeUp 720ms cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }

        [data-motion="storyboard-shell"],
        [data-motion="reveal"] {
          opacity: 0;
          transform: translateY(24px);
          transition: opacity 680ms cubic-bezier(0.22, 1, 0.36, 1), transform 680ms cubic-bezier(0.22, 1, 0.36, 1);
        }

        [data-motion="storyboard-shell"][data-visible="true"],
        [data-motion="reveal"][data-visible="true"] {
          opacity: 1;
          transform: translateY(0);
        }

        .landing-soft-pulse {
          animation: landingSoftPulse 2.8s ease-in-out infinite;
          transform-origin: center;
        }

        .landing-breathe {
          animation: landingBreathe 4.6s ease-in-out infinite;
        }

        @media (prefers-reduced-motion: reduce) {
          .landing-fade-up,
          .landing-soft-pulse,
          .landing-breathe,
          [data-motion="storyboard-shell"],
          [data-motion="reveal"] {
            animation: none !important;
            opacity: 1 !important;
            transform: none !important;
            transition: none !important;
            box-shadow: inherit;
          }
        }
      `}</style>
      <nav className="sticky top-0 z-50 border-b border-sand-200 bg-white/50 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-forest-900">
              <Sparkles className="h-4 w-4 text-accent-400" />
            </div>
            <span className="font-serif text-xl font-bold text-forest-900">
              EasyHR
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Link to="/login" className="text-sm font-medium text-forest-600 hover:text-forest-900">
              Sign in
            </Link>
            <Link to="/login?mode=signup">
              <Button>Get started</Button>
            </Link>
          </div>
        </div>
      </nav>

      <main>
        <section className="mx-auto max-w-7xl px-4 pb-32 pt-24 text-center sm:px-6 lg:px-8">
          <h1 className="landing-fade-up mb-8 text-6xl font-serif leading-[1.1] tracking-tight text-forest-900 md:text-8xl" style={{ animationDelay: "0ms" }}>
            Hire like it's <span className="italic text-accent-600">2030.</span>
          </h1>
          <p className="landing-fade-up mx-auto mb-10 max-w-3xl text-xl font-light leading-relaxed text-fg-muted md:text-2xl" style={{ animationDelay: "100ms" }}>
            The intelligent recruitment platform that turns overwhelming resume piles into
            curated shortlists, predictive scores, and automated outreach in minutes.
          </p>
          <div className="landing-fade-up flex flex-col items-center justify-center gap-4 sm:flex-row" style={{ animationDelay: "180ms" }}>
            <Link to="/login?mode=signup">
              <Button size="lg" className="h-14 px-8 text-base">
                Start your free trial <ChevronRight className="ml-2 inline h-4 w-4 align-middle" />
              </Button>
            </Link>
            <Button size="lg" variant="secondary" className="h-14 px-8 text-base">
              Book a demo
            </Button>
          </div>
        </section>

        <section className="mx-auto mb-32 max-w-6xl px-4 sm:px-6 lg:px-8">
          <OverviewStory />
        </section>

        <section className="border-y border-sand-200 bg-white py-20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 gap-12 md:grid-cols-2 lg:grid-cols-4">
              {[
                {
                  icon: FileText,
                  title: "Resume Parsing",
                  desc: "Instantly extract skills and experience from complex PDFs.",
                },
                {
                  icon: Zap,
                  title: "Smart Scoring",
                  desc: "Rank candidates objectively against your specific job descriptions.",
                },
                {
                  icon: Users,
                  title: "Candidate Pools",
                  desc: "Organize talent into dynamic shortlists that evolve over time.",
                },
                {
                  icon: Sparkles,
                  title: "AI Interview Prep",
                  desc: "Generate tailored interview questions based on candidate profiles.",
                },
              ].map((value, index) => (
                <div key={index} className="space-y-4">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-forest-50 text-forest-700">
                    <value.icon className="h-6 w-6" />
                  </div>
                  <h3 className="font-serif text-xl text-forest-900">{value.title}</h3>
                  <p className="leading-relaxed text-forest-600">{value.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="space-y-32 py-32">
          <div className="mx-auto grid max-w-7xl items-center gap-16 px-4 sm:px-6 lg:grid-cols-2 lg:px-8">
            <div className="space-y-6">
              <Badge variant="neutral" className="bg-forest-100 text-forest-800">Scoring Engine</Badge>
              <h2 className="text-4xl font-serif leading-tight text-forest-900 md:text-5xl">
                Stop guessing. <br />
                <span className="italic text-accent-700">Start measuring.</span>
              </h2>
              <p className="text-lg leading-relaxed text-forest-600">
                Our flagship scoring engine breaks down candidates dimension by dimension.
                Adjust weights for Technical vs Behavioral skills on the fly and see your pool reshuffle instantly.
              </p>
              <ul className="space-y-3">
                {["Objective baseline for all applicants", "Explainable AI rationales", "Identify hidden gems instantly"].map((item, index) => (
                  <li key={index} className="flex items-center text-forest-700">
                    <CheckCircle2 className="mr-3 h-5 w-5 text-accent-600" /> {item}
                  </li>
                ))}
              </ul>
            </div>
            <SupportingSnapshot
              eyebrow="Scoring Engine"
              title="Turn evaluation rules into a readable scorecard."
              description="A static product snapshot shows how recruiters compare candidates and understand rank changes."
              testId="landing-story-scoring"
              variant="scoring"
            />
          </div>

          <div className="mx-auto grid max-w-7xl items-center gap-16 px-4 sm:px-6 lg:grid-cols-2 lg:px-8">
            <div className="order-2 lg:order-1">
              <SupportingSnapshot
                eyebrow="AI Assistant"
                title="Search the talent pool in natural language."
                description="The assistant surface reads like a recruiter query with direct candidate matches."
                testId="landing-story-assistant"
                variant="assistant"
              />
            </div>
            <div className="order-1 space-y-6 lg:order-2">
              <Badge variant="neutral" className="bg-forest-100 text-forest-800">AI Assistant</Badge>
              <h2 className="text-4xl font-serif leading-tight text-forest-900 md:text-5xl">
                Converse with your <br />
                <span className="italic text-accent-700">talent pool.</span>
              </h2>
              <p className="text-lg leading-relaxed text-forest-600">
                Ask natural questions like "Who has 5+ years of React and lives in Europe?"
                The AI recruiter searches, filters, and presents the best matches instantly.
              </p>
            </div>
          </div>
        </section>

        <section className="bg-forest-900 py-24 text-sand-50">
          <div className="mx-auto max-w-4xl space-y-12 px-4 text-center sm:px-6 lg:px-8">
            <div className="mb-16 grid grid-cols-2 gap-8 opacity-50 md:grid-cols-4">
              <div className="h-8 font-serif text-2xl italic">Acme Corp</div>
              <div className="h-8 font-sans text-2xl font-bold tracking-widest">NEXUS</div>
              <div className="h-8 font-serif text-2xl font-black">Lumina</div>
              <div className="h-8 font-mono text-2xl">SYS.IO</div>
            </div>
            <blockquote className="text-3xl font-serif font-light leading-snug md:text-4xl">
              "EasyHR completely transformed our hiring pipeline. What used to take our team two weeks of manual screening now happens flawlessly in an afternoon."
            </blockquote>
            <div className="flex items-center justify-center gap-4">
              <div className="h-12 w-12 overflow-hidden rounded-full bg-sand-200"></div>
              <div className="text-left">
                <div className="font-medium text-white">Sarah Jenkins</div>
                <div className="text-sm text-forest-300">VP of Talent, Nexus</div>
              </div>
            </div>
          </div>
        </section>

        <section className="px-4 py-32 text-center">
          <h2 className="mb-8 text-4xl font-serif text-forest-900 md:text-6xl">
            Ready to find your next great hire?
          </h2>
          <Link to="/login?mode=signup">
            <Button size="lg" className="h-16 px-10 text-lg">
              Get started for free
            </Button>
          </Link>
          <p className="mt-4 text-sm text-forest-500">No credit card required. 14-day free trial.</p>
        </section>
      </main>

      <footer className="border-t border-sand-200 bg-white py-12">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-6 px-4 sm:px-6 lg:flex-row lg:px-8">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-forest-900" />
            <span className="font-serif text-lg font-bold text-forest-900">EasyHR</span>
          </div>
          <div className="text-sm text-forest-500">
            © 2030 EasyHR Platform Inc. All rights reserved.
          </div>
          <div className="flex gap-6 text-sm text-forest-600">
            <a href="#" className="hover:text-forest-900">Privacy</a>
            <a href="#" className="hover:text-forest-900">Terms</a>
            <a href="#" className="hover:text-forest-900">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
}
