import { useEffect, useState } from "react";
import { Link } from "react-router";
import {
  Badge,
  Button,
} from "@/components/ui";
import {
  CheckCircle2,
  ChevronRight,
  FileText,
  MessageSquareText,
  SlidersHorizontal,
  Sparkles,
  Users,
  Zap,
} from "lucide-react";

type StoryStep = {
  id: string;
  title: string;
  detail: string;
  kicker: string;
};

const overviewSteps: StoryStep[] = [
  {
    id: "upload",
    title: "Upload CVs",
    detail: "Drag resumes into one hiring workspace and queue the batch instantly.",
    kicker: "Step 1",
  },
  {
    id: "parse",
    title: "AI parses profiles",
    detail: "Extract skills, years, and role signals without reading every file manually.",
    kicker: "Step 2",
  },
  {
    id: "shortlist",
    title: "Ranked shortlist",
    detail: "Surface the strongest matches and move the team into review mode quickly.",
    kicker: "Step 3",
  },
];

const scoringSteps: StoryStep[] = [
  {
    id: "weights",
    title: "Adjust score weights",
    detail: "Tune technical, domain, and communication fit to match the role.",
    kicker: "Scoring",
  },
  {
    id: "scores",
    title: "See rankings update",
    detail: "Candidate order responds immediately as the weighting changes.",
    kicker: "Scoring",
  },
  {
    id: "reasons",
    title: "Read the rationale",
    detail: "Review why the strongest applicants scored above the rest.",
    kicker: "Scoring",
  },
];

const assistantSteps: StoryStep[] = [
  {
    id: "ask",
    title: "Ask the candidate pool",
    detail: "Type a recruiter question in plain language instead of digging through filters.",
    kicker: "Assistant",
  },
  {
    id: "filter",
    title: "Activate the right filters",
    detail: "The assistant translates the request into location, skills, and seniority signals.",
    kicker: "Assistant",
  },
  {
    id: "match",
    title: "Get matched candidates",
    detail: "Return the strongest shortlist with a concise summary and next actions.",
    kicker: "Assistant",
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

function useAutoplayIndex(length: number, delayMs: number) {
  const prefersReducedMotion = usePrefersReducedMotion();
  const [index, setIndex] = useState(prefersReducedMotion ? length - 1 : 0);

  useEffect(() => {
    setIndex(prefersReducedMotion ? length - 1 : 0);
  }, [length, prefersReducedMotion]);

  useEffect(() => {
    if (prefersReducedMotion || length <= 1) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setIndex((currentIndex) => (currentIndex + 1) % length);
    }, delayMs);

    return () => window.clearTimeout(timeoutId);
  }, [delayMs, index, length, prefersReducedMotion]);

  return index;
}

function StoryStepRail({
  activeIndex,
  steps,
}: {
  activeIndex: number;
  steps: StoryStep[];
}) {
  return (
    <div className="grid gap-3">
      {steps.map((step, index) => {
        const isActive = index === activeIndex;

        return (
          <div
            key={step.id}
            className={`rounded-2xl border px-4 py-3 text-left transition-all duration-700 ${
              isActive
                ? "border-accent-400 bg-accent-50 shadow-lg shadow-accent-200/40 scale-[1.02]"
                : "border-sand-200 bg-white/70 opacity-75"
            }`}
          >
            <div className="flex items-center gap-3">
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-sm font-semibold transition-colors ${
                  isActive ? "bg-accent-500 text-white" : "bg-sand-100 text-forest-700"
                }`}
              >
                {index + 1}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-medium uppercase tracking-[0.2em] text-forest-400">
                  {step.kicker}
                </p>
                <p className="font-serif text-lg text-forest-900">{step.title}</p>
              </div>
            </div>
            <p className="mt-2 text-sm leading-6 text-forest-600">{step.detail}</p>
          </div>
        );
      })}
    </div>
  );
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
  const activeIndex = useAutoplayIndex(overviewSteps.length, 2600);

  return (
    <BrowserFrame>
      <div
        data-testid="landing-story-overview"
        className="grid aspect-[16/10] gap-6 bg-[radial-gradient(circle_at_top_left,_rgba(180,150,78,0.16),_transparent_30%),linear-gradient(180deg,_#fffdf7,_#f6f1e5)] p-5 md:grid-cols-[0.9fr_1.1fr] md:p-8"
      >
        <div className="flex flex-col justify-between gap-5">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-accent-700">
              Timeline Story
            </p>
            <h3 className="mt-3 font-serif text-3xl text-forest-900 md:text-4xl">
              From resume pile to shortlist.
            </h3>
            <p className="mt-3 max-w-md text-sm leading-6 text-forest-600 md:text-base">
              A compact walkthrough of how EasyHR turns raw CV uploads into a ranked review queue.
            </p>
          </div>
          <StoryStepRail activeIndex={activeIndex} steps={overviewSteps} />
        </div>

        <div className="relative overflow-hidden rounded-[28px] border border-sand-200 bg-white/80 p-4 md:p-5">
          <div className="grid grid-cols-3 gap-2">
            {overviewSteps.map((step, index) => {
              const isActive = index === activeIndex;

              return (
                <div key={step.id} className="space-y-2">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-700 ${
                      isActive ? "bg-accent-500" : "bg-sand-200"
                    }`}
                  />
                  <p className="text-xs font-medium text-forest-500">{step.title}</p>
                </div>
              );
            })}
          </div>

          <div className="relative mt-5 h-[280px] md:h-[320px]">
            <div
              aria-hidden={activeIndex !== 0}
              className={`absolute inset-0 rounded-[24px] border border-sand-200 bg-white p-5 transition-all duration-500 ${
                activeIndex === 0 ? "block scale-100 opacity-100" : "hidden"
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-forest-400">Candidate Intake</p>
                  <p className="mt-2 font-serif text-2xl text-forest-900">Upload CVs</p>
                </div>
                <div className="rounded-full bg-forest-50 px-3 py-1 text-xs font-medium text-forest-700">
                  24 files ready
                </div>
              </div>
              <div className="mt-5 grid gap-3">
                {["Frontend_Lead.pdf", "Growth_PM.pdf", "Data_Analyst.pdf"].map((fileName, index) => (
                  <div
                    key={fileName}
                    className={`flex items-center justify-between rounded-2xl border border-sand-200 bg-sand-50 px-4 py-3 transition-transform duration-700 ${
                      activeIndex === 0 && index === 1 ? "scale-[1.03] shadow-lg shadow-accent-100" : ""
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className="rounded-xl bg-white p-2 text-forest-700">
                        <FileText className="h-4 w-4" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-forest-900">{fileName}</p>
                        <p className="text-xs text-forest-500">Queued for parsing</p>
                      </div>
                    </div>
                    <div className="h-2 w-24 rounded-full bg-sand-200">
                      <div className="h-2 rounded-full bg-accent-500" style={{ width: `${60 + index * 10}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div
              aria-hidden={activeIndex !== 1}
              className={`absolute inset-0 rounded-[24px] border border-sand-200 bg-white p-5 transition-all duration-500 ${
                activeIndex === 1 ? "block scale-100 opacity-100" : "hidden"
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-forest-400">Profile Signals</p>
                  <p className="mt-2 font-serif text-2xl text-forest-900">AI parses profiles</p>
                </div>
                <div className="rounded-full bg-accent-50 px-3 py-1 text-xs font-medium text-accent-700">
                  18 profiles extracted
                </div>
              </div>
              <div className="mt-6 grid gap-4 md:grid-cols-[1fr_0.9fr]">
                <div className="rounded-[24px] border border-sand-200 bg-sand-50 p-4 shadow-lg shadow-accent-100/40 transition-transform duration-700 scale-[1.03]">
                  <p className="text-sm font-medium text-forest-900">Avery Chen</p>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {["React", "TypeScript", "Team Lead", "8 years"].map((tag) => (
                      <span key={tag} className="rounded-full bg-white px-3 py-1 text-xs text-forest-700">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="space-y-3">
                  {[
                    ["Skills matched", "12"],
                    ["Relevant years", "8.4"],
                    ["Leadership signals", "High"],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-2xl border border-sand-200 bg-white px-4 py-3">
                      <p className="text-xs uppercase tracking-[0.18em] text-forest-400">{label}</p>
                      <p className="mt-2 font-serif text-2xl text-forest-900">{value}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div
              aria-hidden={activeIndex !== 2}
              className={`absolute inset-0 rounded-[24px] border border-sand-200 bg-white p-5 transition-all duration-500 ${
                activeIndex === 2 ? "block scale-100 opacity-100" : "hidden"
              }`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-forest-400">Ready To Review</p>
                  <p className="mt-2 font-serif text-2xl text-forest-900">Ranked shortlist</p>
                </div>
                <div className="rounded-full bg-forest-900 px-3 py-1 text-xs font-medium text-white">
                  Shortlist locked
                </div>
              </div>
              <div className="mt-5 grid gap-3">
                {[
                  ["Avery Chen", "94", "Frontend leadership and hiring experience"],
                  ["Priya Raman", "91", "Strong product analytics and stakeholder fit"],
                  ["Jordan Lee", "88", "Reliable full-stack depth for scaling team"],
                ].map(([name, score, reason], index) => (
                  <div
                    key={name}
                    className={`rounded-[24px] border px-4 py-4 transition-all duration-700 ${
                      index === 0 ? "border-accent-300 bg-accent-50 scale-[1.03] shadow-xl shadow-accent-100/50" : "border-sand-200 bg-white"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="font-medium text-forest-900">{name}</p>
                        <p className="mt-1 text-sm text-forest-600">{reason}</p>
                      </div>
                      <div className="rounded-2xl bg-forest-900 px-3 py-2 text-lg font-semibold text-white">{score}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </BrowserFrame>
  );
}

function ShowcaseWalkthrough({
  activeIndex,
  steps,
  testId,
  type,
}: {
  activeIndex: number;
  steps: StoryStep[];
  testId: string;
  type: "assistant" | "scoring";
}) {
  const isScoring = type === "scoring";
  const activeStep = steps[activeIndex];

  return (
    <div
      data-testid={testId}
      className="relative aspect-square overflow-hidden rounded-3xl border border-sand-300 bg-[radial-gradient(circle_at_top_right,_rgba(27,55,39,0.08),_transparent_36%),linear-gradient(180deg,_#fffcf4,_#f4efe2)] p-4 md:p-5"
    >
      <div className="rounded-[28px] border border-white/70 bg-white/90 p-4 shadow-xl shadow-forest-900/10">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-forest-400">{isScoring ? "Live walkthrough" : "Conversational search"}</p>
            <h3 className="mt-2 font-serif text-2xl text-forest-900">{activeStep?.title}</h3>
          </div>
          <div className="rounded-full bg-forest-100 px-3 py-1 text-xs font-medium text-forest-700">
            {activeIndex + 1} / {steps.length}
          </div>
        </div>
        <p className="mt-3 text-sm leading-6 text-forest-600">{activeStep?.detail}</p>
      </div>

      <div className="mt-4 rounded-[26px] border border-sand-200 bg-white/85 p-4 shadow-lg shadow-forest-900/5 transition-all duration-700">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-accent-100 text-accent-700">
              {isScoring ? <SlidersHorizontal className="h-4 w-4" /> : <MessageSquareText className="h-4 w-4" />}
            </div>
            <div>
              <p className="font-medium text-forest-900">{activeStep?.title}</p>
              <p className="text-xs uppercase tracking-[0.18em] text-forest-400">{activeStep?.kicker}</p>
            </div>
          </div>
          <div className="h-2.5 w-20 rounded-full bg-accent-500"></div>
        </div>

        {isScoring ? (
          <div className="mt-4 grid gap-2">
            {activeIndex === 0 &&
              [
                ["Technical fit", "80%"],
                ["Domain context", "65%"],
                ["Communication", "55%"],
              ].map(([label, value], barIndex) => (
                <div key={label} className="grid grid-cols-[1fr_auto] items-center gap-3">
                  <p className="text-sm text-forest-600">{label}</p>
                  <p className="text-xs font-medium text-forest-500">{value}</p>
                  <div className="col-span-2 h-2 rounded-full bg-sand-200">
                    <div
                      className="h-2 rounded-full bg-forest-900 transition-all duration-700"
                      style={{ width: `${58 + barIndex * 14}%` }}
                    />
                  </div>
                </div>
              ))}
            {activeIndex === 1 &&
              [
                ["Avery Chen", "92"],
                ["Jordan Lee", "88"],
                ["Priya Raman", "84"],
              ].map(([name, score], rowIndex) => (
                <div
                  key={name}
                  className={`flex items-center justify-between rounded-2xl px-3 py-3 ${
                    rowIndex === 0 ? "bg-accent-50" : "bg-sand-50"
                  }`}
                >
                  <p className="text-sm font-medium text-forest-900">{name}</p>
                  <p className="rounded-xl bg-white px-2 py-1 text-sm font-semibold text-forest-800">{score}</p>
                </div>
              ))}
            {activeIndex === 2 && (
              <div className="rounded-[20px] bg-forest-900 p-4 text-sand-50">
                <p className="text-xs uppercase tracking-[0.18em] text-forest-200">Rationale</p>
                <p className="mt-3 text-sm leading-6">
                  Avery leads because frontend architecture depth and team leadership both clear the hiring threshold.
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="mt-4 grid gap-2">
            {activeIndex === 0 && (
              <div className="rounded-[20px] bg-sand-50 p-4">
                <p className="text-sm text-forest-600">Who has 5+ years of React and lives in Europe?</p>
              </div>
            )}
            {activeIndex === 1 && (
              <div className="flex flex-wrap gap-2">
                {["React", "Europe", "5+ years", "Open to remote"].map((chip) => (
                  <span key={chip} className="rounded-full bg-accent-50 px-3 py-1 text-xs font-medium text-accent-700">
                    {chip}
                  </span>
                ))}
              </div>
            )}
            {activeIndex === 2 &&
              [
                ["Avery Chen", "Frontend Lead"],
                ["Noah Martins", "Senior React Engineer"],
              ].map(([name, role]) => (
                <div key={name} className="flex items-center justify-between rounded-2xl bg-sand-50 px-3 py-3">
                  <div>
                    <p className="text-sm font-medium text-forest-900">{name}</p>
                    <p className="text-xs text-forest-500">{role}</p>
                  </div>
                  <div className="rounded-full bg-white px-3 py-1 text-xs font-medium text-forest-700">Match</div>
                </div>
              ))}
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2">
        {steps.map((step, index) => {
          const isActive = index === activeIndex;

          return (
            <div
              key={step.id}
              className={`rounded-2xl border px-3 py-3 transition-all duration-500 ${
                isActive ? "border-accent-300 bg-accent-50" : "border-sand-200 bg-white/70"
              }`}
            >
              <p className="text-[11px] uppercase tracking-[0.16em] text-forest-400">{step.kicker}</p>
              <p className="mt-2 text-sm font-medium leading-5 text-forest-900">{step.title}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function LandingRoute() {
  const scoringIndex = useAutoplayIndex(scoringSteps.length, 2400);
  const assistantIndex = useAutoplayIndex(assistantSteps.length, 2800);

  return (
    <div className="min-h-screen bg-sand-50 selection:bg-accent-200">
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
          <h1 className="mb-8 text-6xl font-serif leading-[1.1] tracking-tight text-forest-900 md:text-8xl">
            Hire like it's <span className="italic text-accent-600">2030.</span>
          </h1>
          <p className="mx-auto mb-10 max-w-3xl text-xl font-light leading-relaxed text-fg-muted md:text-2xl">
            The intelligent recruitment platform that turns overwhelming resume piles into
            curated shortlists, predictive scores, and automated outreach in minutes.
          </p>
          <div className="flex flex-col items-center justify-center gap-4 sm:flex-row">
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
            <ShowcaseWalkthrough
              activeIndex={scoringIndex}
              steps={scoringSteps}
              testId="landing-story-scoring"
              type="scoring"
            />
          </div>

          <div className="mx-auto grid max-w-7xl items-center gap-16 px-4 sm:px-6 lg:grid-cols-2 lg:px-8">
            <div className="order-2 lg:order-1">
              <ShowcaseWalkthrough
                activeIndex={assistantIndex}
                steps={assistantSteps}
                testId="landing-story-assistant"
                type="assistant"
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
