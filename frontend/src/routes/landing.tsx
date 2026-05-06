import { Link } from "react-router";
import { Button, Badge } from "@/components/ui";
import { Sparkles, Users, FileText, Zap, ChevronRight, CheckCircle2 } from "lucide-react";

export default function LandingRoute() {
  return (
    <div className="min-h-screen bg-sand-50 selection:bg-accent-200">
      {/* Navigation */}
      <nav className="border-b border-sand-200 bg-white/50 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-forest-900 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-accent-400" />
            </div>
            <span className="font-serif font-bold text-xl text-forest-900">
              RecruitAI
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
        {/* Hero Section */}
        <section className="pt-24 pb-32 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto text-center">
          <h1 className="text-6xl md:text-8xl font-serif text-forest-900 tracking-tight leading-[1.1] mb-8">
            Hire like it's <span className="italic text-accent-600">2030.</span>
          </h1>
          <p className="text-xl md:text-2xl text-fg-muted max-w-3xl mx-auto mb-10 font-sans font-light leading-relaxed">
            The intelligent recruitment platform that turns overwhelming resume piles into
            curated shortlists, predictive scores, and automated outreach in minutes.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/login?mode=signup">
              <Button size="lg" className="text-base px-8 h-14">
                Start your free trial <ChevronRight className="inline w-4 h-4 ml-2 align-middle" />
              </Button>
            </Link>
            <Button size="lg" variant="secondary" className="text-base px-8 h-14">
              Book a demo
            </Button>
          </div>
        </section>

        {/* Browser Frame Showcase */}
        <section className="px-4 sm:px-6 lg:px-8 max-w-6xl mx-auto mb-32">
          <div className="rounded-2xl border border-sand-200 bg-white shadow-2xl overflow-hidden shadow-forest-900/10">
            <div className="h-12 bg-sand-100 border-b border-sand-200 flex items-center px-4 gap-2">
              <div className="w-3 h-3 rounded-full bg-red-400"></div>
              <div className="w-3 h-3 rounded-full bg-amber-400"></div>
              <div className="w-3 h-3 rounded-full bg-green-400"></div>
              <div className="mx-auto bg-white border border-sand-200 text-xs text-forest-400 px-4 py-1 rounded-md w-64 text-center">
                app.recruitai.com
              </div>
            </div>
            <div className="aspect-[16/10] bg-sand-50 relative flex items-center justify-center overflow-hidden">
              {/* Abstract UI Representation */}
              <div className="absolute inset-8 bg-white shadow-sm border border-sand-200 rounded-xl flex">
                <div className="w-48 border-r border-sand-200 bg-sand-50 p-4 space-y-4">
                  <div className="h-4 w-24 bg-sand-200 rounded"></div>
                  <div className="h-4 w-32 bg-sand-200 rounded"></div>
                  <div className="h-4 w-20 bg-sand-200 rounded"></div>
                </div>
                <div className="flex-1 p-8 space-y-8">
                  <div className="h-8 w-64 bg-forest-100 rounded"></div>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="h-32 bg-sand-100 rounded-xl"></div>
                    <div className="h-32 bg-sand-100 rounded-xl"></div>
                    <div className="h-32 bg-sand-100 rounded-xl"></div>
                  </div>
                  <div className="h-64 bg-sand-100 rounded-xl"></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4-Column Value Strip */}
        <section className="border-y border-sand-200 bg-white py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
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
              ].map((val, i) => (
                <div key={i} className="space-y-4">
                  <div className="w-12 h-12 rounded-xl bg-forest-50 text-forest-700 flex items-center justify-center">
                    <val.icon className="w-6 h-6" />
                  </div>
                  <h3 className="font-serif text-xl text-forest-900">{val.title}</h3>
                  <p className="text-forest-600 leading-relaxed">{val.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Alternating Feature Deep-dives */}
        <section className="py-32 space-y-32">
          {/* Feature 1 */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid md:grid-cols-2 gap-16 items-center">
            <div className="space-y-6">
              <Badge variant="neutral" className="bg-forest-100 text-forest-800">Scoring Engine</Badge>
              <h2 className="text-4xl md:text-5xl font-serif text-forest-900 leading-tight">
                Stop guessing. <br />
                <span className="italic text-accent-700">Start measuring.</span>
              </h2>
              <p className="text-lg text-forest-600 leading-relaxed">
                Our flagship scoring engine breaks down candidates dimension by dimension.
                Adjust weights for Technical vs Behavioral skills on the fly and see your pool reshuffle instantly.
              </p>
              <ul className="space-y-3">
                {["Objective baseline for all applicants", "Explainable AI rationales", "Identify hidden gems instantly"].map((item, i) => (
                  <li key={i} className="flex items-center text-forest-700">
                    <CheckCircle2 className="w-5 h-5 mr-3 text-accent-600" /> {item}
                  </li>
                ))}
              </ul>
            </div>
            <div className="aspect-square bg-sand-200 rounded-3xl overflow-hidden border border-sand-300 relative">
              {/* Placeholder for actual product screenshot */}
              <div className="absolute inset-0 flex items-center justify-center text-sand-400 font-medium">
                [Scoring Interface Screenshot]
              </div>
            </div>
          </div>

          {/* Feature 2 */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid md:grid-cols-2 gap-16 items-center">
            <div className="aspect-square bg-sand-200 rounded-3xl overflow-hidden border border-sand-300 relative order-2 md:order-1">
              <div className="absolute inset-0 flex items-center justify-center text-sand-400 font-medium">
                [AI Chat Interface Screenshot]
              </div>
            </div>
            <div className="space-y-6 order-1 md:order-2">
              <Badge variant="neutral" className="bg-forest-100 text-forest-800">AI Assistant</Badge>
              <h2 className="text-4xl md:text-5xl font-serif text-forest-900 leading-tight">
                Converse with your <br />
                <span className="italic text-accent-700">talent pool.</span>
              </h2>
              <p className="text-lg text-forest-600 leading-relaxed">
                Ask natural questions like "Who has 5+ years of React and lives in Europe?"
                The AI recruiter searches, filters, and presents the best matches instantly.
              </p>
            </div>
          </div>
        </section>

        {/* Social Proof */}
        <section className="bg-forest-900 text-sand-50 py-24">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 opacity-50 mb-16">
              {/* Fake Logos */}
              <div className="h-8 font-serif italic text-2xl">Acme Corp</div>
              <div className="h-8 font-sans font-bold text-2xl tracking-widest">NEXUS</div>
              <div className="h-8 font-serif font-black text-2xl">Lumina</div>
              <div className="h-8 font-mono text-2xl">SYS.IO</div>
            </div>
            <blockquote className="text-3xl md:text-4xl font-serif font-light leading-snug">
              "RecruitAI completely transformed our hiring pipeline. What used to take our team two weeks of manual screening now happens flawlessly in an afternoon."
            </blockquote>
            <div className="flex items-center justify-center gap-4">
              <div className="w-12 h-12 bg-sand-200 rounded-full overflow-hidden"></div>
              <div className="text-left">
                <div className="font-medium text-white">Sarah Jenkins</div>
                <div className="text-forest-300 text-sm">VP of Talent, Nexus</div>
              </div>
            </div>
          </div>
        </section>

        {/* Final CTA */}
        <section className="py-32 text-center px-4">
          <h2 className="text-4xl md:text-6xl font-serif text-forest-900 mb-8">
            Ready to find your next great hire?
          </h2>
          <Link to="/login?mode=signup">
            <Button size="lg" className="h-16 px-10 text-lg">
              Get started for free
            </Button>
          </Link>
          <p className="mt-4 text-forest-500 text-sm">No credit card required. 14-day free trial.</p>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-sand-200 bg-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-forest-900" />
            <span className="font-serif font-bold text-lg text-forest-900">RecruitAI</span>
          </div>
          <div className="text-forest-500 text-sm">
            © 2030 RecruitAI Platform Inc. All rights reserved.
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
