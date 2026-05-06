import { api } from "@/api";
import { Button } from "@/components/ui";
import { useAuthStore } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { ArrowRight, Sparkles } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router";
import { toast } from "sonner";

const OAUTH_ERROR_MESSAGES: Record<string, string> = {
  google_email_not_verified: "Your Google email is not verified. Please verify it and try again.",
  invalid_state: "Sign-in session expired. Please try again.",
  missing_params: "Sign-in was cancelled or failed. Please try again.",
  oauth_not_configured: "Google sign-in is not configured for this environment yet.",
  oauth_failed: "Google sign-in failed. Please try again.",
  access_denied: "Google sign-in was cancelled.",
};

export default function LoginRoute() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const isSignUp = searchParams.get("mode") === "signup";

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [isShake, setIsShake] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const error = searchParams.get("error");
    if (error) {
      toast.error(OAUTH_ERROR_MESSAGES[error] ?? "Sign-in failed. Please try again.");
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!email.includes("@")) {
      setIsShake(true);
      setTimeout(() => setIsShake(false), 500);
      return;
    }

    setIsLoading(true);
    try {
      const { access_token } = isSignUp
        ? await api.auth.register({ email, password, display_name: name })
        : await api.auth.login({ email, password });
      api.auth.storeToken(access_token);
      try {
        const user = await api.auth.me();
        useAuthStore.getState().setUser(user);
      } catch {
        // non-critical — store hydrated by router loader on next navigation
      }
      const redirect = searchParams.get("redirect") ?? "/dashboard";
      navigate(redirect, { replace: true });
    } catch {
      toast.error(isSignUp ? "Could not create account. Email may already be in use." : "Invalid email or password.");
      setIsShake(true);
      setTimeout(() => setIsShake(false), 500);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex selection:bg-accent-200">
      {/* Left Panel - Editorial (Hidden on mobile) */}
      <div className="hidden lg:flex w-[60%] bg-forest-900 text-sand-50 p-12 flex-col justify-between relative overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-forest-800 rounded-full blur-3xl -translate-y-1/2 translate-x-1/3 opacity-50" />
        <div className="absolute bottom-0 left-0 w-[600px] h-[600px] bg-accent-900 rounded-full blur-3xl translate-y-1/3 -translate-x-1/4 opacity-20" />

        <div className="relative z-10">
          <Link to="/" className="flex items-center gap-2 text-sand-50 hover:text-white transition-colors w-fit">
            <Sparkles className="w-6 h-6" />
            <span className="font-serif font-bold text-2xl">RecruitAI</span>
          </Link>
        </div>

        <div className="relative z-10 max-w-xl space-y-8">
          <h1 className="text-5xl font-serif leading-tight">
            The talent you've been searching for is already in your pipeline.
          </h1>
          <p className="text-xl text-forest-300 font-light leading-relaxed">
            Stop digging through resumes. Start having meaningful conversations with the right candidates.
          </p>
        </div>

        <div className="relative z-10 flex items-center gap-4">
          <div className="flex -space-x-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="w-12 h-12 rounded-full border-2 border-forest-900 bg-sand-200" />
            ))}
          </div>
          <div className="text-sm text-forest-300">
            Join <strong className="text-sand-50">2,000+</strong> recruiters building better teams.
          </div>
        </div>
      </div>

      {/* Right Panel - Form */}
      <div className="flex-1 flex flex-col justify-center px-8 sm:px-16 lg:px-24 bg-white relative">
        {/* Mobile Header */}
        <div className="lg:hidden absolute top-8 left-8 flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-forest-900" />
          <span className="font-serif font-bold text-xl text-forest-900">RecruitAI</span>
        </div>

        <div className="max-w-md w-full mx-auto space-y-8">
          <div>
            <h2 className="text-3xl font-serif text-forest-900 mb-2">
              {isSignUp ? "Create an account" : "Welcome back"}
            </h2>
            <p className="text-forest-600">
              {isSignUp
                ? "Start your 14-day free trial. No credit card required."
                : "Enter your details to sign in to your workspace."}
            </p>
          </div>

          {/* Google OAuth button */}
          <a
            href={api.auth.getGoogleLoginUrl(searchParams.get("redirect") ?? "/dashboard")}
            className="w-full h-12 flex items-center justify-center gap-3 border border-sand-200 rounded-xl hover:bg-sand-50 transition-colors text-sm font-medium text-forest-900"
          >
            <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
              <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
              <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
              <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
              <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
              <path fill="none" d="M0 0h48v48H0z"/>
            </svg>
            Sign in with Google
          </a>

          <div className="flex items-center gap-3 text-xs text-forest-400">
            <div className="flex-1 h-px bg-sand-200" />
            <span>OR</span>
            <div className="flex-1 h-px bg-sand-200" />
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {isSignUp && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-forest-900" htmlFor="name">
                  Full name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Jane Doe"
                  className="w-full px-4 py-3 bg-sand-50 border border-sand-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-accent-500 focus:border-transparent transition-all placeholder:text-sand-400 text-forest-900"
                  required
                />
              </div>
            )}
            
            <div className="space-y-2">
              <label className="text-sm font-medium text-forest-900" htmlFor="email">
                Email address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="jane@company.com"
                className={cn(
                  "w-full px-4 py-3 bg-sand-50 border border-sand-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-accent-500 focus:border-transparent transition-all placeholder:text-sand-400 text-forest-900",
                  isShake && "animate-shake border-red-500 focus:ring-red-500"
                )}
                required
              />
              {isShake && (
                <p className="text-xs text-red-500 mt-1">Please enter a valid email address.</p>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-forest-900" htmlFor="password">
                  Password
                </label>
                {!isSignUp && (
                  <a href="#" className="text-sm text-forest-500 hover:text-forest-900">
                    Forgot password?
                  </a>
                )}
              </div>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-4 py-3 bg-sand-50 border border-sand-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-accent-500 focus:border-transparent transition-all placeholder:text-sand-400 text-forest-900"
                required
              />
            </div>

            <Button type="submit" className="w-full h-12 text-base mt-2" disabled={isLoading}>
              {isLoading ? (isSignUp ? "Creating account…" : "Signing in…") : isSignUp ? "Create account" : "Sign in"}
              {!isLoading && <ArrowRight className="w-4 h-4 ml-2 opacity-50" />}
            </Button>
          </form>

          <div className="text-center text-sm text-forest-600 pt-4 border-t border-sand-100">
            {isSignUp ? (
              <>
                Already have an account?{" "}
                <Link to="/login" className="font-medium text-forest-900 hover:underline">
                  Sign in
                </Link>
              </>
            ) : (
              <>
                Don't have an account?{" "}
                <Link to="/login?mode=signup" className="font-medium text-forest-900 hover:underline">
                  Sign up
                </Link>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Shake animation styles */}
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes shake {
          0%, 100% { transform: translateX(0); }
          25% { transform: translateX(-5px); }
          75% { transform: translateX(5px); }
        }
        .animate-shake {
          animation: shake 0.2s ease-in-out 0s 2;
        }
      `}} />
    </div>
  );
}
