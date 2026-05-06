import { api } from "@/api";
import { useAuthStore } from "@/lib/auth";
import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { toast } from "sonner";

export default function AuthCallbackRoute() {
  const [sp] = useSearchParams();
  const navigate = useNavigate();

  useEffect(() => {
    const token = sp.get("token");
    const redirect = sp.get("redirect") ?? "/dashboard";
    const error = sp.get("error");

    if (error || !token) {
      toast.error("Google sign-in failed. Please try again.");
      navigate("/login", { replace: true });
      return;
    }

    api.auth.storeToken(token);
    api.auth
      .me()
      .then((user) => {
        useAuthStore.getState().setUser(user);
        navigate(redirect, { replace: true });
      })
      .catch(() => {
        api.auth.clearToken();
        toast.error("Could not load profile. Please try again.");
        navigate("/login", { replace: true });
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="min-h-screen flex items-center justify-center bg-white">
      <p className="text-forest-600 text-sm">Signing you in…</p>
    </div>
  );
}
