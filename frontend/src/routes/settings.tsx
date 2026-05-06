import { api } from "@/api";
import { Button } from "@/components/ui";
import { useAuthStore } from "@/lib/auth";
import { cn } from "@/lib/cn";
import { useMutation } from "@tanstack/react-query";
import { Bell, Briefcase, CheckCircle2, Key, ShieldAlert, User } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

const TABS = [
  { id: "profile", label: "Profile", icon: User },
  { id: "workspace", label: "Workspace", icon: Briefcase },
  { id: "apikeys", label: "API Keys", icon: Key },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "danger", label: "Danger Zone", icon: ShieldAlert },
];

const inputCn = cn(
  "w-full px-3 py-2 text-sm font-sans",
  "bg-bg border border-[color:var(--hairline-strong)] rounded-[var(--radius-md)]",
  "focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent outline-none",
  "text-fg"
);

const labelCn = "block text-xs font-medium text-fg-muted mb-1.5";

export default function SettingsRoute() {
  const [activeTab, setActiveTab] = useState("profile");
  const user = useAuthStore((s) => s.user);
  const setUser = useAuthStore((s) => s.setUser);
  const [displayName, setDisplayName] = useState(user?.display_name ?? "");
  const [email, setEmail] = useState(user?.email ?? "");

  const updateMutation = useMutation({
    mutationFn: () => api.auth.updateProfile({ display_name: displayName, email }),
    onSuccess: (updated) => {
      setUser(updated);
      toast.success("Settings saved successfully.");
    },
    onError: () => {
      toast.error("Failed to save settings.");
    },
  });

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    updateMutation.mutate();
  };

  return (
    <div className="flex-1 overflow-auto px-8 py-8 min-h-full">
      <div className="max-w-5xl mx-auto space-y-8">
        <div>
          <h1 className="font-display text-[2rem] font-medium text-fg leading-tight">Settings</h1>
          <p className="text-sm text-fg-muted mt-1 font-sans">
            Manage your account and workspace preferences
          </p>
        </div>

        <div className="flex flex-col md:flex-row gap-8 items-start">
          {/* Sidebar */}
          <div className="w-full md:w-64 flex-shrink-0 rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated overflow-hidden">
            <nav className="p-2 space-y-1">
              {TABS.map((tab) => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={cn(
                      "w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors",
                      isActive
                        ? "bg-[color:var(--hairline)] text-fg"
                        : "text-fg-muted hover:bg-[color:var(--hairline)] hover:text-fg",
                      tab.id === "danger" && isActive && "bg-red-50 text-red-700",
                      tab.id === "danger" && !isActive && "hover:bg-red-50 hover:text-red-600"
                    )}
                  >
                    <tab.icon className={cn("w-4 h-4", isActive ? "opacity-100" : "opacity-70")} />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>

          {/* Content Area */}
          <div className="flex-1 rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated overflow-hidden">
            {activeTab === "profile" && (
              <form onSubmit={handleSave} className="p-8 space-y-6">
                <h2 className="font-display text-lg font-medium text-fg border-b border-[color:var(--hairline)] pb-4">
                  Profile Information
                </h2>
                <div className="space-y-4 max-w-md">
                  <div>
                    <label className={labelCn}>Full name</label>
                    <input type="text" value={displayName} onChange={(e) => setDisplayName(e.target.value)} className={inputCn} />
                  </div>
                  <div>
                    <label className={labelCn}>Email address</label>
                    <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className={inputCn} />
                  </div>
                  <div>
                    <label className={labelCn}>Timezone</label>
                    <select className={inputCn}>
                      <option>Pacific Time (US &amp; Canada)</option>
                      <option>Eastern Time (US &amp; Canada)</option>
                      <option>UTC</option>
                    </select>
                  </div>
                </div>
                <div className="pt-4">
                  <Button type="submit" disabled={updateMutation.isPending}>
                    {updateMutation.isPending ? "Saving…" : "Save changes"}
                  </Button>
                </div>
              </form>
            )}

            {activeTab === "workspace" && (
              <div className="p-8 space-y-6">
                <h2 className="font-display text-lg font-medium text-fg border-b border-[color:var(--hairline)] pb-4">
                  Workspace Settings
                </h2>
                <p className="text-fg-muted text-sm">
                  Configure defaults for your recruitment team.
                </p>
                <div className="h-32 bg-bg border border-[color:var(--hairline)] rounded-[var(--radius-md)] flex items-center justify-center text-fg-muted text-sm">
                  Workspace configuration options
                </div>
              </div>
            )}

            {activeTab === "apikeys" && (
              <div className="p-8 space-y-6">
                <h2 className="font-display text-lg font-medium text-fg border-b border-[color:var(--hairline)] pb-4">
                  API Keys
                </h2>
                <p className="text-fg-muted text-sm">
                  Manage keys for programmatic access to the RecruitAI API.
                </p>
                <div className="border border-[color:var(--hairline)] rounded-[var(--radius-md)] p-4 flex items-center justify-between">
                  <div>
                    <div className="font-medium text-fg text-sm">Production Key</div>
                    <div className="font-mono text-sm text-fg-muted mt-1">sk_prod_••••••••••••••••</div>
                  </div>
                  <Button variant="secondary" size="sm">Reveal</Button>
                </div>
                <Button variant="ghost" className="mt-2 text-accent">
                  <PlusIcon /> Generate new key
                </Button>
              </div>
            )}

            {activeTab === "notifications" && (
              <div className="p-8 space-y-6">
                <h2 className="font-display text-lg font-medium text-fg border-b border-[color:var(--hairline)] pb-4">
                  Notification Preferences
                </h2>
                <div className="space-y-4">
                  {[
                    "When a batch upload finishes processing",
                    "When a candidate score is above 80",
                    "When an outreach email bounces",
                  ].map((label, i) => (
                    <label key={i} className="flex items-center gap-3 cursor-pointer">
                      <div className="relative flex items-center">
                        <input type="checkbox" defaultChecked className="peer sr-only" />
                        <div className="w-5 h-5 border border-[color:var(--hairline-strong)] rounded bg-bg peer-checked:bg-accent peer-checked:border-accent transition-colors"></div>
                        <CheckCircle2 className="w-3.5 h-3.5 text-white absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 peer-checked:opacity-100 transition-opacity" />
                      </div>
                      <span className="text-fg text-sm">{label}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            {activeTab === "danger" && (
              <div className="p-8 space-y-6">
                <h2 className="font-display text-lg font-medium text-red-600 border-b border-red-100 pb-4">
                  Danger Zone
                </h2>
                <div className="space-y-4">
                  <div className="border border-red-200 rounded-xl p-6 bg-red-50 flex items-center justify-between">
                    <div>
                      <h3 className="font-medium text-red-900">Delete Account</h3>
                      <p className="text-sm text-red-700 mt-1">
                        Permanently delete your account and all data. This action cannot be undone.
                      </p>
                    </div>
                    <Button variant="danger">Delete account</Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function PlusIcon() {
  return (
    <svg className="w-4 h-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
    </svg>
  );
}
