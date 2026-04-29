import { useState } from "react";
import { User, Briefcase, Key, Bell, ShieldAlert, CheckCircle2 } from "lucide-react";
import { Button } from "@/components/ui";
import { toast } from "sonner";
import { cn } from "@/lib/cn";

const TABS = [
  { id: "profile", label: "Profile", icon: User },
  { id: "workspace", label: "Workspace", icon: Briefcase },
  { id: "apikeys", label: "API Keys", icon: Key },
  { id: "notifications", label: "Notifications", icon: Bell },
  { id: "danger", label: "Danger Zone", icon: ShieldAlert },
];

export default function SettingsRoute() {
  const [activeTab, setActiveTab] = useState("profile");

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    toast.success("Settings saved successfully.");
  };

  return (
    <div className="flex-1 overflow-auto bg-sand-50 p-8">
      <div className="max-w-5xl mx-auto space-y-8">
        <div>
          <h1 className="text-3xl font-serif text-forest-900">Settings</h1>
          <p className="text-forest-600 mt-1">Manage your account and workspace preferences.</p>
        </div>

        <div className="flex flex-col md:flex-row gap-8 items-start">
          {/* Sidebar */}
          <div className="w-full md:w-64 flex-shrink-0 bg-white rounded-2xl shadow-sm border border-sand-200 overflow-hidden">
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
                        ? "bg-forest-50 text-forest-900"
                        : "text-forest-600 hover:bg-sand-100 hover:text-forest-900",
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
          <div className="flex-1 bg-white rounded-2xl shadow-sm border border-sand-200 overflow-hidden">
            {activeTab === "profile" && (
              <form onSubmit={handleSave} className="p-8 space-y-6">
                <h2 className="text-xl font-serif text-forest-900 border-b border-sand-200 pb-4">
                  Profile Information
                </h2>
                <div className="space-y-4 max-w-md">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-forest-900 block">Full name</label>
                    <input
                      type="text"
                      defaultValue="Hieu"
                      className="w-full px-3 py-2 bg-sand-50 border border-sand-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-500 text-forest-900"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-forest-900 block">Email address</label>
                    <input
                      type="email"
                      defaultValue="hieu@recruitai.com"
                      className="w-full px-3 py-2 bg-sand-50 border border-sand-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-500 text-forest-900"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-forest-900 block">Timezone</label>
                    <select className="w-full px-3 py-2 bg-sand-50 border border-sand-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-accent-500 text-forest-900">
                      <option>Pacific Time (US & Canada)</option>
                      <option>Eastern Time (US & Canada)</option>
                      <option>UTC</option>
                    </select>
                  </div>
                </div>
                <div className="pt-4">
                  <Button type="submit">Save changes</Button>
                </div>
              </form>
            )}

            {activeTab === "workspace" && (
              <div className="p-8 space-y-6">
                <h2 className="text-xl font-serif text-forest-900 border-b border-sand-200 pb-4">
                  Workspace Settings
                </h2>
                <p className="text-forest-600 text-sm">
                  Configure defaults for your recruitment team.
                </p>
                {/* Mock form UI */}
                <div className="h-32 bg-sand-50 border border-sand-200 rounded-xl flex items-center justify-center text-forest-400">
                  Workspace configuration options
                </div>
              </div>
            )}

            {activeTab === "apikeys" && (
              <div className="p-8 space-y-6">
                <h2 className="text-xl font-serif text-forest-900 border-b border-sand-200 pb-4">
                  API Keys
                </h2>
                <p className="text-forest-600 text-sm">
                  Manage keys for programmatic access to the RecruitAI API.
                </p>
                <div className="border border-sand-200 rounded-xl p-4 flex items-center justify-between">
                  <div>
                    <div className="font-medium text-forest-900">Production Key</div>
                    <div className="font-mono text-sm text-forest-500 mt-1">sk_prod_••••••••••••••••</div>
                  </div>
                  <Button variant="secondary" size="sm">Reveal</Button>
                </div>
                <Button variant="ghost" className="mt-2 text-accent-600"><PlusIcon /> Generate new key</Button>
              </div>
            )}

            {activeTab === "notifications" && (
              <div className="p-8 space-y-6">
                <h2 className="text-xl font-serif text-forest-900 border-b border-sand-200 pb-4">
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
                        <div className="w-5 h-5 border border-sand-300 rounded bg-sand-50 peer-checked:bg-forest-900 peer-checked:border-forest-900 transition-colors"></div>
                        <CheckCircle2 className="w-3.5 h-3.5 text-white absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 peer-checked:opacity-100 transition-opacity" />
                      </div>
                      <span className="text-forest-800 text-sm">{label}</span>
                    </label>
                  ))}
                </div>
              </div>
            )}

            {activeTab === "danger" && (
              <div className="p-8 space-y-6">
                <h2 className="text-xl font-serif text-red-600 border-b border-red-100 pb-4">
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
