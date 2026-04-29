import { useState, useEffect } from "react";
import { useNavigate } from "react-router";
import { Search, FileText, User, LayoutDashboard, Command } from "lucide-react";


export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  if (!open) return null;

  const actions = [
    { id: "dashboard", name: "Go to Dashboard", icon: LayoutDashboard, path: "/" },
    { id: "candidates", name: "View Candidates", icon: User, path: "/candidates" },
    { id: "upload", name: "Upload Resumes", icon: FileText, path: "/candidates" },
    { id: "scoring", name: "Score Candidates", icon: Command, path: "/scoring/setup" },
  ];

  const filtered = query
    ? actions.filter((a) => a.name.toLowerCase().includes(query.toLowerCase()))
    : actions;

  const handleSelect = (path: string) => {
    setOpen(false);
    navigate(path);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]">
      <div
        className="fixed inset-0 bg-forest-900/40 backdrop-blur-sm"
        onClick={() => setOpen(false)}
      />
      <div className="relative w-full max-w-xl bg-white rounded-2xl shadow-2xl overflow-hidden border border-sand-200">
        <div className="flex items-center px-4 border-b border-sand-100">
          <Search className="w-5 h-5 text-forest-400" />
          <input
            autoFocus
            className="w-full bg-transparent px-4 py-4 text-forest-900 placeholder-forest-400 focus:outline-none"
            placeholder="Type a command or search..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <kbd className="hidden sm:inline-block bg-sand-100 border border-sand-200 rounded px-2 py-0.5 text-xs text-forest-500 font-mono">
            ESC
          </kbd>
        </div>
        <div className="max-h-[60vh] overflow-y-auto py-2">
          {filtered.length === 0 ? (
            <div className="p-8 text-center text-forest-500 text-sm">
              No results found.
            </div>
          ) : (
            <div className="px-2">
              <div className="px-3 py-2 text-xs font-medium text-forest-500">Suggestions</div>
              {filtered.map((action) => (
                <button
                  key={action.id}
                  onClick={() => handleSelect(action.path)}
                  className="w-full flex items-center gap-3 px-3 py-3 rounded-lg hover:bg-forest-50 text-forest-800 transition-colors text-left"
                >
                  <action.icon className="w-4 h-4 text-forest-400" />
                  <span className="text-sm font-medium">{action.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
