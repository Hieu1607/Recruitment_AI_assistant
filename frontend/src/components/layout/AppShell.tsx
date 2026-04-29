import { Outlet } from "react-router";
import { TopBar } from "./TopBar";
import { Sidebar } from "./Sidebar";
import { CommandPalette } from "../CommandPalette";

export function AppShell() {
  return (
    <div className="flex h-full">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <TopBar />
        <div className="flex-1 overflow-y-auto">
          <div
            className="mx-auto w-full"
            style={{ maxWidth: "var(--content-max)" }}
          >
            <Outlet />
          </div>
        </div>
      </div>
      <CommandPalette />
    </div>
  );
}
