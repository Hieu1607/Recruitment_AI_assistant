import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "sonner";
import { router } from "./router";
import { queryClient } from "./api";
import { ThemeProvider } from "./components/ThemeProvider";
import "./styles/globals.css";

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("Root element #root not found");

createRoot(rootEl).render(
  <StrictMode>
    <ThemeProvider>
      <QueryClientProvider client={queryClient}>
        <RouterProvider router={router} />
        <Toaster position="top-right" duration={4000} closeButton richColors={false} />
      </QueryClientProvider>
    </ThemeProvider>
  </StrictMode>
);
