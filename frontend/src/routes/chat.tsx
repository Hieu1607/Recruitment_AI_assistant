import { RoutePlaceholder } from "@/components/RoutePlaceholder";

export default function ChatRoute() {
  return (
    <RoutePlaceholder
      screen="AI Chat"
      description="Recruiter chatbot — sessions sidebar, prose-style AI messages, inline candidate cards for scoped results, auto-recover on expired sessions."
      phase="Phase 6"
      requirements={[
        "CHAT-01", "CHAT-02", "CHAT-03", "CHAT-04", "CHAT-05",
        "CHAT-06", "CHAT-07", "CHAT-08", "CHAT-09", "CHAT-10",
        "CHAT-11", "CHAT-12",
      ]}
    />
  );
}
