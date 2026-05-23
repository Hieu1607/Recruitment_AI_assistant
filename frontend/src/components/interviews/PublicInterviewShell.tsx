import { api, type PublicInterviewStartResponse } from "@/api";
import { Badge, Button, EmptyState } from "@/components/ui";
import { cn } from "@/lib/cn";
import { useMutation } from "@tanstack/react-query";
import { Mic, MicOff, Play, Send, Volume2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

type InterviewQuestion = {
  key: string;
  prompt: string;
  maxDurationSec: number | null;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionEventLike = {
  resultIndex: number;
  results: ArrayLike<{
    isFinal: boolean;
    0: { transcript: string };
  }>;
};

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}

function extractQuestions(payload: Record<string, unknown>): InterviewQuestion[] {
  const rawQuestions = Array.isArray(payload.questions) ? payload.questions : [];
  return rawQuestions
    .map((item, index) => {
      if (!item || typeof item !== "object") return null;
      const candidate = item as Record<string, unknown>;
      const prompt = typeof candidate.prompt === "string"
        ? candidate.prompt
        : typeof candidate.text === "string"
          ? candidate.text
          : "";
      if (!prompt.trim()) return null;
      return {
        key:
          typeof candidate.key === "string" && candidate.key.trim()
            ? candidate.key.trim()
            : `question_${index + 1}`,
        prompt: prompt.trim(),
        maxDurationSec:
          typeof candidate.max_duration_sec === "number" ? candidate.max_duration_sec : null,
      } satisfies InterviewQuestion;
    })
    .filter((item): item is InterviewQuestion => item !== null);
}

function buildProviderSessionId() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `browser-${Date.now()}`;
}

function formatAttempts(started: PublicInterviewStartResponse | null) {
  if (!started) return null;
  return `${started.invitation.attempt_count}/${started.invitation.max_attempts}`;
}

export function PublicInterviewShell({ token }: { token: string }) {
  const [started, setStarted] = useState<PublicInterviewStartResponse | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answerDraft, setAnswerDraft] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);

  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);

  const questions = useMemo(
    () => (started ? extractQuestions(started.template.question_payload) : []),
    [started],
  );
  const currentQuestion = questions[currentQuestionIndex] ?? null;
  const supportsSpeechRecognition = typeof window !== "undefined" && !!(window.SpeechRecognition || window.webkitSpeechRecognition);

  const startMutation = useMutation({
    mutationFn: async () =>
      api.interviewPublic.start(token, {
        provider: "fake",
        provider_session_id: buildProviderSessionId(),
        browser_metadata: {
          user_agent: navigator.userAgent,
          language: navigator.language,
        },
      }),
    onSuccess: async (response) => {
      setStarted(response);
      setCurrentQuestionIndex(0);
      setAnswerDraft("");
      setErrorMessage(null);
      await promptCurrentQuestion(response, 0, true);
    },
    onError: (error: Error) => setErrorMessage(error.message || "Unable to start interview."),
  });

  const eventsMutation = useMutation({
    mutationFn: (events: Array<{ speaker: string; text: string; question_key?: string | null }>) =>
      api.interviewPublic.ingestEvents(token, {
        provider: "fake",
        events: events.map((event) => ({
          speaker: event.speaker,
          text: event.text,
          question_key: event.question_key ?? null,
        })),
      }),
    onError: (error: Error) => setErrorMessage(error.message || "Unable to save interview transcript."),
  });

  const completeMutation = useMutation({
    mutationFn: async () => api.interviewPublic.complete(token, { provider: "fake" }),
    onSuccess: () => {
      window.speechSynthesis?.cancel();
      setIsListening(false);
      setIsCompleted(true);
    },
    onError: (error: Error) => setErrorMessage(error.message || "Unable to complete interview."),
  });

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop();
      window.speechSynthesis?.cancel();
    };
  }, []);

  async function speak(text: string) {
    const candidate = text.trim();
    if (!candidate || typeof window === "undefined" || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(candidate);
    utterance.lang = started?.template.language_code || "en-US";
    window.speechSynthesis.speak(utterance);
  }

  async function promptCurrentQuestion(
    response: PublicInterviewStartResponse,
    questionIndex: number,
    includeIntro: boolean,
  ) {
    const availableQuestions = extractQuestions(response.template.question_payload);
    const question = availableQuestions[questionIndex];
    if (!question) return;

    const transcriptEvents: Array<{ speaker: string; text: string; question_key?: string | null }> = [];
    const spokenSegments: string[] = [];

    if (includeIntro && response.template.intro_script?.trim()) {
      transcriptEvents.push({ speaker: "agent", text: response.template.intro_script.trim() });
      spokenSegments.push(response.template.intro_script.trim());
    }

    transcriptEvents.push({
      speaker: "agent",
      text: question.prompt,
      question_key: question.key,
    });
    spokenSegments.push(question.prompt);

    await eventsMutation.mutateAsync(transcriptEvents);
    await speak(spokenSegments.join(" "));
  }

  async function advanceInterview() {
    if (!started || !currentQuestion) return;

    const trimmedAnswer = answerDraft.trim();
    if (trimmedAnswer) {
      await eventsMutation.mutateAsync([
        {
          speaker: "user",
          text: trimmedAnswer,
          question_key: currentQuestion.key,
        },
      ]);
    }

    setAnswerDraft("");
    recognitionRef.current?.stop();
    setIsListening(false);

    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < questions.length) {
      setCurrentQuestionIndex(nextQuestionIndex);
      await promptCurrentQuestion(started, nextQuestionIndex, false);
      return;
    }

    if (started.template.closing_script?.trim()) {
      await eventsMutation.mutateAsync([
        {
          speaker: "agent",
          text: started.template.closing_script.trim(),
        },
      ]);
      await speak(started.template.closing_script.trim());
    }

    await completeMutation.mutateAsync();
  }

  function startListening() {
    if (!supportsSpeechRecognition || !started) return;

    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return;

    setErrorMessage(null);
    const recognition = new Recognition();
    recognition.lang = started.template.language_code || "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onresult = (event) => {
      let transcript = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        transcript += event.results[index][0].transcript;
      }
      setAnswerDraft(transcript.trim());
    };
    recognition.onerror = (event) => {
      setErrorMessage(`Microphone transcription error: ${event.error}`);
      setIsListening(false);
    };
    recognition.onend = () => {
      setIsListening(false);
    };
    recognition.start();
    recognitionRef.current = recognition;
    setIsListening(true);
  }

  function stopListening() {
    recognitionRef.current?.stop();
    setIsListening(false);
  }

  if (!token) {
    return (
      <div className="min-h-screen bg-bg px-6 py-10">
        <div className="mx-auto max-w-4xl">
          <EmptyState heading="Interview link is invalid" body="The public interview token is missing." />
        </div>
      </div>
    );
  }

  const attemptText = formatAttempts(started);

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(228,209,167,0.26),_transparent_42%),linear-gradient(180deg,_var(--color-bg-sidebar),_var(--color-bg))] px-6 py-10">
      <div className="mx-auto max-w-4xl space-y-6">
        <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg-elevated p-6 shadow-[0_18px_48px_rgba(15,23,18,0.08)]">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-xs font-medium uppercase tracking-[0.22em] text-fg-muted">Public AI Interview</p>
              <h1 className="mt-2 font-display text-[2.4rem] leading-tight text-fg">
                {started?.template.name ?? "Structured Screening Interview"}
              </h1>
              <p className="mt-3 max-w-2xl text-sm leading-relaxed text-fg-muted">
                This interview uses a fixed recruiter-approved question set. Your responses are transcribed and shared
                with the hiring team for review.
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              {started?.invitation.status && <Badge variant="warning">{started.invitation.status}</Badge>}
              {attemptText && <Badge variant="neutral">Attempt {attemptText}</Badge>}
            </div>
          </div>
        </section>

        {!started ? (
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-6">
            <h2 className="font-display text-xl font-medium text-fg">Before you begin</h2>
            <ul className="mt-4 space-y-2 text-sm leading-relaxed text-fg-muted">
              <li>The interviewer will ask only the questions configured for this role.</li>
              <li>You can answer by voice if your browser supports speech recognition, or type into the fallback box.</li>
              <li>The interview can be attempted only within the invitation limits set by the recruiter.</li>
            </ul>
            {errorMessage && (
              <p className="mt-4 rounded-[var(--radius-md)] border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
                {errorMessage}
              </p>
            )}
            <div className="mt-6 flex flex-wrap gap-3">
              <Button
                onClick={() => startMutation.mutate()}
                loading={startMutation.isPending}
                icon={<Play size={15} strokeWidth={1.9} />}
              >
                Start interview
              </Button>
            </div>
          </section>
        ) : isCompleted ? (
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-6">
            <h2 className="font-display text-2xl font-medium text-fg">Interview completed</h2>
            <p className="mt-3 max-w-2xl text-sm leading-relaxed text-fg-muted">
              Your responses have been submitted. The recruiter can now review the transcript and structured summary.
            </p>
          </section>
        ) : currentQuestion ? (
          <section className="grid gap-6 lg:grid-cols-[1.3fr_0.9fr]">
            <div className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-6">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-xs font-medium uppercase tracking-[0.2em] text-fg-muted">
                    Question {currentQuestionIndex + 1} of {questions.length}
                  </p>
                  <h2 className="mt-2 font-display text-2xl font-medium text-fg">{currentQuestion.prompt}</h2>
                </div>
                {currentQuestion.maxDurationSec ? (
                  <Badge variant="neutral">Target {currentQuestion.maxDurationSec}s</Badge>
                ) : null}
              </div>

              <div className="mt-6 rounded-[var(--radius-lg)] border border-[rgba(74,124,89,0.18)] bg-bg-elevated p-4">
                <div className="flex items-center gap-2 text-sm text-fg-muted">
                  <Volume2 size={15} strokeWidth={1.75} />
                  Agent prompt is spoken when the question loads.
                </div>
                <label className="mt-4 block space-y-2">
                  <span className="text-xs font-medium uppercase tracking-wide text-fg-muted">Answer transcript</span>
                  <textarea
                    aria-label="Answer transcript"
                    value={answerDraft}
                    onChange={(event) => setAnswerDraft(event.target.value)}
                    rows={10}
                    placeholder="Speak or type your answer here."
                    className={cn(
                      "w-full rounded-[var(--radius-md)] border border-[color:var(--hairline-strong)] bg-bg px-3 py-2.5",
                      "text-sm leading-relaxed text-fg outline-none focus:outline focus:outline-2 focus:outline-offset-1 focus:outline-accent",
                    )}
                  />
                </label>
              </div>

              {errorMessage && (
                <p className="mt-4 rounded-[var(--radius-md)] border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
                  {errorMessage}
                </p>
              )}

              <div className="mt-6 flex flex-wrap gap-3">
                {supportsSpeechRecognition ? (
                  isListening ? (
                    <Button variant="secondary" onClick={stopListening} icon={<MicOff size={15} strokeWidth={1.75} />}>
                      Stop listening
                    </Button>
                  ) : (
                    <Button variant="secondary" onClick={startListening} icon={<Mic size={15} strokeWidth={1.75} />}>
                      Start listening
                    </Button>
                  )
                ) : (
                  <Badge variant="neutral">Speech recognition unavailable, using text fallback</Badge>
                )}

                <Button
                  onClick={() => void advanceInterview()}
                  loading={eventsMutation.isPending || completeMutation.isPending}
                  icon={<Send size={15} strokeWidth={1.75} />}
                >
                  {currentQuestionIndex === questions.length - 1 ? "Finish interview" : "Next question"}
                </Button>
              </div>
            </div>

            <aside className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-6">
              <h3 className="font-display text-xl font-medium text-fg">Interview Notes</h3>
              <dl className="mt-4 space-y-3 text-sm">
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-fg-muted">Candidate</dt>
                  <dd className="text-right text-fg">{started?.invitation.candidate_full_name || "Candidate"}</dd>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-fg-muted">Language</dt>
                  <dd className="text-right text-fg">{started?.template.language_code || "n/a"}</dd>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-fg-muted">Session</dt>
                  <dd className="max-w-[12rem] break-all text-right text-fg">{started?.session.id}</dd>
                </div>
              </dl>
            </aside>
          </section>
        ) : (
          <section className="rounded-[var(--radius-lg)] border border-[color:var(--hairline)] bg-bg p-6">
            <EmptyState
              heading="Interview questions unavailable"
              body="The recruiter invitation does not contain a usable question set yet."
            />
          </section>
        )}
      </div>
    </div>
  );
}
