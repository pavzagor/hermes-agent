import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePartPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  useLocalRuntime,
  type ChatModelAdapter,
  type ThreadMessage,
} from "@assistant-ui/react";
import { AlertCircle, Bot, Send, ShieldCheck, Sparkles, User } from "lucide-react";
import { useMemo, type FC } from "react";

import { Typography } from "@/components/NouiTypography";
import { HERMES_BASE_PATH } from "@/lib/api";
import { cn } from "@/lib/utils";

function messageText(message: ThreadMessage): string {
  return message.content
    .map((part) => {
      if (part.type === "text") return part.text;
      if (part.type === "image") return `[Image: ${part.filename ?? "attached image"}]`;
      if (part.type === "file") return `[File: ${part.filename ?? part.mimeType}]`;
      if (part.type === "data") return `[Data: ${part.name}]`;
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

const hermesAdapter: ChatModelAdapter = {
  async run({ messages, abortSignal, unstable_threadId }) {
    const response = await fetch(`${HERMES_BASE_PATH}/api/assistant-chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Hermes-Session-Token": window.__HERMES_SESSION_TOKEN__ ?? "",
      },
      signal: abortSignal,
      body: JSON.stringify({
        thread_id: unstable_threadId,
        messages: messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: messageText(message),
        })),
      }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(`Hermes chat failed (${response.status}): ${text}`);
    }

    const data = (await response.json()) as { text?: string };
    return {
      content: [{ type: "text", text: data.text ?? "" }],
      status: { type: "complete", reason: "stop" },
    };
  },
};

export default function AssistantPage() {
  const runtime = useLocalRuntime(hermesAdapter);
  const hints = useMemo(
    () => [
      "Ask Hermes to inspect files, run commands, or research something.",
      "Access is gated by the dashboard session token and optional Cloudflare Access email allowlist.",
      "The Hermes API key never leaves the server-side dashboard proxy.",
    ],
    [],
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-full min-h-0 flex-col gap-4 p-4 lg:p-6">
        <header className="shrink-0 rounded-xl border border-current/15 bg-background-base/70 p-4 backdrop-blur">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-midground">
                <Sparkles className="h-5 w-5" />
                <Typography className="text-lg font-bold tracking-[0.08em]">
                  Assistant UI Chat
                </Typography>
              </div>
              <p className="max-w-3xl font-sans text-sm normal-case leading-6 text-midground/75">
                A private, browser-native chat surface powered by assistant-ui and the
                Hermes API server. Use this when you want a clean web chat instead of
                the embedded terminal.
              </p>
            </div>
            <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1.5 font-sans text-xs normal-case text-emerald-200">
              <ShieldCheck className="h-4 w-4" />
              Server-side auth proxy
            </div>
          </div>
          <ul className="mt-4 grid gap-2 font-sans text-xs normal-case text-midground/60 lg:grid-cols-3">
            {hints.map((hint) => (
              <li key={hint} className="rounded-lg border border-current/10 bg-black/15 p-3">
                {hint}
              </li>
            ))}
          </ul>
        </header>

        <section className="min-h-0 flex-1 overflow-hidden rounded-xl border border-current/15 bg-background-base/80 shadow-2xl shadow-black/25 backdrop-blur">
          <MinimalThread />
        </section>
      </div>
    </AssistantRuntimeProvider>
  );
}

const MinimalThread: FC = () => {
  return (
    <ThreadPrimitive.Root className="flex h-full min-h-0 flex-col bg-[#071f1f]/75 font-sans normal-case text-midground">
      <ThreadPrimitive.Viewport className="flex min-h-0 flex-1 flex-col overflow-y-auto scroll-smooth">
        <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-4 px-4 py-6">
          <ThreadPrimitive.Empty>
            <div className="mx-auto mt-16 flex max-w-xl flex-col items-center gap-4 text-center">
              <div className="rounded-full border border-current/15 bg-black/20 p-4">
                <Bot className="h-8 w-8 text-midground/80" />
              </div>
              <div>
                <Typography className="text-xl font-bold tracking-[0.06em]">
                  Chat with Hermes
                </Typography>
                <p className="mt-2 text-sm leading-6 text-midground/60">
                  Messages route through the dashboard backend to your localhost-only
                  Hermes API server.
                </p>
              </div>
            </div>
          </ThreadPrimitive.Empty>

          <div className="flex flex-col gap-4 empty:hidden">
            <ThreadPrimitive.Messages components={{ Message: ChatMessage }} />
          </div>

          <ThreadPrimitive.ViewportFooter className="sticky bottom-0 mt-auto bg-gradient-to-t from-[#071f1f] via-[#071f1f]/95 to-transparent pb-4 pt-8">
            <Composer />
          </ThreadPrimitive.ViewportFooter>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const ChatMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="group flex gap-3 data-[role=assistant]:justify-start data-[role=user]:justify-end">
      <MessagePrimitive.If user>
        <div className="order-2 mt-1 rounded-full border border-current/15 bg-midground/10 p-2 text-midground/70">
          <User className="h-4 w-4" />
        </div>
      </MessagePrimitive.If>
      <MessagePrimitive.If assistant>
        <div className="mt-1 rounded-full border border-current/15 bg-emerald-400/10 p-2 text-emerald-200/80">
          <Bot className="h-4 w-4" />
        </div>
      </MessagePrimitive.If>
      <div
        className={cn(
          "max-w-[min(44rem,calc(100%-3rem))] rounded-2xl border px-4 py-3 text-sm leading-6 shadow-sm",
          "data-[role=assistant]:border-emerald-400/15 data-[role=assistant]:bg-black/25",
          "data-[role=user]:border-current/15 data-[role=user]:bg-midground/10",
        )}
      >
        <MessagePrimitive.Parts components={{ Text: TextPart }} />
        <MessagePrimitive.Error>
          <div className="mt-2 flex items-start gap-2 rounded-lg border border-red-400/30 bg-red-400/10 p-3 text-red-100">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>Something went wrong. Check the dashboard server logs.</span>
          </div>
        </MessagePrimitive.Error>
      </div>
    </MessagePrimitive.Root>
  );
};

const TextPart: FC = () => (
  <MessagePartPrimitive.Text
    component="div"
    className="whitespace-pre-wrap break-words"
    smooth
  />
);

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="flex items-end gap-3 rounded-3xl border border-current/15 bg-black/35 p-3 shadow-xl shadow-black/25 backdrop-blur">
      <ComposerPrimitive.Input
        autoFocus
        submitMode="enter"
        placeholder="Message Hermes..."
        className="max-h-40 min-h-12 flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-6 text-midground outline-none placeholder:text-midground/35"
      />
      <ComposerPrimitive.Send className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-midground text-background-base transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:scale-100">
        <Send className="h-4 w-4" />
        <span className="sr-only">Send</span>
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
};
