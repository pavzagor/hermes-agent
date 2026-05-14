import {
  AssistantRuntimeProvider,
  AttachmentPrimitive,
  ComposerPrimitive,
  CompositeAttachmentAdapter,
  MessagePartPrimitive,
  MessagePrimitive,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  ThreadPrimitive,
  useComposer,
  useComposerRuntime,
  useLocalRuntime,
  useThreadRuntime,
  type Attachment,
  type ChatModelAdapter,
  type CompleteAttachment,
  type ThreadMessage,
  type ThreadUserMessagePart,
} from "@assistant-ui/react";
import {
  AlertCircle,
  ArrowUp,
  Bot,
  FileText,
  ImageIcon,
  Paperclip,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, type FC, type FormEvent } from "react";

import { HERMES_BASE_PATH } from "@/lib/api";
import { GatewayClient } from "@/lib/gatewayClient";
import { executeSlash } from "@/lib/slashExec";
import { cn } from "@/lib/utils";

type HermesContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

function partToHermesContent(part: ThreadUserMessagePart): HermesContentPart | null {
  if (part.type === "text") return part.text ? { type: "text", text: part.text } : null;
  if (part.type === "image") return { type: "image_url", image_url: { url: part.image } };
  if (part.type === "file") {
    if (part.mimeType.startsWith("image/") && part.data) {
      return { type: "image_url", image_url: { url: part.data } };
    }
    return {
      type: "text",
      text: `[File attached: ${part.filename ?? part.mimeType}. Non-image file payloads are not supported by this chat proxy.]`,
    };
  }
  if (part.type === "data") return { type: "text", text: `[Data: ${part.name}]` };
  return null;
}

function messageContent(message: ThreadMessage): string | HermesContentPart[] {
  const parts: HermesContentPart[] = [];

  for (const part of message.content) {
    const converted = partToHermesContent(part as ThreadUserMessagePart);
    if (converted) parts.push(converted);
  }

  for (const attachment of message.attachments ?? []) {
    for (const part of attachment.content ?? []) {
      const converted = partToHermesContent(part);
      if (converted) parts.push(converted);
    }
  }

  if (parts.some((part) => part.type === "image_url")) return parts;
  return parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .filter(Boolean)
    .join("\n");
}

type AssistantStreamEvent =
  | { type: "delta"; text: string }
  | { type: "done"; reason?: string }
  | { type: "error"; status?: number; message?: string };

async function* readAssistantStream(response: Response, abortSignal?: AbortSignal) {
  const reader = response.body?.getReader();
  if (!reader) {
    const data = (await response.json()) as { text?: string };
    yield data.text ?? "";
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";

  while (!abortSignal?.aborted) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const event = JSON.parse(trimmed) as AssistantStreamEvent;
      if (event.type === "delta") {
        text += event.text;
        yield text;
      } else if (event.type === "error") {
        throw new Error(event.message ?? `Hermes chat failed (${event.status ?? "stream"})`);
      } else if (event.type === "done") {
        return;
      }
    }
  }

  const tail = buffer.trim();
  if (tail) {
    const event = JSON.parse(tail) as AssistantStreamEvent;
    if (event.type === "delta") {
      text += event.text;
      yield text;
    } else if (event.type === "error") {
      throw new Error(event.message ?? `Hermes chat failed (${event.status ?? "stream"})`);
    }
  }
}

const hermesAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal, unstable_threadId }) {
    const response = await fetch(`${HERMES_BASE_PATH}/api/assistant-chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Hermes-Session-Token": window.__HERMES_SESSION_TOKEN__ ?? "",
      },
      signal: abortSignal,
      body: JSON.stringify({
        thread_id: unstable_threadId,
        stream: true,
        messages: messages.map((message) => ({
          id: message.id,
          role: message.role,
          content: messageContent(message),
        })),
      }),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => response.statusText);
      throw new Error(`Hermes chat failed (${response.status}): ${text}`);
    }

    let latest = "";
    for await (const partial of readAssistantStream(response, abortSignal)) {
      latest = partial;
      yield {
        content: [{ type: "text", text: partial }],
      };
    }

    yield {
      content: [{ type: "text", text: latest }],
      status: { type: "complete", reason: "stop" },
    };
  },
};

const attachmentAdapter = new CompositeAttachmentAdapter([
  new SimpleImageAttachmentAdapter(),
  new SimpleTextAttachmentAdapter(),
]);

const examplePrompts = [
  "Summarize what changed in the last Hermes session",
  "Inspect my gateway status and tell me if messaging is healthy",
  "Draft a plan for making the dashboard service durable",
];

export default function AssistantPage() {
  const runtime = useLocalRuntime(hermesAdapter, {
    adapters: {
      attachments: attachmentAdapter,
    },
  });

  const trustBadges = useMemo(
    () => ["Cloudflare Access", "Local API proxy", "Files stay in this chat turn"],
    [],
  );

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <main className="relative -m-4 flex h-[calc(100dvh-5.5rem)] min-h-[36rem] flex-col overflow-hidden bg-[#050816] font-sans normal-case text-white sm:-m-6 sm:h-[calc(100dvh-6.5rem)] lg:-m-6">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute -left-24 top-[-12rem] h-80 w-80 rounded-full bg-cyan-400/20 blur-3xl" />
          <div className="absolute right-[-8rem] top-20 h-72 w-72 rounded-full bg-fuchsia-500/20 blur-3xl" />
          <div className="absolute bottom-[-10rem] left-1/4 h-96 w-96 rounded-full bg-emerald-400/10 blur-3xl" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(255,255,255,0.08),transparent_32rem)]" />
        </div>

        <header className="relative z-10 shrink-0 border-b border-white/10 bg-[#050816]/75 px-4 py-3 backdrop-blur-2xl sm:px-6 sm:py-4">
          <div className="mx-auto flex w-full max-w-5xl items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-white text-[#050816] shadow-lg shadow-cyan-500/20 sm:h-11 sm:w-11">
                <Sparkles className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-base font-semibold tracking-[-0.01em] sm:text-xl">
                  Hermes Assistant
                </div>
                <p className="truncate text-xs text-white/55 sm:text-sm">
                  Mobile-first assistant-ui chat for your private agent
                </p>
              </div>
            </div>
            <div className="hidden items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1.5 text-xs text-emerald-100 sm:flex">
              <ShieldCheck className="h-3.5 w-3.5" />
              Protected
            </div>
          </div>
          <div className="mx-auto mt-3 flex w-full max-w-5xl gap-2 overflow-x-auto pb-1 sm:mt-4">
            {trustBadges.map((badge) => (
              <span
                key={badge}
                className="shrink-0 rounded-full border border-white/10 bg-white/[0.06] px-3 py-1 text-[11px] text-white/65"
              >
                {badge}
              </span>
            ))}
          </div>
        </header>

        <section className="relative z-10 min-h-0 flex-1">
          <ShowcaseThread />
        </section>
      </main>
    </AssistantRuntimeProvider>
  );
}

const ShowcaseThread: FC = () => {
  useEffect(() => {
    const reset = window.setTimeout(() => {
      const headline = [...document.querySelectorAll("h1")].find((element) =>
        element.textContent?.includes("What can Hermes do for you?"),
      );
      headline?.closest<HTMLElement>(".overflow-y-auto")?.scrollTo({ top: 0 });
    }, 150);
    return () => window.clearTimeout(reset);
  }, []);

  return (
    <ThreadPrimitive.Root className="flex h-full min-h-0 flex-col">
      <ThreadPrimitive.Viewport className="flex min-h-0 flex-1 flex-col overflow-y-auto scroll-smooth px-3 sm:px-5">
        <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-4 py-4 sm:gap-5 sm:py-6">
          <ThreadPrimitive.Empty>
            <WelcomePanel />
          </ThreadPrimitive.Empty>

          <div className="flex flex-col gap-4 empty:hidden sm:gap-5">
            <ThreadPrimitive.Messages components={{ Message: ChatMessage }} />
          </div>

          <ThreadPrimitive.ViewportFooter className="sticky bottom-0 mt-auto bg-gradient-to-t from-[#050816] via-[#050816]/95 to-transparent pb-3 pt-10 sm:pb-5">
            <Composer />
          </ThreadPrimitive.ViewportFooter>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const WelcomePanel: FC = () => {
  return (
    <div className="flex flex-1 items-start px-1 pb-32 pt-4 sm:px-4 sm:pb-36 sm:pt-6">
      <div className="mx-auto w-full max-w-3xl rounded-[2rem] border border-white/10 bg-white/[0.07] p-5 shadow-2xl shadow-black/30 backdrop-blur-2xl sm:p-8">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-3xl bg-white text-[#050816] shadow-xl shadow-cyan-500/20 sm:h-16 sm:w-16">
          <Bot className="h-7 w-7" />
        </div>
        <div className="mt-5 text-center">
          <h1 className="pb-1 text-2xl font-black leading-tight tracking-[0.02em] text-white sm:text-4xl sm:leading-tight">
            What can Hermes do for you?
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-sm leading-6 text-white/60 sm:text-base">
            Ask the agent to reason, research, inspect files, run commands, or attach text/images
            for context. This is the assistant-ui surface, proxied safely through the dashboard.
          </p>
        </div>
        <div className="mt-6 grid gap-2 sm:grid-cols-3">
          {examplePrompts.map((prompt) => (
            <div
              key={prompt}
              className="rounded-2xl border border-white/10 bg-black/20 p-3 text-left text-xs leading-5 text-white/70 sm:p-4"
            >
              {prompt}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const ChatMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="group grid grid-cols-[minmax(0,1fr)] data-[role=assistant]:justify-items-start data-[role=user]:justify-items-end">
      <div
        className={cn(
          "max-w-[min(92%,46rem)] rounded-[1.5rem] border px-4 py-3 shadow-xl backdrop-blur-xl sm:max-w-[min(82%,48rem)] sm:px-5 sm:py-4",
          "data-[role=assistant]:border-white/10 data-[role=assistant]:bg-white/[0.08] data-[role=assistant]:text-white",
          "data-[role=user]:border-cyan-300/20 data-[role=user]:bg-cyan-300/15 data-[role=user]:text-cyan-50",
        )}
      >
        <div className="mb-2 flex items-center gap-2 text-[11px] uppercase tracking-[0.16em] text-white/35">
          <MessagePrimitive.If assistant>
            <span className="flex items-center gap-1.5">
              <Bot className="h-3.5 w-3.5" /> Hermes
            </span>
          </MessagePrimitive.If>
          <MessagePrimitive.If user>
            <span>You</span>
          </MessagePrimitive.If>
        </div>
        <MessagePrimitive.Attachments>
          {({ attachment }) => <MessageAttachment attachment={attachment} />}
        </MessagePrimitive.Attachments>
        <MessagePrimitive.Parts components={{ Text: TextPart }} />
        <MessagePrimitive.Error>
          <div className="mt-3 flex items-start gap-2 rounded-2xl border border-red-300/25 bg-red-500/15 p-3 text-sm text-red-100">
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
    className="whitespace-pre-wrap break-words text-[15px] leading-7 text-inherit sm:text-base"
    smooth
  />
);

function useAssistantSlashSubmit() {
  const composerRuntime = useComposerRuntime();
  const composerText = useComposer((state) => state.text);
  const threadRuntime = useThreadRuntime();
  const gwRef = useRef<GatewayClient | null>(null);
  const slashSessionRef = useRef<string>("");

  useEffect(() => {
    return () => gwRef.current?.close();
  }, []);

  async function ensureSlashSession() {
    let gw = gwRef.current;
    if (!gw || gw.state === "closed" || gw.state === "error") {
      gw?.close();
      gw = new GatewayClient();
      gwRef.current = gw;
    }
    if (gw.state !== "open") {
      await gw.connect();
    }
    if (!slashSessionRef.current) {
      const result = await gw.request<{ session_id?: string }>("session.create", {});
      slashSessionRef.current = result.session_id ?? "";
    }
    return { gw, sessionId: slashSessionRef.current };
  }

  return async (event: FormEvent<HTMLFormElement>) => {
    const command = composerText.trim();
    if (!command.startsWith("/")) return;

    event.preventDefault();
    const attachments = composerRuntime.getState().attachments;
    composerRuntime.reset();

    threadRuntime.append({
      role: "user",
      content: [{ type: "text", text: command }],
      startRun: false,
    });

    if (attachments.length > 0) {
      threadRuntime.append({
        role: "assistant",
        content: [{ type: "text", text: "Slash commands ignore attachments. Send files with a normal message instead." }],
        startRun: false,
      });
    }

    try {
      const { gw, sessionId } = await ensureSlashSession();
      await executeSlash({
        command,
        sessionId,
        gw,
        callbacks: {
          sys(text) {
            threadRuntime.append({
              role: "assistant",
              content: [{ type: "text", text }],
              startRun: false,
            });
          },
          send(message) {
            threadRuntime.append({
              role: "user",
              content: [{ type: "text", text: message }],
              startRun: true,
            });
          },
        },
      });
    } catch (error) {
      threadRuntime.append({
        role: "assistant",
        content: [
          {
            type: "text",
            text: `Slash command backend unavailable: ${error instanceof Error ? error.message : String(error)}\n\nOpen /chat for the exact TUI-backed slash-command surface, or start the dashboard with embedded chat enabled.`,
          },
        ],
        startRun: false,
      });
    }
  };
}

const Composer: FC = () => {
  const handleSlashSubmit = useAssistantSlashSubmit();

  return (
    <ComposerPrimitive.AttachmentDropzone>
      <ComposerPrimitive.Root
        className="relative overflow-hidden rounded-[1.75rem] border border-white/15 bg-white/[0.09] p-2 shadow-2xl shadow-black/35 backdrop-blur-2xl sm:rounded-[2rem] sm:p-3"
        onSubmit={handleSlashSubmit}
      >
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/40 to-transparent" />
        <ComposerPrimitive.Attachments>
          {({ attachment }) => <ComposerAttachment attachment={attachment} />}
        </ComposerPrimitive.Attachments>
        <div className="flex items-end gap-2">
          <ComposerPrimitive.AddAttachment
            multiple
            className="mb-1 inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-white/10 bg-white/[0.08] text-white/70 transition hover:bg-white/[0.14] hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
          >
            <Paperclip className="h-5 w-5" />
            <span className="sr-only">Add attachment</span>
          </ComposerPrimitive.AddAttachment>
          <ComposerPrimitive.Input
            submitMode="enter"
            placeholder="Message Hermes..."
            className="max-h-36 min-h-12 flex-1 resize-none bg-transparent px-1 py-3 text-[16px] leading-6 text-white outline-none placeholder:text-white/35 sm:max-h-44 sm:px-2 sm:text-[15px]"
          />
          <ComposerPrimitive.Send className="mb-1 inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-white text-[#050816] shadow-lg shadow-cyan-500/20 transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-35 disabled:hover:scale-100">
            <ArrowUp className="h-5 w-5" />
            <span className="sr-only">Send message</span>
          </ComposerPrimitive.Send>
        </div>
        <p className="px-14 pb-1 text-[10px] leading-4 text-white/35 sm:text-xs">
          Attach images or text files. Press Enter to send, Shift+Enter for a new line.
        </p>
      </ComposerPrimitive.Root>
    </ComposerPrimitive.AttachmentDropzone>
  );
};

function attachmentIcon(attachment: Pick<Attachment, "type" | "contentType">) {
  if (attachment.type === "image" || attachment.contentType?.startsWith("image/")) {
    return <ImageIcon className="h-4 w-4" />;
  }
  return <FileText className="h-4 w-4" />;
}

function attachmentLabel(attachment: Pick<Attachment, "name" | "contentType">) {
  return attachment.contentType ? `${attachment.name} · ${attachment.contentType}` : attachment.name;
}

const ComposerAttachment: FC<{ attachment: Attachment }> = ({ attachment }) => {
  const running = attachment.status.type === "running";
  return (
    <AttachmentPrimitive.Root className="mb-2 flex items-center gap-2 rounded-2xl border border-white/10 bg-black/25 px-3 py-2 text-xs text-white/75">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-white/10 text-white/75">
        {attachmentIcon(attachment)}
      </span>
      <div className="min-w-0 flex-1">
        <span className="block truncate font-medium text-white/85">
          <AttachmentPrimitive.Name />
        </span>
        <span className="block truncate text-[11px] text-white/40">
          {running ? "Uploading..." : attachmentLabel(attachment)}
        </span>
      </div>
      <AttachmentPrimitive.Remove className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-white/45 transition hover:bg-white/10 hover:text-white">
        <X className="h-4 w-4" />
        <span className="sr-only">Remove attachment</span>
      </AttachmentPrimitive.Remove>
    </AttachmentPrimitive.Root>
  );
};

const MessageAttachment: FC<{ attachment: CompleteAttachment }> = ({ attachment }) => {
  return (
    <div className="mb-3 inline-flex max-w-full items-center gap-2 rounded-2xl border border-white/10 bg-black/20 px-3 py-2 text-xs text-white/65">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-white/10 text-white/70">
        {attachmentIcon(attachment)}
      </span>
      <span className="truncate">{attachmentLabel(attachment)}</span>
    </div>
  );
};
