import {
  AssistantRuntimeProvider,
  AttachmentPrimitive,
  ComposerPrimitive,
  CompositeAttachmentAdapter,
  MessagePrimitive,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  ThreadPrimitive,
  useComposer,
  useComposerRuntime,
  useLocalRuntime,
  useMessage,
  useThread,
  useThreadRuntime,
  type Attachment,
  type ChatModelAdapter,
  type CompleteAttachment,
  type ThreadHistoryAdapter,
  type ThreadMessage,
  type ThreadUserMessagePart,
  type ThreadMessageLike,
} from "@assistant-ui/react";
import {
  AlertCircle,
  ArrowUp,
  Bot,
  Clock,
  FileText,
  ImageIcon,
  Loader2,
  Menu,
  Paperclip,
  PanelLeftClose,
  Plus,
  Sparkles,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FC,
  type FormEvent,
  type KeyboardEvent,
} from "react";

import { SlashPopover, type SlashPopoverHandle } from "@/components/SlashPopover";
import { api, HERMES_BASE_PATH, type SessionInfo, type SessionMessage } from "@/lib/api";
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
    try {
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

      let sawText = false;
      for await (const delta of readAssistantStream(response, abortSignal)) {
        sawText = true;
        yield {
          content: [{ type: "text", text: delta }],
        };
      }

      yield {
        content: sawText ? undefined : [{ type: "text", text: "Hermes returned an empty response." }],
        status: { type: "complete", reason: "stop" },
      };
    } catch (error) {
      yield {
        content: [
          {
            type: "text",
            text: `Hermes chat backend unavailable: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        status: { type: "incomplete", reason: "error" },
      };
    }
  },
};

const attachmentAdapter = new CompositeAttachmentAdapter([
  new SimpleImageAttachmentAdapter(),
  new SimpleTextAttachmentAdapter(),
]);

const examplePrompts = [
  "What can Hermes do from this dashboard?",
  "Check gateway status and summarize anything unhealthy",
  "Explain how slash commands work here",
];

const threadStorageKey = "hermes.assistant-ui.thread.v1";

function readableSessionTitle(session: SessionInfo): string {
  const title = session.title?.trim();
  if (title && title !== "Untitled") return title;
  const preview = session.preview?.trim();
  if (preview) return preview;
  return session.id;
}

function formatSessionTime(seconds: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(seconds * 1000));
}

function sessionMessageToThreadMessage(message: SessionMessage, index: number): ThreadMessageLike | null {
  const role = message.role === "tool" ? "assistant" : message.role;
  const toolCallNames = message.tool_calls?.map((call) => call.function.name).filter(Boolean) ?? [];
  const content =
    message.content?.trim() ||
    (toolCallNames.length > 0 ? `[Tool call: ${toolCallNames.join(", ")}]` : "");
  if (!content) return null;

  const text = message.role === "tool" && message.tool_name ? `[Tool: ${message.tool_name}]\n${content}` : content;

  return {
    id: `session-${message.timestamp ?? "no-time"}-${index}`,
    role,
    content: [{ type: "text", text }],
    createdAt: message.timestamp ? new Date(message.timestamp * 1000) : undefined,
    ...(role === "assistant" ? { status: { type: "complete" as const, reason: "unknown" as const } } : {}),
  };
}

type StoredThread = Awaited<ReturnType<ThreadHistoryAdapter["load"]>>;

function reviveThreadMessage(message: ThreadMessage): ThreadMessage {
  if (typeof message.createdAt === "string") {
    return { ...message, createdAt: new Date(message.createdAt) };
  }
  return message;
}

async function loadStoredThread(): Promise<StoredThread> {
  if (typeof window === "undefined") return { messages: [] };
  const raw = window.localStorage.getItem(threadStorageKey);
  if (!raw) return { messages: [] };

  try {
    const parsed = JSON.parse(raw) as StoredThread;
    return {
      ...parsed,
      messages: parsed.messages.map((item) => ({
        ...item,
        message: reviveThreadMessage(item.message),
      })),
    };
  } catch {
    window.localStorage.removeItem(threadStorageKey);
    return { messages: [] };
  }
}

const browserThreadHistory: ThreadHistoryAdapter = {
  async load() {
    return loadStoredThread();
  },
  async append(item) {
    if (typeof window === "undefined") return;
    const current = await loadStoredThread();
    const messages = current.messages.filter((entry) => entry.message.id !== item.message.id);
    messages.push(item);
    window.localStorage.setItem(
      threadStorageKey,
      JSON.stringify({
        headId: item.message.id,
        messages,
      }),
    );
  },
};

const appFontStack =
  'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

export default function AssistantPage() {
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    if (typeof window === "undefined") return true;
    return window.matchMedia("(min-width: 768px)").matches;
  });
  const runtime = useLocalRuntime(hermesAdapter, {
    adapters: {
      attachments: attachmentAdapter,
      history: browserThreadHistory,
    },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <main
        className="-m-4 flex h-[calc(100dvh-5.5rem)] min-h-[36rem] overflow-hidden bg-[#09090b] text-[#f4f4f5] antialiased normal-case sm:-m-6 sm:h-[calc(100dvh-6.5rem)] lg:-m-6"
        style={{ fontFamily: appFontStack }}
      >
        <AssistantSidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <section className="flex min-w-0 flex-1 flex-col bg-[#0b0b0f]">
          <AssistantHeader sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen((open) => !open)} />
          <ShowcaseThread />
        </section>
      </main>
    </AssistantRuntimeProvider>
  );
}

const AssistantSidebar: FC<{ open: boolean; onClose: () => void }> = ({ open, onClose }) => {
  const threadRuntime = useThreadRuntime();
  const messages = useThread((state) => state.messages);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionsError, setSessionsError] = useState<string | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const firstUserMessage = messages.find((message) => message.role === "user");
  const title =
    firstUserMessage?.content
      .map((part) => (part.type === "text" ? part.text : ""))
      .join(" ")
      .trim() || "Current conversation";

  useEffect(() => {
    let cancelled = false;

    async function loadSessions() {
      try {
        setSessionsLoading(true);
        setSessionsError(null);
        const response = await api.getSessions(30);
        if (!cancelled) {
          setSessions(response.sessions.filter((session) => session.message_count > 0));
        }
      } catch (error) {
        if (!cancelled) {
          setSessionsError(error instanceof Error ? error.message : "Could not load previous chats");
        }
      } finally {
        if (!cancelled) setSessionsLoading(false);
      }
    }

    void loadSessions();
    return () => {
      cancelled = true;
    };
  }, []);

  const closeMobileSidebar = () => {
    if (window.matchMedia("(max-width: 767px)").matches) onClose();
  };

  const startNewThread = () => {
    window.localStorage.removeItem(threadStorageKey);
    setSelectedSessionId(null);
    threadRuntime.reset();
    closeMobileSidebar();
  };

  const restoreCurrentThread = async () => {
    const stored = await loadStoredThread();
    threadRuntime.import(stored);
    setSelectedSessionId(null);
    closeMobileSidebar();
  };

  const openPreviousChat = async (session: SessionInfo) => {
    setLoadingSessionId(session.id);
    setSessionsError(null);
    try {
      const response = await api.getSessionMessages(session.id);
      const restoredMessages = response.messages
        .map((message, index) => sessionMessageToThreadMessage(message, index))
        .filter((message): message is ThreadMessageLike => message !== null);
      threadRuntime.reset(restoredMessages);
      setSelectedSessionId(session.id);
      closeMobileSidebar();
    } catch (error) {
      setSessionsError(error instanceof Error ? error.message : "Could not open previous chat");
    } finally {
      setLoadingSessionId(null);
    }
  };

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close sidebar overlay"
          className="fixed inset-0 z-30 bg-black/55 backdrop-blur-sm md:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-80 max-w-[86vw] shrink-0 flex-col border-r border-[#27272a] bg-[#111113] shadow-2xl shadow-black/60 transition-transform duration-200 md:relative md:z-auto md:w-72 md:max-w-none md:shadow-none md:transition-[width,opacity]",
          open ? "translate-x-0" : "-translate-x-full md:w-0 md:translate-x-0 md:overflow-hidden md:border-r-0 md:opacity-0",
        )}
        aria-hidden={!open}
      >
      <div className="flex h-14 items-center gap-2 px-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-[#27272a] bg-[#111113] shadow-sm shadow-black/30">
          <Sparkles className="h-4 w-4 text-[#f4f4f5]" />
        </div>
        <span className="text-sm font-semibold tracking-tight">assistant-ui</span>
        <button
          type="button"
          onClick={onClose}
          className="ml-auto inline-flex h-8 w-8 items-center justify-center rounded-md text-[#a1a1aa] transition hover:bg-[#18181b] hover:text-[#f4f4f5] md:hidden"
          aria-label="Close previous chats"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="px-3 pb-3">
        <button
          type="button"
          onClick={startNewThread}
          className="flex h-9 w-full items-center justify-center gap-2 rounded-md bg-[#f4f4f5] px-3 text-sm font-medium text-[#09090b] shadow-sm transition hover:bg-[#e4e4e7]"
        >
          <Plus className="h-4 w-4" />
          New Thread
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        <div className="mb-2 px-2 text-xs font-medium text-[#a1a1aa]">Current</div>
        <button
          type="button"
          onClick={() => void restoreCurrentThread()}
          className={cn(
            "w-full truncate rounded-md px-2 py-2 text-left text-sm text-[#f4f4f5] shadow-sm ring-1 ring-[#27272a] transition hover:bg-[#18181b]",
            selectedSessionId === null ? "bg-[#18181b]" : "bg-transparent",
          )}
          title={title}
        >
          {title}
        </button>

        <div className="mb-2 mt-5 flex items-center justify-between px-2 text-xs font-medium text-[#a1a1aa]">
          <span>Previous chats</span>
          {sessionsLoading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
        </div>
        {sessionsError && (
          <p className="mb-2 rounded-md border border-red-950/60 bg-red-950/30 px-2 py-1.5 text-xs leading-5 text-red-300">
            {sessionsError}
          </p>
        )}
        <div className="flex flex-col gap-1">
          {sessions.map((session) => {
            const sessionTitle = readableSessionTitle(session);
            const selected = selectedSessionId === session.id;
            const loading = loadingSessionId === session.id;
            return (
              <button
                key={session.id}
                type="button"
                onClick={() => void openPreviousChat(session)}
                className={cn(
                  "group rounded-md px-2 py-2 text-left text-sm transition hover:bg-[#18181b]",
                  selected ? "bg-[#18181b] text-[#f4f4f5] ring-1 ring-[#27272a]" : "text-[#d4d4d8]",
                )}
                title={sessionTitle}
              >
                <span className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate">{sessionTitle}</span>
                  {loading && <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-[#a1a1aa]" />}
                </span>
                <span className="mt-1 flex items-center gap-1 text-[11px] text-[#71717a]">
                  <Clock className="h-3 w-3" />
                  {formatSessionTime(session.last_active)} · {session.message_count} msgs
                </span>
              </button>
            );
          })}
        </div>
        {!sessionsLoading && sessions.length === 0 && !sessionsError && (
          <p className="px-2 text-xs leading-5 text-[#71717a]">No previous chats yet.</p>
        )}
      </div>

      <div className="border-t border-[#27272a] p-3">
        <button
          type="button"
          onClick={onClose}
          className="flex h-9 w-full items-center gap-2 rounded-md px-2 text-sm text-[#a1a1aa] transition hover:bg-[#18181b] hover:text-[#f4f4f5]"
        >
          <PanelLeftClose className="h-4 w-4" />
          Hide sidebar
        </button>
      </div>
      </aside>
    </>
  );
};

const AssistantHeader: FC<{ sidebarOpen: boolean; onToggleSidebar: () => void }> = ({
  sidebarOpen,
  onToggleSidebar,
}) => {
  return (
    <header className="relative z-20 flex h-14 shrink-0 items-center justify-between border-b border-[#27272a] bg-[#111113] px-3 sm:px-4">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onToggleSidebar}
          className="inline-flex h-9 w-9 items-center justify-center rounded-md text-[#a1a1aa] transition hover:bg-[#18181b] hover:text-[#f4f4f5]"
          aria-label={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
        >
          <Menu className="h-4 w-4 md:hidden" />
          <PanelLeftClose className="hidden h-4 w-4 md:block" />
        </button>
        <div className="inline-flex h-9 items-center gap-2 rounded-md px-2 text-sm font-medium tracking-tight text-[#f4f4f5]">
          <Bot className="h-4 w-4" />
          Hermes Agent
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden h-9 items-center gap-2 rounded-md border border-[#27272a] bg-[#111113] px-3 text-sm text-[#a1a1aa] shadow-sm sm:flex">
          <span className="h-2 w-2 rounded-full bg-emerald-500" />
          Streaming via Hermes
        </div>
      </div>
    </header>
  );
};

const ShowcaseThread: FC = () => {
  useEffect(() => {
    const reset = window.setTimeout(() => {
      const headline = [...document.querySelectorAll("h1")].find((element) =>
        element.textContent?.includes("Hello there!"),
      );
      headline?.closest<HTMLElement>(".overflow-y-auto")?.scrollTo({ top: 0 });
    }, 150);
    return () => window.clearTimeout(reset);
  }, []);

  return (
    <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col">
      <ThreadPrimitive.Viewport className="flex min-h-0 flex-1 flex-col overflow-y-auto scroll-smooth bg-[#0b0b0f] px-3 sm:px-6">
        <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-6 py-6 sm:py-8">
          <ThreadPrimitive.Empty>
            <WelcomePanel />
          </ThreadPrimitive.Empty>

          <div className="flex flex-col gap-6 empty:hidden">
            <ThreadPrimitive.Messages components={{ Message: ChatMessage }} />
          </div>

          <ThreadPrimitive.ViewportFooter className="sticky bottom-0 mt-auto bg-gradient-to-t from-[#0b0b0f] via-[#0b0b0f]/95 to-transparent pb-4 pt-10 sm:pb-6">
            <Composer />
            <p className="mt-2 text-center text-xs text-[#a1a1aa]">
              Streaming replies are on. Slash commands run through the Hermes gateway bridge.
            </p>
          </ThreadPrimitive.ViewportFooter>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
};

const WelcomePanel: FC = () => {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-1 pb-32 pt-10 text-center sm:pb-40">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl border border-[#27272a] bg-[#111113] shadow-sm shadow-black/30">
        <Bot className="h-6 w-6 text-[#f4f4f5]" />
      </div>
      <h1 className="mt-6 text-3xl font-semibold tracking-tight text-[#f4f4f5] sm:text-4xl">
        Hello there!
      </h1>
      <p className="mt-3 max-w-xl text-sm leading-6 text-[#a1a1aa] sm:text-base">
        Ask Hermes to research, inspect files, run tools, or use `/help`, `/status`, and other
        slash commands without leaving the assistant-ui surface.
      </p>
      <div className="mt-8 grid w-full max-w-2xl gap-2 sm:grid-cols-3">
        {examplePrompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            className="rounded-xl border border-[#27272a] bg-[#111113] p-4 text-left text-sm leading-5 text-[#a1a1aa] shadow-sm transition hover:border-[#3f3f46] hover:bg-[#18181b] hover:text-[#f4f4f5]"
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
};

const ChatMessage: FC = () => {
  const role = useMessage((state) => state.role);
  const isUser = role === "user";

  return (
    <MessagePrimitive.Root
      className={cn(
        "grid grid-cols-[minmax(0,1fr)]",
        isUser ? "justify-items-end" : "justify-items-start",
      )}
    >
      <div className={cn("flex max-w-[min(92%,46rem)] flex-col gap-1.5", isUser ? "items-end" : "items-start")}>
        <div
          className={cn(
            "flex items-center gap-1.5 px-1 text-xs font-medium",
            isUser ? "text-[#a1a1aa]" : "text-[#d4d4d8]",
          )}
        >
          {isUser ? (
            <span>You</span>
          ) : (
            <span className="flex items-center gap-1.5">
              <Bot className="h-3.5 w-3.5" /> Hermes
            </span>
          )}
        </div>
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-[15px] leading-7 shadow-sm sm:px-5",
            isUser
              ? "rounded-br-md bg-[#f4f4f5] text-[#09090b] shadow-black/20"
              : "rounded-bl-md border border-[#27272a] bg-[#111113] text-[#f4f4f5] shadow-black/30",
          )}
        >
          <MessagePrimitive.Attachments>
            {({ attachment }) => <MessageAttachment attachment={attachment} />}
          </MessagePrimitive.Attachments>
          <ChatTextContent />
          <MessagePrimitive.Error>
            <div className="mt-3 flex items-start gap-2 rounded-xl border border-red-950/60 bg-red-950/30 p-3 text-sm text-red-300">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>Something went wrong. Check the dashboard server logs.</span>
            </div>
          </MessagePrimitive.Error>
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};

const ChatTextContent: FC = () => {
  const text = useMessage((state) =>
    state.content
      .map((part) => (part.type === "text" ? part.text : ""))
      .filter(Boolean)
      .join("\n"),
  );

  if (!text) return null;
  return <div className="whitespace-pre-wrap break-words text-[15px] leading-7 text-inherit sm:text-base">{text}</div>;
};

function useAssistantSlashBridge() {
  const composerRuntime = useComposerRuntime();
  const composerText = useComposer((state) => state.text);
  const threadRuntime = useThreadRuntime();
  const gwRef = useRef<GatewayClient | null>(null);
  const slashSessionRef = useRef<string>("");
  const [gateway, setGateway] = useState<GatewayClient | null>(null);

  const ensureGateway = useCallback(async () => {
    let gw = gwRef.current;
    if (!gw || gw.state === "closed" || gw.state === "error") {
      gw?.close();
      gw = new GatewayClient();
      gwRef.current = gw;
      setGateway(gw);
    }
    if (gw.state !== "open") {
      await gw.connect();
      setGateway(gw);
    }
    return gw;
  }, []);

  const ensureSlashSession = useCallback(async () => {
    const gw = await ensureGateway();
    if (!slashSessionRef.current) {
      const result = await gw.request<{ session_id?: string }>("session.create", {});
      slashSessionRef.current = result.session_id ?? "";
    }
    return { gw, sessionId: slashSessionRef.current };
  }, [ensureGateway]);

  useEffect(() => {
    return () => gwRef.current?.close();
  }, []);

  useEffect(() => {
    if (!composerText.startsWith("/")) return;
    void ensureGateway().catch(() => undefined);
  }, [composerText, ensureGateway]);

  const applySlashCompletion = useCallback(
    (nextInput: string) => {
      composerRuntime.setText(nextInput);
    },
    [composerRuntime],
  );

  const handleSlashSubmit = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
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
    },
    [composerRuntime, composerText, ensureSlashSession, threadRuntime],
  );

  return {
    composerText,
    gateway,
    handleSlashSubmit,
    applySlashCompletion,
  };
}

const Composer: FC = () => {
  const { composerText, gateway, handleSlashSubmit, applySlashCompletion } = useAssistantSlashBridge();
  const slashPopoverRef = useRef<SlashPopoverHandle>(null);
  const handleInputKeyDown = useCallback((event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (slashPopoverRef.current?.handleKey(event)) return;
  }, []);

  return (
    <ComposerPrimitive.AttachmentDropzone>
      <ComposerPrimitive.Root
        className="relative rounded-2xl border border-[#27272a] bg-[#111113] p-2 shadow-[0_18px_60px_rgba(0,0,0,0.45)]"
        onSubmit={handleSlashSubmit}
      >
        <SlashPopover
          ref={slashPopoverRef}
          input={composerText}
          gw={gateway}
          onApply={applySlashCompletion}
        />
        <ComposerPrimitive.Attachments>
          {({ attachment }) => <ComposerAttachment attachment={attachment} />}
        </ComposerPrimitive.Attachments>
        <div className="flex items-end gap-2">
          <ComposerPrimitive.AddAttachment
            multiple
            className="mb-1 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[#a1a1aa] transition hover:bg-[#18181b] hover:text-[#f4f4f5] disabled:cursor-not-allowed disabled:opacity-40"
          >
            <Paperclip className="h-4 w-4" />
            <span className="sr-only">Add attachment</span>
          </ComposerPrimitive.AddAttachment>
          <ComposerPrimitive.Input
            submitMode="enter"
            placeholder="Message Hermes or type / for commands..."
            onKeyDown={handleInputKeyDown}
            className="max-h-36 min-h-10 flex-1 resize-none bg-transparent px-1 py-2.5 text-[16px] leading-6 text-[#f4f4f5] outline-none placeholder:text-[#a1a1aa] sm:max-h-44 sm:text-[15px]"
          />
          <ComposerPrimitive.Send className="mb-1 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#f4f4f5] text-[#09090b] shadow-sm transition hover:bg-[#e4e4e7] disabled:cursor-not-allowed disabled:opacity-35">
            <ArrowUp className="h-4 w-4" />
            <span className="sr-only">Send message</span>
          </ComposerPrimitive.Send>
        </div>
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
    <AttachmentPrimitive.Root className="mb-2 flex items-center gap-2 rounded-xl border border-[#27272a] bg-[#18181b] px-3 py-2 text-xs text-[#a1a1aa]">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#27272a] text-[#a1a1aa] shadow-sm">
        {attachmentIcon(attachment)}
      </span>
      <div className="min-w-0 flex-1">
        <span className="block truncate font-medium text-[#f4f4f5]">
          <AttachmentPrimitive.Name />
        </span>
        <span className="block truncate text-[11px] text-[#a1a1aa]">
          {running ? "Uploading..." : attachmentLabel(attachment)}
        </span>
      </div>
      <AttachmentPrimitive.Remove className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[#a1a1aa] transition hover:bg-[#3f3f46] hover:text-[#f4f4f5]">
        <X className="h-4 w-4" />
        <span className="sr-only">Remove attachment</span>
      </AttachmentPrimitive.Remove>
    </AttachmentPrimitive.Root>
  );
};

const MessageAttachment: FC<{ attachment: CompleteAttachment }> = ({ attachment }) => {
  return (
    <div className="mb-3 inline-flex max-w-full items-center gap-2 rounded-xl border border-[#27272a] bg-[#18181b] px-3 py-2 text-xs text-[#a1a1aa]">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[#27272a] text-[#a1a1aa] shadow-sm">
        {attachmentIcon(attachment)}
      </span>
      <span className="truncate">{attachmentLabel(attachment)}</span>
    </div>
  );
};
