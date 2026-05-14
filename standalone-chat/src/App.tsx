import {
  AssistantRuntimeProvider,
  AttachmentPrimitive,
  ComposerPrimitive,
  CompositeAttachmentAdapter,
  MessagePrimitive,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  ThreadPrimitive,
  useLocalRuntime,
  useMessage,
  useThread,
  useThreadRuntime,
  type Attachment,
  type ChatModelAdapter,
  type CompleteAttachment,
  type ThreadHistoryAdapter,
  type ThreadMessage,
  type ThreadMessageLike,
  type ThreadUserMessagePart,
} from "@assistant-ui/react";
import { AlertCircle, ArrowUp, Bot, FileText, ImageIcon, Menu, Paperclip, Plus, X } from "lucide-react";
import { useEffect, useState, type FC } from "react";
import "./styles.css";

type HermesContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

type SessionInfo = {
  id: string;
  source: string | null;
  model: string | null;
  title: string | null;
  started_at: number;
  ended_at: number | null;
  last_active: number;
  is_active: boolean;
  message_count: number;
  tool_call_count: number;
  input_tokens: number;
  output_tokens: number;
  preview: string | null;
};

type SessionMessage = {
  role: "user" | "assistant" | "system" | "tool";
  content: string | null;
  tool_calls?: Array<{ id: string; function: { name: string; arguments: string } }>;
  tool_name?: string;
  tool_call_id?: string;
  timestamp?: number;
};

type AssistantStreamEvent =
  | { type: "delta"; text: string }
  | { type: "done"; reason?: string }
  | { type: "error"; status?: number; message?: string };

const attachmentAdapter = new CompositeAttachmentAdapter([
  new SimpleImageAttachmentAdapter(),
  new SimpleTextAttachmentAdapter(),
]);

const threadStorageKey = "hermes.standalone-chat.thread.v1";

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function partToHermesContent(part: ThreadUserMessagePart): HermesContentPart | null {
  if (part.type === "text") return part.text ? { type: "text", text: part.text } : null;
  if (part.type === "image") return { type: "image_url", image_url: { url: part.image } };
  if (part.type === "file") {
    if (part.mimeType.startsWith("image/") && part.data) {
      return { type: "image_url", image_url: { url: part.data } };
    }
    return {
      type: "text",
      text: `[File attached: ${part.filename ?? part.mimeType}. Non-image files are not sent to Hermes yet.]`,
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
  return parts.map((part) => (part.type === "text" ? part.text : "")).filter(Boolean).join("\n");
}

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
}

const hermesAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal, unstable_threadId }) {
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
        yield { content: [{ type: "text", text: delta }] };
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
  const content = message.content?.trim() || (toolCallNames.length > 0 ? `[Tool call: ${toolCallNames.join(", ")}]` : "");
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
  if (typeof message.createdAt === "string") return { ...message, createdAt: new Date(message.createdAt) };
  return message;
}

async function loadStoredThread(): Promise<StoredThread> {
  const raw = window.localStorage.getItem(threadStorageKey);
  if (!raw) return { messages: [] };
  try {
    const parsed = JSON.parse(raw) as StoredThread;
    return {
      ...parsed,
      messages: parsed.messages.map((item) => ({ ...item, message: reviveThreadMessage(item.message) })),
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
    const current = await loadStoredThread();
    const messages = current.messages.filter((entry) => entry.message.id !== item.message.id);
    messages.push(item);
    window.localStorage.setItem(threadStorageKey, JSON.stringify({ headId: item.message.id, messages }));
  },
};

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json() as Promise<T>;
}

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const runtime = useLocalRuntime(hermesAdapter, {
    adapters: { attachments: attachmentAdapter, history: browserThreadHistory },
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="app-shell">
        <HistoryDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} />
        <main className="chat-shell">
          <Header onMenu={() => setDrawerOpen(true)} />
          <ChatThread />
        </main>
      </div>
    </AssistantRuntimeProvider>
  );
}

const Header: FC<{ onMenu: () => void }> = ({ onMenu }) => (
  <header className="topbar">
    <button type="button" className="icon-button" onClick={onMenu} aria-label="Open previous chats">
      <Menu size={20} />
    </button>
    <div className="brand">
      <span className="brand-icon"><Bot size={17} /></span>
      <div>
        <div className="brand-title">Hermes Chat</div>
        <div className="brand-subtitle">mobile-first assistant-ui</div>
      </div>
    </div>
    <div className="status-pill"><span /> Streaming</div>
  </header>
);

const HistoryDrawer: FC<{ open: boolean; onClose: () => void }> = ({ open, onClose }) => {
  const threadRuntime = useThreadRuntime();
  const messages = useThread((state) => state.messages);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const firstUserMessage = messages.find((message) => message.role === "user");
  const currentTitle = firstUserMessage?.content.map((part) => (part.type === "text" ? part.text : "")).join(" ").trim() || "Current chat";

  useEffect(() => {
    let cancelled = false;
    async function loadSessions() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetchJSON<{ sessions: SessionInfo[] }>("/api/sessions?limit=40");
        if (!cancelled) setSessions(response.sessions.filter((session) => session.message_count > 0));
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Could not load previous chats");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void loadSessions();
    return () => {
      cancelled = true;
    };
  }, []);

  const newChat = () => {
    window.localStorage.removeItem(threadStorageKey);
    threadRuntime.reset();
    onClose();
  };

  const restoreCurrent = async () => {
    const stored = await loadStoredThread();
    threadRuntime.import(stored);
    onClose();
  };

  const openSession = async (session: SessionInfo) => {
    setLoadingId(session.id);
    setError(null);
    try {
      const response = await fetchJSON<{ messages: SessionMessage[] }>(`/api/sessions/${encodeURIComponent(session.id)}/messages`);
      const restored = response.messages
        .map((message, index) => sessionMessageToThreadMessage(message, index))
        .filter((message): message is ThreadMessageLike => message !== null);
      threadRuntime.reset(restored);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not open previous chat");
    } finally {
      setLoadingId(null);
    }
  };

  return (
    <>
      {open && <button type="button" className="drawer-backdrop" aria-label="Close previous chats" onClick={onClose} />}
      <aside className={cn("drawer", open && "drawer-open")} aria-hidden={!open}>
        <div className="drawer-head">
          <div>
            <div className="drawer-title">Chats</div>
            <div className="drawer-subtitle">Current + Hermes history</div>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close previous chats"><X size={18} /></button>
        </div>
        <button type="button" className="new-chat" onClick={newChat}><Plus size={16} /> New chat</button>
        <div className="drawer-section-label">Current</div>
        <button type="button" className="session-item active" onClick={() => void restoreCurrent()} title={currentTitle}>
          <span>{currentTitle}</span>
          <small>local browser thread</small>
        </button>
        <div className="drawer-section-label">Previous chats</div>
        {loading && <div className="drawer-note">Loading history…</div>}
        {error && <div className="drawer-error">{error}</div>}
        <div className="session-list">
          {sessions.map((session) => (
            <button key={session.id} type="button" className="session-item" onClick={() => void openSession(session)}>
              <span>{loadingId === session.id ? "Opening…" : readableSessionTitle(session)}</span>
              <small>{formatSessionTime(session.started_at)} · {session.message_count} msgs</small>
            </button>
          ))}
        </div>
      </aside>
    </>
  );
};

const ChatThread: FC = () => (
  <ThreadPrimitive.Root className="thread-root">
    <ThreadPrimitive.Viewport className="thread-viewport">
      <div className="thread-inner">
        <ThreadPrimitive.Empty>
          <Welcome />
        </ThreadPrimitive.Empty>
        <div className="messages-stack">
          <ThreadPrimitive.Messages components={{ Message: ChatMessage }} />
        </div>
        <ThreadPrimitive.ViewportFooter className="composer-footer">
          <Composer />
          <p className="fine-print">Protected separately. API keys stay server-side.</p>
        </ThreadPrimitive.ViewportFooter>
      </div>
    </ThreadPrimitive.Viewport>
  </ThreadPrimitive.Root>
);

const Welcome: FC = () => (
  <section className="welcome">
    <div className="welcome-icon"><Bot size={28} /></div>
    <h1>What should Hermes do?</h1>
    <p>Ask from your phone or desktop. Streaming is enabled, and previous Hermes sessions are in the menu.</p>
  </section>
);

const ChatMessage: FC = () => {
  const role = useMessage((state) => state.role);
  const isUser = role === "user";
  return (
    <MessagePrimitive.Root className={cn("message-row", isUser ? "message-user" : "message-hermes")}>
      <div className="message-wrap">
        <div className="message-label">{isUser ? "You" : <><Bot size={14} /> Hermes</>}</div>
        <div className="message-bubble">
          <MessagePrimitive.Attachments>{({ attachment }) => <MessageAttachment attachment={attachment} />}</MessagePrimitive.Attachments>
          <ChatTextContent />
          <MessagePrimitive.Error>
            <div className="message-error"><AlertCircle size={16} /> Something went wrong.</div>
          </MessagePrimitive.Error>
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};

const ChatTextContent: FC = () => {
  const text = useMessage((state) => state.content.map((part) => (part.type === "text" ? part.text : "")).filter(Boolean).join("\n"));
  if (!text) return null;
  return <div className="message-text">{text}</div>;
};

const Composer: FC = () => (
  <ComposerPrimitive.AttachmentDropzone>
    <ComposerPrimitive.Root className="composer">
      <ComposerPrimitive.Attachments>{({ attachment }) => <ComposerAttachment attachment={attachment} />}</ComposerPrimitive.Attachments>
      <div className="composer-row">
        <ComposerPrimitive.AddAttachment multiple className="composer-icon-button">
          <Paperclip size={18} />
          <span className="sr-only">Add attachment</span>
        </ComposerPrimitive.AddAttachment>
        <ComposerPrimitive.Input submitMode="enter" placeholder="Message Hermes…" className="composer-input" />
        <ComposerPrimitive.Send className="send-button">
          <ArrowUp size={18} />
          <span className="sr-only">Send</span>
        </ComposerPrimitive.Send>
      </div>
    </ComposerPrimitive.Root>
  </ComposerPrimitive.AttachmentDropzone>
);

function attachmentIcon(attachment: Pick<Attachment, "type" | "contentType">) {
  if (attachment.type === "image" || attachment.contentType?.startsWith("image/")) return <ImageIcon size={16} />;
  return <FileText size={16} />;
}

function attachmentLabel(attachment: Pick<Attachment, "name" | "contentType">) {
  return attachment.contentType ? `${attachment.name} · ${attachment.contentType}` : attachment.name;
}

const ComposerAttachment: FC<{ attachment: Attachment }> = ({ attachment }) => (
  <AttachmentPrimitive.Root className="attachment-chip">
    {attachmentIcon(attachment)}
    <span><AttachmentPrimitive.Name /></span>
    <AttachmentPrimitive.Remove className="attachment-remove"><X size={14} /></AttachmentPrimitive.Remove>
  </AttachmentPrimitive.Root>
);

const MessageAttachment: FC<{ attachment: CompleteAttachment }> = ({ attachment }) => (
  <div className="attachment-chip message-attachment">
    {attachmentIcon(attachment)}
    <span>{attachmentLabel(attachment)}</span>
  </div>
);

export default App;
