import type { GatewayClient } from "@/lib/gatewayClient";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { ChevronRight } from "lucide-react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

/**
 * Slash-command autocomplete popover, rendered above the composer in ChatPage.
 * Mirrors the completion UX of the Ink TUI — type `/`, see matching commands,
 * arrow keys or click to select, Tab to apply, Enter to submit.
 *
 * The parent owns all keyboard handling via `ref.handleKey`, which returns
 * true when the popover consumed the event, so the composer's Enter/arrow
 * logic stays in one place.
 */

export interface CompletionItem {
  display: string;
  text: string;
  meta?: string;
}

export interface SlashPopoverHandle {
  /** Returns true if the key was consumed by the popover. */
  handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>): boolean;
}

interface Props {
  input: string;
  gw: GatewayClient | null;
  onApply(nextInput: string): void;
}

interface CompletionResponse {
  items?: CompletionItem[];
  replace_from?: number;
}

const DEBOUNCE_MS = 60;

const FALLBACK_COMMANDS: CompletionItem[] = [
  { display: "/help", text: "/help", meta: "Show slash-command help" },
  { display: "/commands", text: "/commands", meta: "Browse available commands" },
  { display: "/model", text: "/model", meta: "View or switch the active model" },
  { display: "/status", text: "/status", meta: "Show session status" },
  { display: "/new", text: "/new", meta: "Start a fresh session" },
  { display: "/retry", text: "/retry", meta: "Retry the last assistant turn" },
  { display: "/undo", text: "/undo", meta: "Remove the last exchange" },
  { display: "/tools", text: "/tools", meta: "Manage enabled tools" },
  { display: "/skills", text: "/skills", meta: "Search and install skills" },
  { display: "/cron", text: "/cron", meta: "Manage scheduled jobs" },
];

function fallbackItems(input: string): CompletionItem[] {
  const query = input.slice(1).trimStart().toLowerCase();
  return FALLBACK_COMMANDS.filter((item) => item.display.toLowerCase().startsWith(`/${query}`));
}

function applyCompletionText(input: string, replaceFrom: number, item: CompletionItem): string {
  if (replaceFrom <= 1 && item.text.startsWith("/")) return item.text;
  return input.slice(0, replaceFrom) + item.text;
}

export const SlashPopover = forwardRef<SlashPopoverHandle, Props>(
  function SlashPopover({ input, gw, onApply }, ref) {
    const [items, setItems] = useState<CompletionItem[]>([]);
    const [selected, setSelected] = useState(0);
    const [replaceFrom, setReplaceFrom] = useState(1);
    const lastInputRef = useRef<string>("");

    // Debounced completion fetch. We never clear `items` in the effect body
    // (doing so would flag react-hooks/set-state-in-effect); instead the
    // render guard below hides stale items once the input stops matching.
    useEffect(() => {
      const trimmed = input ?? "";

      if (!trimmed.startsWith("/")) {
        lastInputRef.current = "";
        return;
      }

      if (!gw) {
        lastInputRef.current = trimmed;
        setItems(fallbackItems(trimmed));
        setReplaceFrom(1);
        setSelected(0);
        return;
      }

      if (trimmed === lastInputRef.current) {
        return;
      }
      lastInputRef.current = trimmed;

      const timer = window.setTimeout(async () => {
        if (lastInputRef.current !== trimmed) return;
        try {
          const r = await gw.request<CompletionResponse>("complete.slash", {
            text: trimmed,
          });
          if (lastInputRef.current !== trimmed) return;
          setItems(r?.items ?? []);
          setReplaceFrom(r?.replace_from ?? 1);
          setSelected(0);
        } catch {
          if (lastInputRef.current === trimmed) {
            setItems(fallbackItems(trimmed));
            setReplaceFrom(1);
            setSelected(0);
          }
        }
      }, DEBOUNCE_MS);

      return () => window.clearTimeout(timer);
    }, [input, gw]);

    const apply = useCallback(
      (item: CompletionItem) => {
        onApply(applyCompletionText(input, replaceFrom, item));
      },
      [input, replaceFrom, onApply],
    );

    // Only consume keys when the popover is actually visible. Stale items from
    // a previous slash prefix are ignored once the user deletes the "/".
    const visible = items.length > 0 && input.startsWith("/");

    useImperativeHandle(
      ref,
      () => ({
        handleKey: (e) => {
          if (!visible) return false;

          switch (e.key) {
            case "ArrowDown":
              e.preventDefault();
              setSelected((s) => (s + 1) % items.length);
              return true;

            case "ArrowUp":
              e.preventDefault();
              setSelected((s) => (s - 1 + items.length) % items.length);
              return true;

            case "Tab": {
              e.preventDefault();
              const item = items[selected];
              if (item) apply(item);
              return true;
            }

            case "Escape":
              e.preventDefault();
              setItems([]);
              return true;

            default:
              return false;
          }
        },
      }),
      [visible, items, selected, apply],
    );

    if (!visible) return null;

    return (
      <div
        className="absolute bottom-full left-0 right-0 mb-2 max-h-72 overflow-hidden rounded-md border border-border bg-popover shadow-xl text-sm"
        role="listbox"
      >
        <div className="flex items-center justify-between border-b border-border/60 px-3 py-2 text-[0.7rem] text-muted-foreground">
          <span>Slash commands</span>
          <span>↑↓ choose · Tab apply · Esc close</span>
        </div>
        <div className="max-h-56 overflow-y-auto">
          {items.map((it, i) => {
            const active = i === selected;

            return (
              <ListItem
                key={`${it.text}-${i}`}
                active={active}
                role="option"
                aria-selected={active}
                onMouseEnter={() => setSelected(i)}
                onClick={() => apply(it)}
                className="px-3 py-1.5"
              >
                <ChevronRight
                  className={`h-3 w-3 shrink-0 ${active ? "text-primary" : "text-transparent"}`}
                />

                <span className="font-mono text-xs shrink-0 truncate">
                  {it.display}
                </span>

                {it.meta && (
                  <span className="text-[0.7rem] text-muted-foreground/70 truncate ml-auto">
                    {it.meta}
                  </span>
                )}
              </ListItem>
            );
          })}
        </div>
      </div>
    );
  },
);
