// One stream for the whole app (Live Monitor and Brain View share it, and it survives page changes).
// Live: WebSocket /stream/{subject}/{condition}. Demo: replays public/demo/stream_*.json with the same
// message sequence and timing, so both modes look identical to the components.
import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { isDemo, wsUrl } from "./api";
import type { CognitiveStateOutput, StreamMessage, StreamMeta } from "./types";

export const DISPLAY_SECONDS = 10;

export interface RawBuffer {
  data: number[][]; // channels x samples (display rate)
  fs: number;
  tEnd: number; // signal time (s) of the last sample
  receivedAt: number; // performance.now() when the last chunk arrived
}

interface StreamState {
  status: "idle" | "connecting" | "streaming" | "ended" | "error";
  meta: StreamMeta | null;
  outputs: CognitiveStateOutput[];
  error: string | null;
  speed: number;
  raw: React.RefObject<RawBuffer | null>;
  rawVersion: number;
  start: (o: { subject: number; condition: string; model: string; speed: number }) => void;
  stop: () => void;
}

const Ctx = createContext<StreamState | null>(null);

export function StreamProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<StreamState["status"]>("idle");
  const [meta, setMeta] = useState<StreamMeta | null>(null);
  const [outputs, setOutputs] = useState<CognitiveStateOutput[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1);
  const [rawVersion, setRawVersion] = useState(0);
  const raw = useRef<RawBuffer | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const demoTimer = useRef<number | null>(null);
  const runId = useRef(0);

  const handle = useCallback((m: StreamMessage) => {
    if (m.type === "meta") {
      setMeta(m);
      setStatus("streaming");
    } else if (m.type === "raw") {
      const prev = raw.current;
      const maxLen = DISPLAY_SECONDS * m.fs + m.fs * 4;
      const data = m.data.map((ch, i) => {
        const merged = (prev?.data[i] ?? []).concat(ch);
        return merged.length > maxLen ? merged.slice(merged.length - maxLen) : merged;
      });
      raw.current = { data, fs: m.fs, tEnd: m.t_start + m.data[0].length / m.fs, receivedAt: performance.now() };
      setRawVersion((v) => v + 1);
    } else if (m.type === "state") {
      setOutputs((o) => [...o, m.output]);
    } else if (m.type === "end") {
      setStatus("ended");
    } else if (m.type === "error") {
      setError(m.message);
      setStatus("error");
    }
  }, []);

  const stop = useCallback(() => {
    runId.current += 1;
    ws.current?.close();
    ws.current = null;
    if (demoTimer.current) window.clearTimeout(demoTimer.current);
    demoTimer.current = null;
    setStatus((s) => (s === "streaming" || s === "connecting" ? "ended" : s));
  }, []);

  const start = useCallback(
    ({ subject, condition, model, speed }: { subject: number; condition: string; model: string; speed: number }) => {
      stop();
      const id = ++runId.current;
      setOutputs([]);
      setMeta(null);
      setError(null);
      setSpeed(speed);
      raw.current = null;
      setStatus("connecting");
      if (isDemo()) {
        const key = `sub${String(subject).padStart(2, "0")}_${condition}`;
        fetch(`demo/stream_${key}.json`)
          .then((r) => {
            if (!r.ok) throw new Error(`No offline recording for ${key}. Demo mode includes only a few recordings.`);
            return r.json();
          })
          .then((msgs: StreamMessage[]) => {
            let i = 0;
            const tick = () => {
              if (id !== runId.current) return;
              // send messages up to and including the next "state", then wait one step
              while (i < msgs.length) {
                const m = msgs[i++];
                handle(m.type === "meta" ? { ...m, speed } : m);
                if (m.type === "state") break;
              }
              if (i < msgs.length) demoTimer.current = window.setTimeout(tick, 2000 / speed);
            };
            tick();
          })
          .catch((e: Error) => {
            setError(e.message);
            setStatus("error");
          });
        return;
      }
      const sock = new WebSocket(wsUrl(subject, condition, model, speed));
      ws.current = sock;
      sock.onmessage = (ev) => id === runId.current && handle(JSON.parse(ev.data) as StreamMessage);
      sock.onerror = () => {
        if (id !== runId.current) return;
        setError("WebSocket error — is the API running on port 8000? Switch on demo mode to run offline.");
        setStatus("error");
      };
      sock.onclose = () => id === runId.current && setStatus((s) => (s === "streaming" ? "ended" : s));
    },
    [handle, stop],
  );

  useEffect(() => () => stop(), [stop]);

  return (
    <Ctx.Provider value={{ status, meta, outputs, error, speed, raw, rawVersion, start, stop }}>{children}</Ctx.Provider>
  );
}

export function useStream() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useStream outside StreamProvider");
  return c;
}
