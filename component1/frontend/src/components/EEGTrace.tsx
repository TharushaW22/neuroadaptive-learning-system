// Scrolling 14-channel EEG trace on a canvas. The display runs 2 s behind the newest data so the
// traces can scroll smoothly between the 2-s chunks the stream delivers.
import { useEffect, useRef } from "react";
import { CHANNELS } from "../lib/eeg";
import { DISPLAY_SECONDS, useStream } from "../lib/stream";

export default function EEGTrace({ badChannels = [], interpolated = [], height = 380, simple = false }: {
  badChannels?: string[]; interpolated?: string[]; height?: number; simple?: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const { raw, speed, status } = useStream();
  const bad = useRef(new Set<string>());
  const interp = useRef(new Set<string>());
  bad.current = new Set(badChannels);
  interp.current = new Set(interpolated);

  useEffect(() => {
    let frame = 0;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const c = canvas.current;
      if (!c) return;
      const dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth;
      const h = c.clientHeight;
      if (c.width !== w * dpr || c.height !== h * dpr) {
        c.width = w * dpr;
        c.height = h * dpr;
      }
      const ctx = c.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const css = getComputedStyle(document.documentElement);
      const col = (n: string) => css.getPropertyValue(n).trim();
      ctx.clearRect(0, 0, w, h);
      const labelW = 44;
      const plotW = w - labelW - 8;
      const rowH = h / CHANNELS.length;

      // channel bands + labels
      CHANNELS.forEach((ch, i) => {
        const y = i * rowH;
        if (bad.current.has(ch)) {
          ctx.fillStyle = col("--bad-soft");
          ctx.fillRect(labelW, y, plotW, rowH);
        }
        ctx.fillStyle = bad.current.has(ch) ? col("--bad") : col("--text-2");
        ctx.font = "500 12px IBM Plex Mono, monospace";
        ctx.textBaseline = "middle";
        ctx.fillText(ch + (interp.current.has(ch) ? "*" : ""), 4, y + rowH / 2);
      });
      // grid: one line per second
      const buf = raw.current;
      ctx.strokeStyle = col("--grid");
      ctx.lineWidth = 1;
      const rightT = buf ? buf.tEnd - 2 + Math.min(((performance.now() - buf.receivedAt) / 1000) * speed, 2) : DISPLAY_SECONDS;
      const leftT = rightT - DISPLAY_SECONDS;
      const xOf = (t: number) => labelW + ((t - leftT) / DISPLAY_SECONDS) * plotW;
      for (let s = Math.ceil(leftT); s <= rightT; s++) {
        const x = xOf(s);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
        if (s % 2 === 0) {
          ctx.fillStyle = col("--text-3");
          ctx.font = "11px IBM Plex Mono, monospace";
          ctx.fillText(`${s}s`, x + 2, h - 6);
        }
      }
      if (!buf) return;
      // traces: fixed scale, ±75 uV per row (clipped), so amplitudes are comparable across channels
      const scale = rowH / 2 / 75;
      const n = buf.data[0].length;
      const t0 = buf.tEnd - n / buf.fs;
      ctx.lineWidth = 1;
      buf.data.forEach((ch, i) => {
        const mid = i * rowH + rowH / 2;
        ctx.strokeStyle = bad.current.has(CHANNELS[i]) ? col("--bad") : col("--trace");
        ctx.beginPath();
        let started = false;
        for (let k = 0; k < n; k++) {
          const t = t0 + k / buf.fs;
          if (t < leftT || t > rightT) continue;
          const v = Math.max(-75, Math.min(75, ch[k]));
          const x = xOf(t);
          const y = mid - v * scale;
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [raw, speed]);

  return (
    <div>
      <canvas ref={canvas} style={{ width: "100%", height }} aria-label="Scrolling 14-channel EEG trace" role="img" />
      {simple ? (
        status === "idle" && <div className="mt-1 text-sm text-ink-3">Press Start to play a recording.</div>
      ) : (
        <div className="mt-1 flex flex-wrap justify-between gap-2 text-xs text-ink-3">
          <span>Filtered 0.5–45 Hz, shown at 64 Hz · each row ±75 µV (clipped) · grid = 1 s</span>
          <span>{status === "idle" ? "Press Start to stream a recording" : "red rows = bad channels in the current window · * = interpolated"}</span>
        </div>
      )}
    </div>
  );
}
