import { CHANNELS } from "../lib/eeg";
import type { CognitiveStateOutput } from "../lib/types";
import { FlagPill } from "./ui";

export default function QualityIndicator({ output }: { output?: CognitiveStateOutput }) {
  const q = output?.signal_quality;
  const bad = new Set(q?.bad_channels ?? []);
  const interp = new Set(q?.interpolated_channels ?? []);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs text-ink-3">Signal quality</span>
        {q ? <FlagPill flag={q.flag} /> : <span className="text-xs text-ink-3">—</span>}
        {q && <span className="num text-xs text-ink-2">{(q.overall * 100).toFixed(0)}% channels OK</span>}
      </div>
      <div className="flex flex-wrap gap-1" aria-label="Channel status">
        {CHANNELS.map((c) => (
          <span key={c}
            title={bad.has(c) ? `${c}: failing a quality check in this window` : interp.has(c) ? `${c}: interpolated (persistently bad)` : `${c}: OK`}
            className={`num rounded border px-1.5 py-0.5 text-[11px] ${bad.has(c) ? "border-bad/50 bg-bad-soft text-bad" : interp.has(c) ? "border-warn/50 bg-warn-soft text-warn" : "border-line text-ink-3"}`}>
            {c}{interp.has(c) ? "*" : ""}{bad.has(c) ? " ✕" : ""}
          </span>
        ))}
      </div>
    </div>
  );
}
