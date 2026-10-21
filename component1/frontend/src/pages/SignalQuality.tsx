import { useMemo, useState } from "react";
import { Card, Explain, FlagPill, Notice, Segmented, Stat } from "../components/ui";
import { CHANNELS } from "../lib/eeg";
import type { QualityRow, QualityWindow } from "../lib/types";
import { useApi } from "../lib/useApi";

type SortKey = "recording" | "usable_fraction" | "bad" | "mean_score";

export default function SignalQuality() {
  const q = useApi<QualityRow[]>("/quality");
  const [sort, setSort] = useState<SortKey>("usable_fraction");
  const [filter, setFilter] = useState<"all" | "lo" | "hi" | "low">("all");
  const [sel, setSel] = useState<QualityRow | null>(null);
  const win = useApi<QualityWindow[]>(sel ? `/quality/${sel.subject_id}/${sel.condition}` : null);

  const rows = useMemo(() => {
    let r = [...(q.data ?? [])];
    if (filter === "lo" || filter === "hi") r = r.filter((x) => x.condition === filter);
    if (filter === "low") r = r.filter((x) => x.usable_fraction < 0.5);
    r.sort((a, b) => (sort === "recording" ? a.recording.localeCompare(b.recording) : sort === "bad" ? b.bad - a.bad : (a[sort] as number) - (b[sort] as number)));
    return r;
  }, [q.data, sort, filter]);

  const all = q.data ?? [];
  const tot = all.reduce((a, r) => ({ g: a.g + r.good, w: a.w + r.warn, b: a.b + r.bad }), { g: 0, w: 0, b: 0 });
  const n = tot.g + tot.w + tot.b || 1;

  return (
    <div className="space-y-4">
      <Card title="Signal quality" subtitle="After 0.5–45 Hz filtering and interpolation of persistently bad channels. Windows are flagged, never deleted.">
        {q.error && <Notice kind="error">{q.error}</Notice>}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
          <Stat label="Recordings" value={all.length} />
          <Stat label="Good windows" value={`${((tot.g / n) * 100).toFixed(1)}%`} sub="no channel failing" />
          <Stat label="Warn windows" value={`${((tot.w / n) * 100).toFixed(1)}%`} sub="blinks or ≤ 2 bad channels" />
          <Stat label="Bad windows" value={`${((tot.b / n) * 100).toFixed(1)}%`} sub="> 2 hard-bad channels" />
          <Stat label="Interpolated channels" value={all.reduce((a, r) => a + r.interpolated.length, 0)}
            sub={`in ${all.filter((r) => r.interpolated.length).length} recordings`} />
        </div>
        <Explain>How clean the headset signal is across all 96 recordings. Noisy stretches are marked, never thrown away.</Explain>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-3" title="Per recording"
          right={<div className="flex flex-wrap gap-2">
            <Segmented value={filter} onChange={setFilter} options={[{ value: "all", label: "all" }, { value: "lo", label: "rest" }, { value: "hi", label: "task" }, { value: "low", label: "< 50% usable" }]} />
            <Segmented value={sort} onChange={setSort} options={[{ value: "usable_fraction", label: "usable ↑" }, { value: "bad", label: "bad ↓" }, { value: "recording", label: "name" }]} />
          </div>}>
          <div className="max-h-[560px] overflow-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface text-left text-xs text-ink-3">
                <tr><th className="py-1.5">Recording</th><th>Good</th><th>Warn</th><th>Bad</th><th>Usable</th><th>Interpolated</th></tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.recording} onClick={() => setSel(r)}
                    className={`cursor-pointer border-t border-line hover:bg-surface-2 ${sel?.recording === r.recording ? "bg-accent-soft" : ""}`}>
                    <td className="num py-1.5">{r.recording}</td>
                    <td className="num">{r.good}</td><td className="num">{r.warn}</td><td className="num">{r.bad}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-16 rounded bg-surface-2"><div className="h-1.5 rounded" style={{ width: `${r.usable_fraction * 100}%`, background: r.usable_fraction < 0.5 ? "var(--bad)" : "var(--good)" }} /></div>
                        <span className="num text-xs">{(r.usable_fraction * 100).toFixed(0)}%</span>
                      </div>
                    </td>
                    <td className="num text-xs" title={Object.entries(r.interpolation_reasons).map(([c, why]) => `${c}: ${why.join("+")}`).join("\n")}>
                      {r.interpolated.join(", ") || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Explain>One row per recording: how many 4-second pieces were clean, a bit noisy or bad. Click a row to see it in detail.</Explain>
        </Card>

        <Card className="xl:col-span-2" title={sel ? `Windows of ${sel.recording}` : "Flagged windows"}
          subtitle={sel ? "Each column is a 4-s window (2-s step); red cells = channel failing a check" : "Select a recording on the left"}>
          {win.data && (
            <>
              <div className="overflow-x-auto">
                <svg viewBox={`0 0 ${win.data.length * 6 + 40} ${CHANNELS.length * 10 + 22}`} width="100%" role="img" aria-label="Channel by window quality grid">
                  {CHANNELS.map((ch, i) => <text key={ch} x={0} y={i * 10 + 18} fontSize={7} fill="var(--text-3)">{ch}</text>)}
                  {win.data.map((w, j) => (
                    <g key={j}>
                      <rect x={36 + j * 6} y={0} width={5} height={6} fill={`var(--${w.flag})`}><title>{`${w.t_start}-${w.t_end}s ${w.flag}`}</title></rect>
                      {CHANNELS.map((ch, i) => (
                        <rect key={ch} x={36 + j * 6} y={i * 10 + 10} width={5} height={8}
                          fill={w.bad_channels.includes(ch) ? "var(--bad)" : "var(--surface-2)"} />
                      ))}
                    </g>
                  ))}
                </svg>
              </div>
              <div className="mt-3 max-h-64 overflow-auto text-xs">
                {win.data.filter((w) => w.flag !== "good").slice(0, 40).map((w) => (
                  <div key={w.window_index} className="flex items-center gap-2 border-t border-line py-1">
                    <span className="num w-20">{w.t_start}–{w.t_end}s</span><FlagPill flag={w.flag} />
                    <span className="num text-ink-3">{w.bad_channels.join(" ")}</span>
                  </div>
                ))}
              </div>
            </>
          )}
          {win.error && <Notice kind="error">{win.error}</Notice>}
          <Explain>Time runs left to right; each row is one sensor. Red marks a sensor that was noisy at that moment.</Explain>
        </Card>
      </div>
    </div>
  );
}
