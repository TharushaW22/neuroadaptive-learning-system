// 14-electrode head map (10-20 positions, nose up) coloured by band power (log scale, per window).
// simple = plain "weak -> strong" legend with no numbers or units (default viva view).
import { CHANNELS, POS, SEQ, seqColor } from "../lib/eeg";

export default function HeadMap({ values, unit = "log10 µV²", size = 320, highlight = [], simple = false }: {
  values: Record<string, number> | null; unit?: string; size?: number; highlight?: string[]; simple?: boolean;
}) {
  const R = size * 0.42;
  const c = size / 2;
  const xy = (ch: string) => [c + POS[ch][0] * R, c - POS[ch][1] * R] as const;
  const vals = values ? CHANNELS.map((ch) => values[ch]) : [];
  const lo = vals.length ? Math.min(...vals) : 0;
  const hi = vals.length ? Math.max(...vals) : 1;
  const norm = (v: number) => (hi - lo < 1e-9 ? 0.5 : (v - lo) / (hi - lo));
  const hl = new Set(highlight);
  return (
    <div className="flex flex-col items-center">
      <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ maxWidth: size }} role="img" aria-label="Head map of band power">
        {/* head, nose, ears */}
        <path d={`M ${c - 14} ${c - R - 2} L ${c} ${c - R - 18} L ${c + 14} ${c - R - 2}`} fill="none" stroke="var(--text-3)" strokeWidth={1.5} />
        <ellipse cx={c - R - 6} cy={c} rx={8} ry={22} fill="none" stroke="var(--text-3)" strokeWidth={1.5} />
        <ellipse cx={c + R + 6} cy={c} rx={8} ry={22} fill="none" stroke="var(--text-3)" strokeWidth={1.5} />
        <circle cx={c} cy={c} r={R} fill="var(--surface-2)" stroke="var(--text-3)" strokeWidth={1.5} />
        {CHANNELS.map((ch) => {
          const [x, y] = xy(ch);
          const v = values?.[ch];
          return (
            <g key={ch}>
              <circle cx={x} cy={y} r={size * 0.055} fill={v === undefined ? "var(--surface)" : seqColor(norm(v))}
                stroke={hl.has(ch) ? "var(--bad)" : "var(--surface)"} strokeWidth={hl.has(ch) ? 3 : 2}>
                <title>{simple ? ch : `${ch}: ${v === undefined ? "—" : v.toFixed(2)} ${unit}`}</title>
              </circle>
              <text x={x} y={y + 3.5} textAnchor="middle" fontSize={size * 0.032} fontWeight={600}
                fill={v === undefined ? "var(--text)" : norm(v) > 0.55 ? "#fff" : "#16161a"} style={{ pointerEvents: "none" }}>{ch}</text>
            </g>
          );
        })}
      </svg>
      {simple ? (
        <div className="mt-3 flex items-center gap-2 text-sm text-ink-3">
          <span>weak</span>
          <div className="flex h-3 w-48 overflow-hidden rounded">{SEQ.map((s) => <div key={s} className="flex-1" style={{ background: s }} />)}</div>
          <span>strong</span>
        </div>
      ) : (
        <div className="mt-2 flex items-center gap-2 text-xs text-ink-3">
          <span className="num">{values ? lo.toFixed(2) : "—"}</span>
          <div className="flex h-2 w-40 overflow-hidden rounded">{SEQ.map((s) => <div key={s} className="flex-1" style={{ background: s }} />)}</div>
          <span className="num">{values ? hi.toFixed(2) : "—"}</span>
          <span>{unit} (scaled per window)</span>
        </div>
      )}
    </div>
  );
}
