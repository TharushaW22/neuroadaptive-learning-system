// Brain-network graph: electrodes as nodes, PLV above the threshold as edges (width/opacity = PLV).
import { CHANNELS, POS } from "../lib/eeg";

export default function NetworkGraph({ matrix, channels = CHANNELS, threshold, size = 320 }: {
  matrix: number[][] | null; channels?: string[]; threshold: number; size?: number;
}) {
  const R = size * 0.42;
  const c = size / 2;
  const xy = (ch: string) => [c + POS[ch][0] * R, c - POS[ch][1] * R] as const;
  const edges: { a: string; b: string; w: number }[] = [];
  if (matrix) {
    for (let i = 0; i < channels.length; i++)
      for (let j = i + 1; j < channels.length; j++)
        if (matrix[i][j] >= threshold) edges.push({ a: channels[i], b: channels[j], w: matrix[i][j] });
  }
  const strength = Object.fromEntries(channels.map((ch, i) => [ch, matrix ? matrix[i].reduce((s, v, j) => (j !== i && v >= threshold ? s + v : s), 0) : 0]));
  const maxS = Math.max(1e-6, ...Object.values(strength));
  return (
    <div className="flex flex-col items-center">
      <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ maxWidth: size }} role="img" aria-label="Brain connectivity network">
        <circle cx={c} cy={c} r={R} fill="none" stroke="var(--border)" strokeWidth={1.5} />
        {edges.map((e) => {
          const [x1, y1] = xy(e.a);
          const [x2, y2] = xy(e.b);
          const t = (e.w - threshold) / Math.max(1e-6, 1 - threshold);
          return (
            <line key={e.a + e.b} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--accent)" strokeLinecap="round"
              strokeWidth={1 + 3 * t} strokeOpacity={0.3 + 0.6 * t}>
              <title>{`${e.a}–${e.b}: PLV ${e.w.toFixed(2)}`}</title>
            </line>
          );
        })}
        {channels.map((ch) => {
          const [x, y] = xy(ch);
          const r = size * (0.03 + 0.03 * (strength[ch] / maxS));
          return (
            <g key={ch}>
              <circle cx={x} cy={y} r={r} fill="var(--surface)" stroke="var(--accent)" strokeWidth={2}>
                <title>{`${ch}: strength ${strength[ch].toFixed(2)}`}</title>
              </circle>
              <text x={x} y={y - r - 3} textAnchor="middle" fontSize={size * 0.03} fill="var(--text-2)">{ch}</text>
            </g>
          );
        })}
      </svg>
      <div className="mt-2 text-[11px] text-ink-3">{edges.length} of 91 edges with PLV ≥ {threshold.toFixed(2)} · node size = strength</div>
    </div>
  );
}
