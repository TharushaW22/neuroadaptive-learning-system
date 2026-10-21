// Brain-network graph: electrodes as nodes, PLV above the threshold as edges (width/opacity = PLV).
// topK (default view) replaces the threshold: only the K strongest connections are drawn.
import { CHANNELS, POS } from "../lib/eeg";

export default function NetworkGraph({ matrix, channels = CHANNELS, threshold: fixedThreshold, size = 320, topK }: {
  matrix: number[][] | null; channels?: string[]; threshold: number; size?: number; topK?: number;
}) {
  // with topK, the threshold becomes the K-th largest connection value of this window
  let threshold = fixedThreshold;
  if (topK && matrix) {
    const all: number[] = [];
    for (let i = 0; i < channels.length; i++) for (let j = i + 1; j < channels.length; j++) all.push(matrix[i][j]);
    all.sort((a, b) => b - a);
    threshold = all[Math.min(topK, all.length) - 1] ?? fixedThreshold;
  }
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
  // largest drawn edge (scales line width/opacity in the top-K view); empty list -> threshold
  const maxW: number = edges.length ? Math.max(...edges.map((e) => e.w)) : threshold;
  return (
    <div className="flex flex-col items-center">
      <svg viewBox={`0 0 ${size} ${size}`} width="100%" style={{ maxWidth: size }} role="img" aria-label="Brain connectivity network">
        <circle cx={c} cy={c} r={R} fill="none" stroke="var(--border)" strokeWidth={1.5} />
        {edges.map((e) => {
          const [x1, y1] = xy(e.a);
          const [x2, y2] = xy(e.b);
          // top-K view: scale within the shown edges so even the weakest of them stays visible
          const t = (e.w - threshold) / Math.max(1e-6, (topK ? maxW : 1) - threshold);
          return (
            <line key={e.a + e.b} x1={x1} y1={y1} x2={x2} y2={y2} stroke="var(--accent)" strokeLinecap="round"
              strokeWidth={topK ? 1.5 + 4 * t : 1 + 3 * t} strokeOpacity={topK ? 0.5 + 0.5 * t : 0.3 + 0.6 * t}>
              <title>{topK ? `${e.a}–${e.b}` : `${e.a}–${e.b}: PLV ${e.w.toFixed(2)}`}</title>
            </line>
          );
        })}
        {channels.map((ch) => {
          const [x, y] = xy(ch);
          const r = size * (0.03 + 0.03 * (strength[ch] / maxS));
          return (
            <g key={ch}>
              <circle cx={x} cy={y} r={r} fill="var(--surface)" stroke="var(--accent)" strokeWidth={2}>
                <title>{topK ? ch : `${ch}: strength ${strength[ch].toFixed(2)}`}</title>
              </circle>
              <text x={x} y={y - r - 3} textAnchor="middle" fontSize={size * 0.03} fill="var(--text-2)">{ch}</text>
            </g>
          );
        })}
      </svg>
      {topK ? (
        <div className="mt-3 text-sm text-ink-3">The {topK} strongest connections right now · bigger circle = more connected sensor</div>
      ) : (
        <div className="mt-2 text-xs text-ink-3">{edges.length} of 91 edges with PLV ≥ {threshold.toFixed(2)} · node size = strength</div>
      )}
    </div>
  );
}
