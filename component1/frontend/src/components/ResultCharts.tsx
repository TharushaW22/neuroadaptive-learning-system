import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const tooltipStyle = { background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 };
const axis = { tick: { fill: "var(--text-3)", fontSize: 11 }, stroke: "var(--border)" };

/** Horizontal bars of one metric with ± std error bars. */
export function MetricBars({ rows, height = 260, domain = [0, 1] as [number, number] }: {
  rows: { name: string; value: number | null; err?: number | null; color?: string }[]; height?: number; domain?: [number, number];
}) {
  const data = rows.map((r) => ({ ...r, value: r.value ?? 0, err: r.err ?? 0 }));
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }} barCategoryGap={6}>
          <CartesianGrid stroke="var(--grid)" horizontal={false} />
          <XAxis type="number" domain={domain} {...axis} />
          <YAxis type="category" dataKey="name" width={170} {...axis} />
          <ReferenceLine x={0.5} stroke="var(--text-3)" strokeDasharray="3 3" />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: number, _n, p) => [`${v.toFixed(3)}${p.payload.err ? ` ± ${p.payload.err.toFixed(3)}` : ""}`, "value"]} />
          <Bar dataKey="value" fill="var(--accent)" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            <ErrorBar dataKey="err" width={4} stroke="var(--text-2)" direction="x" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Reliability diagram: accuracy vs confidence before/after temperature scaling. */
export function Reliability({ before, after, eceB, eceA, height = 260 }: {
  before: { confidence: number[]; accuracy: number[] }; after: { confidence: number[]; accuracy: number[] };
  eceB: number; eceA: number; height?: number;
}) {
  const pts = (b: { confidence: number[]; accuracy: number[] }) => b.confidence.map((c, i) => ({ c, a: b.accuracy[i] }));
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart margin={{ top: 8, right: 16, bottom: 16, left: -8 }}>
          <CartesianGrid stroke="var(--grid)" />
          <XAxis type="number" dataKey="c" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} {...axis}
            label={{ value: "confidence", position: "insideBottom", offset: -8, fill: "var(--text-3)", fontSize: 11 }} />
          <YAxis type="number" domain={[0, 1]} ticks={[0, 0.5, 1]} {...axis} />
          <Line data={[{ c: 0, a: 0 }, { c: 1, a: 1 }]} dataKey="a" stroke="var(--text-3)" strokeDasharray="4 4" dot={false} name="perfect" isAnimationActive={false} />
          <Line data={pts(before)} dataKey="a" stroke="var(--series-low)" strokeWidth={2} dot={{ r: 3 }} name={`before (ECE ${eceB.toFixed(3)})`} isAnimationActive={false} />
          <Line data={pts(after)} dataKey="a" stroke="var(--series-high)" strokeWidth={2} dot={{ r: 3 }} name={`after T-scaling (ECE ${eceA.toFixed(3)})`} isAnimationActive={false} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => v.toFixed(3)} labelFormatter={(v) => `confidence ${Number(v).toFixed(2)}`} />
          <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Confusion matrix as a grid with counts and row-normalised shading. */
export function Confusion({ m, classes, title }: { m: number[][]; classes: string[]; title?: string }) {
  return (
    <div>
      {title && <div className="mb-1 text-xs text-ink-3">{title}</div>}
      <table className="num text-xs">
        <thead>
          <tr>
            <th className="p-1 text-left font-normal text-ink-3">true \ pred</th>
            {classes.map((c) => <th key={c} className="p-1 font-medium text-ink-2">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {m.map((row, i) => {
            const tot = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={i}>
                <th className="p-1 text-left font-medium text-ink-2">{classes[i]}</th>
                {row.map((v, j) => {
                  const f = v / tot;
                  return (
                    <td key={j} className="h-11 w-16 rounded text-center"
                      style={{ background: `color-mix(in srgb, var(--accent) ${Math.round(f * 85)}%, var(--surface-2))`, color: f > 0.55 ? "#fff" : "var(--text)" }}>
                      {v}<div className="text-[10px] opacity-80">{(f * 100).toFixed(0)}%</div>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
