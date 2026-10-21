// Probability of the higher-workload class over time, with a confidence band.
// simple = probability line only, shown as a percentage (default viva view).
// Band = smoothed probability ± 1 SD of the last 5 per-window (unsmoothed) predictions: it widens when
// consecutive windows disagree, i.e. when the estimate is unstable.
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CognitiveStateOutput } from "../lib/types";

export default function WorkloadTimeline({ outputs, classes, height = 220, simple = false }: {
  outputs: CognitiveStateOutput[]; classes: string[]; height?: number; simple?: boolean;
}) {
  const hi = classes[classes.length - 1];
  const data = outputs.map((o, i) => {
    const recent = outputs.slice(Math.max(0, i - 4), i + 1).map((x) => x.states.workload.raw_probabilities[hi]);
    const m = recent.reduce((a, b) => a + b, 0) / recent.length;
    const sd = Math.sqrt(recent.reduce((a, b) => a + (b - m) ** 2, 0) / recent.length);
    const p = o.states.workload.probabilities[hi];
    return {
      t: o.t_end_s,
      p,
      raw: o.states.workload.raw_probabilities[hi],
      band: [Math.max(0, p - sd), Math.min(1, p + sd)] as [number, number],
      flag: o.signal_quality.flag,
    };
  });
  return (
    <div style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <CartesianGrid stroke="var(--grid)" vertical={false} />
          <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} tickFormatter={(v) => `${v}s`}
            tick={{ fill: "var(--text-3)", fontSize: 11 }} stroke="var(--border)" />
          <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tick={{ fill: "var(--text-3)", fontSize: simple ? 13 : 11 }} stroke="var(--border)"
            width={simple ? 56 : 60} tickFormatter={simple ? (v: number) => `${Math.round(v * 100)} %` : undefined} />
          <ReferenceLine y={0.5} stroke="var(--text-3)" strokeDasharray="3 3" />
          {!simple && <Area dataKey="band" stroke="none" fill="var(--series-high)" fillOpacity={0.15} isAnimationActive={false} />}
          {!simple && <Line dataKey="raw" stroke="var(--text-3)" strokeWidth={1} dot={false} strokeDasharray="2 3" isAnimationActive={false} />}
          <Line dataKey="p" stroke="var(--series-high)" strokeWidth={simple ? 3 : 2} dot={false} isAnimationActive={false} />
          <Tooltip
            contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }}
            labelFormatter={(v) => (simple ? `at ${v} s` : `window ending ${v} s`)}
            formatter={(v, name) => {
              if (simple) return [`${Math.round((v as number) * 100)} %`, `chance of ${hi} workload`];
              if (name === "band") return [`${(v as number[]).map((x) => x.toFixed(2)).join(" – ")}`, "band"];
              return [(v as number).toFixed(3), name === "p" ? `P(${hi}) smoothed` : `P(${hi}) this window`];
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
