import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import type { CognitiveStateOutput } from "../lib/types";
import { HeuristicBadge, ModelBadge } from "./ui";

function Spark({ values, color }: { values: number[]; color: string }) {
  return (
    <div className="h-10">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={values.map((v, i) => ({ i, v }))} margin={{ top: 4, bottom: 4, left: 0, right: 0 }}>
          <YAxis hide domain={["auto", "auto"]} />
          <Line dataKey="v" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const HEUR: { key: "attention" | "fatigue" | "readiness"; title: string; desc: string }[] = [
  { key: "attention", title: "Attention", desc: "engagement β/(α+θ)" },
  { key: "fatigue", title: "Fatigue", desc: "(θ+α)/β, posterior" },
  { key: "readiness", title: "Learning readiness", desc: "0.5·E/(1+E) + 0.5/(1+F)" },
];

export default function StateCards({ outputs }: { outputs: CognitiveStateOutput[] }) {
  const last = outputs[outputs.length - 1];
  const w = last?.states.workload;
  const classes = w ? Object.keys(w.probabilities) : [];
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <div className="rounded-lg border border-accent/40 bg-surface p-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold">Workload</h3>
          <ModelBadge />
        </div>
        {w ? (
          <>
            <div className="mt-2 text-3xl font-semibold capitalize" style={{ color: w.label === classes[classes.length - 1] ? "var(--series-high)" : "var(--series-low)" }}>
              {w.label}
            </div>
            <div className="num mt-1 text-xs text-ink-3">confidence {(w.confidence * 100).toFixed(0)}%</div>
            <div className="mt-3 space-y-1">
              {classes.map((c, i) => (
                <div key={c} className="flex items-center gap-2 text-xs">
                  <span className="w-16 text-ink-2">{c}</span>
                  <div className="h-2 flex-1 rounded bg-surface-2">
                    <div className="h-2 rounded" style={{ width: `${w.probabilities[c] * 100}%`, background: i === classes.length - 1 ? "var(--series-high)" : "var(--series-low)" }} />
                  </div>
                  <span className="num w-10 text-right">{w.probabilities[c].toFixed(2)}</span>
                </div>
              ))}
            </div>
            <div className="mt-2 truncate text-[11px] text-ink-3" title={w.model_version}>{w.model_version}</div>
          </>
        ) : <div className="mt-3 text-sm text-ink-3">waiting for the first window…</div>}
      </div>
      {HEUR.map((h) => {
        const s = last?.states[h.key];
        return (
          <div key={h.key} className="rounded-lg border border-line bg-surface p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-sm font-semibold">{h.title}</h3>
            </div>
            <div className="mt-1"><HeuristicBadge /></div>
            <div className="num mt-2 text-2xl font-medium">{s ? s.value.toFixed(3) : "—"}</div>
            <div className="text-[11px] text-ink-3">{h.desc}</div>
            <Spark values={outputs.map((o) => o.states[h.key].value)} color="var(--text-2)" />
          </div>
        );
      })}
    </div>
  );
}
