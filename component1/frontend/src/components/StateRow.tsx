// Live Monitor: the four cognitive states as four equal cards (same size, layout and type), each with a
// big decimal value, a sparkline of the last 30 windows, one plain sentence and a SOURCE badge saying
// where the number comes from. Above the row: the signal-quality light.
import type { ReactNode } from "react";
import { Line, LineChart, ResponsiveContainer, YAxis } from "recharts";
import type { CognitiveStateOutput, Flag } from "../lib/types";

const SPARK_WINDOWS = 30;

const QUALITY: Record<Flag, { text: string; color: string; soft: string }> = {
  good: { text: "Signal good", color: "var(--good)", soft: "var(--good-soft)" },
  warn: { text: "Some noise", color: "var(--warn)", soft: "var(--warn-soft)" },
  bad: { text: "Poor signal", color: "var(--bad)", soft: "var(--bad-soft)" },
};

export function QualityLight({ flag, score }: { flag?: Flag; score?: number }) {
  if (!flag) return <span className="text-base text-ink-3">Signal: —</span>;
  const q = QUALITY[flag];
  return (
    <span className="inline-flex items-center gap-2 rounded-full px-3 py-1 text-base font-medium" style={{ background: q.soft, color: q.color }}>
      <span className="inline-block h-3 w-3 rounded-full" style={{ background: q.color }} aria-hidden />
      {q.text}
      {score !== undefined && <span className="tabular-nums">({score.toFixed(2)})</span>}
    </span>
  );
}

function Spark({ values }: { values: number[] }) {
  return (
    <div className="h-12">
      {values.length > 1 && (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={values.map((v, i) => ({ i, v }))} margin={{ top: 4, bottom: 4, left: 0, right: 0 }}>
            <YAxis hide domain={["auto", "auto"]} />
            <Line dataKey="v" stroke="var(--text-2)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

function StateCard({ title, value, sub, formula, spark, sentence, source, detail }: {
  title: string; value: number | undefined; sub: ReactNode; formula: string; spark: number[]; sentence: string; source: string; detail?: string;
}) {
  return (
    <div className="flex h-full flex-col rounded-lg border border-line bg-surface p-5">
      <h3 className="text-base font-semibold text-ink">{title}</h3>
      <div className="mt-2 text-4xl font-semibold tabular-nums text-ink">{value === undefined ? "—" : value.toFixed(2)}</div>
      <div className="mt-1 min-h-[1.5rem] text-sm text-ink-2">{sub}</div>
      <div className="mt-1 text-xs leading-snug text-ink-3"><span className="font-semibold">Formula: </span>{formula}</div>
      <div className="mt-2"><Spark values={spark} /></div>
      <p className="mt-2 text-sm leading-relaxed text-ink-3">{sentence}</p>
      {detail && <p className="mt-2 break-words text-xs text-ink-3">{detail}</p>}
      <div className="mt-auto pt-4">
        <span className="inline-block rounded border border-line bg-surface-2 px-2 py-0.5 text-xs text-ink-2">
          <span className="font-semibold uppercase tracking-wide text-ink-3">Source </span>{source}
        </span>
      </div>
    </div>
  );
}

/** advanced = also show the exact formula / model version under each sentence. */
export default function StateRow({ outputs, advanced = false }: { outputs: CognitiveStateOutput[]; advanced?: boolean }) {
  const last = outputs[outputs.length - 1];
  const recent = outputs.slice(-SPARK_WINDOWS);
  const w = last?.states.workload;
  const classes = w ? Object.keys(w.probabilities) : [];
  const top = classes[classes.length - 1] ?? "high"; // highest workload level
  const pTop = (o: CognitiveStateOutput) => o.states.workload.probabilities[top] ?? 0;
  // the formula and badge name XGBoost only when that is the model actually running
  const isXgb = !w || w.model_version.startsWith("xgb");
  const workloadSource = isXgb ? "Trained model (XGBoost, tested on held-out people)" : `Trained model (${w.model_version})`;
  const workloadFormula = isXgb
    ? `Trained XGBoost model on 1,499 EEG features → probability of ${top.toUpperCase()} (0–1)`
    : `Trained ${w.model_version} model → probability of ${top.toUpperCase()} (0–1)`;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <QualityLight flag={last?.signal_quality.flag} score={last?.signal_quality.overall} />
        {!last && <span className="text-base text-ink-3">Press Start to begin.</span>}
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StateCard
          title="Workload"
          value={last ? pTop(last) : undefined}
          sub={w ? <>{w.label.toUpperCase()} · confidence <span className="tabular-nums">{w.confidence.toFixed(2)}</span></> : "—"}
          formula={workloadFormula}
          spark={recent.map(pTop)}
          sentence={`Chance that the brain is under ${top} workload right now (0 to 1).`}
          source={workloadSource}
          detail={advanced && w ? `P(${top}) · ${w.model_version} · task ${w.task}` : undefined}
        />
        <StateCard
          title="Attention"
          value={last?.states.attention.value}
          sub="higher = more engaged"
          formula="Engagement = β / (α + θ) — all 14 sensors"
          spark={recent.map((o) => o.states.attention.value)}
          sentence="How engaged the brain looks, from the balance of fast and slow brain rhythms."
          source="Published EEG index — Pope et al., 1995"
          detail={advanced ? last?.states.attention.formula : undefined}
        />
        <StateCard
          title="Fatigue"
          value={last?.states.fatigue.value}
          sub="higher = more tired"
          formula="Fatigue = (θ + α) / β — sensors P7, P8, O1, O2"
          spark={recent.map((o) => o.states.fatigue.value)}
          sentence="How tired the brain looks, from slow rhythms at the back of the head."
          source="Published EEG index — Jap et al., 2009"
          detail={advanced ? last?.states.fatigue.formula : undefined}
        />
        <StateCard
          title="Learning readiness"
          value={last?.states.readiness.value}
          sub="scale 0–1 · higher = more ready to learn"
          formula="Readiness = 0.5 × E/(1+E) + 0.5 × 1/(1+F),  E = engagement, F = fatigue"
          spark={recent.map((o) => o.states.readiness.value)}
          sentence="Attention and fatigue folded into one number between 0 and 1."
          source="Project-defined placeholder — not validated"
          detail={advanced ? last?.states.readiness.formula : undefined}
        />
      </div>
      <p className="text-xs text-ink-3">θ = theta (4–8 Hz), α = alpha (8–13 Hz), β = beta (13–30 Hz)</p>
    </div>
  );
}
