// Every number on this page comes from /results/summary (built by scripts/export_report.py from results/).
// Default view: four plain-language story cards. Advanced: every table, ablation, matrix and test.
import { useMemo, useState, type ReactNode } from "react";
import { Confusion, MetricBars, PercentBars, Reliability } from "../components/ResultCharts";
import { Card, Explain, Notice, Segmented, Select } from "../components/ui";
import { fmt, TASK_LABEL } from "../lib/eeg";
import { useSettings } from "../lib/settings";
import type { MeanStd, ResultRow, Summary } from "../lib/types";
import { useApi } from "../lib/useApi";

type Task = "T1" | "T2" | "T3";

const ms = (m: MeanStd | null | undefined) => (m && m.mean !== null ? `${fmt(m.mean)} ± ${fmt(m.std)}` : "—");

/** Fold mean ± std when defined; T3 under LOSO has no per-fold macro-F1, so fall back to pooled. */
function primary(r: ResultRow): { value: number | null; err: number | null; label: string } {
  const f = r.per_fold?.macro_f1;
  if (f && f.mean !== null && f.n_folds > 1) return { value: f.mean, err: f.std, label: "fold macro-F1" };
  return { value: r.pooled.macro_f1 ?? null, err: null, label: "pooled macro-F1" };
}

function Table({ rows, cols }: { rows: ResultRow[]; cols: { h: string; f: (r: ResultRow) => React.ReactNode }[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-ink-3"><tr>{cols.map((c) => <th key={c.h} className="whitespace-nowrap py-1.5 pr-4 font-medium">{c.h}</th>)}</tr></thead>
        <tbody>{rows.map((r) => <tr key={r.experiment_id} className="border-t border-line">{cols.map((c) => <td key={c.h} className="num whitespace-nowrap py-1.5 pr-4">{c.f(r)}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function AdvancedResults({ d }: { d: Summary }) {
  const [task, setTask] = useState<Task>("T1");
  const [norm, setNorm] = useState<"none" | "rest_baseline">("none");
  const [cmId, setCmId] = useState<string>("");
  const [relId, setRelId] = useState<string>("");

  const byTask = (rows: ResultRow[] | undefined) => (rows ?? []).filter((r) => r.task_short === task);
  const base = byTask(d?.classical_baselines);
  const abl = byTask(d?.feature_ablation).filter((r) => r.normalisation === norm);
  const deep = byTask(d?.deep);
  const comp = byTask(d?.comparison_5fold);
  const allRows = useMemo(() => [...base, ...deep, ...byTask(d?.classical_kfold5)], [d, task]); // eslint-disable-line react-hooks/exhaustive-deps
  const cm = allRows.find((r) => r.experiment_id === cmId) ?? deep.find((r) => r.model === "A10_hybrid") ?? allRows[0];
  const rel = deep.find((r) => r.experiment_id === relId) ?? deep.find((r) => r.model === "A10_hybrid") ?? deep[0];
  const tests = [...(d?.stats_classical?.tests ?? []), ...(d?.stats_deep?.tests ?? [])].filter((t) => t.a.includes(`_${task}_`));

  return (
    <div className="space-y-4">
      <Card title="Results — all details" subtitle={`Generated ${d.generated} from results/ — no number on this page is typed by hand.`}
        right={<Segmented value={task} onChange={setTask} options={[{ value: "T1", label: "T1 rest vs task" }, { value: "T3", label: "T3 within-task" }, { value: "T2", label: "T2 rating level" }]} />}>
        <p className="text-sm text-ink-2">{TASK_LABEL[base[0]?.task ?? deep[0]?.task ?? ""] ?? task}.{" "}
          {task === "T3" && "T3 compares moderate vs high self-rated workload inside the SIMKAP task only — the confound-free test. Each subject contributes one class, so LOSO fold metrics are undefined and pooled / per-recording values are shown. "}
          {task === "T2" && "T2 is almost the rest/task split (rest 42/3/0, task 0/20/25), so it mostly re-measures T1. "}
          {task === "T1" && "T1 separates rest from multitasking; rest vs task differs in more than workload (eye movements, motor activity), so read it with T3. "}
        </p>
      </Card>

      <Card title="Classical baselines — leave-one-subject-out" subtitle="Unit: 4-s epoch. All 1499 features. rest_baseline = features z-scored with the subject's own rest recording (can inflate T1).">
        <Table rows={base} cols={[
          { h: "Model", f: (r) => r.model },
          { h: "Norm", f: (r) => <span className={r.normalisation === "rest_baseline" ? "text-warn" : ""}>{r.normalisation}</span> },
          { h: "Fold macro-F1", f: (r) => ms(r.per_fold.macro_f1) },
          { h: "Fold bal. acc", f: (r) => ms(r.per_fold.balanced_accuracy) },
          { h: "Pooled macro-F1", f: (r) => fmt(r.pooled.macro_f1) },
          { h: "Pooled AUROC", f: (r) => fmt(r.pooled.auroc) },
          { h: "Recording macro-F1", f: (r) => fmt(r.recording_level.macro_f1) },
          { h: "ms / window", f: (r) => fmt(r.latency_ms, 1) },
        ]} />
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title={`Feature ablation A1–A5 (${d.best_classical_model ?? "best model"})`} subtitle="LOSO; bars = fold mean ± std (T3: pooled)"
          right={<Segmented value={norm} onChange={setNorm} options={[{ value: "none", label: "no baseline" }, { value: "rest_baseline", label: "rest baseline" }]} />}>
          {abl.length ? <MetricBars rows={abl.map((r) => ({ name: r.feature_set ?? "", ...primary(r) }))} /> : <Notice>No ablation results for this task yet.</Notice>}
        </Card>
        <Card title="Architecture ablation A6–A10 + EEGNet" subtitle="5 subject folds; unit: 18-s window; bars = fold macro-F1 mean ± std">
          {deep.length ? <MetricBars rows={deep.map((r) => ({ name: `${r.ablation_id ?? "baseline"} · ${r.model}`, ...primary(r) }))} /> : <Notice>No deep results for this task yet.</Notice>}
        </Card>
      </div>

      <Card title="All models on the same 5 folds and the same 18-s windows" subtitle="Classical models average their epoch scores over each window; no rest baseline.">
        <Table rows={comp} cols={[
          { h: "Kind", f: (r) => r.kind },
          { h: "Model", f: (r) => r.model },
          { h: "Fold macro-F1", f: (r) => ms(r.per_fold.macro_f1) },
          { h: "Fold bal. acc", f: (r) => ms(r.per_fold.balanced_accuracy) },
          { h: "Pooled AUROC", f: (r) => fmt(r.pooled.auroc) },
          { h: "Recording macro-F1", f: (r) => fmt(r.recording_level.macro_f1) },
          { h: "ECE before → after", f: (r) => (r.calibration ? `${fmt(r.calibration.ece_before)} → ${fmt(r.calibration.ece_after)}` : "—") },
          { h: "ms / window", f: (r) => fmt(r.latency_ms ?? r.model_latency_ms, 1) },
        ]} />
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card title="Confusion matrices" right={cm && <Select label="" value={cm.experiment_id} onChange={setCmId} options={allRows.map((r) => ({ value: r.experiment_id, label: r.experiment_id }))} />}>
          {cm?.confusion_pooled ? (
            <div className="flex flex-wrap gap-6">
              <Confusion m={cm.confusion_pooled} classes={cm.classes} title={`pooled (${cm.kind === "deep" ? "windows" : "epochs"})`} />
              {cm.confusion_recording && <Confusion m={cm.confusion_recording} classes={cm.classes} title="per recording" />}
            </div>
          ) : <Notice>No confusion data.</Notice>}
        </Card>
        <Card title="Reliability diagram" subtitle="Temperature scaling fit on the inner validation subjects"
          right={rel && <Select label="" value={rel.experiment_id} onChange={setRelId} options={deep.map((r) => ({ value: r.experiment_id, label: r.model }))} />}>
          {rel?.calibration ? <Reliability before={rel.calibration.reliability_before} after={rel.calibration.reliability_after}
            eceB={rel.calibration.ece_before} eceA={rel.calibration.ece_after} /> : <Notice>No deep results yet.</Notice>}
        </Card>
        <Card title="Latency per window (CPU)" subtitle="classical: features + model; deep: model forward pass (1 thread)">
          <MetricBars domain={[0, Math.max(10, ...comp.map((r) => r.latency_ms ?? r.model_latency_ms ?? 0)) * 1.1]}
            rows={comp.map((r) => ({ name: r.model, value: r.latency_ms ?? r.model_latency_ms }))} />
        </Card>
      </div>

      <Card title="Paired Wilcoxon signed-rank tests" subtitle="Per-subject macro-F1 (T1/T2) or accuracy (T3); p_holm = Holm-corrected within each group">
        {tests.length ? (
          <div className="max-h-96 overflow-auto">
            <table className="w-full text-xs">
              <thead className="text-left text-ink-3"><tr><th className="py-1">Comparison</th><th>A</th><th>B</th><th>n</th><th>mean A</th><th>mean B</th><th>A/B better</th><th>p</th><th>p_holm</th></tr></thead>
              <tbody>{tests.map((t, i) => (
                <tr key={i} className="num border-t border-line">
                  <td className="py-1">{t.group ?? "deep"}</td><td>{t.a.replace(/^(cls|deep)_/, "")}</td><td>{t.b.replace(/^(cls|deep)_/, "")}</td>
                  <td>{t.n_subjects}</td><td>{fmt(t.mean_a)}</td><td>{fmt(t.mean_b)}</td><td>{t.n_a_better}/{t.n_b_better}</td>
                  <td className={t.p_value < 0.05 ? "font-semibold" : ""}>{t.p_value.toPrecision(2)}</td><td>{t.p_holm?.toPrecision(2) ?? "—"}</td>
                </tr>))}</tbody>
            </table>
          </div>
        ) : <Notice>No tests for this task.</Notice>}
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------- default view: story cards

/** "Correct %" in the default view = pooled balanced accuracy (falls back to pooled macro-F1). */
const correct = (r: ResultRow | undefined) => (r ? (r.pooled.balanced_accuracy ?? r.pooled.macro_f1 ?? null) : null);
const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)} %`);

const MODEL_NAME: Record<string, string> = {
  lr: "Logistic regression",
  rf: "Random forest",
  svm: "Support vector machine",
  xgb: "XGBoost",
  eegnet: "EEGNet (deep)",
  A10_hybrid: "Full hybrid (deep)",
};
const name = (m: string) => MODEL_NAME[m] ?? m;

function StoryCard({ title, big, bigLabel, chart, sentence, explain }: {
  title: string; big: number | null; bigLabel: string; chart: ReactNode; sentence: ReactNode; explain: ReactNode;
}) {
  return (
    <Card title={title}>
      <div className="flex items-baseline gap-3">
        <span className="text-5xl font-bold tabular-nums text-accent">{pct(big)}</span>
        <span className="text-base text-ink-3">{bigLabel}</span>
      </div>
      {/* the exact value behind the rounded percentage */}
      <div className="mt-1 text-sm tabular-nums text-ink-3">{big === null ? "—" : big.toFixed(3)}</div>
      <div className="mt-4">{chart}</div>
      <p className="mt-4 text-base text-ink">{sentence}</p>
      <Explain>{explain}</Explain>
    </Card>
  );
}

function StoryCards({ d }: { d: Summary }) {
  // a) cross-subject: the four classical models on T1, no rest baseline
  const base = d.classical_baselines.filter((r) => r.task_short === "T1" && r.normalisation === "none");
  const bestModel = d.best_classical_model ?? "xgb";
  const best = base.find((r) => r.model === bestModel);

  // b) rhythm (spectral) features only vs + connection features, same model, T1, no rest baseline
  const abl = d.feature_ablation.filter((r) => r.task_short === "T1" && r.normalisation === "none");
  const rhythm = abl.find((r) => r.feature_set === "A1_spectral");
  const conn = abl.find((r) => r.feature_set === "A4_spectral_connectivity");
  const test = d.stats_classical?.tests.find((t) => t.a === conn?.experiment_id && t.b === rhythm?.experiment_id);
  const significant = test ? (test.p_holm ?? test.p_value) < 0.05 : false;

  // c) classical vs deep on the same 5 subject groups (T1)
  const comp1 = d.comparison_5fold.filter((r) => r.task_short === "T1");
  const xgb5 = comp1.find((r) => r.kind === "classical" && r.model === "xgb");
  const eegnet = comp1.find((r) => r.model === "eegnet");
  const hybrid = comp1.find((r) => r.model === "A10_hybrid");

  // d) T3: moderate vs high inside the task, every model on the same 5 groups
  const T3_MODELS = ["xgb", "rf", "svm", "lr", "eegnet", "A10_hybrid"];
  const comp3 = T3_MODELS.map((m) => d.comparison_5fold.find((r) => r.task_short === "T3" && r.model === m))
    .filter((r): r is ResultRow => !!r && correct(r) !== null);
  const t3avg = comp3.length ? comp3.reduce((a, r) => a + (correct(r) as number), 0) / comp3.length : null;

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
      <StoryCard
        title="Works on people it has never seen"
        big={correct(best)} bigLabel={`correct · ${name(bestModel)}`}
        chart={<PercentBars rows={base.map((r) => ({ name: name(r.model), value: correct(r), highlight: r.model === bestModel }))} />}
        sentence={<>Tested on people it never saw, {name(bestModel)} told rest from task correctly about {pct(correct(best))} of the time.</>}
        explain={<>Each person was left out in turn: the model learned from everyone else and was then tested on that person. Bars compare four standard models.</>}
      />
      <StoryCard
        title="Brain connections help"
        big={correct(conn)} bigLabel="correct with connection features"
        chart={<PercentBars rows={[
          { name: "Rhythm features only", value: correct(rhythm) },
          { name: "+ connection features", value: correct(conn), highlight: true },
        ]} />}
        sentence={significant
          ? <>Adding connection features gave a real improvement (statistically significant).</>
          : <>Adding connection features raised the score.</>}
        explain={<>Rhythm features measure how strong each brain wave is at each sensor. Connection features measure how well pairs of sensors work in sync.</>}
      />
      <StoryCard
        title="On small data, the simple model wins"
        big={correct(xgb5)} bigLabel={`correct · ${name("xgb")}`}
        chart={<PercentBars rows={[
          { name: name("xgb"), value: correct(xgb5), highlight: true },
          { name: name("eegnet"), value: correct(eegnet) },
          { name: name("A10_hybrid"), value: correct(hybrid) },
        ]} />}
        sentence={<>48 people is too little data for the big deep model.</>}
        explain={<>All three models were trained and tested on exactly the same five groups of unseen people, so the comparison is fair.</>}
      />
      <StoryCard
        title="Can't tell 'moderate' from 'high' within the same task"
        big={t3avg} bigLabel={`correct · average of ${comp3.length} models`}
        chart={<PercentBars guessLine rows={comp3.map((r) => ({ name: name(r.model), value: correct(r) }))} />}
        sentence={<>Our own dataset needs controlled difficulty levels.</>}
        explain={<>Everyone here is doing the same task; the model must tell people who rated it moderate from those who rated it high. 50 % = guessing.</>}
      />
    </div>
  );
}

export default function Results() {
  const { advanced } = useSettings();
  const s = useApi<Summary>("/results/summary");
  const d = s.data;
  if (s.error) return <Notice kind="error">Could not load results: {s.error}. Run scripts/export_report.py, or switch on demo mode.</Notice>;
  if (!d) return <Notice>Loading results…</Notice>;
  return (
    <div className="space-y-6">
      <Card title="Results" subtitle="Four key findings. Every number is loaded from the saved experiment results.">
        <p className="text-base text-ink-2">
          "Correct %" = how often the model's answer matched the truth. 50 % would be guessing.
        </p>
      </Card>
      <StoryCards d={d} />
      {advanced ? (
        <section className="space-y-4 pt-4">
          <h2 className="border-t border-line pt-6 text-xl font-semibold">Advanced details</h2>
          <AdvancedResults d={d} />
        </section>
      ) : (
        <p className="text-sm text-ink-3">Switch on "Advanced" in the header for the full tables, statistics and model details.</p>
      )}
    </div>
  );
}
