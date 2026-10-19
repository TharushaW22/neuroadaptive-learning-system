// Every number on this page comes from /results/summary (built by scripts/export_report.py from results/).
import { useMemo, useState } from "react";
import { Confusion, MetricBars, Reliability } from "../components/ResultCharts";
import { Card, Notice, Segmented, Select } from "../components/ui";
import { fmt, TASK_LABEL } from "../lib/eeg";
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

export default function Results() {
  const s = useApi<Summary>("/results/summary");
  const [task, setTask] = useState<Task>("T1");
  const [norm, setNorm] = useState<"none" | "rest_baseline">("none");
  const [cmId, setCmId] = useState<string>("");
  const [relId, setRelId] = useState<string>("");

  const d = s.data;
  const byTask = (rows: ResultRow[] | undefined) => (rows ?? []).filter((r) => r.task_short === task);
  const base = byTask(d?.classical_baselines);
  const abl = byTask(d?.feature_ablation).filter((r) => r.normalisation === norm);
  const deep = byTask(d?.deep);
  const comp = byTask(d?.comparison_5fold);
  const allRows = useMemo(() => [...base, ...deep, ...byTask(d?.classical_kfold5)], [d, task]); // eslint-disable-line react-hooks/exhaustive-deps
  const cm = allRows.find((r) => r.experiment_id === cmId) ?? deep.find((r) => r.model === "A10_hybrid") ?? allRows[0];
  const rel = deep.find((r) => r.experiment_id === relId) ?? deep.find((r) => r.model === "A10_hybrid") ?? deep[0];
  const tests = [...(d?.stats_classical?.tests ?? []), ...(d?.stats_deep?.tests ?? [])].filter((t) => t.a.includes(`_${task}_`));

  if (s.error) return <Notice kind="error">Could not load results: {s.error}. Run scripts/export_report.py, or switch on demo mode.</Notice>;
  if (!d) return <Notice>Loading results…</Notice>;

  return (
    <div className="space-y-4">
      <Card title="Results" subtitle={`Generated ${d.generated} from results/ — no number on this page is typed by hand.`}
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
