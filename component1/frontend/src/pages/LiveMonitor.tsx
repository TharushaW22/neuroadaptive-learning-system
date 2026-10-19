import { useEffect, useMemo, useState } from "react";
import EEGTrace from "../components/EEGTrace";
import QualityIndicator from "../components/QualityIndicator";
import StateCards from "../components/StateCards";
import { Button, Card, Notice, Select } from "../components/ui";
import WorkloadTimeline from "../components/WorkloadTimeline";
import { TASK_LABEL } from "../lib/eeg";
import { useSettings } from "../lib/settings";
import { useStream } from "../lib/stream";
import type { ModelInfo, Subject } from "../lib/types";
import { useApi } from "../lib/useApi";

export function StreamControls() {
  const { demo } = useSettings();
  const stream = useStream();
  const subjects = useApi<Subject[]>("/subjects");
  const models = useApi<ModelInfo[]>("/models");
  const demoIndex = useApi<{ subject: number; condition: string; model: string }[]>(demo ? "/stream_index" : null);
  const [subject, setSubject] = useState(1);
  const [condition, setCondition] = useState<"lo" | "hi">("hi");
  const [model, setModel] = useState<string>("");
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    if (!model && models.data?.length) setModel(models.data[0].id);
  }, [models.data, model]);

  const available = useMemo(() => {
    if (!demo || !demoIndex.data) return null;
    return new Set(demoIndex.data.map((d) => `${d.subject}_${d.condition}`));
  }, [demo, demoIndex.data]);

  const subjOptions = (subjects.data ?? [])
    .filter((s) => !available || available.has(`${s.subject_id}_lo`) || available.has(`${s.subject_id}_hi`))
    .map((s) => ({
      value: s.subject_id,
      label: `Subject ${String(s.subject_id).padStart(2, "0")}${s.rated ? ` · rest ${s.rating_lo} / task ${s.rating_hi}` : " · no rating"}`,
    }));
  const running = stream.status === "streaming" || stream.status === "connecting";
  const modelInfo = models.data?.find((m) => m.id === model);

  return (
    <div className="flex flex-wrap items-end gap-3">
      <Select label="Subject" value={subject} onChange={setSubject} options={subjOptions.length ? subjOptions : [{ value: 1, label: "Subject 01" }]} disabled={running} />
      <Select label="Recording" value={condition} onChange={setCondition} disabled={running}
        options={[{ value: "lo", label: "lo — rest" }, { value: "hi", label: "hi — SIMKAP task" }]} />
      <Select label="Model" value={model} onChange={setModel} disabled={running}
        options={(models.data ?? []).map((m) => ({ value: m.id, label: `${m.id} (${m.kind})` }))} />
      <Select label="Speed" value={speed} onChange={setSpeed} disabled={running}
        options={[1, 2, 5, 10].map((v) => ({ value: v, label: `${v}× real time` }))} />
      {running
        ? <Button variant="ghost" onClick={stream.stop}>Stop</Button>
        : <Button onClick={() => model && stream.start({ subject, condition, model, speed })} disabled={!model}>Start</Button>}
      {modelInfo && <span className="pb-2 text-xs text-ink-3">{TASK_LABEL[modelInfo.task] ?? modelInfo.task}</span>}
      {(subjects.error || models.error) && <Notice kind="error">{subjects.error || models.error}</Notice>}
    </div>
  );
}

export default function LiveMonitor() {
  const { outputs, meta, status, error } = useStream();
  const last = outputs[outputs.length - 1];
  return (
    <div className="space-y-4">
      <Card title="Live Monitor" subtitle="Replays a STEW recording through the engine window by window (4-s windows every 2 s), exactly as a live headset would be processed.">
        <StreamControls />
        {error && <div className="mt-3"><Notice kind="error">{error}</Notice></div>}
        {meta && (
          <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-ink-2">
            <span>Subject <b className="num">{String(meta.subject).padStart(2, "0")}</b> · {meta.condition === "lo" ? "rest" : "SIMKAP task"}</span>
            <span>self-rated workload: <b className="num">{meta.rating ?? "not rated"}</b></span>
            <span>window <b className="num">{last ? last.window_index + 1 : 0}</b> / 74</span>
            <span>latency <b className="num">{last ? last.latency_ms.toFixed(0) : "—"} ms</b></span>
            <span className="capitalize">status: {status}</span>
          </div>
        )}
      </Card>

      <StateCards outputs={outputs} />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-3" title="EEG — 14 channels" right={<QualityIndicator output={last} />}>
          <EEGTrace badChannels={last?.signal_quality.bad_channels} interpolated={last?.signal_quality.interpolated_channels} />
        </Card>
        <Card className="xl:col-span-2" title="Workload probability timeline"
          subtitle={meta ? `P(${meta.classes[meta.classes.length - 1]}) — solid: EMA-smoothed; dotted: this window; band: ±1 SD of the last 5 windows` : "Start a stream to see the timeline"}>
          <WorkloadTimeline outputs={outputs} classes={meta?.classes ?? ["low", "high"]} height={300} />
          <div className="mt-2 flex flex-wrap gap-1">
            {outputs.map((o) => (
              <span key={o.window_index} title={`window ${o.window_index}: ${o.signal_quality.flag}`}
                className="h-2 w-2 rounded-sm" style={{ background: `var(--${o.signal_quality.flag})` }} />
            ))}
          </div>
          <div className="mt-1 text-[11px] text-ink-3">squares = quality flag of each window (green good · amber warn · red bad)</div>
        </Card>
      </div>
    </div>
  );
}
