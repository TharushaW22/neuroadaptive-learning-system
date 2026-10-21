import { useEffect, useMemo, useState } from "react";
import EEGTrace from "../components/EEGTrace";
import ErrorBoundary from "../components/ErrorBoundary";
import QualityIndicator from "../components/QualityIndicator";
import { Button, Card, Explain, Notice, Select } from "../components/ui";
import StateRow from "../components/StateRow";
import WorkloadTimeline from "../components/WorkloadTimeline";
import { TASK_LABEL } from "../lib/eeg";
import { useSettings } from "../lib/settings";
import { useStream } from "../lib/stream";
import type { ModelInfo, Subject } from "../lib/types";
import { useApi } from "../lib/useApi";

// Used whenever the Advanced switch is off (the model and speed pickers are hidden then).
const DEFAULT_MODEL = "T1_xgb_all";
const DEFAULT_SPEED = 2;

export function StreamControls() {
  const { demo, advanced } = useSettings();
  const stream = useStream();
  const subjects = useApi<Subject[]>("/subjects");
  const models = useApi<ModelInfo[]>("/models");
  const demoIndex = useApi<{ subject: number; condition: string; model: string }[]>(demo ? "/stream_index" : null);
  const [subject, setSubject] = useState(1);
  const [condition, setCondition] = useState<"lo" | "hi">("hi");
  const [model, setModel] = useState<string>("");
  const [speed, setSpeed] = useState(DEFAULT_SPEED);

  // default: T1_xgb_all (most accurate in cross-subject evaluation), else any classical T1 model, else the first one
  const defaultModel = useMemo(() => {
    const list = models.data ?? [];
    return (list.find((m) => m.id === DEFAULT_MODEL) ?? list.find((m) => m.kind === "classical" && m.task === "workload_binary") ?? list[0])?.id ?? "";
  }, [models.data]);
  useEffect(() => {
    if (!model && defaultModel) setModel(defaultModel);
  }, [defaultModel, model]);

  const available = useMemo(() => {
    if (!demo || !demoIndex.data) return null;
    return new Set(demoIndex.data.map((d) => `${d.subject}_${d.condition}`));
  }, [demo, demoIndex.data]);

  const subjOptions = (subjects.data ?? [])
    .filter((s) => !available || available.has(`${s.subject_id}_lo`) || available.has(`${s.subject_id}_hi`))
    .map((s) => {
      const id = String(s.subject_id).padStart(2, "0");
      return {
        value: s.subject_id,
        label: advanced ? `Subject ${id}${s.rated ? ` · rest ${s.rating_lo} / task ${s.rating_hi}` : " · no rating"}` : `Person ${id}`,
      };
    });
  const running = stream.status === "streaming" || stream.status === "connecting";
  const useModel = advanced ? model : defaultModel;
  const useSpeed = advanced ? speed : DEFAULT_SPEED;
  const modelInfo = models.data?.find((m) => m.id === useModel);

  return (
    <div className="flex flex-wrap items-end gap-4">
      <Select label={advanced ? "Subject" : "Person"} value={subject} onChange={setSubject} disabled={running}
        options={subjOptions.length ? subjOptions : [{ value: 1, label: advanced ? "Subject 01" : "Person 01" }]} />
      <Select label="Recording" value={condition} onChange={setCondition} disabled={running}
        options={advanced
          ? [{ value: "lo", label: "lo — rest" }, { value: "hi", label: "hi — SIMKAP task" }]
          : [{ value: "lo", label: "Resting" }, { value: "hi", label: "Doing a task" }]} />
      {advanced && (
        <>
          <Select label="Model" value={model} onChange={setModel} disabled={running}
            options={(models.data ?? []).map((m) => ({ value: m.id, label: `${m.id} (${m.kind})` }))} />
          <Select label="Speed" value={speed} onChange={setSpeed} disabled={running}
            options={[1, 2, 5, 10].map((v) => ({ value: v, label: `${v}× real time` }))} />
        </>
      )}
      {running
        ? <Button variant="ghost" onClick={stream.stop}>Stop</Button>
        : <Button onClick={() => useModel && stream.start({ subject, condition, model: useModel, speed: useSpeed })} disabled={!useModel}>Start</Button>}
      {advanced && modelInfo && <span className="pb-2 text-xs text-ink-3">{TASK_LABEL[modelInfo.task] ?? modelInfo.task}</span>}
      {(subjects.error || models.error) && <Notice kind="error">{subjects.error || models.error}</Notice>}
    </div>
  );
}

export default function LiveMonitor() {
  const { advanced } = useSettings();
  const { outputs, meta, status, error } = useStream();
  const last = outputs[outputs.length - 1];
  return (
    <div className="space-y-6">
      <Card title="Live Monitor"
        subtitle={advanced
          ? "Replays a STEW recording through the engine window by window (4-s windows every 2 s), exactly as a live headset would be processed."
          : "Plays back a real brain recording as if the person were wearing the headset right now."}>
        <StreamControls />
        {error && <div className="mt-3"><Notice kind="error">{error}</Notice></div>}
        {meta && (advanced ? (
          <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-ink-2">
            <span>Subject <b className="num">{String(meta.subject).padStart(2, "0")}</b> · {meta.condition === "lo" ? "rest" : "SIMKAP task"}</span>
            <span>self-rated workload: <b className="num">{meta.rating ?? "not rated"}</b></span>
            <span>window <b className="num">{last ? last.window_index + 1 : 0}</b> / 74</span>
            <span>latency <b className="num">{last ? last.latency_ms.toFixed(0) : "—"} ms</b></span>
            <span className="capitalize">status: {status}</span>
          </div>
        ) : (
          <div className="mt-4 text-base text-ink-2">
            Person {String(meta.subject).padStart(2, "0")} · {meta.condition === "lo" ? "resting" : "doing a task"}
            {status === "ended" && " · finished"}
          </div>
        ))}
      </Card>

      <section>
        <ErrorBoundary label="states"><StateRow outputs={outputs} advanced={advanced} /></ErrorBoundary>
        <Explain>
          Four estimates of the person's mental state, updated every 2 seconds; the small line shows the last minute.
          The SOURCE tag says where each number comes from. The light shows whether the headset signal is clean enough to trust.
        </Explain>
      </section>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-3" title={advanced ? "EEG — 14 channels" : "Brain signal"}
          right={advanced ? <QualityIndicator output={last} /> : undefined}>
          <EEGTrace simple={!advanced} badChannels={last?.signal_quality.bad_channels} interpolated={last?.signal_quality.interpolated_channels} />
          <Explain>Cleaned brain signal from 14 sensors. A red row means that sensor is noisy right now.</Explain>
        </Card>
        <Card className="xl:col-span-2" title={advanced ? "Workload probability timeline" : "Workload over time"}
          subtitle={advanced
            ? (meta ? `P(${meta.classes[meta.classes.length - 1]}) — solid: EMA-smoothed; dotted: this window; band: ±1 SD of the last 5 windows` : "Start a stream to see the timeline")
            : undefined}>
          <WorkloadTimeline simple={!advanced} outputs={outputs} classes={meta?.classes ?? ["low", "high"]} height={300} />
          {advanced && (
            <>
              <div className="mt-2 flex flex-wrap gap-1">
                {outputs.map((o) => (
                  <span key={o.window_index} title={`window ${o.window_index}: ${o.signal_quality.flag}`}
                    className="h-2 w-2 rounded-sm" style={{ background: `var(--${o.signal_quality.flag})` }} />
                ))}
              </div>
              <div className="mt-1 text-xs text-ink-3">squares = quality flag of each window (green good · amber warn · red bad)</div>
            </>
          )}
          <Explain>How the workload estimate changes over time. Above the dashed middle line, the model leans towards high workload.</Explain>
        </Card>
      </div>
    </div>
  );
}
