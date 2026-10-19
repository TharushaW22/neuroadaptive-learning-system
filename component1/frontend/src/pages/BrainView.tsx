import { useState } from "react";
import HeadMap from "../components/HeadMap";
import NetworkGraph from "../components/NetworkGraph";
import { Card, Notice, Segmented } from "../components/ui";
import { BANDS, BAND_LABEL, CHANNELS } from "../lib/eeg";
import { useStream } from "../lib/stream";
import { StreamControls } from "./LiveMonitor";

type Band = (typeof BANDS)[number];

export default function BrainView() {
  const { outputs } = useStream();
  const [band, setBand] = useState<Band>("alpha");
  const [mode, setMode] = useState<"abs" | "rel">("rel");
  const [threshold, setThreshold] = useState(0.6);
  const last = outputs[outputs.length - 1];

  let values: Record<string, number> | null = null;
  if (last) {
    values = {};
    for (const ch of CHANNELS) {
      const bp = last.band_power[ch];
      const total = BANDS.reduce((s, b) => s + bp[b], 0);
      values[ch] = mode === "abs" ? Math.log10(bp[band] + 1e-12) : bp[band] / total;
    }
  }

  return (
    <div className="space-y-4">
      <Card title="Brain View" subtitle="Updates with every window of the stream started here or on the Live Monitor.">
        <StreamControls />
      </Card>
      {!last && <Notice>Start a stream to populate the head map and the network.</Notice>}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="Band power head map" subtitle={BAND_LABEL[band]}
          right={<div className="flex flex-wrap gap-2">
            <Segmented value={band} onChange={setBand} options={BANDS.map((b) => ({ value: b, label: b }))} />
            <Segmented value={mode} onChange={setMode} options={[{ value: "rel", label: "relative" }, { value: "abs", label: "absolute" }]} />
          </div>}>
          <HeadMap values={values} unit={mode === "abs" ? "log10 µV²" : "fraction of 1–45 Hz power"} size={380}
            highlight={last?.signal_quality.bad_channels} />
          <p className="mt-3 text-xs text-ink-3">Red outline = channel failing a quality check in this window. Colours are scaled to this window's range.</p>
        </Card>
        <Card title="Brain network (phase-locking value)" subtitle={`PLV in the ${last?.connectivity.band ?? "alpha"} band, current window`}
          right={<label className="flex items-center gap-2 text-xs text-ink-3">threshold
            <input type="range" min={0.2} max={0.95} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            <span className="num w-8 text-ink">{threshold.toFixed(2)}</span></label>}>
          <NetworkGraph matrix={last?.connectivity.matrix ?? null} channels={last?.connectivity.channels} threshold={threshold} size={380} />
          <p className="mt-3 text-xs text-ink-3">PLV = |mean phase difference phasor| between two electrodes: 0 = unrelated phases, 1 = locked. Volume conduction makes neighbouring electrodes look coupled, so long-range edges are the more informative ones.</p>
        </Card>
      </div>
    </div>
  );
}
