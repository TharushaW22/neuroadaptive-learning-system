import { useState } from "react";
import HeadMap from "../components/HeadMap";
import NetworkGraph from "../components/NetworkGraph";
import { Card, Explain, Notice, Segmented } from "../components/ui";
import { BANDS, BAND_LABEL, CHANNELS } from "../lib/eeg";
import { useSettings } from "../lib/settings";
import { useStream } from "../lib/stream";
import { StreamControls } from "./LiveMonitor";

type Band = (typeof BANDS)[number];

// Default view: three rhythms, relative power, and a fixed "top 15 connections" network.
const SIMPLE_BANDS: Band[] = ["theta", "alpha", "beta"];
const TOP_EDGES = 15;
const capital = (s: string) => s[0].toUpperCase() + s.slice(1);

export default function BrainView() {
  const { advanced } = useSettings();
  const { outputs } = useStream();
  const [bandChoice, setBand] = useState<Band>("alpha");
  const [modeChoice, setMode] = useState<"abs" | "rel">("rel");
  const [threshold, setThreshold] = useState(0.6);
  const last = outputs[outputs.length - 1];

  // Delta/gamma and absolute power are Advanced-only; fall back to alpha / relative otherwise.
  const band: Band = advanced || SIMPLE_BANDS.includes(bandChoice) ? bandChoice : "alpha";
  const mode = advanced ? modeChoice : "rel";

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
    <div className="space-y-6">
      <Card title="Brain View"
        subtitle={advanced ? "Updates with every window of the stream started here or on the Live Monitor." : "Updates every 2 seconds while a recording is playing (here or on the Live Monitor)."}>
        <StreamControls />
      </Card>
      {!last && <Notice>{advanced ? "Start a stream to populate the head map and the network." : "Press Start to fill the head map and the network."}</Notice>}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card title={advanced ? "Band power head map" : "Brain rhythms on the head"} subtitle={advanced ? BAND_LABEL[band] : undefined}
          right={<div className="flex flex-wrap gap-2">
            <Segmented value={band} onChange={setBand}
              options={(advanced ? [...BANDS] : SIMPLE_BANDS).map((b) => ({ value: b, label: advanced ? b : capital(b) }))} />
            {advanced && <Segmented value={mode} onChange={setMode} options={[{ value: "rel", label: "relative" }, { value: "abs", label: "absolute" }]} />}
          </div>}>
          <HeadMap simple={!advanced} values={values} unit={mode === "abs" ? "log10 µV²" : "fraction of 1–45 Hz power"} size={380}
            highlight={last?.signal_quality.bad_channels} />
          {advanced && <p className="mt-3 text-xs text-ink-3">Red outline = channel failing a quality check in this window. Colours are scaled to this window's range.</p>}
          <Explain>Top view of the head (nose at the top). Colour = strength of the selected brain rhythm under each sensor.</Explain>
        </Card>
        <Card title={advanced ? "Brain network (phase-locking value)" : "Brain network"}
          subtitle={advanced ? `PLV in the ${last?.connectivity.band ?? "alpha"} band, current window` : undefined}
          right={advanced ? <label className="flex items-center gap-2 text-xs text-ink-3">threshold
            <input type="range" min={0.2} max={0.95} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
            <span className="num w-8 text-ink">{threshold.toFixed(2)}</span></label> : undefined}>
          <NetworkGraph matrix={last?.connectivity.matrix ?? null} channels={last?.connectivity.channels} threshold={threshold} size={380}
            topK={advanced ? undefined : TOP_EDGES} />
          {advanced && <p className="mt-3 text-xs text-ink-3">PLV = |mean phase difference phasor| between two electrodes: 0 = unrelated phases, 1 = locked. Volume conduction makes neighbouring electrodes look coupled, so long-range edges are the more informative ones.</p>}
          <Explain>Lines join sensors that work in sync. Thicker = stronger.</Explain>
        </Card>
      </div>
    </div>
  );
}
