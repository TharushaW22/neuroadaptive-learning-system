import { Card } from "../components/ui";

const STEPS = [
  { t: "Acquisition", d: "Emotiv EPOC, 14 ch, 128 Hz" },
  { t: "Signal quality", d: "flat · amplitude · variance · EMG · line noise" },
  { t: "Preprocessing", d: "DC removal · 0.5–45 Hz FIR · 50 Hz notch · bad-channel interpolation" },
  { t: "Epoching", d: "4-s windows, 50 % overlap" },
  { t: "Features", d: "spectral · entropy · wavelet · PLV · coherence · graph" },
  { t: "Models", d: "LR · SVM · RF · XGBoost · EEGNet · CNN-LSTM-Transformer-GAT" },
  { t: "Calibration", d: "temperature scaling · EMA smoothing" },
  { t: "Output", d: "timestamped JSON → fusion · XAI · cueing · dashboard" },
];

function Pipeline() {
  const w = 150, h = 74, gap = 22, cols = 4;
  const W = cols * w + (cols - 1) * gap;
  const H = 2 * h + 40;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Pipeline: acquisition to output">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 10 5 0 10z" fill="var(--text-3)" />
        </marker>
      </defs>
      {STEPS.map((s, i) => {
        const row = Math.floor(i / cols);
        const col = row === 0 ? i % cols : cols - 1 - (i % cols); // snake layout
        const x = col * (w + gap);
        const y = row * (h + 40);
        return (
          <g key={s.t}>
            <rect x={x} y={y} width={w} height={h} rx={8} fill={i === 5 ? "var(--accent-soft)" : "var(--surface-2)"} stroke={i === 5 ? "var(--accent)" : "var(--border)"} />
            <text x={x + 10} y={y + 22} fontSize={13} fontWeight={600} fill="var(--text)">{i + 1}. {s.t}</text>
            <foreignObject x={x + 10} y={y + 28} width={w - 18} height={h - 30}>
              <div style={{ fontSize: 10.5, lineHeight: 1.3, color: "var(--text-2)" }}>{s.d}</div>
            </foreignObject>
          </g>
        );
      })}
      {/* arrows: row 1 left->right, down, row 2 right->left */}
      {[0, 1, 2].map((i) => <line key={i} x1={i * (w + gap) + w + 2} y1={h / 2} x2={(i + 1) * (w + gap) - 3} y2={h / 2} stroke="var(--text-3)" strokeWidth={1.5} markerEnd="url(#arr)" />)}
      <line x1={W - w / 2} y1={h + 2} x2={W - w / 2} y2={h + 37} stroke="var(--text-3)" strokeWidth={1.5} markerEnd="url(#arr)" />
      {[3, 2, 1].map((i) => <line key={`b${i}`} x1={i * (w + gap) - 2} y1={h + 40 + h / 2} x2={(i - 1) * (w + gap) + w + 3} y2={h + 40 + h / 2} stroke="var(--text-3)" strokeWidth={1.5} markerEnd="url(#arr)" />)}
    </svg>
  );
}

export default function About() {
  return (
    <div className="space-y-4">
      <Card title="EEG Neuro-Cognitive Understanding Engine" subtitle="SLIIT IT4010 · J26-DS-333 · Component 1 of the Neuroadaptive Learning System (EEG, Computer Vision, Agentic AI)">
        <Pipeline />
      </Card>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="Dataset — STEW">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-2">
            <li>Simultaneous Task EEG Workload (Lim et al., 2018): 48 subjects, Emotiv EPOC, 14 channels at 128 Hz.</li>
            <li>Each subject: 2.5 min at rest (<code>lo</code>) and 2.5 min of the SIMKAP multitasking test (<code>hi</code>).</li>
            <li>Self-rated workload 1–9 after each recording; subjects 5, 24 and 42 have no rating (kept for T1, masked for T2/T3).</li>
            <li>Tasks: <b>T1</b> rest vs task · <b>T2</b> rating level low/medium/high · <b>T3</b> moderate vs high workload within the task.</li>
            <li>Evaluation is always across subjects: leave-one-subject-out for classical models, 5 fixed subject folds for every model in the paired comparison.</li>
          </ul>
        </Card>
        <Card title="Limitations — read before interpreting results">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-ink-2">
            <li><b>Workload-only labels.</b> STEW labels workload only. Attention, fatigue and learning readiness are shown as established band-power indices, marked “heuristic index — not a trained prediction”. Their model heads exist but stay masked until the team collects labelled data.</li>
            <li><b>Rest-vs-task confound.</b> T1 separates sitting still from doing a demanding visual-motor task. Eye movements, muscle activity and arousal differ, not just workload, so high T1 scores overstate what is learned about workload.</li>
            <li><b>T2 ≈ T1.</b> Rest recordings are rated low (42/3/0 low/medium/high) and task recordings medium or high (0/20/25), so T2 mostly re-measures T1.</li>
            <li><b>Why T3 matters.</b> T3 keeps the condition fixed (task only) and asks whether EEG tracks how hard the task felt. It is the confound-free workload test — but with only 45 labelled recordings (one per subject) its estimates are noisy.</li>
            <li><b>Rest baseline.</b> Normalising with a subject's own rest recording is a realistic calibration step but uses one of the two T1 classes; results are reported with and without it.</li>
            <li><b>Consumer headset.</b> Saline electrodes and no EOG channel: blinks and muscle noise are frequent (flagged per window), and 30 channels in 26 recordings needed interpolation.</li>
            <li><b>CPU-only training.</b> Deep models are small and lightly tuned; better hyper-parameters may exist.</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
