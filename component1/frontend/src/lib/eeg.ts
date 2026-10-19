// EEG constants shared by the views.
export const CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"];
export const BANDS = ["delta", "theta", "alpha", "beta", "gamma"] as const;
export const BAND_LABEL: Record<string, string> = {
  delta: "δ delta 1–4 Hz",
  theta: "θ theta 4–8 Hz",
  alpha: "α alpha 8–13 Hz",
  beta: "β beta 13–30 Hz",
  gamma: "γ gamma 30–45 Hz",
};

// 2-D head positions (nose up), projected from the MNE 10-20 montage and centred; radius ~0.95.
export const POS: Record<string, [number, number]> = {
  AF3: [-0.279, 0.735], F7: [-0.627, 0.508], F3: [-0.381, 0.508], FC5: [-0.595, 0.258], T7: [-0.716, -0.006],
  P7: [-0.592, -0.465], O1: [-0.24, -0.743], O2: [0.21, -0.741], P8: [0.562, -0.464], T8: [0.691, -0.003],
  FC6: [0.587, 0.263], F4: [0.371, 0.515], F8: [0.626, 0.523], AF4: [0.271, 0.74],
};

// Sequential single-hue ramp (light -> dark blue) for magnitude, 7 steps.
export const SEQ = ["#e4eefb", "#c4d9f5", "#9cc0ee", "#6fa3e3", "#4385d4", "#2a68b8", "#1c4c8c"];
export const seqColor = (t: number) => SEQ[Math.max(0, Math.min(SEQ.length - 1, Math.floor(t * SEQ.length)))];

export const TASK_LABEL: Record<string, string> = {
  workload_binary: "T1 · rest vs task (low / high)",
  workload_level: "T2 · rating level (low / medium / high)",
  workload_within_task: "T3 · within-task workload (moderate / high)",
};

export const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(d));
