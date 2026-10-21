// Types mirror src/eegengine/inference/schema.py (output contract v1.0.0) and the API responses.

export type Flag = "good" | "warn" | "bad";

export interface HeuristicState {
  value: number;
  raw_value: number;
  source: "heuristic_index";
  formula: string;
  note: string;
}

export interface CognitiveStateOutput {
  schema_version: string;
  participant_id: string;
  session_id: string;
  window_index: number;
  timestamp_start: string;
  timestamp_end: string;
  t_start_s: number;
  t_end_s: number;
  signal_quality: { overall: number; flag: Flag; bad_channels: string[]; interpolated_channels: string[] };
  states: {
    workload: {
      label: string;
      probabilities: Record<string, number>;
      raw_probabilities: Record<string, number>;
      confidence: number;
      source: "model";
      model_version: string;
      task: string;
    };
    attention: HeuristicState;
    fatigue: HeuristicState;
    readiness: HeuristicState;
  };
  band_power: Record<string, Record<string, number>>;
  band_power_unit: string;
  connectivity: { metric: string; band: string; channels: string[]; matrix: number[][] };
  latency_ms: number;
}

export interface StreamMeta {
  type: "meta";
  subject: number;
  condition: "lo" | "hi";
  rating: number | null;
  model: string;
  task: string;
  classes: string[];
  fs: number;
  display_fs: number;
  channels: string[];
  duration_s: number;
  speed: number;
}

export type StreamMessage =
  | StreamMeta
  | { type: "raw"; t_start: number; fs: number; data: number[][] }
  | { type: "state"; output: CognitiveStateOutput }
  | { type: "end" }
  | { type: "error"; message: string };

export interface ModelInfo {
  id: string;
  file: string;
  kind: "classical" | "deep";
  version: string;
  task: string;
  classes: string[];
  model?: string;
  feature_set?: string;
  temperature?: number;
}

export interface Subject {
  subject_id: number;
  rating_lo: number | null;
  rating_hi: number | null;
  rated: boolean;
}

export interface QualityRow {
  subject_id: number;
  condition: "lo" | "hi";
  recording: string;
  n_windows: number;
  good: number;
  warn: number;
  bad: number;
  usable_fraction: number;
  mean_score: number;
  interpolated: string[];
  interpolation_reasons: Record<string, string[]>;
}

export interface QualityWindow {
  window_index: number;
  t_start: number;
  t_end: number;
  flag: Flag;
  score: number;
  bad_channels: string[];
}

export interface MeanStd { mean: number | null; std: number | null; n_folds: number }
type MetricKeys = "macro_f1" | "balanced_accuracy" | "precision_macro" | "recall_macro" | "auroc";

export interface ResultRow {
  experiment_id: string;
  kind: "classical" | "deep";
  task: string;
  task_short: "T1" | "T2" | "T3";
  model: string;
  feature_set: string | null;
  normalisation: string | null;
  ablation_id: string | null;
  cv: string;
  n_folds: number;
  unit: string | null;
  per_fold: Record<MetricKeys, MeanStd | null | undefined>;
  pooled: Record<MetricKeys, number | null | undefined>;
  recording_level: Record<MetricKeys, number | null | undefined>;
  confusion_pooled: number[][] | null;
  confusion_recording: number[][] | null;
  classes: string[];
  latency_ms: number | null;
  model_latency_ms: number | null;
  calibration: {
    ece_before: number;
    ece_after: number;
    reliability_before: { confidence: number[]; accuracy: number[]; count: number[] };
    reliability_after: { confidence: number[]; accuracy: number[]; count: number[] };
    temperatures: number[];
  } | null;
  n_params: number | null;
}

export interface StatTest {
  group?: string;
  a: string;
  b: string;
  n_subjects: number;
  mean_a: number;
  mean_b: number;
  n_a_better: number;
  n_b_better: number;
  p_value: number;
  p_holm?: number;
}

export interface Summary {
  generated: string;
  notes: Record<string, string>;
  best_classical_model: string | null;
  classical_baselines: ResultRow[];
  feature_ablation: ResultRow[];
  classical_kfold5: ResultRow[];
  deep: ResultRow[];
  comparison_5fold: ResultRow[];
  stats_classical: { best_model: string; tests: StatTest[] } | null;
  stats_deep: { tests: StatTest[] } | null;
  dataset_summary?: Record<string, unknown>;
  preprocess_summary?: Record<string, unknown>;
  label_counts?: Record<string, unknown>;
  feature_summary?: Record<string, unknown>;
}
