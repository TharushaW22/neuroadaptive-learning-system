import { Card, Notice } from "../components/ui";
import { API_URL, isDemo } from "../lib/api";
import { useStream } from "../lib/stream";
import type { CognitiveStateOutput } from "../lib/types";
import { useApi } from "../lib/useApi";

interface JsonSchema {
  properties?: Record<string, { description?: string; type?: string; $ref?: string; title?: string }>;
  $defs?: Record<string, { properties?: Record<string, { description?: string; type?: string; $ref?: string }> }>;
}

function flatten(schema: JsonSchema): { path: string; type: string; desc: string }[] {
  const out: { path: string; type: string; desc: string }[] = [];
  const walk = (props: JsonSchema["properties"], prefix: string, depth: number) => {
    for (const [k, v] of Object.entries(props ?? {})) {
      const path = prefix ? `${prefix}.${k}` : k;
      const ref = v.$ref?.split("/").pop();
      out.push({ path, type: ref ?? v.type ?? "object", desc: v.description ?? "" });
      if (ref && schema.$defs?.[ref] && depth < 3) walk(schema.$defs[ref].properties, path, depth + 1);
    }
  };
  walk(schema.properties, "", 0);
  return out;
}

const ENDPOINTS = [
  ["GET", "/health", "liveness, number of models"],
  ["GET", "/subjects", "STEW subjects, ratings, labels"],
  ["GET", "/models", "deployable models"],
  ["GET", "/results/summary", "all result tables"],
  ["GET", "/results/{experiment_id}", "one experiment (+ /files/{name})"],
  ["GET", "/quality, /quality/{s}/{c}", "signal quality per recording / window"],
  ["GET", "/schema", "JSON Schema of the output"],
  ["GET", "/contract/sample", "one real output"],
  ["POST", "/predict", "upload EEG → list of outputs"],
  ["WS", "/stream/{subject}/{condition}?model=&speed=", "real-time replay: raw chunks + outputs"],
];

export default function Integration() {
  const schema = useApi<JsonSchema>("/schema");
  const sample = useApi<CognitiveStateOutput>("/contract/sample");
  const { outputs } = useStream();
  const live = outputs[outputs.length - 1];
  const shown = live ?? sample.data;
  return (
    <div className="space-y-4">
      <Card title="Integration / API" subtitle="For the fusion, XAI and cueing components. Full description: docs/output_contract.md.">
        <p className="text-sm text-ink-2">
          One <code className="num">CognitiveStateOutput</code> per 4-s window, every 2 s. <b>Only <code>workload</code> is a model prediction</b>{" "}
          (<code>source: "model"</code>); attention, fatigue and readiness carry <code>source: "heuristic_index"</code> and must not be treated as validated predictions.
          Branch on <code>source</code>: once team data trains those heads, they switch to <code>"model"</code>.
        </p>
        <div className="mt-3 overflow-x-auto">
          <table className="text-sm">
            <tbody>{ENDPOINTS.map(([m, p, d]) => (
              <tr key={p} className="border-t border-line">
                <td className="num py-1 pr-3 text-xs font-semibold text-accent">{m}</td>
                <td className="num py-1 pr-4">{p}</td><td className="py-1 text-ink-3">{d}</td>
              </tr>))}</tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-ink-3">Base URL: <span className="num">{isDemo() ? "(demo mode — offline snapshot)" : API_URL}</span></p>
      </Card>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card title="Output contract fields" subtitle="Rendered from the JSON Schema served by /schema">
          {schema.error && <Notice kind="error">{schema.error}</Notice>}
          {schema.data && (
            <div className="max-h-[640px] overflow-auto">
              <table className="w-full text-xs">
                <thead className="text-left text-ink-3"><tr><th className="py-1">Field</th><th>Type</th><th>Description</th></tr></thead>
                <tbody>{flatten(schema.data).map((f) => (
                  <tr key={f.path} className="border-t border-line align-top">
                    <td className="num py-1 pr-3">{f.path}</td><td className="num pr-3 text-ink-3">{f.type}</td><td className="text-ink-2">{f.desc}</td>
                  </tr>))}</tbody>
              </table>
            </div>
          )}
        </Card>
        <Card title={live ? `Live JSON — window ${live.window_index}` : "Sample JSON (engine output)"}
          subtitle={live ? "The latest output of the running stream" : "Start a stream to see live outputs; this is a real output from /contract/sample"}>
          {shown ? (
            <pre className="num max-h-[640px] overflow-auto rounded bg-surface-2 p-3 text-[11px] leading-relaxed">
              {JSON.stringify({ ...shown, connectivity: { ...shown.connectivity, matrix: `[14 x 14 matrix — first row: ${shown.connectivity.matrix[0].slice(0, 4).join(", ")}, …]` } }, null, 2)}
            </pre>
          ) : <Notice>{sample.error ?? "Loading…"}</Notice>}
        </Card>
      </div>
    </div>
  );
}
