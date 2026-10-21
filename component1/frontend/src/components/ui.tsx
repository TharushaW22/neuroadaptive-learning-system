import type { ReactNode } from "react";
import ErrorBoundary from "./ErrorBoundary";
import type { Flag } from "../lib/types";

export function Card({ title, subtitle, right, children, className = "" }: {
  title?: ReactNode; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-lg border border-line bg-surface ${className}`}>
      {(title || right) && (
        <header className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-5 py-4">
          <div>
            {title && <h2 className="text-base font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-1 text-sm text-ink-3">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className="p-5"><ErrorBoundary label={typeof title === "string" ? title : undefined}>{children}</ErrorBoundary></div>
    </section>
  );
}

export function HeuristicBadge() {
  return (
    <span
      title="Established EEG band-power index. STEW has no labels for this state, so it is not a trained prediction."
      className="inline-flex items-center gap-1 rounded border border-warn/40 bg-warn-soft px-1.5 py-0.5 text-[11px] font-medium text-warn"
    >
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><path d="M5 1 9 9H1z" fill="none" stroke="currentColor" strokeWidth="1.3" /><path d="M5 4v2.2M5 7.4v.1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" /></svg>
      heuristic index — not a trained prediction
    </span>
  );
}

export function ModelBadge() {
  return (
    <span className="inline-flex items-center rounded border border-accent/40 bg-accent-soft px-1.5 py-0.5 text-[11px] font-medium text-accent">
      model prediction
    </span>
  );
}

const FLAG_STYLE: Record<Flag, string> = {
  good: "bg-good-soft text-good border-good/40",
  warn: "bg-warn-soft text-warn border-warn/40",
  bad: "bg-bad-soft text-bad border-bad/40",
};
const FLAG_ICON: Record<Flag, ReactNode> = {
  good: <path d="M2 5.2 4.2 7.4 8 3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
  warn: <><path d="M5 2v3.6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /><circle cx="5" cy="7.8" r=".9" fill="currentColor" /></>,
  bad: <path d="m2.5 2.5 5 5m0-5-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
};

/** Status always shown with icon + text, never colour alone. */
export function FlagPill({ flag, label }: { flag: Flag; label?: string }) {
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${FLAG_STYLE[flag]}`}>
      <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>{FLAG_ICON[flag]}</svg>
      {label ?? flag}
    </span>
  );
}

export function Select<T extends string | number>({ label, value, onChange, options, disabled }: {
  label: string; value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; disabled?: boolean;
}) {
  return (
    <label className="flex flex-col gap-1 text-xs text-ink-3">
      {label}
      <select
        className="min-w-0 rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink focus:outline-2 focus:outline-accent disabled:opacity-50"
        value={String(value)}
        disabled={disabled}
        onChange={(e) => {
          const o = options.find((x) => String(x.value) === e.target.value);
          if (o) onChange(o.value);
        }}
      >
        {options.map((o) => <option key={String(o.value)} value={String(o.value)}>{o.label}</option>)}
      </select>
    </label>
  );
}

export function Button({ children, onClick, variant = "primary", disabled }: {
  children: ReactNode; onClick?: () => void; variant?: "primary" | "ghost"; disabled?: boolean;
}) {
  const cls = variant === "primary"
    ? "bg-accent text-white hover:opacity-90"
    : "border border-line bg-surface text-ink hover:bg-surface-2";
  return (
    <button type="button" onClick={onClick} disabled={disabled}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-40 ${cls}`}>
      {children}
    </button>
  );
}

export function Segmented<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { value: T; label: string }[];
}) {
  return (
    <div className="inline-flex rounded-md border border-line bg-surface-2 p-0.5" role="tablist">
      {options.map((o) => (
        <button key={o.value} type="button" role="tab" aria-selected={o.value === value} onClick={() => onChange(o.value)}
          className={`rounded px-2.5 py-1 text-xs font-medium transition ${o.value === value ? "bg-surface text-ink shadow-sm" : "text-ink-3 hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Notice({ kind = "info", children }: { kind?: "info" | "error"; children: ReactNode }) {
  return (
    <div className={`rounded-md border px-3 py-2 text-sm ${kind === "error" ? "border-bad/40 bg-bad-soft text-bad" : "border-line bg-surface-2 text-ink-2"}`}>
      {children}
    </div>
  );
}

export function Stat({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-xs text-ink-3">{label}</div>
      <div className="num mt-0.5 text-2xl font-medium text-ink">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-ink-3">{sub}</div>}
    </div>
  );
}

/** Plain-English caption shown under a panel: "What am I looking at?" + one sentence. */
export function Explain({ children }: { children: ReactNode }) {
  return (
    <p className="mt-4 text-sm leading-relaxed text-ink-3">
      <span className="font-medium text-ink-2">What am I looking at? </span>
      {children}
    </p>
  );
}
