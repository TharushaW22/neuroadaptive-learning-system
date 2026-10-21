import { NavLink, Outlet, useLocation } from "react-router-dom";
import ErrorBoundary from "./ErrorBoundary";
import { useSettings } from "../lib/settings";
import { useStream } from "../lib/stream";

// Pages marked advanced only appear in the nav when the "Advanced" switch is on.
const NAV = [
  { to: "/", label: "Live Monitor" },
  { to: "/brain", label: "Brain View" },
  { to: "/quality", label: "Signal Quality", advanced: true },
  { to: "/results", label: "Results" },
  { to: "/integration", label: "Integration / API", advanced: true },
  { to: "/about", label: "About" },
];

function Toggle({ label, on, set, title }: { label: string; on: boolean; set: (v: boolean) => void; title?: string }) {
  return (
    <button type="button" title={title} onClick={() => set(!on)} aria-pressed={on}
      className={`rounded-md border px-2.5 py-1 text-sm font-medium transition ${on ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-3 hover:text-ink"}`}>
      {label}
    </button>
  );
}

export default function Layout() {
  const s = useSettings();
  const { status } = useStream();
  const { pathname } = useLocation();
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <img src="./favicon.svg" alt="" width={28} height={28} />
            <div className="leading-tight">
              <div className="text-base font-semibold">EEG Neuro-Cognitive Engine</div>
              <div className="text-xs text-ink-3">J26-DS-333 · Component 1</div>
            </div>
          </div>
          <nav className="flex flex-wrap gap-1" aria-label="Pages">
            {NAV.filter((n) => s.advanced || !n.advanced).map((n) => (
              <NavLink key={n.to} to={n.to} end
                className={({ isActive }) => `rounded-md px-3 py-1.5 text-base transition ${isActive ? "bg-accent-soft font-medium text-accent" : "text-ink-2 hover:bg-surface-2 hover:text-ink"}`}>
                {n.label}
                {n.to === "/" && status === "streaming" && <span className="ml-1.5 inline-block h-2 w-2 rounded-full bg-good align-middle" aria-label="streaming" />}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <span className="text-xs text-ink-3">
              {s.apiOnline === null ? "checking API…" : s.apiOnline ? "API online" : "API offline"}
            </span>
            <Toggle label={s.demo ? "Demo (offline)" : "Live API"} on={s.demo} set={s.setDemoMode}
              title="Demo mode reads precomputed outputs from the bundled snapshot instead of the API" />
            <Toggle label="Projector" on={s.projector} set={s.setProjector} title="Larger text for projection" />
            <Toggle label={s.dark ? "Dark" : "Light"} on={s.dark} set={s.setDark} title="Colour theme" />
            <Toggle label="Advanced" on={s.advanced} set={s.setAdvanced}
              title="Show technical details: exact metrics, formulas, model choice, signal-quality and API pages" />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1500px] px-4 py-6">
        <ErrorBoundary key={pathname} label="page"><Outlet /></ErrorBoundary>
      </main>
    </div>
  );
}
