import { NavLink, Outlet } from "react-router-dom";
import { useSettings } from "../lib/settings";
import { useStream } from "../lib/stream";

const NAV = [
  { to: "/", label: "Live Monitor" },
  { to: "/brain", label: "Brain View" },
  { to: "/quality", label: "Signal Quality" },
  { to: "/results", label: "Results" },
  { to: "/integration", label: "Integration / API" },
  { to: "/about", label: "About" },
];

function Toggle({ label, on, set, title }: { label: string; on: boolean; set: (v: boolean) => void; title?: string }) {
  return (
    <button type="button" title={title} onClick={() => set(!on)} aria-pressed={on}
      className={`rounded-md border px-2 py-1 text-xs font-medium transition ${on ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-3 hover:text-ink"}`}>
      {label}
    </button>
  );
}

export default function Layout() {
  const s = useSettings();
  const { status } = useStream();
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5">
          <div className="flex items-center gap-2.5">
            <img src="./favicon.svg" alt="" width={26} height={26} />
            <div className="leading-tight">
              <div className="text-sm font-semibold">EEG Neuro-Cognitive Engine</div>
              <div className="text-[11px] text-ink-3">J26-DS-333 · Component 1</div>
            </div>
          </div>
          <nav className="flex flex-wrap gap-1" aria-label="Pages">
            {NAV.map((n) => (
              <NavLink key={n.to} to={n.to} end
                className={({ isActive }) => `rounded-md px-2.5 py-1.5 text-sm transition ${isActive ? "bg-accent-soft font-medium text-accent" : "text-ink-2 hover:bg-surface-2 hover:text-ink"}`}>
                {n.label}
                {n.to === "/" && status === "streaming" && <span className="ml-1.5 inline-block h-1.5 w-1.5 rounded-full bg-good align-middle" aria-label="streaming" />}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <span className="text-[11px] text-ink-3">
              {s.apiOnline === null ? "checking API…" : s.apiOnline ? "API online" : "API offline"}
            </span>
            <Toggle label={s.demo ? "Demo (offline)" : "Live API"} on={s.demo} set={s.setDemoMode}
              title="Demo mode reads precomputed outputs from the bundled snapshot instead of the API" />
            <Toggle label="Projector" on={s.projector} set={s.setProjector} title="Larger text for projection" />
            <Toggle label={s.dark ? "Dark" : "Light"} on={s.dark} set={s.setDark} title="Colour theme" />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-[1500px] px-4 py-5">
        <Outlet />
      </main>
    </div>
  );
}
