// App-wide settings: theme (light/dark), projector mode (larger type), demo mode and the
// "Advanced" switch (off = simple viva view; on = every technical detail).
// Remembered per browser in localStorage; every access is guarded because storage can be blocked.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { apiReachable, setDemo } from "./api";

interface Settings {
  dark: boolean;
  setDark: (v: boolean) => void;
  projector: boolean;
  setProjector: (v: boolean) => void;
  demo: boolean;
  setDemoMode: (v: boolean) => void;
  advanced: boolean;
  setAdvanced: (v: boolean) => void;
  apiOnline: boolean | null;
}

const Ctx = createContext<Settings | null>(null);

function load(key: string, fallback: boolean): boolean {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : v === "1";
  } catch {
    return fallback;
  }
}
function save(key: string, v: boolean) {
  try {
    localStorage.setItem(key, v ? "1" : "0");
  } catch {
    /* storage unavailable — setting lasts for this visit only */
  }
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [dark, setDarkS] = useState(() =>
    load("eeg.dark", typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches),
  );
  const [projector, setProjectorS] = useState(() => load("eeg.projector", false));
  const [demo, setDemoS] = useState(() => load("eeg.demo", false));
  const [advanced, setAdvancedS] = useState(() => load("eeg.advanced", false));
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);
  useEffect(() => {
    document.documentElement.classList.toggle("projector", projector);
  }, [projector]);
  setDemo(demo);

  useEffect(() => {
    let alive = true;
    apiReachable().then((ok) => {
      if (!alive) return;
      setApiOnline(ok);
      if (!ok) {
        setDemo(true); // nothing to talk to: fall back to the offline snapshot
        setDemoS(true);
      }
    });
    return () => {
      alive = false;
    };
  }, []);

  const value: Settings = {
    dark,
    setDark: (v) => (setDarkS(v), save("eeg.dark", v)),
    projector,
    setProjector: (v) => (setProjectorS(v), save("eeg.projector", v)),
    demo,
    setDemoMode: (v) => (setDemo(v), setDemoS(v), save("eeg.demo", v)),
    advanced,
    setAdvanced: (v) => (setAdvancedS(v), save("eeg.advanced", v)),
    apiOnline,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSettings() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useSettings outside SettingsProvider");
  return c;
}
