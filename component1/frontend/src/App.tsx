import type { ReactNode } from "react";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { SettingsProvider, useSettings } from "./lib/settings";
import { StreamProvider } from "./lib/stream";
import About from "./pages/About";
import BrainView from "./pages/BrainView";
import Integration from "./pages/Integration";
import LiveMonitor from "./pages/LiveMonitor";
import Results from "./pages/Results";
import SignalQuality from "./pages/SignalQuality";

// Advanced-only pages send the visitor back to the Live Monitor while the Advanced switch is off.
function AdvancedOnly({ children }: { children: ReactNode }) {
  return useSettings().advanced ? children : <Navigate to="/" replace />;
}

// HashRouter: the built dashboard works from any static folder, including offline demo mode.
export default function App() {
  return (
    <SettingsProvider>
      <StreamProvider>
        <HashRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<LiveMonitor />} />
              <Route path="brain" element={<BrainView />} />
              <Route path="quality" element={<AdvancedOnly><SignalQuality /></AdvancedOnly>} />
              <Route path="results" element={<Results />} />
              <Route path="integration" element={<AdvancedOnly><Integration /></AdvancedOnly>} />
              <Route path="about" element={<About />} />
            </Route>
          </Routes>
        </HashRouter>
      </StreamProvider>
    </SettingsProvider>
  );
}
