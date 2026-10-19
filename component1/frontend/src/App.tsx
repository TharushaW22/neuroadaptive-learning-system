import { HashRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { SettingsProvider } from "./lib/settings";
import { StreamProvider } from "./lib/stream";
import About from "./pages/About";
import BrainView from "./pages/BrainView";
import Integration from "./pages/Integration";
import LiveMonitor from "./pages/LiveMonitor";
import Results from "./pages/Results";
import SignalQuality from "./pages/SignalQuality";

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
              <Route path="quality" element={<SignalQuality />} />
              <Route path="results" element={<Results />} />
              <Route path="integration" element={<Integration />} />
              <Route path="about" element={<About />} />
            </Route>
          </Routes>
        </HashRouter>
      </StreamProvider>
    </SettingsProvider>
  );
}
