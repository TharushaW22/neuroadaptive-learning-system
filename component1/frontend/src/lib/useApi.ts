import { useEffect, useState } from "react";
import { getJSON } from "./api";
import { useSettings } from "./settings";

/** Fetch JSON from the API (or the demo snapshot). Refetches when demo mode changes. */
export function useApi<T>(path: string | null) {
  const { demo } = useSettings();
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    if (!path) return;
    let alive = true;
    setLoading(true);
    setError(null);
    getJSON<T>(path)
      .then((d) => alive && setData(d))
      .catch((e: Error) => alive && setError(e.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [path, demo]);
  return { data, error, loading };
}
