// Catches a render error in one panel so it shows a short message instead of blanking the whole page.
// (React only supports error boundaries as class components.)
import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props { children: ReactNode; label?: string }
interface State { error: Error | null }

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`Panel failed${this.props.label ? ` (${this.props.label})` : ""}:`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="rounded-md border border-bad/40 bg-bad-soft px-4 py-3 text-sm text-bad">
        <div className="font-medium">This panel failed to load.</div>
        <div className="mt-1 text-xs opacity-80">{this.state.error.message}</div>
        <button type="button" onClick={() => this.setState({ error: null })}
          className="mt-2 rounded border border-bad/40 px-2 py-0.5 text-xs font-medium hover:bg-surface">
          Try again
        </button>
      </div>
    );
  }
}
