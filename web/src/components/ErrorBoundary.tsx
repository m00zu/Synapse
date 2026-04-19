import React from "react";

interface Props { children: React.ReactNode; }
interface State { error: Error | null; }

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Logged to devtools so developers can inspect — toasts don't carry stacks.
    console.error("uncaught React error:", error, info);
  }

  private reset = () => {
    this.setState({ error: null });
    window.location.reload();
  };

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen bg-bg text-fg flex items-center justify-center p-6">
          <div className="max-w-md w-full bg-bg2 border border-red-500 rounded p-4">
            <h2 className="text-red-400 font-semibold mb-2">Synapse crashed</h2>
            <div className="text-xs font-mono text-fg/80 mb-4 whitespace-pre-wrap">
              {this.state.error.message}
            </div>
            <button
              onClick={this.reset}
              className="px-3 py-1 bg-accent text-bg rounded text-sm"
            >
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
