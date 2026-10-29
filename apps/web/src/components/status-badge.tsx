import type { ServerStatus } from "../lib/use-server-status";

const LABELS: Record<ServerStatus, string> = {
  checking: "Checking the server…",
  healthy: "Server healthy",
  degraded: "Server degraded: a dependency is not ready",
  unreachable: "Server unreachable",
};

/** Server status in words (never colour alone), announced politely on change. */
export function StatusBadge({ status }: { status: ServerStatus }) {
  return (
    <p role="status" aria-live="polite" className="status-badge" data-status={status}>
      <span className="status-badge__dot" aria-hidden="true" />
      {LABELS[status]}
    </p>
  );
}
