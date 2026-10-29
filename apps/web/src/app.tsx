import { StatusBadge } from "./components/status-badge";
import { API_BASE_URL } from "./lib/api/client";
import { useServerStatus } from "./lib/use-server-status";

export function App() {
  const status = useServerStatus();

  return (
    <main className="page">
      <h1>Parlora Tutor</h1>
      <p>
        This is the Parlora Tutor dev client, a minimal page for development and demos. The
        learner-facing app is Parlora Hub.
      </p>
      <StatusBadge status={status} />
      <p className="page__meta">
        API: <code>{API_BASE_URL}</code>
      </p>
    </main>
  );
}
