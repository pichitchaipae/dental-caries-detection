import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/global.css';

// Intercepts POST/GET /process at the network level (MSW) so the real,
// unmodified processClient.ts can be built and demoed against before Naris's
// backend routes exist — see docs-md/task-pm-phase1.md's INT-2/msw note.
// Never active in a production build (`vite build`/`preview`): import.meta.env.DEV
// is false there, so this whole branch is dead code, and Vite tree-shakes the
// dynamic import away.
async function enableMockingIfNeeded(): Promise<void> {
  const mockingFlag = import.meta.env.VITE_API_MOCKING ?? 'enabled';
  if (!import.meta.env.DEV || mockingFlag !== 'enabled') return;

  const { worker } = await import('./mocks/browser');
  await worker.start({ onUnhandledRequest: 'bypass' });
}

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element #root not found');
}

void enableMockingIfNeeded().then(() => {
  createRoot(rootElement).render(
    <StrictMode>
      <App />
    </StrictMode>
  );
});
