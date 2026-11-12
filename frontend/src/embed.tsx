import {useEffect, useMemo, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {A2uiSurface, type ReactCatalogComponent} from '@a2ui/react';
import {Catalog, MessageProcessor, type SurfaceModel} from '@a2ui/web_core';
import {BASIC_FUNCTIONS, basicCatalog} from '@a2ui/web_core/catalogs/basic/v1';
import {renderMarkdown} from '@a2ui/markdown-it';
import {setMarkdownRenderer} from '@a2ui/web_core/v0_9/basic_catalog';

// Sanitized markdown rendering for the Basic catalog's Text components.
setMarkdownRenderer(renderMarkdown);

// The host (Streamlit iframe) injects the model's A2UI messages here:
declare global {
  interface Window {
    A2UI_MESSAGES?: unknown[];
  }
}

const catalog = new Catalog<ReactCatalogComponent>(
  'https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json',
  '1.0',
  [...basicCatalog.components.values()],
  BASIC_FUNCTIONS,
);

function Embed() {
  const processor = useMemo(
    () => new MessageProcessor<ReactCatalogComponent>([catalog], undefined, {}),
    [],
  );
  const [surface, setSurface] = useState<SurfaceModel<ReactCatalogComponent>>();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const sub = processor.onSurfaceCreated(setSurface);
    const messages = window.A2UI_MESSAGES;
    if (!Array.isArray(messages) || messages.length === 0) {
      setError('No A2UI messages were passed to the renderer.');
    } else {
      try {
        processor.processMessages(messages as Record<string, unknown>[]);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }
    return () => sub.unsubscribe();
  }, [processor]);

  if (error) return <div style={styles.error}>⚠️ {error}</div>;
  return surface ? (
    <div style={styles.wrap}>
      <A2uiSurface surface={surface} />
    </div>
  ) : (
    <div style={styles.hint}>Rendering A2UI…</div>
  );
}

const styles = {
  wrap: {
    padding: '16px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
  },
  error: {
    padding: 16,
    color: '#b91c1c',
    fontFamily: 'system-ui, sans-serif',
  },
  hint: {padding: 16, color: '#94a3b8', fontFamily: 'system-ui, sans-serif'},
} as const;

createRoot(document.getElementById('root')!).render(<Embed />);
