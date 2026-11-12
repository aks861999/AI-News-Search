import {useEffect, useMemo, useState} from 'react';
import {A2uiSurface, type ReactCatalogComponent} from '@a2ui/react';
import {Catalog, MessageProcessor, type SurfaceModel} from '@a2ui/web_core';
import {BASIC_FUNCTIONS, basicCatalog} from '@a2ui/web_core/catalogs/basic/v1';
import {renderMarkdown} from '@a2ui/markdown-it';
import {setMarkdownRenderer} from '@a2ui/web_core/v0_9/basic_catalog';

// Sanitized markdown rendering for the Basic catalog's Text components
// (falls back to plain text if not configured).
setMarkdownRenderer(renderMarkdown);

// The Basic Catalog covers every component the backend's A2UI_SYSTEM_PROMPT
// asks the model to emit (Card, Column, Row, Text, Icon, Divider, List...).
const catalog = new Catalog<ReactCatalogComponent>(
  'https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json',
  '1.0',
  [...basicCatalog.components.values()],
  BASIC_FUNCTIONS,
);

type AskResponse = {
  model: string;
  a2ui: unknown[];
  raw_sources_count: number;
};

export default function App() {
  const processor = useMemo(
    () =>
      new MessageProcessor<ReactCatalogComponent>([catalog], undefined, {
        outboundListener: (message) => console.log('Outbound:', message),
      }),
    [],
  );

  const [surface, setSurface] = useState<SurfaceModel<ReactCatalogComponent>>();
  const [question, setQuestion] = useState('MercadoLibre revenue growth');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<AskResponse | null>(null);

  useEffect(() => {
    const sub = processor.onSurfaceCreated(setSurface);
    return () => sub.unsubscribe();
  }, [processor]);

  async function ask(q: string) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q, category: 'Web (text)', timelimit: 'm', max_results: 8}),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `HTTP ${res.status}`);
      }
      const data: AskResponse = await res.json();
      setMeta(data);
      // Feed the model's A2UI v1.0 messages into the processor.
      // processMessages takes a payload (single message or array); order matters
      // (createSurface before updateComponents).
      const messages = Array.isArray(data.a2ui) ? data.a2ui : [data.a2ui];
      processor.processMessages(messages as Record<string, unknown>[]);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>📈 Yahoo Finance A2UI</h1>
        <p style={styles.subtitle}>
          Ask a finance question — live Yahoo results, synthesized by{' '}
          <code style={styles.code}>{meta?.model ?? 'Nemotron'}</code> into A2UI.
        </p>
      </header>

      <form
        style={styles.form}
        onSubmit={(e) => {
          e.preventDefault();
          if (question.trim()) void ask(question.trim());
        }}
      >
        <input
          style={styles.input}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. MercadoLibre revenue growth"
          disabled={loading}
        />
        <button style={styles.button} type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Searching…' : 'Ask'}
        </button>
      </form>

      {error && <div style={styles.error}>⚠️ {error}</div>}
      {loading && <div style={styles.loading}>🔎 Fetching Yahoo results and synthesizing A2UI…</div>}

      <main style={styles.surfaceArea}>
        {surface ? (
          <A2uiSurface surface={surface} />
        ) : (
          !loading &&
          !error && (
            <div style={styles.placeholder}>
              Ask a question to render an A2UI surface here.
            </div>
          )
        )}
      </main>

      {meta && (
        <footer style={styles.footer}>
          model: <code>{meta.model}</code> · sources: {meta.raw_sources_count}
        </footer>
      )}
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    maxWidth: 860,
    margin: '0 auto',
    padding: '24px 16px 48px',
    fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
  },
  header: {marginBottom: 16},
  title: {margin: '0 0 4px', fontSize: 26},
  subtitle: {margin: 0, color: '#64748b', fontSize: 14},
  code: {background: '#f1f5f9', padding: '1px 5px', borderRadius: 4, fontSize: 13},
  form: {display: 'flex', gap: 8, margin: '16px 0'},
  input: {
    flex: 1,
    padding: '10px 12px',
    fontSize: 15,
    border: '1px solid #cbd5e1',
    borderRadius: 8,
  },
  button: {
    padding: '10px 18px',
    fontSize: 15,
    border: 'none',
    borderRadius: 8,
    background: '#2563eb',
    color: 'white',
    cursor: 'pointer',
  },
  error: {
    background: '#fef2f2',
    color: '#b91c1c',
    border: '1px solid #fecaca',
    borderRadius: 8,
    padding: 12,
    margin: '12px 0',
  },
  loading: {color: '#64748b', margin: '12px 0'},
  surfaceArea: {minHeight: 200, marginTop: 8},
  placeholder: {
    border: '2px dashed #e2e8f0',
    borderRadius: 12,
    padding: 40,
    textAlign: 'center' as const,
    color: '#94a3b8',
  },
  footer: {marginTop: 24, fontSize: 12, color: '#94a3b8'},
};
