import { useEffect, useMemo, useState } from 'react'

type ModelInfo = {
  model_id: string
  genome: string
  vocab_size: number
  created_at: string
  size_bytes: number
  metadata: Record<string, unknown>
  loaded: boolean
  registered: boolean
  auto_load: boolean
}

type RuntimeStats = {
  storage_path: string
  manifest_path: string
  registered_models: string[]
  loaded_models: string[]
  registry: Record<string, number>
  orchestrator: {
    models: string[]
    graph: Record<string, { inputs: string[]; outputs: string[]; execution_count: number; output_type: string }>
  }
}

type Prediction = {
  token: string
  probability: number
}

type PredictResponse = {
  model_id: string
  context: string
  predictions: Prediction[]
  runtime: {
    execution_time_ms?: number
    strategy?: string
    models_executed?: string[]
  }
}

type SampleResponse = {
  model_id: string
  seed_text: string
  generated_text: string
  n_tokens: number
  runtime: {
    execution_time_ms?: number
    strategy?: string
    models_executed?: string[]
  }
}

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
    ...options,
  })

  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    throw new Error(payload?.detail ?? `Request failed: ${response.status}`)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json()
}

function formatBytes(bytes: number) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

function pickDefaultModel(models: ModelInfo[]) {
  return models.find((model) => model.loaded)?.model_id ?? models[0]?.model_id ?? ''
}

export function App() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [stats, setStats] = useState<RuntimeStats | null>(null)
  const [selectedModel, setSelectedModel] = useState('')
  const [context, setContext] = useState('1')
  const [seedText, setSeedText] = useState('1')
  const [sampleTokens, setSampleTokens] = useState(64)
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null)
  const [sampleResult, setSampleResult] = useState<SampleResponse | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const graphEdges = useMemo(() => {
    if (!stats) return []
    return Object.entries(stats.orchestrator.graph).flatMap(([source, node]) =>
      node.outputs.length ? node.outputs.map((target) => `${source} → ${target}`) : [`${source} · isolated`],
    )
  }, [stats])

  async function refresh() {
    const [modelPayload, runtimePayload] = await Promise.all([
      request<{ models: ModelInfo[]; total: number }>('/api/v1/models/?load_metadata=true'),
      request<RuntimeStats>('/api/v1/runtime/stats'),
    ])
    setModels(modelPayload.models)
    setStats(runtimePayload)
    setSelectedModel((current) => current || pickDefaultModel(modelPayload.models))
  }

  async function runAction(label: string, action: () => Promise<void>) {
    setBusy(label)
    setError(null)
    try {
      await action()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(null)
    }
  }

  useEffect(() => {
    runAction('boot', refresh)
  }, [])

  async function loadModel(modelId: string) {
    await runAction(`load-${modelId}`, async () => {
      await request(`/api/v1/models/${modelId}/load`, { method: 'POST' })
      await refresh()
    })
  }

  async function unloadModel(modelId: string) {
    await runAction(`unload-${modelId}`, async () => {
      await request(`/api/v1/models/${modelId}/unload`, { method: 'DELETE' })
      await refresh()
    })
  }

  async function registerModel(modelId: string) {
    await runAction(`register-${modelId}`, async () => {
      await request(`/api/v1/models/${modelId}/register`, {
        method: 'POST',
        body: JSON.stringify({ auto_load: true }),
      })
      await refresh()
    })
  }

  async function predict() {
    if (!selectedModel) return
    await runAction('predict', async () => {
      const result = await request<PredictResponse>('/api/v1/inference/predict', {
        method: 'POST',
        body: JSON.stringify({ model_id: selectedModel, context, top_k: 8 }),
      })
      setPredictResult(result)
      await refresh()
    })
  }

  async function sample() {
    if (!selectedModel) return
    await runAction('sample', async () => {
      const result = await request<SampleResponse>('/api/v1/inference/sample', {
        method: 'POST',
        body: JSON.stringify({
          model_id: selectedModel,
          seed_text: seedText,
          n_tokens: sampleTokens,
          temperature: 0.75,
        }),
      })
      setSampleResult(result)
      await refresh()
    })
  }

  return (
    <main className="shell">
      <section className="hero panel">
        <div>
          <p className="eyebrow">MagicBrain Platform</p>
          <h1>Neural runtime command deck</h1>
          <p className="subhead">
            Управление моделями, persistent runtime registry и inference playground в одном интерфейсе.
          </p>
        </div>
        <div className="heroDial">
          <span>{stats?.loaded_models.length ?? 0}</span>
          <small>loaded</small>
        </div>
      </section>

      {error && <div className="error">{error}</div>}

      <section className="metrics">
        <Metric title="Storage" value={stats?.storage_path ?? '—'} tone="green" />
        <Metric title="Registered" value={String(stats?.registered_models.length ?? 0)} tone="amber" />
        <Metric title="Runtime graph" value={`${Object.keys(stats?.orchestrator.graph ?? {}).length} nodes`} tone="blue" />
      </section>

      <section className="grid">
        <div className="panel modelsPanel">
          <div className="sectionHeader">
            <div>
              <p className="eyebrow">Registry</p>
              <h2>Models</h2>
            </div>
            <button className="ghost" onClick={() => runAction('refresh', refresh)} disabled={!!busy}>
              Refresh
            </button>
          </div>

          <div className="modelList">
            {models.length === 0 && <div className="empty">Нет моделей в storage. Создайте модель через API.</div>}
            {models.map((model) => (
              <article
                key={model.model_id}
                className={`modelCard ${selectedModel === model.model_id ? 'selected' : ''}`}
                onClick={() => setSelectedModel(model.model_id)}
              >
                <div>
                  <h3>{model.model_id}</h3>
                  <p>{formatBytes(model.size_bytes)} · vocab {model.vocab_size || '—'}</p>
                </div>
                <div className="badges">
                  <span className={model.loaded ? 'badge hot' : 'badge'}>{model.loaded ? 'loaded' : 'cold'}</span>
                  <span className={model.registered ? 'badge green' : 'badge'}>
                    {model.registered ? 'registered' : 'untracked'}
                  </span>
                  {model.auto_load && <span className="badge amber">auto</span>}
                </div>
                <div className="actions">
                  <button onClick={(event) => { event.stopPropagation(); loadModel(model.model_id) }}>Load</button>
                  <button onClick={(event) => { event.stopPropagation(); unloadModel(model.model_id) }}>Unload</button>
                  <button onClick={(event) => { event.stopPropagation(); registerModel(model.model_id) }}>Auto</button>
                </div>
              </article>
            ))}
          </div>
        </div>

        <div className="panel playground">
          <div className="sectionHeader">
            <div>
              <p className="eyebrow">Inference</p>
              <h2>Playground</h2>
            </div>
            <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
              <option value="">Select model</option>
              {models.map((model) => (
                <option key={model.model_id} value={model.model_id}>{model.model_id}</option>
              ))}
            </select>
          </div>

          <label>
            Context for prediction
            <input value={context} onChange={(event) => setContext(event.target.value)} placeholder="Last known token" />
          </label>
          <button className="primary" onClick={predict} disabled={!selectedModel || !!busy}>
            Predict next token
          </button>

          {predictResult && (
            <div className="predictionBars">
              {predictResult.predictions.map((item) => (
                <div className="bar" key={`${item.token}-${item.probability}`}>
                  <span>{item.token}</span>
                  <div><i style={{ width: `${Math.max(2, item.probability * 100)}%` }} /></div>
                  <strong>{(item.probability * 100).toFixed(1)}%</strong>
                </div>
              ))}
            </div>
          )}

          <div className="splitInputs">
            <label>
              Seed
              <input value={seedText} onChange={(event) => setSeedText(event.target.value)} />
            </label>
            <label>
              Tokens
              <input
                type="number"
                min="1"
                max="512"
                value={sampleTokens}
                onChange={(event) => setSampleTokens(Number(event.target.value))}
              />
            </label>
          </div>
          <button className="primary secondary" onClick={sample} disabled={!selectedModel || !!busy}>
            Generate sample
          </button>

          {sampleResult && <pre className="sampleBox">{sampleResult.generated_text}</pre>}
        </div>

        <div className="panel runtimePanel">
          <div className="sectionHeader">
            <div>
              <p className="eyebrow">Orchestrator</p>
              <h2>Runtime graph</h2>
            </div>
            <span className="pulse">{busy ? 'working' : 'online'}</span>
          </div>

          <div className="graphBox">
            {graphEdges.length === 0 && <div className="empty">Graph is empty until models are loaded.</div>}
            {graphEdges.map((edge) => <div className="edge" key={edge}>{edge}</div>)}
          </div>

          <div className="manifest">
            <h3>Manifest</h3>
            <p>{stats?.manifest_path ?? '—'}</p>
          </div>
        </div>
      </section>
    </main>
  )
}

function Metric({ title, value, tone }: { title: string; value: string; tone: 'green' | 'amber' | 'blue' }) {
  return (
    <article className={`metric ${tone}`}>
      <span>{title}</span>
      <strong>{value}</strong>
    </article>
  )
}
