import { useEffect, useMemo, useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type RunResult = {
  job_id?: string
  status?: string
  terminal_reason?: string
  done?: boolean
  planning_complete?: boolean
  planned_first?: boolean
  steps_run?: number
  cycles_run?: number
  next_action?: string
  next_continue_command?: string | null
  next_supervise_command?: string | null
  provider_preflight?: Record<string, unknown>
  stop_summary?: Record<string, unknown>
  pm_decision?: PmDecision | null
  pm_interventions?: PmDecision[]
  summary?: {
    progress_ratio?: number
    pending_task_count?: number
    completed_task_count?: number
    planning_summary?: {
      ready_for_implementation?: boolean
      task_graph_valid?: boolean | null
      uncovered_small_parts?: Array<Record<string, unknown>>
      uncovered_acceptance_tests?: Array<Record<string, unknown>>
      blocking_items?: Array<Record<string, unknown>>
    }
    next_task?: {
      id?: string
      title?: string
      role?: string
    } | null
  }
  error?: string
}

type PmDecision = {
  action?: string
  reason?: string
  strategy?: string
  summary?: string
  applied?: boolean
  can_apply_automatically?: boolean
  focus_task_id?: string | null
  repeated_cycle_count?: number
  intervention_index?: number
}

type ProgressResult = {
  job_id?: string
  status?: string
  history?: string[]
  outputs_keys?: string[]
  completed_task_ids?: string[]
  checkpoint_count?: number
  last_error?: string | null
  updated_at?: string
  summary?: RunResult['summary'] & {
    total_tasks?: number
    last_error?: string | null
  }
  failure_diagnosis?: {
    classification?: string
    root_cause?: string
    recommended_fix_strategy?: string
    retry_mode?: string
    confidence?: number
    failure_signature?: string | null
    same_failure_repeats?: number
  } | null
  pm_interventions?: PmDecision[]
  recent_audit_events?: Array<{
    timestamp?: string
    event_type?: string
    role?: string
    action?: string
    status?: string
  }>
}

type BackgroundRun = {
  run_id?: string
  status?: string
  job_id?: string
  batches_run?: number
  stop_requested?: boolean
  created_at?: string
  updated_at?: string
  last_result?: RunResult | null
  error?: string
}

type PersistedUiState = {
  requestText?: string
  repoPath?: string
  jobId?: string
  backgroundRunId?: string
  unlimitedCycles?: boolean
  planFirst?: boolean
  usePreflight?: boolean
  maxCycles?: number
  preflightTimeout?: number
}

const defaultRequestTemplate = `作りたいアプリや機能をここに書いてください。

要件:
- まず最小の動く核を作る
- 機能を小さく分割して追加する
- 各ステップでテストを書く
- README にローカル起動手順を書く
`

const defaultRepoPath = '\\\\wsl.localhost\\Ubuntu\\home\\jalan\\wip\\acos-runs\\new-acos-app'
const defaultJobId = 'new-acos-app'
const uiStateStorageKey = 'acos.frontend.uiState.v1'

function normalizeJobId(value: string) {
  const normalized = value
    .trim()
    .replace(/^[\\/]+/, '')
    .replace(/[\\/:\s]+/g, '-')
    .replace(/[^A-Za-z0-9._-]/g, '-')
    .replace(/-+/g, '-')
    .replace(/^[.-]+|[.-]+$/g, '')
  return normalized.slice(0, 128) || defaultJobId
}

function readPersistedUiState(): PersistedUiState {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(uiStateStorageKey)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as PersistedUiState
    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

function persistUiState(state: PersistedUiState) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(uiStateStorageKey, JSON.stringify(state))
}

const initialUiState = readPersistedUiState()

function compactJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

const stageOrder = [
  { key: 'submitted', label: '受付' },
  { key: 'analyzing', label: '要件' },
  { key: 'designing', label: '設計' },
  { key: 'planning', label: '計画' },
  { key: 'implementing', label: '実装' },
  { key: 'writing_tests', label: 'テスト作成' },
  { key: 'reviewing', label: 'レビュー' },
  { key: 'testing', label: 'テスト' },
  { key: 'fixing', label: '修正' },
  { key: 'finalizing', label: '完了処理' },
  { key: 'done', label: '完了' },
]

const activeStatuses = new Set([
  'queued',
  'running',
  'stopping',
  'recovering',
  'diagnosing',
  'replanning',
  'strategy_change',
])
const terminalStatuses = new Set(['done', 'paused', 'stopped', 'error', 'cancelled', 'policy_hard_stop'])

function progressPercent(progress: ProgressResult | null, result: RunResult | null) {
  const ratio = progress?.summary?.progress_ratio ?? result?.summary?.progress_ratio
  if (typeof ratio === 'number' && Number.isFinite(ratio)) {
    return Math.max(0, Math.min(100, Math.round(ratio * 100)))
  }
  const completed = progress?.summary?.completed_task_count ?? result?.summary?.completed_task_count
  const total = progress?.summary?.total_tasks
  if (typeof completed === 'number' && typeof total === 'number' && total > 0) {
    return Math.max(0, Math.min(100, Math.round((completed / total) * 100)))
  }
  return 0
}

function currentStatus(
  progress: ProgressResult | null,
  result: RunResult | null,
  backgroundRun: BackgroundRun | null,
) {
  return progress?.status || result?.status || backgroundRun?.status || 'waiting'
}

function statusTone(status: string) {
  if (['done'].includes(status)) return 'good'
  if (['error', 'policy_hard_stop'].includes(status)) return 'bad'
  if (['paused', 'stopped', 'stopping', 'blocked', 'stuck', 'failed'].includes(status)) return 'warn'
  if (
    [
      'running',
      'queued',
      'submitted',
      'analyzing',
      'designing',
      'planning',
      'replanning',
      'recovering',
      'diagnosing',
      'strategy_change',
      'implementing',
      'writing_tests',
      'reviewing',
      'testing',
      'fixing',
      'finalizing',
    ].includes(status)
  )
    return 'live'
  return 'idle'
}

function stageState(stageKey: string, status: string, history: string[] = []) {
  if (stageKey === status) return 'current'
  if (history.includes(stageKey) || stageOrder.findIndex((stage) => stage.key === stageKey) < stageOrder.findIndex((stage) => stage.key === status)) {
    return 'done'
  }
  return 'pending'
}

function formatTime(value?: string) {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function describeProgressGap(
  progress: ProgressResult | null,
  progressError: string | null,
  backgroundRun: BackgroundRun | null,
) {
  if (progress) return null
  if (progressError) return progressError
  if (backgroundRun?.status === 'running' && (backgroundRun.batches_run ?? 0) === 0) {
    return 'ジョブ初期化または最初の計画レスポンス待ち'
  }
  return '実行を開始すると、ジョブ状態を数秒ごとに読み込みます。'
}

function App() {
  const [requestText, setRequestText] = useState(initialUiState.requestText || defaultRequestTemplate)
  const [repoPath, setRepoPath] = useState(initialUiState.repoPath || defaultRepoPath)
  const [jobId, setJobId] = useState(normalizeJobId(initialUiState.jobId || defaultJobId))
  const [maxCycles, setMaxCycles] = useState(initialUiState.maxCycles || 12)
  const [unlimitedCycles, setUnlimitedCycles] = useState(initialUiState.unlimitedCycles ?? false)
  const [batchesRun, setBatchesRun] = useState(0)
  const [preflightTimeout, setPreflightTimeout] = useState(initialUiState.preflightTimeout || 180)
  const [usePreflight, setUsePreflight] = useState(initialUiState.usePreflight ?? true)
  const [planFirst, setPlanFirst] = useState(initialUiState.planFirst ?? true)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<RunResult | null>(null)
  const [progress, setProgress] = useState<ProgressResult | null>(null)
  const [backgroundRun, setBackgroundRun] = useState<BackgroundRun | null>(
    initialUiState.backgroundRunId
      ? { run_id: initialUiState.backgroundRunId, status: 'queued' }
      : null,
  )
  const [progressError, setProgressError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const payload = useMemo(
    () => ({
      request_text: requestText,
      repo_path: repoPath,
      workspace_root: repoPath,
      target_branch: `acos/${normalizeJobId(jobId)}`,
      job_id: normalizeJobId(jobId),
      title: normalizeJobId(jobId),
      jobs_dir: '.acos/jobs-ui',
      max_cycles: maxCycles,
      steps_per_cycle: 1,
      max_stalled_cycles: 3,
      pm_stall_recovery: true,
      max_runtime_seconds: 3600,
      max_autonomous_stages: 256,
      summary_file: '.acos/ui-last-summary.json',
      summary_dir: '.acos/ui-cycles',
      plan_first: planFirst,
      preflight_provider: usePreflight ? 'local_ornith' : null,
      preflight_timeout: preflightTimeout,
      require_prd_quality: true,
      stage_review: true,
      metadata: {
        source: 'acos_frontend',
        requested_app: 'english_vocab_test',
      },
    }),
    [jobId, maxCycles, planFirst, preflightTimeout, repoPath, requestText, usePreflight],
  )

  const command = useMemo(() => {
    const commandJobId = normalizeJobId(jobId)
    const args = [
      'acos',
      'run-supervised',
      '--request',
      `"${requestText.replaceAll('"', '\\"')}"`,
      '--repo-path',
      repoPath,
      '--job-id',
      commandJobId,
      '--jobs-dir',
      '.acos/jobs-ui',
      '--max-cycles',
      String(maxCycles),
      '--steps-per-cycle',
      '1',
      '--pm-stall-recovery',
      '--summary-file',
      '.acos/ui-last-summary.json',
      '--summary-dir',
      '.acos/ui-cycles',
    ]
    if (planFirst) args.push('--plan-first')
    if (usePreflight) {
      args.push('--preflight-provider', 'local_ornith')
      args.push('--preflight-timeout', String(preflightTimeout))
    }
    const baseCommand = args.join(' ')
    if (!unlimitedCycles) return baseCommand
    return `${baseCommand}\n# 無制限モード: APIサーバー側のバックグラウンドワーカーで継続実行`
  }, [jobId, maxCycles, planFirst, preflightTimeout, repoPath, requestText, unlimitedCycles, usePreflight])

  async function runJob(event: FormEvent) {
    event.preventDefault()
    const jobIdForRun = normalizeJobId(jobId)
    setJobId(jobIdForRun)
    const runPayload = {
      ...payload,
      target_branch: `acos/${jobIdForRun}`,
      job_id: jobIdForRun,
      title: jobIdForRun,
    }
    persistUiState({
      requestText,
      repoPath,
      jobId: jobIdForRun,
      unlimitedCycles,
      planFirst,
      usePreflight,
      maxCycles,
      preflightTimeout,
      backgroundRunId: undefined,
    })
    setIsRunning(true)
    setBatchesRun(0)
    setError(null)
    setResult(null)
    setProgress(null)
    setBackgroundRun(null)
    setProgressError(null)
    if (unlimitedCycles) {
      try {
        const response = await fetch('/api/jobs/supervised/background', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...runPayload, max_batches: null }),
        })
        const body = await response.json()
        if (!response.ok) {
          throw new Error(body.detail || 'ACOS background request failed')
        }
        setBackgroundRun(body)
        persistUiState({
          requestText,
          repoPath,
          jobId: jobIdForRun,
          unlimitedCycles,
          planFirst,
          usePreflight,
          maxCycles,
          preflightTimeout,
          backgroundRunId: body.run_id,
        })
        setBatchesRun(body.batches_run ?? 0)
        void refreshProgress()
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : 'Unknown error')
        setIsRunning(false)
      }
      return
    }
    try {
      const response = await fetch('/api/jobs/supervised', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(runPayload),
      })
      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail || 'ACOS API request failed')
      }
      setResult(body)
      persistUiState({
        requestText,
        repoPath,
        jobId: jobIdForRun,
        unlimitedCycles,
        planFirst,
        usePreflight,
        maxCycles,
        preflightTimeout,
        backgroundRunId: undefined,
      })
      setBatchesRun(1)
      void refreshProgress()
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unknown error')
    } finally {
      setIsRunning(false)
    }
  }

  async function refreshBackgroundRun(runId: string) {
    try {
      const response = await fetch(`/api/background-runs/${encodeURIComponent(runId)}`)
      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail || 'background status request failed')
      }
      setBackgroundRun(body)
      if (body.job_id) {
        setJobId(normalizeJobId(body.job_id))
      }
      setBatchesRun(body.batches_run ?? 0)
      if (body.last_result) setResult(body.last_result)
      const nextStatus = String(body.status || '')
      if (terminalStatuses.has(nextStatus)) {
        setIsRunning(false)
      } else if (activeStatuses.has(nextStatus)) {
        setIsRunning(true)
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'background status error')
      setBackgroundRun(null)
      setIsRunning(false)
    }
  }

  async function reconnectBackgroundRun() {
    if (!jobId.trim()) return
    try {
      const currentJobId = normalizeJobId(jobId)
      const response = await fetch(
        `/api/background-runs?job_id=${encodeURIComponent(currentJobId)}`,
      )
      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail || 'background list request failed')
      }
      const latest = Array.isArray(body) ? body[0] : null
      if (!latest) return
      setBackgroundRun(latest)
      if (latest.job_id) {
        setJobId(normalizeJobId(latest.job_id))
      }
      setBatchesRun(latest.batches_run ?? 0)
      if (latest.last_result) setResult(latest.last_result)
      setIsRunning(activeStatuses.has(String(latest.status)))
    } catch (caught) {
      setProgressError(caught instanceof Error ? caught.message : 'background reconnect error')
    }
  }

  async function requestStop() {
    if (!backgroundRun?.run_id) return
    try {
      const response = await fetch(
        `/api/background-runs/${encodeURIComponent(backgroundRun.run_id)}/stop`,
        { method: 'POST' },
      )
      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail || 'stop request failed')
      }
      setBackgroundRun(body)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'stop request error')
    }
  }

  async function refreshProgress() {
    const currentJobId = normalizeJobId(jobId)
    if (!currentJobId) return
    try {
      const response = await fetch(
        `/api/jobs/${encodeURIComponent(currentJobId)}/progress?jobs_dir=${encodeURIComponent(
          payload.jobs_dir,
        )}`,
      )
      if (response.status === 404) {
        setProgressError('ジョブファイル作成待ち')
        return
      }
      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail || 'progress request failed')
      }
      setProgress(body)
      setProgressError(null)
    } catch (caught) {
      setProgressError(caught instanceof Error ? caught.message : 'progress error')
    }
  }

  useEffect(() => {
    if (!isRunning) return
    void refreshProgress()
    const id = window.setInterval(() => {
      void refreshProgress()
    }, 3000)
    return () => window.clearInterval(id)
  }, [isRunning, jobId, payload.jobs_dir])

  useEffect(() => {
    if (!backgroundRun?.run_id) return
    void refreshBackgroundRun(backgroundRun.run_id)
    const id = window.setInterval(() => {
      void refreshBackgroundRun(backgroundRun.run_id!)
      void refreshProgress()
    }, 3000)
    return () => window.clearInterval(id)
  }, [backgroundRun?.run_id])

  useEffect(() => {
    void reconnectBackgroundRun()
    void refreshProgress()
  }, [])

  useEffect(() => {
    persistUiState({
      requestText,
      repoPath,
      jobId: normalizeJobId(jobId),
      unlimitedCycles,
      planFirst,
      usePreflight,
      maxCycles,
      preflightTimeout,
      backgroundRunId: backgroundRun?.run_id,
    })
  }, [
    backgroundRun?.run_id,
    jobId,
    maxCycles,
    planFirst,
    preflightTimeout,
    repoPath,
    requestText,
    unlimitedCycles,
    usePreflight,
  ])

  useEffect(() => {
    if (isRunning || backgroundRun?.run_id) return
    const id = window.setInterval(() => {
      void refreshProgress()
    }, 10000)
    return () => window.clearInterval(id)
  }, [backgroundRun?.run_id, isRunning, jobId, payload.jobs_dir])

  const liveSummary = progress?.summary ?? result?.summary
  const planning = liveSummary?.planning_summary
  const stop = result?.stop_summary
  const status = currentStatus(progress, result, backgroundRun)
  const tone = statusTone(status)
  const percent = progressPercent(progress, result)
  const taskTotal = progress?.summary?.total_tasks ?? 0
  const taskDone = progress?.summary?.completed_task_count ?? result?.summary?.completed_task_count ?? 0
  const taskPending = progress?.summary?.pending_task_count ?? result?.summary?.pending_task_count ?? 0
  const progressGap = describeProgressGap(progress, progressError, backgroundRun)
  const lastError = progress?.last_error ?? progress?.summary?.last_error ?? result?.error ?? backgroundRun?.error
  const diagnosis = progress?.failure_diagnosis
  const workerActive = backgroundRun?.status ? activeStatuses.has(backgroundRun.status) : false
  const workerTerminal = backgroundRun?.status ? terminalStatuses.has(backgroundRun.status) : false
  const latestPmDecision =
    progress?.pm_interventions?.slice(-1)[0] ??
    result?.pm_decision ??
    result?.pm_interventions?.slice(-1)[0]

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="kicker">ACOS Frontend</p>
            <h1>要件定義から自律実行まで流す</h1>
          </div>
          <div className="status-pill">
            {isRunning
              ? `実行中${backgroundRun?.status ? `: ${backgroundRun.status}` : progress?.status ? `: ${progress.status}` : ''}`
              : '待機中'}
          </div>
        </header>

        <section className={`mobile-status-card ${tone}`}>
          <div className="mobile-status-main">
            <div>
              <span>status</span>
              <strong>{status}</strong>
            </div>
            <div>
              <span>tasks</span>
              <strong>
                {taskDone}/{taskTotal || '-'}
              </strong>
            </div>
            <div>
              <span>worker</span>
              <strong>{backgroundRun?.status || (isRunning ? 'running' : 'idle')}</strong>
            </div>
          </div>
          <div className="mobile-progress-bar" aria-label="mobile job progress">
            <div style={{ width: `${percent}%` }} />
          </div>
          <div className="mobile-status-detail">
            <span>{percent}%</span>
            <span>{formatTime(progress?.updated_at || backgroundRun?.updated_at)}</span>
          </div>
          {progress?.summary?.next_task?.id && (
            <p>
              {progress.summary.next_task.id}: {progress.summary.next_task.title}
            </p>
          )}
          {lastError && <p className="mobile-error">last error: {lastError}</p>}
          {diagnosis?.root_cause && <p className="mobile-diagnosis">{diagnosis.root_cause}</p>}
          <div className="mobile-actions">
            <button className="ghost-button" type="button" onClick={() => void refreshProgress()}>
              更新
            </button>
            <button className="ghost-button" type="button" onClick={() => void reconnectBackgroundRun()}>
              再接続
            </button>
            {backgroundRun?.run_id && !workerTerminal && (
              <button className="ghost-button danger-action" type="button" onClick={requestStop}>
                停止
              </button>
            )}
          </div>
        </section>

        <form className="layout" onSubmit={runJob}>
          <section className="panel requirement-panel">
            <div className="panel-heading">
              <div>
                <p className="section-label">Requirement</p>
                <h2>ユーザー要件</h2>
              </div>
              <button
                className="ghost-button"
                type="button"
                onClick={() => setRequestText(defaultRequestTemplate)}
              >
                テンプレート
              </button>
            </div>
            <textarea
              value={requestText}
              onChange={(event) => setRequestText(event.target.value)}
              spellCheck={false}
            />
          </section>

          <aside className="panel controls-panel">
            <p className="section-label">Run Settings</p>
            <label>
              生成先 repo_path
              <input value={repoPath} onChange={(event) => setRepoPath(event.target.value)} />
            </label>
            <label>
              job_id
              <input
                value={jobId}
                onBlur={() => setJobId(normalizeJobId(jobId))}
                onChange={(event) => setJobId(event.target.value)}
              />
            </label>
            <div className="split">
              <label>
                {unlimitedCycles ? 'cycles / batch' : 'cycles'}
                <input
                  min={1}
                  type="number"
                  value={maxCycles}
                  onChange={(event) => setMaxCycles(Number(event.target.value))}
                />
              </label>
              <label>
                preflight 秒
                <input
                  min={1}
                  type="number"
                  value={preflightTimeout}
                  onChange={(event) => setPreflightTimeout(Number(event.target.value))}
                />
              </label>
            </div>
            <label className="check-row">
              <input
                checked={unlimitedCycles}
                type="checkbox"
                onChange={(event) => setUnlimitedCycles(event.target.checked)}
              />
              無制限に続ける
            </label>
            <label className="check-row">
              <input
                checked={planFirst}
                type="checkbox"
                onChange={(event) => setPlanFirst(event.target.checked)}
              />
              plan-first で要件定義を先に通す
            </label>
            <label className="check-row">
              <input
                checked={usePreflight}
                type="checkbox"
                onChange={(event) => setUsePreflight(event.target.checked)}
              />
              Ornith の事前疎通を確認する
            </label>
            <button className="run-button" disabled={isRunning || !requestText.trim()} type="submit">
              {isRunning ? 'ACOS 実行中' : 'ACOS に渡す'}
            </button>
            {isRunning && unlimitedCycles && (
              <button className="ghost-button" type="button" onClick={requestStop}>
                強制停止
              </button>
            )}
          </aside>
        </form>

        <section className="panel command-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">Equivalent CLI</p>
              <h2>同等の CLI コマンド</h2>
            </div>
          </div>
          <pre>{command}</pre>
        </section>

        <section className="panel live-panel">
          <div className="panel-heading">
            <div>
              <p className="section-label">Live Progress</p>
              <h2>実行中の進捗</h2>
            </div>
            <button className="ghost-button" type="button" onClick={() => void refreshProgress()}>
              更新
            </button>
            <button className="ghost-button" type="button" onClick={() => void reconnectBackgroundRun()}>
              worker再接続
            </button>
          </div>
          <div className={`progress-overview ${tone}`}>
            <div className="progress-headline">
              <div>
                <span>現在の状態</span>
                <strong>{status}</strong>
              </div>
              <div>
                <span>タスク</span>
                <strong>
                  {taskDone}/{taskTotal || '-'}
                </strong>
              </div>
              <div>
                <span>worker</span>
                <strong>{backgroundRun?.status || (isRunning ? 'running' : 'idle')}</strong>
              </div>
              <div>
                <span>最終更新</span>
                <strong>{formatTime(progress?.updated_at || backgroundRun?.updated_at)}</strong>
              </div>
            </div>
            <div className="progress-bar" aria-label="job progress">
              <div style={{ width: `${percent}%` }} />
            </div>
            <div className="progress-caption">
              <span>{percent}%</span>
              <span>
                {workerActive
                  ? '処理中'
                  : workerTerminal
                    ? 'worker停止'
                    : '待機'}
                {taskPending ? ` / pending ${taskPending}` : ''}
              </span>
            </div>
          </div>
          <div className="stage-strip">
            {stageOrder.map((stage) => (
              <div
                className={`stage ${stageState(stage.key, status, progress?.history)}`}
                key={stage.key}
              >
                <span />
                <strong>{stage.label}</strong>
              </div>
            ))}
          </div>
          {progressGap && !progress && <div className="progress-note">{progressGap}</div>}
          {lastError && <div className="error-box live-error">last error: {lastError}</div>}
          {diagnosis && (
            <div className="diagnosis-card">
              <div>
                <span>diagnosis</span>
                <strong>{diagnosis.classification || '-'}</strong>
              </div>
              <div>
                <span>root cause</span>
                <strong>{diagnosis.root_cause || '-'}</strong>
              </div>
              <div>
                <span>fix strategy</span>
                <strong>{diagnosis.recommended_fix_strategy || '-'}</strong>
              </div>
              <div>
                <span>retry</span>
                <strong>
                  {[diagnosis.retry_mode, diagnosis.failure_signature]
                    .filter(Boolean)
                    .join(' / ') || '-'}
                </strong>
              </div>
            </div>
          )}
          {progress && (
            <>
              <div className="live-metrics">
                <div>
                  <span>status</span>
                  <strong>{progress.status || '-'}</strong>
                </div>
                <div>
                  <span>tasks</span>
                  <strong>
                    {progress.summary?.completed_task_count ?? 0}/
                    {progress.summary?.total_tasks ?? 0}
                  </strong>
                </div>
                <div>
                  <span>pending</span>
                  <strong>{taskPending}</strong>
                </div>
                <div>
                  <span>outputs</span>
                  <strong>{progress.outputs_keys?.length ?? 0}</strong>
                </div>
                <div>
                  <span>batches</span>
                  <strong>{batchesRun}</strong>
                </div>
                {backgroundRun && (
                  <div>
                    <span>worker</span>
                    <strong>{backgroundRun.status || '-'}</strong>
                  </div>
                )}
              </div>
              {backgroundRun?.stop_requested && (
                <div className="progress-note">停止要求を受け付けました。現在のバッチが終わり次第止まります。</div>
              )}
              <div className="progress-detail">
                <div>
                  <span>次のタスク</span>
                  <strong>
                    {progress.summary?.next_task?.id
                      ? `${progress.summary.next_task.id}: ${progress.summary.next_task.title}`
                      : '-'}
                  </strong>
                </div>
                <div>
                  <span>履歴</span>
                  <strong>{progress.history?.join(' -> ') || '-'}</strong>
                </div>
                <div>
                  <span>保存済み outputs</span>
                  <strong>{progress.outputs_keys?.join(', ') || '-'}</strong>
                </div>
                <div>
                  <span>最終更新</span>
                  <strong>{progress.updated_at || '-'}</strong>
                </div>
              </div>
              {latestPmDecision && (
                <div className="pm-decision">
                  <div>
                    <span>PM判断</span>
                    <strong>{latestPmDecision.strategy || latestPmDecision.action || '-'}</strong>
                  </div>
                  <p>{latestPmDecision.summary || latestPmDecision.reason || '-'}</p>
                  <small>
                    applied: {String(latestPmDecision.applied ?? false)}
                    {latestPmDecision.focus_task_id ? ` / task: ${latestPmDecision.focus_task_id}` : ''}
                  </small>
                </div>
              )}
              <div className="event-list">
                {(progress.recent_audit_events || []).slice(-6).map((event, index) => (
                  <div key={`${event.timestamp}-${index}`}>
                    <span>{formatTime(event.timestamp)}</span>
                    <strong>{event.role || 'system'}</strong>
                    <em>{event.event_type || '-'}</em>
                    <b>{event.status || '-'}</b>
                    <small>{event.action || '-'}</small>
                  </div>
                ))}
              </div>
            </>
          )}
        </section>

        <section className="result-grid">
          <article className="panel result-card">
            <p className="section-label">Result</p>
            {error && <div className="error-box">{error}</div>}
            {!error && !result && (
              <div className="empty-box">要件を入力して ACOS に渡すと、ここに結果が出ます。</div>
            )}
            {result && (
              <div className="metrics">
                <div>
                  <span>status</span>
                  <strong>{result.status || '-'}</strong>
                </div>
                <div>
                  <span>terminal</span>
                  <strong>{result.terminal_reason || '-'}</strong>
                </div>
                <div>
                  <span>planning</span>
                  <strong>{String(result.planning_complete ?? '-')}</strong>
                </div>
                <div>
                  <span>cycles</span>
                  <strong>{result.cycles_run ?? 0}</strong>
                </div>
              </div>
            )}
          </article>

          <article className="panel result-card">
            <p className="section-label">Next Action</p>
            {result ? (
              <>
                <h2>{result.next_action || stop?.resume_action?.toString() || 'none'}</h2>
                <p>{stop?.operator_command?.toString() || result.next_continue_command || '次の手動操作はありません。'}</p>
              </>
            ) : (
              <div className="empty-box">実行後に次アクションを表示します。</div>
            )}
          </article>

          <article className="panel result-card wide">
            <p className="section-label">Planning Gate</p>
            {planning ? (
              <div className="planning-list">
                <div>
                  <span>ready</span>
                  <strong>{String(planning.ready_for_implementation)}</strong>
                </div>
                <div>
                  <span>task graph</span>
                  <strong>{String(planning.task_graph_valid)}</strong>
                </div>
                <div>
                  <span>uncovered small parts</span>
                  <strong>{planning.uncovered_small_parts?.length ?? 0}</strong>
                </div>
                <div>
                  <span>uncovered tests</span>
                  <strong>{planning.uncovered_acceptance_tests?.length ?? 0}</strong>
                </div>
              </div>
            ) : (
              <div className="empty-box">計画ゲートの結果がここに出ます。</div>
            )}
          </article>
        </section>

        {result && (
          <section className="panel json-panel">
            <p className="section-label">Raw JSON</p>
            <pre>{compactJson(result)}</pre>
          </section>
        )}
      </section>
    </main>
  )
}

export default App
