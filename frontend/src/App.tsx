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
  last_result?: RunResult | null
  error?: string
}

const englishVocabTemplate = `英単語テストアプリを作りたい。

対象:
- 生徒が英単語を覚えるための Web アプリ
- 先生が単語セットとテストを管理できる

生徒向け機能:
- 日英、英日、それぞれ4択問題を出す
- 文章の空欄に入るべき英単語を4択で選ばせる
- 英日では択一だけでなく、日本語を自由入力して答える問題を出す
- 日本語自由入力は、漢字・ひらがな・言い換えで不当に不正解にならないよう、軽量LLMで意味採点する
- LLM採点が使えない場合は、同義語・表記ゆれを少し許容するフォールバックを用意する
- 生徒ごとに、単語を覚えたか、苦手か、最後に解いた日、正答率を管理する
- 覚えていない単語は何度も出題する
- 忘却曲線に沿って、時間が経つと復習対象に戻す
- 学習画面では今日やるべき単語、正答率、復習予定を見られる

先生向け機能:
- 単語セットを複数管理できる
- セットごとに単語、英語、日本語訳、例文、選択肢用の誤答候補を編集できる
- クラスまたは生徒ごとの進捗を見られる
- テストに使う単語セットを指定できる
- 生徒が苦手な単語を確認できる

LLM採点:
- 軽い OpenAI-compatible なローカルモデルを想定する
- 採点APIのURL、モデル名、APIキーは環境変数で変更できる
- 採点結果は correct, confidence, reason を返す
- 採点時は正解例、日本語訳、学習者の回答を渡し、意味が合っているかを見る

実装要件:
- フロントエンドつき
- まず最小の動く核を作る
- 機能を小さく分割して追加する
- 各ステップでテストを書く
- バックエンド、フロントエンド、READMEを含める
- 開発用にサンプル単語セットを複数用意する
- ローカルで起動できる手順をREADMEに書く
`

const defaultRepoPath = '/Users/tachibanashunta/wip/acos/tmp_runs/english-vocab-test-app'

function compactJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function App() {
  const [requestText, setRequestText] = useState(englishVocabTemplate)
  const [repoPath, setRepoPath] = useState(defaultRepoPath)
  const [jobId, setJobId] = useState('english-vocab-test-app')
  const [maxCycles, setMaxCycles] = useState(12)
  const [unlimitedCycles, setUnlimitedCycles] = useState(false)
  const [batchesRun, setBatchesRun] = useState(0)
  const [preflightTimeout, setPreflightTimeout] = useState(180)
  const [usePreflight, setUsePreflight] = useState(true)
  const [planFirst, setPlanFirst] = useState(true)
  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState<RunResult | null>(null)
  const [progress, setProgress] = useState<ProgressResult | null>(null)
  const [backgroundRun, setBackgroundRun] = useState<BackgroundRun | null>(null)
  const [progressError, setProgressError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const payload = useMemo(
    () => ({
      request_text: requestText,
      repo_path: repoPath,
      workspace_root: repoPath,
      target_branch: 'acos/english-vocab-test-app',
      job_id: jobId || undefined,
      title: 'English Vocabulary Test App',
      jobs_dir: '.acos/jobs-ui',
      max_cycles: maxCycles,
      steps_per_cycle: 1,
      max_stalled_cycles: 3,
      pm_stall_recovery: true,
      max_runtime_seconds: 3600,
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
    const args = [
      'acos',
      'run-supervised',
      '--request',
      `"${requestText.replaceAll('"', '\\"')}"`,
      '--repo-path',
      repoPath,
      '--job-id',
      jobId,
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
          body: JSON.stringify({ ...payload, max_batches: null }),
        })
        const body = await response.json()
        if (!response.ok) {
          throw new Error(body.detail || 'ACOS background request failed')
        }
        setBackgroundRun(body)
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
        body: JSON.stringify(payload),
      })
      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail || 'ACOS API request failed')
      }
      setResult(body)
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
      setBatchesRun(body.batches_run ?? 0)
      if (body.last_result) setResult(body.last_result)
      if (body.status && ['done', 'paused', 'stopped', 'error'].includes(body.status)) {
        setIsRunning(false)
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'background status error')
      setIsRunning(false)
    }
  }

  async function reconnectBackgroundRun() {
    if (!jobId.trim()) return
    try {
      const response = await fetch(
        `/api/background-runs?job_id=${encodeURIComponent(jobId.trim())}`,
      )
      const body = await response.json()
      if (!response.ok) {
        throw new Error(body.detail || 'background list request failed')
      }
      const latest = Array.isArray(body) ? body[0] : null
      if (!latest) return
      setBackgroundRun(latest)
      setBatchesRun(latest.batches_run ?? 0)
      if (latest.last_result) setResult(latest.last_result)
      setIsRunning(['queued', 'running', 'stopping'].includes(String(latest.status)))
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
    if (!jobId.trim()) return
    try {
      const response = await fetch(
        `/api/jobs/${encodeURIComponent(jobId.trim())}/progress?jobs_dir=${encodeURIComponent(
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
  }, [])

  const liveSummary = progress?.summary ?? result?.summary
  const planning = liveSummary?.planning_summary
  const stop = result?.stop_summary
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
                onClick={() => setRequestText(englishVocabTemplate)}
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
              <input value={jobId} onChange={(event) => setJobId(event.target.value)} />
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
          {progressError && !progress && <div className="progress-note">{progressError}</div>}
          {!progress && !progressError && (
            <div className="progress-note">実行を開始すると、ジョブ状態を数秒ごとに読み込みます。</div>
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
                  <strong>{progress.summary?.pending_task_count ?? 0}</strong>
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
                    <span>{event.role || 'system'}</span>
                    <strong>{event.event_type || '-'}</strong>
                    <em>{event.action || '-'}</em>
                    <b>{event.status || '-'}</b>
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
