# ACOS

ACOS is an Autonomous Coding Operating System. A user provides product
requirements in natural language, and ACOS decomposes the work across explicit
roles such as PM, Architect, Planner, Implementer, Test Writer, Reviewer,
Security Reviewer, Fixer, Summarizer, and Release Manager.

ACOS は Autonomous Coding Operating System です。ユーザーが自然言語で要件を与えると、
PM、Architect、Planner、Implementer、Test Writer、Reviewer、
Security Reviewer、Fixer、Summarizer、Release Manager といった明示的な役割に
作業を分解して進行します。

The orchestrator owns state transitions, retries, stuck and blocked handling,
tool permissions, and model selection. LLMs do not own workflow control.

状態遷移、リトライ、stuck / blocked 判定、ツール権限、モデル選択は Orchestrator が担当し、
LLM 自体はワークフロー制御を持ちません。

## Durable Runtime / 永続実行

ACOS now includes a durable runtime backed by SQLite. `acos jobs submit --file job.yaml`
persists the job in `.acos/acos.sqlite3`, and `acos worker run --forever` or
`acos daemon start --foreground` can continue execution even if your terminal,
browser, SSH session, or Codex UI disconnects.

ACOS は SQLite ベースの Durable Runtime を持ちます。`acos jobs submit --file job.yaml`
で投入した job は `.acos/acos.sqlite3` に永続化され、`acos worker run --forever`
または `acos daemon start --foreground` によって、ターミナルやブラウザ、SSH、
Codex UI が切断されても再接続後に継続できます。

Key behaviors:

- approval-required operations move the job to `waiting_approval`
- provider outages move the job to `waiting_runtime` or `provider_unavailable`
- stale heartbeat / lease detection moves interrupted jobs to `recovering`
- restart resumes from checkpoints instead of replaying every step

主な挙動:

- 承認が必要な操作は `waiting_approval`
- provider 障害は `waiting_runtime` または `provider_unavailable`
- stale heartbeat / lease は `recovering`
- 再起動後は checkpoint から再開

Important limitation:

- If the host Mac sleeps or loses power, execution stops while the machine is
  unavailable. After the host starts again, the worker can recover unfinished
  jobs from checkpoints.

重要な制限:

- 実行ホストの Mac がスリープまたは電源断になると、その間の実行は止まります。
  起動後に worker が checkpoint から復旧できます。

## Architecture Overview / アーキテクチャ概要

- `packages/orchestrator/`: job runner, policy engine, audit, context builder
- `packages/agents/`: role runner and role-specific prompt/config glue
- `packages/llm/`: provider registry, model adapters, routing, budgets
- `packages/mcp_client/`: local MCP-style router and fake tool environment
- `mcp_servers/`: repo, git, test, memory, and notify server skeletons
- `apps/`: CLI, API, and worker entrypoints

主なディレクトリの役割は次の通りです。

- `packages/orchestrator/`: ジョブ実行、ポリシー判定、監査、コンテキスト構築
- `packages/agents/`: 各ロールの実行とプロンプト設定
- `packages/llm/`: プロバイダ登録、モデルアダプタ、ルーティング、予算管理
- `packages/mcp_client/`: ローカル MCP 風ルータとフェイク実行環境
- `mcp_servers/`: repo / git / test / memory / notify サーバの骨組み
- `apps/`: CLI、API、worker のエントリポイント

## Model Routing Design / モデルルーティング設計

ACOS does not fix itself to one model family. Every role resolves models
through:

ACOS は単一のモデル系列に固定されません。各ロールは次の仕組みを通じて
利用モデルを解決します。

- `ModelProviderConfig`: provider endpoint and capability metadata
- `ModelConfig`: concrete model entry with context/output limits
- `AgentModelConfig`: role-to-model mapping
- `ModelRegistry`: YAML-backed provider/model/agent catalog
- `ModelRouter`: role-aware selection, fallback, escalation, capability checks
- `ModelAdapter`: provider-specific execution layer

- `ModelProviderConfig`: プロバイダのエンドポイントと機能メタデータ
- `ModelConfig`: コンテキスト長や出力上限を含むモデル定義
- `AgentModelConfig`: ロールとモデルの対応付け
- `ModelRegistry`: YAML ベースの provider / model / agent カタログ
- `ModelRouter`: ロール単位の選択、フォールバック、エスカレーション、能力判定
- `ModelAdapter`: プロバイダごとの実行層

This allows patterns such as:

これにより、たとえば次のような運用ができます。

- every role can share one long-context model while keeping role-specific
  prompts, tools, schemas, and sampling
- `max_output_tokens` can stay `auto` and resolve per request instead of using
  a fixed `1024` or `2048`
- the orchestrator can still add escalation and fallback later through config

- 全ロールで 1 つの long-context モデルを共有しつつ、ロールごとの
  プロンプト、ツール、スキーマ、サンプリングを分離できる
- `max_output_tokens` は固定 `1024` / `2048` ではなく、`auto` のまま
  リクエストごとに解決できる
- エスカレーションやフォールバックは後から設定で追加できる

See [docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md) for the detailed routing
design.

詳細は [docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md) を参照してください。

## Setup / セットアップ

### 1. Install Dependencies / 依存関係のインストール

```bash
uv sync --group dev
```

### 2. Configure Environment / 環境変数の設定

```bash
cp .env.example .env
```

Set the API key variables expected by `configs/model_providers.yaml`.

`configs/model_providers.yaml` で参照している API キー環境変数を設定します。

Example:

例:

```bash
export OPENAI_API_KEY=replace-me
export MOCK_API_KEY=dummy
```

### 3. Point ACOS at Your OpenAI-Compatible API / OpenAI 互換 API の接続先を設定

Edit [configs/model_providers.yaml](configs/model_providers.yaml). For a local
Qwen endpoint, update the provider `base_url` fields to match your
OpenAI-compatible server.

[configs/model_providers.yaml](configs/model_providers.yaml) を編集し、
利用する OpenAI 互換 API に合わせて `base_url` を設定します。ローカルの
Qwen エンドポイントを使う場合は、その URL に合わせて書き換えます。

Example provider shape:

設定例:

```yaml
providers:
  local_qwen:
    type: openai_compatible
    base_url: "https://msi.tail5c01da.ts.net/v1"
    api_key_env: "OPENAI_API_KEY"
    allow_empty_api_key: true
    default_api_key: "EMPTY"
    timeout_seconds: 900
    extra_body:
      chat_template_kwargs:
        enable_thinking: false
    supports_tools: true
    supports_json_mode: false
```

## Config Files / 設定ファイル

### `configs/model_providers.yaml`

Defines:

内容:

- provider type
- base URL
- API key env var
- timeout
- tool support
- JSON mode support
- context and output limits
- provider-specific request body extensions such as Qwen thinking flags

- provider の種類
- base URL
- API キーを読む環境変数名
- タイムアウト
- ツール呼び出し対応
- JSON mode 対応
- コンテキスト長と出力上限
- Qwen thinking 設定のような provider 固有 request body 拡張

### `configs/agents.yaml`

Defines, per role:

ロールごとに次を定義します。

- primary model
- fallback models
- sampling settings
- `max_output_tokens` or `auto`
- context budget
- allowed tools
- output schema

- primary model
- fallback models
- サンプリング設定
- `max_output_tokens` または `auto`
- コンテキスト予算
- 使用可能ツール
- 出力スキーマ

Example idea:

例:

- `pm.primary_model = qwen_35b`
- every role can set `max_output_tokens: auto`
- every role can set `context_budget_tokens: 262144`

### `configs/model_routing.yaml`

Controls:

制御対象:

- fallback errors such as `timeout`, `rate_limit`, `invalid_json`
- escalation triggers such as repeated failures
- roles that require tool calling
- roles that require strict JSON

- `timeout`, `rate_limit`, `invalid_json` などのフォールバック条件
- 失敗回数に応じたエスカレーション条件
- ツール呼び出しが必須のロール
- strict JSON が必須のロール

### `configs/policies.yaml`

Defines:

内容:

- deny-by-default tool policy
- release branch rules
- test command restrictions
- forbidden patch targets
- protected file and secret-path restrictions

### `configs/runtime.yaml`

Defines provider health checks, waiting-runtime behavior, provider recovery,
token budget policy, and whether recovery auto-resumes or waits for a manual
`acos jobs resume`.

## Token Budgeting / トークン予算

ACOS treats API `max_tokens` as an output cap and `max_context_tokens` as the
model context window. They are not the same value.

ACOS は API の `max_tokens` を出力上限、`max_context_tokens` をモデルの
コンテキスト長として別物で扱います。

Before each LLM call, ACOS resolves the integer `max_tokens` with:

各 LLM call の直前に、ACOS は次の式で整数の `max_tokens` を解決します。

```text
max_tokens = min(
  configured max_output_tokens or remaining context,
  model.max_context_tokens - estimated_input_tokens - safety_margin_tokens
)
```

Current long-context defaults:

- `qwen_35b.max_context_tokens = 262144`
- `qwen_35b.max_output_tokens = auto`
- every role sets `max_output_tokens = auto`
- every role sets `context_budget_tokens = 262144`
- runtime keeps `safety_margin_tokens = 4096`
- provider `timeout_seconds = 900`

If Qwen spends too long in thinking mode and returns `finish_reason=length`,
ACOS records `output_truncated` instead of treating that case as plain
`invalid_json`.

Qwen が thinking mode で長引いて `finish_reason=length` を返した場合、
ACOS はそれを単なる `invalid_json` ではなく `output_truncated` として記録します。

Debug the resolved budget with:

```bash
acos debug token-budget --role pm --file job.yaml
acos debug token-budget --role implementer --job-id <job_id>
```

### `configs/worker.yaml`

Defines worker polling, heartbeat, lease TTL, stale recovery thresholds, and
the default SQLite/log paths used by daemon mode.

## Daemon Mode / 常駐実行

Foreground:

```bash
acos daemon start --foreground --workspace .
```

Detached background process:

```bash
acos daemon start --detach --workspace .
acos daemon status --workspace .
acos daemon logs --workspace .
```

launchd plist generation on macOS:

```bash
acos daemon install-launchd --workspace .
acos daemon uninstall-launchd --workspace .
```

The generated plist does not embed API keys or raw secrets.

## Runtime Recovery / 復旧

Check provider and model health:

```bash
acos check-provider --provider local_qwen
acos check-model --model qwen_35b
acos runtime status --workspace .
acos runtime check --workspace .
```

If a provider recovers and auto-resume is disabled, continue manually:

```bash
acos jobs resume <job_id> --workspace .
```

- デフォルト拒否のツールポリシー
- リリース系ブランチのルール
- テストコマンドの制限
- パッチ禁止対象
- 保護ファイルと秘密情報パスの制限

## MCP Server Overview / MCP サーバ概要

ACOS routes repo, git, test, memory, and notification actions through MCP-style
tools rather than arbitrary shell execution.

ACOS は repo / git / test / memory / notification の操作を、任意の shell 実行ではなく
MCP 風のツール呼び出し経由で扱います。

Current MVP state:

現状の MVP は次の状態です。

- fake in-process MCP environment is implemented and used by tests
- `mcp_servers/*` contains skeleton wrappers for future standalone servers

- インプロセスの fake MCP 環境が実装済みで、テストで利用されている
- `mcp_servers/*` には将来の standalone server 用スケルトンが入っている

## Local Execution / ローカル実行

### Validate Config / 設定検証

```bash
acos validate-config
```

### Inspect Models / モデル一覧確認

```bash
acos list-models
```

### Inspect Role Assignments / ロール割り当て確認

```bash
acos list-agents
```

### Resolve A Model For A Role / ロールごとのモデル解決

```bash
acos resolve-model --role implementer
acos resolve-model --role fixer --repeated-failures 2
```

### Explain Routing In Human Terms / ルーティング理由の確認

```bash
acos explain-routing --role implementer
```

### Run A Job From YAML / YAML からジョブを実行

Start from [job.yaml.example](job.yaml.example):

[job.yaml.example](job.yaml.example) をコピーして使います。

```bash
cp job.yaml.example job.yaml
acos run-job --file job.yaml
```

`run-job` accepts both:

`run-job` は次の 2 形式に対応しています。

- direct `JobSpec` fields such as `request_text`, `repo_path`, `target_branch`
- a friendlier job file format using `requester_input`, `title`,
  `base_branch`, `notification_channel`, `constraints`, and `workspace_root`

- `request_text`, `repo_path`, `target_branch` などの `JobSpec` 直指定
- `requester_input`, `title`, `base_branch`, `notification_channel`,
  `constraints`, `workspace_root` を使う、より人が書きやすい形式

### Approval Commands / 承認コマンド

When ACOS hits a high-risk action, the job moves to `waiting_approval`.

高リスク操作に到達すると、ジョブは `waiting_approval` に遷移します。

```bash
acos approvals list --workspace .
acos approvals show <approval_id> --workspace .
acos approvals approve <approval_id> --workspace .
acos approvals reject <approval_id> --workspace . --reason "not acceptable"
acos jobs resume <job_id> --workspace .
```

Inside the configured workspace, normal development work is auto-allowed.
Approval is reserved for high-risk actions such as large patches, mass delete,
release-like operations, or external send operations. Critical actions remain
denied.

設定された workspace 内では、通常の開発操作は自動許可されます。承認が必要になるのは、
大規模パッチ、一括削除、リリース相当の操作、外部送信などの高リスク操作です。
重大な操作は引き続き拒否されます。

### Start The API / API を起動

```bash
acos api
```

### Start The Worker / Worker を起動

```bash
acos worker
acos worker --request "READMEにセットアップ手順を追加してください" --repo .
acos worker --file job.yaml.example
```

In the MVP, the worker is a simple entrypoint for running one job locally.

MVP の worker は、ローカルで 1 ジョブを実行するためのシンプルなエントリポイントです。

## CLI Reference / CLI リファレンス

- `acos validate-config`
- `acos list-models`
- `acos list-agents`
- `acos resolve-model --role implementer`
- `acos resolve-model --role fixer --repeated-failures 2`
- `acos explain-routing --role implementer`
- `acos run-job --file job.yaml`
- `acos approvals list`
- `acos approvals show <approval_id>`
- `acos approvals approve <approval_id>`
- `acos approvals reject <approval_id> --reason "..."`
- `acos jobs resume <job_id>`
- `acos api`
- `acos worker`

## Testing / テスト

```bash
python3 -m compileall acos
.venv/bin/pytest
```

For a deterministic smoke test without a real provider:

実際のモデルプロバイダなしで決定的なスモークテストを行う場合:

```bash
python3 -m apps.cli run-demo --workspace /tmp/acos-demo
```

## Security Policy / セキュリティポリシー

ACOS enforces:

ACOS では次を強制します。

- deny-by-default tool policy
- workspace-only file access
- symlink escape blocking
- workspace-auto approval gateway for high-risk operations
- secret redaction before prompt, memory, audit, and notification use
- allowlisted test commands only
- protected branch restrictions
- release commit restriction to `release_manager`
- patch restrictions on tests and dependency manifests

- デフォルト拒否のツールポリシー
- workspace 外アクセスの禁止
- symlink escape の遮断
- 高リスク操作に対する workspace 内承認ゲート
- prompt / memory / audit / notification 前の secret redact
- 許可済みテストコマンドのみ実行可
- protected branch 制約
- `release_manager` のみがリリースコミット可能
- テストや依存定義へのパッチ制限

See [docs/SECURITY.md](docs/SECURITY.md).

詳細は [docs/SECURITY.md](docs/SECURITY.md) を参照してください。

## Current Limitations / 現状の制約

- `mcp_servers/*` are still skeletons for standalone server mode
- Docker sandbox execution is not implemented yet
- tests do not call real external model APIs
- `run-job` needs a reachable configured provider unless you switch configs to
  mock models or use `run-demo`

- `mcp_servers/*` は standalone server モード向けのスケルトン段階
- Docker sandbox 実行は未実装
- テストは実外部モデル API を呼ばない
- `run-job` を使うには、mock モデル構成へ切り替えるか `run-demo` を使わない限り、
  到達可能なプロバイダ設定が必要

## Roadmap / ロードマップ

- standalone MCP server transports
- persistent audit and job history
- stronger worker queue and background processing
- real sandbox execution
- richer semantic review and test-quality gates

- standalone MCP server transport
- 永続化された audit と job history
- より強い worker queue とバックグラウンド処理
- 実 sandbox 実行
- より高度な semantic review と test-quality gate

## More Detail / 詳細資料

- [docs/QUICKSTART.md](docs/QUICKSTART.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/MODEL_ROUTING.md](docs/MODEL_ROUTING.md)
- [docs/SECURITY.md](docs/SECURITY.md)
