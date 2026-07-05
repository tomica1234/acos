# Quickstart

## Install

```bash
uv sync --group dev
```

## Verify

```bash
make compile
make pytest
acos validate-config
acos list-models
acos list-agents
acos resolve-model --role implementer
acos resolve-model --role fixer --repeated-failures 2
acos explain-routing --role implementer
```

Repeated implementer/fixer failures should route to `ncmoe40_q4`. If an
escalation rule points to the same model as a role primary, `acos
validate-config` fails so the route cannot silently become a no-op.

## Run A Demo Job

```bash
python -m apps.cli run-demo --workspace /tmp/acos-demo
```

## Run A Job From YAML

```bash
acos run-job --file job.yaml
```

For an end-to-end autonomous run that should keep changing strategy until the
job is complete or a policy hard stop occurs:

```bash
acos run-supervised --request "Build the app from this PRD" --repo-path . --jobs-dir .acos/jobs --autonomous-until-done --summary-file .acos/final-summary.json --summary-dir .acos/cycles --preflight-provider local_ornith
```

Use `job-status --json` to inspect `resume`, `failure_analysis`,
`failure_diagnosis`, `autonomous_recovery_plan`, and `pm_interventions` while
the job is running.

## Start The API

```bash
acos api
```

The API exposes a minimal ACOS MVP surface for submitting and inspecting jobs.
