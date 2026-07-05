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

## Run A Demo Job

```bash
python -m apps.cli run-demo --workspace /tmp/acos-demo
```

## Run A Job From YAML

```bash
acos run-job --file job.yaml
```

## Start The API

```bash
acos api
```

The API exposes a minimal ACOS MVP surface for submitting and inspecting jobs.
