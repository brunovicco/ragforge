.PHONY: install lint type test quality bench bench-live api dashboard infra-up infra-down

install:
	uv sync --all-groups

lint:
	uv run ruff check src tests apps
	uv run ruff format --check src tests apps

type:
	uv run mypy src

test:
	uv run pytest

quality:
	uv run python scripts/quality_gate.py

# Deterministic benchmark: replays the versioned LLM cache (ADR-0004). No API key required.
bench:
	uv run python -m ragforge.evaluation.run --mode cache --config configs/experiments/benchmark-v01.yaml

# Live benchmark: calls providers. Expect ±2pp tolerance vs README numbers (ADR-0004).
bench-live:
	uv run python -m ragforge.evaluation.run --mode live --config configs/experiments/benchmark-v01.yaml

api:
	uv run uvicorn apps.api.main:app --reload

dashboard:
	uv run streamlit run apps/dashboard/main.py

infra-up:
	docker compose --profile core --profile search up -d

infra-down:
	docker compose --profile core --profile search down
