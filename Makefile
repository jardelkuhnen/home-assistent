.PHONY: setup install venv system-deps format lint typecheck test test-ci build vuln-check ci act run-brain run-satellite run-telegram check

# Python used to create the venv. Override with: make setup PYTHON=python3.11
PYTHON ?= python3.12
VENV_BIN := .venv/bin

# Full bootstrap from a fresh checkout:
#   1. system deps (PortAudio on Linux, for sounddevice/faster-whisper)
#   2. virtualenv (.venv) with the pinned Python
#   3. pip upgrade (sidesteps PYSEC-2026-3721) + project + dev deps from pyproject.toml
#   4. pre-commit hooks (ruff, ruff-format, mypy, gitleaks)
# Idempotent — safe to re-run anytime deps change.
setup: system-deps venv install
	$(VENV_BIN)/pre-commit install

# Install Python + project/dev dependencies declared in pyproject.toml.
# Uses the venv's pip explicitly so it works without `activate`.
install: venv
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -e ".[dev]"
	# openwakeword puxa tflite-runtime (sem wheel p/ Python >=3.12 no Linux): só ONNX, sem deps.
	$(VENV_BIN)/pip install --no-deps "openwakeword>=0.6.0,<0.7"

# Create the virtualenv if it doesn't exist.
venv:
	@test -x $(VENV_BIN)/python || { \
		echo ">> Criando .venv com $(PYTHON) ..."; \
		$(PYTHON) -m venv .venv || { \
			echo "!! $(PYTHON) não encontrado. Instale o Python >=3.11 ou rode:"; \
			echo "   make setup PYTHON=/caminho/do/python3.12"; \
			exit 1; }; }

# OS-level dependencies for sounddevice/faster-whisper (satelite.py).
# PortAudio ships with macOS; on Debian/Ubuntu install libportaudio2 + dev headers.
system-deps:
	@if command -v apt-get >/dev/null 2>&1; then \
		echo ">> Linux: instalando PortAudio (sudo pode pedir senha) ..."; \
		sudo apt-get update && sudo apt-get install -y portaudio19-dev libportaudio2; \
	elif command -v brew >/dev/null 2>&1; then \
		echo ">> macOS: PortAudio vem no sistema; garantindo portaudio via brew (opcional) ..."; \
		brew install portaudio 2>/dev/null || true; \
	else \
		echo ">> SO não reconhecido — instale PortAudio manualmente se o satelite.py falhar."; \
	fi

format:
	$(VENV_BIN)/ruff format .
	$(VENV_BIN)/ruff check --fix .

lint:
	$(VENV_BIN)/ruff check .

typecheck:
	$(VENV_BIN)/mypy --strict src

test:
	$(VENV_BIN)/pytest

# Test with coverage, mirroring the CI pipeline (.github/workflows/ci.yml).
test-ci:
	$(VENV_BIN)/pytest --cov=src --cov-report=term

# Build check mirroring the CI build job.
build:
	$(VENV_BIN)/python -m compileall src api.py satelite.py telegram_bot.py

# Vulnerability scan mirroring the CI vuln-check job.
vuln-check:
	$(VENV_BIN)/pip-audit

# Run the full CI pipeline locally (same stages as .github/workflows/ci.yml,
# minus the gitleaks secret-scan job — run `act` for that).
ci: build lint typecheck test-ci vuln-check

# Run the real GitHub Actions workflow locally via act (needs Docker + act).
act:
	act -j test

run-brain:
	./.venv/bin/uvicorn api:app --reload --host 0.0.0.0 --port 8000 --log-level info

run-satellite:
	$(VENV_BIN)/python satelite.py

run-telegram:
	$(VENV_BIN)/python telegram_bot.py

check: lint typecheck test
