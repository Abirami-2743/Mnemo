PY ?= python3

.PHONY: install test lint train evaluate serve frontend all
install:
	$(PY) -m pip install -e ".[all]"
	cd frontend && npm install

test:
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src tests
train:            ## 3 PPO seeds + ablations (~30 min on 2 CPU cores)
	scripts/train_all.sh
evaluate:         ## test/OOD/sweeps/ablations -> results/RESULTS.md, results.json, figures
	$(PY) -m mnemo.run_eval
frontend:
	cd frontend && npm run build
serve: frontend
	$(PY) -m mnemo.cli serve --port 8000
all: train evaluate test
