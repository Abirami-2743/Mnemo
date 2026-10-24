# --- dashboard
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- API (+ served dashboard)
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu ".[rl,service]"
COPY configs ./configs
COPY models/mnemo_ppo ./models/mnemo_ppo
COPY results ./results
COPY --from=web /web/dist ./frontend/dist
ENV FRONTEND_DIST=/app/frontend/dist MODEL_PATH=/app/models/mnemo_ppo/best_model.zip RESULTS_DIR=/app/results
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health')" || exit 1
CMD ["python", "-m", "mnemo.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
