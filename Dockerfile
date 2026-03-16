FROM ghcr.io/astral-sh/uv:bookworm-slim

# Copy the project into the image
COPY . /app

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

# Create non-root user
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8082/health')" || exit 1

CMD ["uv", "run", "start_proxy.py"]
