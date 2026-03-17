FROM ghcr.io/astral-sh/uv:bookworm-slim

WORKDIR /app

# Copy only dependency manifests first for better layer caching.
# The `uv sync` layer is now invalidated only when dependencies change,
# not on every source file edit (saves 30-60s on typical CI rebuilds).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked

# Now copy the full project source
COPY . /app

# Create non-root user
RUN groupadd --system appgroup && useradd --system --gid appgroup appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8082/health')" || exit 1

CMD ["uv", "run", "start_proxy.py"]
