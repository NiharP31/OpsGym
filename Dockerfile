# OpsGym environment image: the workspace + tools + MCP server, nothing else.
FROM python:3.14-slim

# No .pyc writes (lets us run with --read-only); unbuffered so stdio MCP isn't held in a buffer.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# /app is read-only to the non-root user (by design), so pytest must not try to write its cache.
ENV PYTEST_ADDOPTS="-p no:cacheprovider"
WORKDIR /app

# Deps first: this layer stays cached when only our code changes.
COPY requirements-env.txt .
RUN pip install --no-cache-dir -r requirements-env.txt

COPY opsgym/ opsgym/
COPY tests/ tests/

# Non-root: if a future tool ever had a bug that touched the filesystem, it can't modify /app.
RUN useradd --create-home agent
USER agent

# stdio MCP server: run with `docker run -i` (stdin must stay open).
CMD ["python", "-m", "opsgym.mcp_server"]
