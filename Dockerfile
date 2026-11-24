FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY eval ./eval
COPY data ./data

RUN useradd --create-home --uid 10001 sqlagent && mkdir /audit && chown sqlagent /audit
USER sqlagent

# MCP server over stdio. The database is opened read-only by the server's locked-down connection.
ENTRYPOINT ["sqlagent-server", "--db", "/app/data/chinook.sqlite", "--audit", "/audit/audit.jsonl"]
