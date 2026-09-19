"""Registro dos turnos do ``POST /chat`` para o dashboard de observabilidade.

Só metadados (canal, nós, tools, durações, status) — nunca texto de pedido/
resposta nem argumentos de tool. Turnos em andamento vivem em memória; ao
``finish`` o turno e seus passos são gravados no SQLite numa transação.

A escrita é *best-effort*: falha de banco é logada e engolida para o dashboard
nunca derrubar nem atrasar um ``/chat``. A leitura propaga o erro.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Mesmo logger do ``api.py``: os logs aparecem no console do brain (uvicorn).
logger = logging.getLogger("uvicorn.error")

RETENTION_DAYS = 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    duration_ms INTEGER,
    source TEXT NOT NULL,
    session_id TEXT,
    status TEXT NOT NULL,
    spoken INTEGER NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS run_steps (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    node TEXT NOT NULL,
    tool TEXT,
    duration_ms INTEGER,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs (started_at);
CREATE INDEX IF NOT EXISTS idx_run_steps_run_seq ON run_steps (run_id, seq);
"""

# Falhas que a escrita best-effort engole (sqlite, ou diretório do banco inacessível).
_WRITE_ERRORS = (sqlite3.Error, OSError)


def utc_iso(moment: datetime | None = None) -> str:
    """ISO-8601 UTC de largura fixa — ordena lexicograficamente."""
    return (moment or datetime.now(UTC)).isoformat(timespec="microseconds")


@dataclass
class _ActiveRun:
    id: str
    source: str
    session_id: str | None
    started_at: str
    started_mono: float = field(default_factory=time.monotonic)
    steps: list[dict[str, Any]] = field(default_factory=list)


class RunStore:
    """Turnos ativos (memória) + histórico (SQLite)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._active: dict[str, _ActiveRun] = {}

    # --- conexão ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _read(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with closing(self._connect()) as conn:
            return conn.execute(sql, params).fetchall()

    # --- escrita ---------------------------------------------------------

    async def init(self) -> None:
        """Cria o schema e aplica a retenção. Falha é logada; o Cérebro sobe assim mesmo."""
        try:
            await asyncio.to_thread(self._init_sync)
        except _WRITE_ERRORS as exc:
            logger.warning("runs: falha ao inicializar o banco: %s: %s", type(exc).__name__, exc)

    def _init_sync(self) -> None:
        cutoff = utc_iso(datetime.now(UTC) - timedelta(days=RETENTION_DAYS))
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "DELETE FROM run_steps WHERE run_id IN (SELECT id FROM runs WHERE started_at < ?)",
                (cutoff,),
            )
            conn.execute("DELETE FROM runs WHERE started_at < ?", (cutoff,))

    def start(self, source: str, session_id: str | None) -> str:
        """Abre um turno em memória e devolve o ``run_id``."""
        run = _ActiveRun(uuid.uuid4().hex, source, session_id, utc_iso())
        self._active[run.id] = run
        return run.id

    def record_node(
        self,
        run_id: str,
        node: str,
        duration_ms: int,
        tools: Sequence[str],
        tool_errors: Mapping[str, str],
    ) -> None:
        """Anexa ao turno ativo os passos de um nó concluído.

        O nó ``tools`` gera um passo por tool chamada (todos com a duração do
        nó — limite superior, pois as tools podem rodar em paralelo); demais
        nós geram um passo sem tool. ``tool_errors`` casa por nome da tool.
        """
        run = self._active.get(run_id)
        if run is None:
            return
        tool_slots: Sequence[str | None] = tools or [None]
        for tool in tool_slots:
            error = tool_errors.get(tool) if tool else None
            run.steps.append(
                {
                    "seq": len(run.steps),
                    "node": node,
                    "tool": tool,
                    "duration_ms": duration_ms,
                    "status": "error" if error else "ok",
                    "error": error,
                }
            )

    async def finish(self, run_id: str, status: str, spoken: bool, error: str | None) -> None:
        """Encerra o turno: sai da memória e é gravado (turno + passos) numa transação."""
        run = self._active.pop(run_id, None)
        if run is None:
            return
        duration_ms = int((time.monotonic() - run.started_mono) * 1000)
        try:
            await asyncio.to_thread(self._finish_sync, run, status, spoken, error, duration_ms)
        except _WRITE_ERRORS as exc:
            logger.warning(
                "runs: falha ao gravar turno %s: %s: %s", run_id, type(exc).__name__, exc
            )

    def _finish_sync(
        self, run: _ActiveRun, status: str, spoken: bool, error: str | None, duration_ms: int
    ) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO runs (id, started_at, finished_at, duration_ms, source, session_id,"
                " status, spoken, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.started_at,
                    utc_iso(),
                    duration_ms,
                    run.source,
                    run.session_id,
                    status,
                    int(spoken),
                    error,
                ),
            )
            conn.executemany(
                "INSERT INTO run_steps (run_id, seq, node, tool, duration_ms, status, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run.id,
                        s["seq"],
                        s["node"],
                        s["tool"],
                        s["duration_ms"],
                        s["status"],
                        s["error"],
                    )
                    for s in run.steps
                ],
            )

    # --- leitura ---------------------------------------------------------

    def active(self) -> list[dict[str, Any]]:
        """Turnos em andamento, do mais antigo ao mais novo."""
        now = time.monotonic()
        return [
            {
                "id": r.id,
                "source": r.source,
                "started_at": r.started_at,
                "elapsed_ms": int((now - r.started_mono) * 1000),
                "steps": list(r.steps),
            }
            for r in self._active.values()
        ]

    async def recent(self, limit: int) -> list[dict[str, Any]]:
        """Últimos turnos concluídos, do mais novo ao mais antigo, com as tools usadas."""
        return await asyncio.to_thread(self._recent_sync, limit)

    def _recent_sync(self, limit: int) -> list[dict[str, Any]]:
        rows = self._read(
            "SELECT id, started_at, source, duration_ms, status, spoken FROM runs"
            " ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        tools: dict[str, list[str]] = {row["id"]: [] for row in rows}
        placeholders = ",".join("?" * len(tools))
        for step in self._read(
            f"SELECT run_id, tool FROM run_steps WHERE tool IS NOT NULL"  # noqa: S608
            f" AND run_id IN ({placeholders}) ORDER BY seq",
            list(tools),
        ):
            tools[step["run_id"]].append(step["tool"])
        return [
            {
                "id": row["id"],
                "started_at": row["started_at"],
                "source": row["source"],
                "duration_ms": row["duration_ms"],
                "tools": tools[row["id"]],
                "status": row["status"],
                "spoken": bool(row["spoken"]),
            }
            for row in rows
        ]

    async def get(self, run_id: str) -> dict[str, Any] | None:
        """Turno ativo (``status="running"``) ou concluído, com passos; ``None`` se inexistente."""
        active = self._active.get(run_id)
        if active is not None:
            return {
                "id": active.id,
                "started_at": active.started_at,
                "finished_at": None,
                "duration_ms": None,
                "source": active.source,
                "session_id": active.session_id,
                "status": "running",
                "spoken": None,
                "error": None,
                "steps": list(active.steps),
            }
        return await asyncio.to_thread(self._get_sync, run_id)

    def _get_sync(self, run_id: str) -> dict[str, Any] | None:
        rows = self._read("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not rows:
            return None
        run = dict(rows[0])
        run["spoken"] = bool(run["spoken"])
        run["steps"] = [
            dict(step)
            for step in self._read(
                "SELECT seq, node, tool, duration_ms, status, error FROM run_steps"
                " WHERE run_id = ? ORDER BY seq",
                (run_id,),
            )
        ]
        return run

    async def stats(self, since: str) -> dict[str, int]:
        """Turnos e turnos com erro desde ``since`` (ISO UTC)."""
        return await asyncio.to_thread(self._stats_sync, since)

    def _stats_sync(self, since: str) -> dict[str, int]:
        [row] = self._read(
            "SELECT count(*) AS runs, coalesce(sum(status = 'error'), 0) AS errors"
            " FROM runs WHERE started_at >= ?",
            (since,),
        )
        return {"runs": row["runs"], "errors": row["errors"]}

    async def tool_stats(self) -> dict[str, dict[str, Any]]:
        """Por tool: chamadas, erros e último uso (início do turno)."""
        return await asyncio.to_thread(self._tool_stats_sync)

    def _tool_stats_sync(self) -> dict[str, dict[str, Any]]:
        rows = self._read(
            "SELECT s.tool AS tool, count(*) AS calls,"
            " coalesce(sum(s.status = 'error'), 0) AS errors, max(r.started_at) AS last_used"
            " FROM run_steps s JOIN runs r ON r.id = s.run_id"
            " WHERE s.tool IS NOT NULL GROUP BY s.tool"
        )
        return {
            row["tool"]: {
                "calls": row["calls"],
                "errors": row["errors"],
                "last_used": row["last_used"],
            }
            for row in rows
        }

    async def channel_activity(self, since: str) -> dict[str, dict[str, Any]]:
        """Por canal (``source``): última atividade e turnos desde ``since``."""
        return await asyncio.to_thread(self._channel_activity_sync, since)

    def _channel_activity_sync(self, since: str) -> dict[str, dict[str, Any]]:
        rows = self._read(
            "SELECT source, max(started_at) AS last_activity,"
            " coalesce(sum(started_at >= ?), 0) AS runs_24h FROM runs GROUP BY source",
            (since,),
        )
        return {
            row["source"]: {"last_activity": row["last_activity"], "runs_24h": row["runs_24h"]}
            for row in rows
        }
