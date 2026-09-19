"""RunStore — registro de turnos do /chat (SQLite) e consultas do dashboard."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.services.runs import RunStore


def _iso(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).isoformat(timespec="microseconds")


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[RunStore]:
    runs = RunStore(tmp_path / "data" / "runs.db")
    await runs.init()
    yield runs


async def test_run_lifecycle_start_record_finish(store: RunStore) -> None:
    run_id = store.start("telegram", "telegram_1")

    [active] = store.active()
    assert active["id"] == run_id
    assert active["source"] == "telegram"
    assert active["steps"] == []

    store.record_node(run_id, "chatbot", 120, [], {})
    store.record_node(run_id, "tools", 40, ["get_weather", "web_search"], {})
    assert [s["node"] for s in store.active()[0]["steps"]] == ["chatbot", "tools", "tools"]

    await store.finish(run_id, "ok", False, None)

    assert store.active() == []
    run = await store.get(run_id)
    assert run is not None
    assert run["status"] == "ok"
    assert run["spoken"] is False
    assert run["source"] == "telegram"
    assert run["session_id"] == "telegram_1"
    assert run["finished_at"] is not None
    assert isinstance(run["duration_ms"], int)
    # Um passo por tool chamada, todas com a duração do nó ``tools``.
    assert [
        (s["seq"], s["node"], s["tool"], s["duration_ms"], s["status"]) for s in run["steps"]
    ] == [
        (0, "chatbot", None, 120, "ok"),
        (1, "tools", "get_weather", 40, "ok"),
        (2, "tools", "web_search", 40, "ok"),
    ]


async def test_get_active_run_reports_running(store: RunStore) -> None:
    run_id = store.start("satellite", None)
    store.record_node(run_id, "chatbot", 10, [], {})

    run = await store.get(run_id)

    assert run is not None
    assert run["status"] == "running"
    assert run["finished_at"] is None
    assert [s["node"] for s in run["steps"]] == ["chatbot"]


async def test_get_unknown_run_is_none(store: RunStore) -> None:
    assert await store.get("nao-existe") is None


async def test_tool_error_marks_step_not_run(store: RunStore) -> None:
    run_id = store.start("satellite", None)
    store.record_node(
        run_id, "tools", 30, ["control_device"], {"control_device": "404 HTTPStatusError"}
    )
    await store.finish(run_id, "ok", True, None)

    run = await store.get(run_id)
    assert run is not None
    assert run["status"] == "ok"
    [step] = run["steps"]
    assert (step["tool"], step["status"], step["error"]) == (
        "control_device",
        "error",
        "404 HTTPStatusError",
    )
    stats = await store.tool_stats()
    assert stats["control_device"]["calls"] == 1
    assert stats["control_device"]["errors"] == 1
    assert stats["control_device"]["last_used"] == run["started_at"]


async def test_recent_is_newest_first_with_tools_and_limit(store: RunStore) -> None:
    first = store.start("satellite", None)
    store.record_node(first, "tools", 5, ["get_weather"], {})
    await store.finish(first, "ok", True, None)
    second = store.start("telegram", None)
    await store.finish(second, "error", False, "Boom")

    recent = await store.recent(10)

    assert [r["id"] for r in recent] == [second, first]
    assert recent[0]["status"] == "error"
    assert recent[0]["tools"] == []
    assert recent[1]["tools"] == ["get_weather"]
    assert recent[1]["spoken"] is True
    assert len(await store.recent(1)) == 1


async def test_stats_and_channel_activity_respect_since(store: RunStore) -> None:
    ok = store.start("satellite", None)
    await store.finish(ok, "ok", True, None)
    bad = store.start("telegram", None)
    await store.finish(bad, "error", False, "Boom")
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "INSERT INTO runs (id, started_at, finished_at, duration_ms, source, status, spoken)"
            " VALUES ('velho', ?, ?, 1, 'satellite', 'error', 0)",
            (_iso(-timedelta(days=2)), _iso(-timedelta(days=2))),
        )

    since = _iso(-timedelta(hours=24))

    assert await store.stats(since) == {"runs": 2, "errors": 1}
    activity = await store.channel_activity(since)
    assert activity["satellite"]["runs_24h"] == 1
    assert activity["telegram"]["runs_24h"] == 1
    assert activity["telegram"]["last_activity"] is not None


async def test_init_deletes_runs_older_than_retention(store: RunStore) -> None:
    old = _iso(-timedelta(days=31))
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "INSERT INTO runs (id, started_at, finished_at, duration_ms, source, status, spoken)"
            " VALUES ('velho', ?, ?, 1, 'satellite', 'ok', 0)",
            (old, old),
        )
        conn.execute(
            "INSERT INTO run_steps (run_id, seq, node, duration_ms, status)"
            " VALUES ('velho', 0, 'chatbot', 1, 'ok')"
        )
    recent = store.start("satellite", None)
    await store.finish(recent, "ok", True, None)

    await store.init()

    assert await store.get("velho") is None
    with sqlite3.connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM run_steps WHERE run_id='velho'").fetchone() == (
            0,
        )
    assert await store.get(recent) is not None


async def test_writes_are_best_effort_but_reads_raise(tmp_path: Path) -> None:
    # Um diretório no lugar do arquivo: sqlite não consegue abrir o banco.
    store = RunStore(tmp_path)
    await store.init()  # não lança

    run_id = store.start("satellite", None)
    store.record_node(run_id, "chatbot", 1, [], {})
    await store.finish(run_id, "ok", True, None)  # não lança

    assert store.active() == []  # o turno sai da memória mesmo sem gravar
    with pytest.raises(sqlite3.Error):
        await store.recent(10)
