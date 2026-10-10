"""
A2A long-task handling: a blocking message/send that outlives the detach
window answers ``working`` (with a liveness report) instead of failing, the
client follows the task through tasks/get, and a session that stops showing
activity is failed as stalled rather than left to a blind timeout.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

import hermes_yaml as yaml
from plugins.platforms.a2a import protocol, tools
from tests.plugins.test_a2a_plugin import _make_live_adapter, _post_json, _send_body

_RUNNING = {"health": "running", "idleSeconds": 4, "turnActive": True,
            "lastActivity": "executing tool terminal", "toolCalls": 3,
            "lastNote": "Reading the protocol now."}


def _setup(monkeypatch, detach="1"):
    monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("A2A_PEER_TOKENS", raising=False)
    monkeypatch.delenv("A2A_REPLY_TIMEOUT", raising=False)
    monkeypatch.setenv("A2A_DETACH_AFTER", detach)
    monkeypatch.setattr(tools, "_POLL_INTERVAL", 0.2)
    adapter, base = _make_live_adapter(monkeypatch, reply_fn=lambda e: None)
    monkeypatch.setattr(adapter, "_task_activity", lambda rec: dict(_RUNNING))
    return adapter, base


def _only_pending(adapter):
    with adapter._pending_lock:
        (task_id, (ctx, _fut)), = adapter._pending.items()
    return task_id, ctx


async def _wait_for_state(adapter, task_id, state):
    for _ in range(50):
        if adapter.tasks.get(task_id)["state"] == state:
            break
        await asyncio.sleep(0.05)
    return adapter.tasks.get(task_id)


def test_send_detaches_with_working_status_then_records_outcome(monkeypatch):
    adapter, base = _setup(monkeypatch)

    async def run():
        assert await adapter.connect() is True
        t0 = time.time()
        resp = await asyncio.to_thread(_post_json, base + "/", _send_body("long job"))
        task = resp["result"]
        assert 0.9 < time.time() - t0 < 5
        assert task["status"]["state"] == protocol.STATE_WORKING
        assert task["metadata"]["hermes"]["health"] == "running"
        task_id, ctx = _only_pending(adapter)
        assert task["id"] == task_id

        got = await asyncio.to_thread(_post_json, base + "/", {
            "jsonrpc": "2.0", "id": "g", "method": "tasks/get", "params": {"id": task_id}})
        assert got["result"]["status"]["state"] == protocol.STATE_WORKING
        assert got["result"]["metadata"]["hermes"]["toolCalls"] == 3

        # No request owns the task any more, yet neither sweep may fail it: it is
        # old but its session is alive.
        adapter.tasks._tasks[task_id]["created_at"] -= 100_000
        assert adapter._fail_orphans_once() == []
        assert adapter._fail_stalled_once() == []

        # The agent answers after the HTTP request already returned.
        await adapter.send(ctx, "all done", metadata={"notify": True})
        rec = await _wait_for_state(adapter, task_id, protocol.STATE_COMPLETED)
        assert rec["state"] == protocol.STATE_COMPLETED and rec["reply"] == "all done"
        with adapter._pending_lock:
            assert task_id not in adapter._pending
            assert task_id not in adapter._active_tasks
        done = await asyncio.to_thread(_post_json, base + "/", {
            "jsonrpc": "2.0", "id": "g", "method": "tasks/get", "params": {"id": task_id}})
        assert "metadata" not in done["result"]
        await adapter.disconnect()

    asyncio.run(run())


def test_v1_send_message_detaches_inside_the_task_wrapper(monkeypatch):
    adapter, base = _setup(monkeypatch)

    async def run():
        assert await adapter.connect() is True
        body = dict(_send_body("long job"), method="SendMessage")
        resp = await asyncio.to_thread(_post_json, base + "/", body)
        task = resp["result"]["task"]
        assert task["status"]["state"] == protocol.STATE_WORKING
        assert task["metadata"]["hermes"]["lastNote"] == "Reading the protocol now."
        await adapter.disconnect()

    asyncio.run(run())


def test_a2a_call_follows_running_task_to_its_reply(monkeypatch):
    adapter, base = _setup(monkeypatch)

    async def run():
        assert await adapter.connect() is True
        call = asyncio.create_task(asyncio.to_thread(
            tools.a2a_call, {"agent": base, "message": "long job"}))
        await asyncio.sleep(2.0)          # past the detach window: client is polling
        assert not call.done()
        _task_id, ctx = _only_pending(adapter)
        await adapter.send(ctx, "final answer", metadata={"notify": True})
        out = await asyncio.wait_for(call, 10)
        assert "final answer" in out and "completed" in out
        await adapter.disconnect()

    asyncio.run(run())


def test_budget_exhausted_reports_running_and_a2a_status_picks_it_up(monkeypatch):
    adapter, base = _setup(monkeypatch)
    peer = {"url": base, "auth": {}, "timeout": 3, "capabilities": []}
    monkeypatch.setattr(tools, "_resolve_peer", lambda agent: dict(peer))

    async def run():
        assert await adapter.connect() is True
        out = await asyncio.to_thread(tools.a2a_call, {"agent": "peer", "message": "long job"})
        task_id, ctx = _only_pending(adapter)
        assert "has NOT finished yet" in out and "STILL RUNNING" in out
        assert task_id in out and "a2a_status" in out and "working" in out.splitlines()[0]
        assert "Reading the protocol now." in out

        peek = await asyncio.to_thread(
            tools.a2a_status, {"agent": "peer", "task_id": task_id, "wait_s": 0})
        assert "STILL RUNNING" in peek

        waiter = asyncio.create_task(asyncio.to_thread(
            tools.a2a_status, {"agent": "peer", "task_id": task_id, "wait_s": 3}))
        await asyncio.sleep(0.5)
        await adapter.send(ctx, "late result", metadata={"notify": True})
        done = await asyncio.wait_for(waiter, 10)
        assert "late result" in done and "completed" in done and ctx in done

        missing = await asyncio.to_thread(
            tools.a2a_status, {"agent": "peer", "task_id": "no-such-task", "wait_s": 0})
        assert "no-such-task" in missing and "task not found" in missing
        await adapter.disconnect()

    asyncio.run(run())


def test_stalled_task_is_failed_and_caller_is_told(monkeypatch):
    adapter, base = _setup(monkeypatch)
    state = {"act": dict(_RUNNING)}
    monkeypatch.setattr(adapter, "_task_activity", lambda rec: dict(state["act"]))

    async def run():
        assert await adapter.connect() is True
        failed_before = protocol.metrics.tasks_failed
        call = asyncio.create_task(asyncio.to_thread(
            tools.a2a_call, {"agent": base, "message": "long job"}))
        await asyncio.sleep(1.8)
        task_id, _ctx = _only_pending(adapter)

        assert adapter._fail_stalled_once(min_age=0) == []      # healthy: must be left alone
        assert adapter.tasks.get(task_id)["state"] == protocol.STATE_WORKING

        state["act"] = {"health": "stalled", "reason": "no activity for 1200s"}
        assert adapter._fail_stalled_once() == []               # younger than the age floor
        assert adapter._fail_stalled_once(min_age=0) == [task_id]
        out = await asyncio.wait_for(call, 10)
        assert "FAILED" in out and "agent stalled" in out and "1200s" in out
        assert adapter.tasks.get(task_id)["state"] == protocol.STATE_FAILED
        assert protocol.metrics.tasks_failed == failed_before + 1
        assert adapter._fail_stalled_once(min_age=0) == []
        await adapter.disconnect()

    asyncio.run(run())


def test_task_activity_classification(monkeypatch, tmp_path):
    con = sqlite3.connect(tmp_path / "state.db")
    con.executescript("""
        CREATE TABLE sessions (id TEXT, source TEXT, chat_id TEXT, started_at REAL,
            last_activity_at REAL, last_activity_description TEXT,
            message_count INTEGER, tool_call_count INTEGER);
        CREATE TABLE session_turn_leases (conversation_id TEXT, holder TEXT,
            acquired_at REAL, expires_at REAL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT);
    """)
    now = time.time()

    def seed(ctx, idle, lease_in, note="working on it"):
        sid = "s-" + ctx
        con.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                    (sid, "a2a", ctx, now - 2000, now - idle, "executing tool x", 9, 4))
        if lease_in is not None:
            con.execute("INSERT INTO session_turn_leases VALUES (?,?,?,?)",
                        (sid, "h", now - 2000, now + lease_in))
        con.execute("INSERT INTO messages (session_id, role, content) VALUES (?,?,?)",
                    (sid, "assistant", note))
        con.commit()

    seed("busy", 5, 200, note="working   on it\nsk-ant-api03-" + "a" * 40)
    seed("quiet", 400, 200)
    seed("silent", 1500, 200)
    seed("ended", 400, None)
    seed("expired", 400, -30)

    adapter, _base = _make_live_adapter(monkeypatch)
    adapter._state_db_path = str(tmp_path / "state.db")
    rec = lambda ctx, age=2000: {"context_id": ctx, "created_at": now - age}

    busy = adapter._task_activity(rec("busy"))
    assert busy["health"] == "running" and busy["turnActive"] and busy["toolCalls"] == 4
    assert busy["lastActivity"] == "executing tool x" and busy["elapsedSeconds"] >= 2000
    assert busy["lastNote"].startswith("working on it") and "a" * 40 not in busy["lastNote"]
    assert adapter._task_activity(rec("quiet"))["health"] == "quiet"
    silent = adapter._task_activity(rec("silent"))
    assert silent["health"] == "stalled" and "no activity" in silent["reason"]
    for ctx in ("ended", "expired"):
        ended = adapter._task_activity(rec(ctx))
        assert ended["health"] == "stalled" and "without producing a reply" in ended["reason"]
    assert adapter._task_activity(rec("nosuch", age=10))["health"] == "starting"
    assert adapter._task_activity(rec("nosuch", age=900))["health"] == "stalled"

    monkeypatch.setenv("A2A_STALL_AFTER", "300")
    assert adapter._task_activity(rec("quiet"))["health"] == "stalled"

    # An unreadable state.db is not evidence of a stall.
    adapter._state_db_path = str(tmp_path / "missing" / "state.db")
    assert adapter._task_activity(rec("busy")) == {"elapsedSeconds": busy["elapsedSeconds"], "health": "unknown"}


def test_state_db_is_the_served_profiles_captured_at_construction(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter, _base = _make_live_adapter(monkeypatch)
    assert Path(adapter._state_db_path) == tmp_path / "state.db"


def test_liveness_report_only_for_tasks_awaiting_a_reply(monkeypatch):
    """A task forwarded to another profile, or an orphan, has no session in this
    profile's state.db; reporting on it would call a healthy task stalled."""
    adapter, _base = _make_live_adapter(monkeypatch)
    monkeypatch.setattr(adapter, "_task_activity", lambda rec: dict(_RUNNING))
    adapter.tasks.create("t-forwarded", "c1", "peer")
    adapter._activate_task("t-forwarded")
    adapter.tasks.create("t-awaited", "c2", "peer")
    adapter._add_pending("t-awaited", "c2")

    assert "metadata" not in adapter._task_view(adapter.tasks.get("t-forwarded"))
    assert adapter._task_view(adapter.tasks.get("t-awaited"))["metadata"] == {"hermes": _RUNNING}
    assert adapter._fail_stalled_once(min_age=0) == []


@pytest.mark.parametrize("detach, reply, expected", [
    ("1", "300", protocol.STATE_WORKING),   # detach window first: not a failure
    ("0", "1", protocol.STATE_FAILED),      # disabled
    ("5", "1", protocol.STATE_FAILED),      # reply timeout first: fails as before
    ("1", "1", protocol.STATE_FAILED),
])
def test_detach_only_applies_inside_the_reply_timeout(monkeypatch, detach, reply, expected):
    from plugins.platforms.a2a import adapter as mod

    monkeypatch.setenv("A2A_DETACH_AFTER", detach)
    monkeypatch.setenv("A2A_REPLY_TIMEOUT", reply)
    adapter, _base = _make_live_adapter(monkeypatch)
    pending = {"future": adapter._add_pending("t1", "c1"), "started": time.time() - 0.9}
    state, _reply = adapter._await_reply(pending, detach_after=mod._detach_after())
    assert state == expected


def test_follow_task_gives_up_after_repeated_poll_failures(monkeypatch):
    monkeypatch.setattr(tools, "_POLL_INTERVAL", 0)
    calls = []

    def flaky(*_args):
        calls.append(1)
        raise OSError("connection refused")

    monkeypatch.setattr(tools, "_get_task", flaky)
    with pytest.raises(RuntimeError) as exc:
        tools._follow_task("peer", "http://x", {}, "", "t-9", {}, deadline=time.time() + 30)
    assert len(calls) == tools._POLL_MAX_ERRORS
    assert "lost contact" in str(exc.value) and "a2a_status(agent='peer', task_id='t-9')" in str(exc.value)

    # A transient failure does not count against a task that keeps answering.
    seq = iter([OSError("blip"), {"status": {"state": protocol.STATE_WORKING}},
                OSError("blip"), {"status": {"state": protocol.STATE_COMPLETED}}])

    def recovering(*_args):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(tools, "_get_task", recovering)
    task = tools._follow_task("peer", "http://x", {}, "", "t-9", {}, deadline=time.time() + 30)
    assert task["status"]["state"] == protocol.STATE_COMPLETED


def test_progress_text_reports_stuck_and_unknown_peers():
    stuck = tools._progress_text("peer", "t-1", {"metadata": {"hermes": {
        "health": "stalled", "reason": "no activity for 950s"}}}, waited=12)
    assert "STUCK — no activity for 950s" in stuck and "waited 12s" in stuck
    # A non-Hermes peer sends no liveness report.
    unknown = tools._progress_text("peer", "t-1", {"status": {"state": protocol.STATE_WORKING}}, waited=1)
    assert "Status: unknown" in unknown and "a2a_status(agent='peer', task_id='t-1')" in unknown


def test_a2a_status_argument_errors(monkeypatch):
    monkeypatch.setattr(tools, "_load_config", lambda: {})
    assert "required" in tools.a2a_status({"agent": "peer"})
    assert "unknown agent 'ghost'" in tools.a2a_status({"agent": "ghost", "task_id": "t-1"})


def test_plugin_manifest_lists_every_client_tool():
    manifest = yaml.safe_load((Path(tools.__file__).parent / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["provides_tools"] == list(tools._TOOLS)
    assert "a2a_status" in tools._TOOLS
