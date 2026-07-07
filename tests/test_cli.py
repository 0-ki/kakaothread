"""CLI 보조 기능 테스트 — dry-run 추정, 부분 실행 트리밍, 재개/중복 감지."""
import json
import sys
from datetime import datetime, timedelta

import pytest

from kakaothread import cli, outputs
from kakaothread.cli import EXTRA_CALLS, _fmt_dur, preview_stats
from kakaothread.config import Connection, Slot
from kakaothread.pipeline import limit_chunks
from kakaothread.preprocess import Message

CONN = Connection(name="TEST", base_url="", api_key="k")


def _msgs(n: int) -> list[Message]:
    base = datetime(2026, 5, 11, 9, 0)
    return [Message(msg_id=i, dt=base + timedelta(minutes=i), sender="a", text="안녕")
            for i in range(n)]


def _chunks(msgs, *cuts):
    """(세션0 고정) msgs 를 cuts 경계로 청크 분할."""
    out, prev = [], 0
    for c in [*cuts, len(msgs)]:
        out.append((0, msgs[prev:c]))
        prev = c
    return out


def test_fmt_dur():
    assert _fmt_dur(5) == "5초"
    assert _fmt_dur(200) == "3분 20초"
    assert _fmt_dur(3700) == "1시간 1분"


def test_limit_chunks_trims_messages_too():
    msgs = _msgs(10)
    chunks = _chunks(msgs, 3, 6)  # [0:3], [3:6], [6:10]
    m2, c2, total = limit_chunks(msgs, chunks, 2)
    assert total == 3
    assert len(c2) == 2
    assert len(m2) == 6  # 처리 범위 밖 메시지는 리포트에서 제외


def test_limit_chunks_noop_when_none_or_large():
    msgs = _msgs(6)
    chunks = _chunks(msgs, 3)
    assert limit_chunks(msgs, chunks, None) == (msgs, chunks, 2)
    assert limit_chunks(msgs, chunks, 99) == (msgs, chunks, 2)


def test_preview_stats_counts_and_estimates():
    msgs = _msgs(10)
    chunks = _chunks(msgs, 5)
    tier = [Slot(name=n, conn=CONN, model=n, priority=1, rpm=5) for n in ("a", "b", "c")]
    st = preview_stats(12, 10, chunks, tier)
    assert st["n_chunks"] == 2
    assert st["n_calls"] == 2 + EXTRA_CALLS
    assert st["tok_in"] > 0 and st["tok_out"] > 0
    assert st["cost"] == 0.0  # 무료 슬롯
    # 레이트리밋 하한: 4콜 ÷ (5rpm×3슬롯) = 16초
    assert st["min_seconds"] == (2 + EXTRA_CALLS) * 60 / 15


def test_preview_stats_no_slots_and_unlimited_rpm():
    msgs = _msgs(4)
    chunks = _chunks(msgs)
    st = preview_stats(4, 4, chunks, [])
    assert st["cost"] is None and st["min_seconds"] is None
    st = preview_stats(4, 4, chunks, [Slot(name="a", conn=CONN, model="a", rpm=0)])
    assert st["cost"] == 0.0 and st["min_seconds"] is None  # rpm=0 → 하한 없음


def test_warn_duplicate(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(outputs, "RUNS_DIR", tmp_path)
    run1 = tmp_path / "20260101_000000"
    run1.mkdir()
    (run1 / "meta.json").write_text(json.dumps({"source_sha256": "abc"}), encoding="utf-8")

    cli._warn_duplicate("abc")
    assert "기존 실행" in capsys.readouterr().out
    cli._warn_duplicate("zzz")
    assert capsys.readouterr().out == ""  # 다른 해시면 조용


def test_cli_arg_validation(monkeypatch):
    # --resume 과 input 동시 지정 불가
    monkeypatch.setattr(sys, "argv", ["kakaothread", "in.txt", "--resume", "d"])
    with pytest.raises(SystemExit):
        cli.main()
    # input 도 --resume 도 없으면 에러
    monkeypatch.setattr(sys, "argv", ["kakaothread"])
    with pytest.raises(SystemExit):
        cli.main()
