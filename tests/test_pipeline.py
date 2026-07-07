"""pipeline.integrate 와 outputs.save_and_report 방어 로직 테스트."""
import json
from datetime import datetime

from kakaothread.llm_segment import Assignment, ChunkResult, Thread
from kakaothread.outputs import save_and_report
from kakaothread.pipeline import NOISE_THREAD_ID, IdAlloc, integrate
from kakaothread.preprocess import Message


def _msg(i: int) -> Message:
    return Message(msg_id=i, dt=datetime(2026, 5, 11, 9, i), sender="a", text=f"m{i}")


def test_integrate_reuses_open_thread_without_redeclaration():
    """열린 스레드(전역 id)를 재선언 없이 배정에 쓰면 그 id를 유지한다."""
    open_threads = [Thread(thread_id=5, category="A", topic="a", summary="")]
    result = ChunkResult(assignments=[Assignment(msg_id=0, thread_id=5)], threads=[])
    _, _, assigns = integrate(result, open_threads, {5: open_threads[0]}, IdAlloc(6))
    assert assigns == {0: 5}


def test_integrate_undeclared_id_routed_to_noise():
    """threads 에 없고 열린 스레드도 아닌 id로의 배정은 잡담(0)으로 보낸다 (id 유출 방지)."""
    result = ChunkResult(
        assignments=[Assignment(msg_id=0, thread_id=1), Assignment(msg_id=1, thread_id=3)],
        threads=[Thread(thread_id=1, category="A", topic="a", summary="")],
    )
    _, all_threads, assigns = integrate(result, [], {}, IdAlloc(1))
    assert assigns[0] == 1        # 정상 선언된 스레드
    assert assigns[1] == NOISE_THREAD_ID  # 미선언 id 3 → 잡담 (전역 유출 안 함)
    assert 3 not in all_threads


def test_save_and_report_drops_ghost_and_unknown(tmp_path):
    """멤버 없는 유령 스레드는 직렬화 제외, 존재하지 않는 msg_id 배정은 무시."""
    messages = [_msg(0), _msg(1)]
    all_threads = {
        1: Thread(thread_id=1, category="A", topic="a", summary=""),
        2: Thread(thread_id=2, category="B", topic="b", summary=""),  # 유령 (멤버 없음)
    }
    assignments = {0: 1, 1: 1, 999: 1}  # 999 는 존재하지 않는 msg_id
    save_and_report(messages, all_threads, assignments, tmp_path, source="x")

    payload = json.loads((tmp_path / "threads.json").read_text(encoding="utf-8"))
    assert [t["thread_id"] for t in payload] == [1]      # 유령 스레드 2 제외
    assert payload[0]["msg_ids"] == [0, 1]               # 999 는 빠짐
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_threads"] == 1
