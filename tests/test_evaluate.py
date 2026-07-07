"""평가 지표 테스트 — NMI/pairwise F1/purity, 로더, 템플릿."""
import json

import pytest

from kakaothread.evaluate import (
    load_gold,
    load_pred,
    make_template,
    nmi,
    pairwise_prf,
    purity,
    score,
)


def test_perfect_match():
    gold = ["a", "a", "b", "b"]
    pred = [1, 1, 2, 2]  # 라벨 이름이 달라도 구조가 같으면 만점
    assert nmi(gold, pred) == pytest.approx(1.0)
    assert pairwise_prf(gold, pred) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert purity(gold, pred) == 1.0


def test_all_in_one_cluster():
    """전부 한 덩어리로 뭉치면: recall 완벽, precision 나쁨, NMI 0."""
    gold = ["a", "a", "b", "b"]
    pred = [1, 1, 1, 1]
    prf = pairwise_prf(gold, pred)
    assert prf["recall"] == 1.0
    assert prf["precision"] == pytest.approx(2 / 6)  # 쌍 6개 중 옳은 쌍 2개
    assert nmi(gold, pred) == pytest.approx(0.0)
    assert purity(gold, pred) == 0.5


def test_all_singletons():
    """전부 낱개로 쪼개면: precision(쌍 없음→1.0 관례), recall 0."""
    gold = ["a", "a", "b", "b"]
    pred = [1, 2, 3, 4]
    prf = pairwise_prf(gold, pred)
    assert prf["recall"] == 0.0
    assert purity(gold, pred) == 1.0  # purity 는 과분할에 관대 — 단독 지표로 쓰지 말 것


def test_score_handles_missing_pred():
    gold = {0: "a", 1: "a", 2: "b"}
    pred = {0: 1, 1: 1}  # msg 2 예측 누락 → UNASSIGNED 클러스터
    r = score(gold, pred)
    assert r["n_evaluated"] == 3
    assert r["n_missing_pred"] == 1
    assert r["f1"] == 1.0  # 누락 항목이 홀로 클러스터가 되어 쌍 지표엔 영향 없음


def test_score_empty_gold_raises():
    with pytest.raises(ValueError):
        score({}, {0: 1})


def test_loaders_and_template_roundtrip(tmp_path):
    # messages.jsonl → 템플릿 생성 → 라벨 채워 gold 로 로드
    mj = tmp_path / "messages.jsonl"
    mj.write_text(
        '{"msg_id": 0, "dt": "2026-05-11T09:00:00", "sender": "a", "text": "hi"}\n'
        '{"msg_id": 1, "dt": "2026-05-11T09:01:00", "sender": "b", "text": "yo"}\n',
        encoding="utf-8",
    )
    tpl = tmp_path / "gold.json"
    assert make_template(mj, tpl) == 2

    rows = json.loads(tpl.read_text(encoding="utf-8"))
    rows[0]["thread"] = "인사"
    rows[1]["thread"] = ""  # 미라벨 → 평가 제외
    tpl.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    assert load_gold(tpl) == {0: "인사"}

    # dict 형태도 허용
    d = tmp_path / "gold2.json"
    d.write_text('{"0": "인사", "1": " "}', encoding="utf-8")
    assert load_gold(d) == {0: "인사"}

    # threads.json 로더
    th = tmp_path / "threads.json"
    th.write_text(json.dumps([
        {"thread_id": 1, "category": "c", "topic": "t", "summary": "", "msg_ids": [0, 1]},
        {"thread_id": 0, "category": "잡담", "topic": "잡담", "summary": "", "msg_ids": [2]},
    ]), encoding="utf-8")
    assert load_pred(th) == {0: 1, 1: 1, 2: 0}
