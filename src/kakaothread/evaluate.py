"""평가 하네스 — gold 라벨 vs 분류 결과(threads.json) 비교. 표준 라이브러리만.

사용 흐름:
  1) 템플릿 생성:  python -m kakaothread.evaluate template data/runs/<ts>/messages.jsonl
     → gold_template.json 의 각 항목에 사람이 thread 라벨을 채운다
       (라벨 문자열은 자유 — 같은 주제면 같은 문자열이기만 하면 됨. 잡담은 "잡담").
  2) 채점:        python -m kakaothread.evaluate score gold.json data/runs/<ts>/threads.json

지표:
  - NMI(정규화 상호정보량, arithmetic 평균 정규화) — 클러스터링 유사도
  - pairwise P/R/F1 — '같은 스레드로 묶인 메시지 쌍'의 정밀도/재현율
  - purity — 예측 클러스터별 최다 gold 라벨 비율
빈 라벨("")은 미라벨로 간주해 평가에서 제외한다.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

UNASSIGNED = -1  # gold 에는 있는데 예측에 없는 msg_id 가 흘러들 때의 표식


# ── 지표 (라벨 시퀀스 기반) ─────────────────────────────────────────
def _entropy(counts: list[int], n: int) -> float:
    return -sum((c / n) * math.log(c / n) for c in counts if c)


def nmi(gold: list, pred: list) -> float:
    """정규화 상호정보량. 정규화: I / mean(H_gold, H_pred) (sklearn 기본과 동일)."""
    assert len(gold) == len(pred) and gold
    n = len(gold)
    cg, cp = Counter(gold), Counter(pred)
    joint = Counter(zip(gold, pred))
    mi = sum((nij / n) * math.log(n * nij / (cg[g] * cp[p]))
             for (g, p), nij in joint.items())
    hg, hp = _entropy(list(cg.values()), n), _entropy(list(cp.values()), n)
    if hg == 0.0 and hp == 0.0:
        return 1.0  # 둘 다 단일 클러스터 — 완전 일치로 간주
    denom = (hg + hp) / 2
    return mi / denom if denom else 0.0


def _pairs(count: int) -> int:
    return count * (count - 1) // 2


def pairwise_prf(gold: list, pred: list) -> dict[str, float]:
    """같은 클러스터로 묶인 '메시지 쌍' 기준 precision/recall/F1."""
    assert len(gold) == len(pred) and gold
    joint = Counter(zip(gold, pred))
    tp = sum(_pairs(c) for c in joint.values())
    pred_pairs = sum(_pairs(c) for c in Counter(pred).values())
    gold_pairs = sum(_pairs(c) for c in Counter(gold).values())
    precision = tp / pred_pairs if pred_pairs else (1.0 if gold_pairs == 0 else 0.0)
    recall = tp / gold_pairs if gold_pairs else (1.0 if pred_pairs == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def purity(gold: list, pred: list) -> float:
    """예측 클러스터마다 최다 gold 라벨이 차지하는 비율의 합 / 전체."""
    assert len(gold) == len(pred) and gold
    by_pred: dict = {}
    for g, p in zip(gold, pred):
        by_pred.setdefault(p, Counter())[g] += 1
    return sum(c.most_common(1)[0][1] for c in by_pred.values()) / len(gold)


def score(gold: dict[int, str], pred: dict[int, int]) -> dict:
    """msg_id 정렬 후 전 지표 계산. gold 에만 있는 id 는 UNASSIGNED 클러스터로."""
    ids = sorted(gold)
    if not ids:
        raise ValueError("gold 라벨이 비어 있습니다 (thread 값을 채웠는지 확인).")
    g = [gold[i] for i in ids]
    p = [pred.get(i, UNASSIGNED) for i in ids]
    return {
        "n_evaluated": len(ids),
        "n_missing_pred": sum(1 for x in p if x == UNASSIGNED),
        "nmi": nmi(g, p),
        **pairwise_prf(g, p),
        "purity": purity(g, p),
    }


# ── 파일 로더 ───────────────────────────────────────────────────────
def load_gold(path: str | Path) -> dict[int, str]:
    """gold 라벨 로드. 템플릿 리스트형과 {msg_id: label} 딕셔너리형 모두 허용."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = ((int(k), v) for k, v in data.items())
    else:
        items = ((int(r["msg_id"]), r.get("thread", "")) for r in data)
    return {mid: str(label).strip() for mid, label in items if str(label).strip()}


def load_pred(threads_path: str | Path) -> dict[int, int]:
    """threads.json(payload 리스트) → msg_id -> thread_id."""
    payload = json.loads(Path(threads_path).read_text(encoding="utf-8"))
    return {mid: t["thread_id"] for t in payload for mid in t["msg_ids"]}


def make_template(messages_jsonl: str | Path, out: str | Path) -> int:
    """messages.jsonl 로부터 라벨링 템플릿 생성. 반환: 항목 수."""
    rows = []
    with open(messages_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append({"msg_id": r["msg_id"], "sender": r["sender"],
                         "text": r["text"], "thread": ""})
    Path(out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m kakaothread.evaluate",
        description="gold 라벨 대비 분류 결과 채점 / 라벨링 템플릿 생성",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="messages.jsonl → 사람이 채울 gold 템플릿 생성")
    t.add_argument("messages", help="run 폴더의 messages.jsonl 경로")
    t.add_argument("-o", "--out", default="gold_template.json")

    s = sub.add_parser("score", help="gold 라벨 vs threads.json 채점")
    s.add_argument("gold", help="라벨을 채운 gold JSON")
    s.add_argument("threads", help="run 폴더의 threads.json")

    args = parser.parse_args()
    if args.cmd == "template":
        n = make_template(args.messages, args.out)
        print(f"템플릿 생성: {args.out} ({n}개 발화 — thread 필드를 채우세요. 잡담은 \"잡담\")")
        return

    result = score(load_gold(args.gold), load_pred(args.threads))
    print(f"평가 대상   : {result['n_evaluated']}개 발화 "
          f"(예측 누락 {result['n_missing_pred']}개)")
    print(f"NMI         : {result['nmi']:.4f}")
    print(f"pairwise P/R: {result['precision']:.4f} / {result['recall']:.4f}  "
          f"F1: {result['f1']:.4f}")
    print(f"purity      : {result['purity']:.4f}")


if __name__ == "__main__":
    main()
