"""실행 산출물 저장 — threads.json / messages.jsonl / report.html / meta.json + 콘솔 요약."""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import report
from .cost import UsageTracker
from .llm_segment import Thread
from .pipeline import NOISE_THREAD_ID
from .preprocess import Message, dump_messages

logger = logging.getLogger(__name__)

# 수행별 산출물 폴더 (실행마다 타임스탬프로 분리)
RUNS_DIR = Path("data/runs")


def new_run_dir() -> Path:
    base = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    # 초 단위 타임스탬프라 같은 초에 두 번 불리면 충돌한다 (잡 worker 연속 실행 등).
    # 고유 폴더를 보장해 서로 다른 실행이 같은 run_dir/체크포인트를 공유하지 않게 한다.
    d = base
    n = 2
    while d.exists():
        d = base.with_name(f"{base.name}-{n}")
        n += 1
    d.mkdir(parents=True)
    return d


def save_and_report(
    messages: list[Message],
    all_threads: dict[int, Thread],
    assignments: dict[int, int],
    run_dir: Path,
    source: str = "",
    tracker: UsageTracker | None = None,
    merge_map: dict[str, str] | None = None,
    elapsed: float | None = None,
    chunks_done: int | None = None,
    chunks_total: int | None = None,
    source_sha256: str | None = None,
    anonymize: bool | None = None,
    note: str | None = None,
) -> None:
    by_msg = {m.msg_id: m for m in messages}
    members: dict[int, list[int]] = defaultdict(list)
    unknown = 0
    for mid, tid in assignments.items():
        if mid not in by_msg:  # LLM이 환각한 존재하지 않는 msg_id — 버린다
            unknown += 1
            continue
        members[tid].append(mid)
    if unknown:
        logger.warning("존재하지 않는 msg_id 배정 %d건 무시", unknown)

    # 저장: 카테고리 > 스레드 > 메시지 id.
    # 멤버가 없는 스레드(유령)는 직렬화하지 않는다 — 증분 경계에서 배정이 전부 접두 밖으로
    # 떨어진 스레드가 msg_ids:[] 로 남아 리포트·다음 증분 재로드에 누적되는 것을 막는다.
    payload = [
        {
            "thread_id": tid,
            "category": t.category,
            "topic": t.topic,
            "summary": t.summary,
            "msg_ids": sorted(members[tid]),
        }
        for tid, t in sorted(all_threads.items())
        if members.get(tid)
    ]
    ghosts = len(all_threads) - len(payload)
    if ghosts:
        logger.info("멤버 없는 스레드 %d개 제외", ghosts)
    threads_path = run_dir / "threads.json"
    threads_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("결과 저장: %s (스레드 %d개)", threads_path, len(all_threads))

    # msg_id↔본문 매핑 영속화 (원본 재파싱 없이 run 폴더만으로 해석 가능)
    dump_messages(messages, run_dir / "messages.jsonl")

    # 정리부 병합 규칙 저장 (어떤 category가 어디로 합쳐졌는지 사람이 확인 가능)
    if merge_map:
        (run_dir / "category_merges.json").write_text(
            json.dumps(merge_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # HTML 리포트 (메시지 매핑 포함)
    meta = {
        "source": source,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if source_sha256:
        meta["source_sha256"] = source_sha256  # 동일 원본 재실행 감지용
    if anonymize is not None:
        meta["anonymize"] = anonymize  # 증분 처리 시 동일 전처리 강제용
    if note:
        meta["note"] = note  # 예: 증분 실행 표시
    if chunks_total is not None:
        meta["chunks_done"] = chunks_done
        meta["chunks_total"] = chunks_total
        if chunks_done is not None and chunks_done < chunks_total:
            # 부분 실행 산출물이 전체 실행으로 오인되지 않도록 명시
            meta["partial"] = f"전체 {chunks_total}청크 중 앞 {chunks_done}청크만 처리 (부분 실행)"
    if elapsed is not None:
        meta["elapsed_seconds"] = elapsed
    if tracker is not None:
        meta["models"] = ", ".join(
            f"{s['model']}×{s['calls']}" for s in tracker.stats.values()
        )
        meta["cost"] = tracker.cost
        meta["tok_in"] = tracker.tok_in
        meta["tok_out"] = tracker.tok_out
        meta["model_stats"] = [  # 모델별 상세 표용
            {"name": name, "model": s["model"], "calls": s["calls"],
             "tok_in": s["in"], "tok_out": s["out"]}
            for name, s in tracker.stats.items()
        ]
    report.write_report(messages, payload, run_dir / "report.html", meta)

    # 실행 메타 저장 (재현/추적용) — n_threads 는 실제 직렬화된(멤버 있는) 스레드 수
    (run_dir / "meta.json").write_text(
        json.dumps({**meta, "n_threads": len(payload), "n_messages": len(messages)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 출력: 카테고리별로 묶어서 미리보기 (멤버 있는 스레드만)
    live_threads = [all_threads[t["thread_id"]] for t in payload]
    by_category: dict[str, list[Thread]] = defaultdict(list)
    for t in live_threads:
        by_category[t.category].append(t)

    if meta.get("partial"):
        print(f"\n⚠️ {meta['partial']}")
    print(f"\n총 스레드 {len(payload)}개 / 카테고리 {len(by_category)}개 -> {run_dir}\n")
    for category in sorted(by_category):
        print(f"[{category}]")
        for t in sorted(by_category[category], key=lambda t: t.thread_id):
            ids = sorted(members.get(t.thread_id, []))
            m0 = by_msg.get(ids[0]) if ids else None
            sample = m0.text[:40] if m0 else ""
            print(f"  #{t.thread_id} {t.topic} ({len(ids)} msgs)  예: {sample}")
    noise = members.get(NOISE_THREAD_ID, [])
    print(f"\n잡담(thread {NOISE_THREAD_ID}): {len(noise)} msgs")
    print(f"\n리포트: {run_dir / 'report.html'}")
