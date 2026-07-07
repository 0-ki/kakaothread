"""명령행 진입점.

즉시 실행 (서브커맨드 생략 시 run 으로 해석):
    kakaothread data/raw/KakaoTalkChats.txt              # 전체 실행
    kakaothread data/raw/KakaoTalkChats.txt --dry-run    # LLM 없이 통계·예상 비용만
    kakaothread data/raw/KakaoTalkChats.txt --limit 5    # 앞 5청크만 (테스트)
    kakaothread data/raw/KakaoTalkChats.txt --parallel 4 # 세션 병렬 (체크포인트 기록)
    kakaothread --resume data/runs/20260706_123154       # 중단된 병렬 실행 이어서
    kakaothread new.txt --continue-from data/runs/<ts>   # 재내보내기 증분 처리

잡 시스템 (방 1개 = job 1개):
    kakaothread submit chat.txt --job 등산모임             # 큐에 등록
    kakaothread worker                                   # pending 잡 순차 실행
    kakaothread jobs                                     # 잡 목록/상태
    kakaothread cancel 등산모임                            # pending 잡 취소
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from . import config, jobs, outputs
from .chunking import _msg_chars, make_session_chunks
from .incremental import NothingToUpdate, run_incremental
from .logging_setup import setup_logging
from .outputs import new_run_dir, save_and_report
from .parallel import (
    delete_checkpoint,
    file_sha256,
    load_checkpoint,
    options_from_checkpoint,
    run_parallel,
)
from .pipeline import PipelineOptions
from .preprocess import Message, merge_consecutive, parse_kakao
from .segment_graph import run

SUBCOMMANDS = {"run", "submit", "worker", "jobs", "cancel"}

# ── dry-run 추정용 대략치 — 정확한 예측이 아니라 자릿수 가늠용 ──────
PROMPT_OVERHEAD_TOKENS = 600  # 청크당 고정 프롬프트 (규칙·예시·열린 스레드)
OUT_TOKENS_PER_MSG = 15       # 메시지당 배정 JSON
OUT_TOKENS_BASE = 200         # 청크당 스레드 메타 등
EXTRA_CALLS = 2               # 도메인탐색 1 + janitor 1


def _fmt_dur(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}시간 {m}분"
    if m:
        return f"{m}분 {s}초"
    return f"{s}초"


def preview_stats(
    n_raw: int,
    n_merged: int,
    chunks: list[tuple[int, list[Message]]],
    slots: list[config.Slot],
    use_domain: bool = True,
) -> dict:
    """dry-run 통계: 청킹 결과 + 예상 토큰/비용/소요시간(레이트리밋 하한)."""
    sizes = [len(c) for _, c in chunks]
    extra = EXTRA_CALLS if use_domain else EXTRA_CALLS - 1  # 도메인 탐색 콜 생략 시 -1
    n_calls = len(chunks) + extra
    chars = sum(_msg_chars(m) for _, c in chunks for m in c)
    tok_in = chars // config.CHARS_PER_TOKEN + n_calls * PROMPT_OVERHEAD_TOKENS
    tok_out = sum(n * OUT_TOKENS_PER_MSG + OUT_TOKENS_BASE for n in sizes) if sizes else 0

    st = {
        "n_raw": n_raw,
        "n_merged": n_merged,
        "n_sessions": len({sidx for sidx, _ in chunks}),
        "n_chunks": len(chunks),
        "sizes": sizes,
        "n_calls": n_calls,
        "tok_in": tok_in,
        "tok_out": tok_out,
        "cost": None,          # 슬롯 없으면 추정 불가
        "min_seconds": None,   # 레이트리밋 없으면(rpm=0) 하한 없음
    }
    if slots:
        # 호출 대부분이 최상위 티어에서 소화된다고 보고 그 티어의 평균 단가로 환산
        top = min(s.priority for s in slots)
        tier = [s for s in slots if s.priority == top]
        pin = statistics.mean(s.price_in for s in tier)
        pout = statistics.mean(s.price_out for s in tier)
        st["cost"] = tok_in / 1e6 * pin + tok_out / 1e6 * pout
        if all(s.rpm > 0 for s in tier):
            st["min_seconds"] = n_calls * 60 / sum(s.rpm for s in tier)
    return st


def _dry_run(path: str, *, limit: int | None = None, use_domain: bool = True) -> None:
    raw = parse_kakao(path)
    messages = merge_consecutive(raw)
    chunks = make_session_chunks(messages)
    # 같은 명령의 --limit / --no-domain 을 추정에도 반영 (안 하면 전체 기준 과대 추정)
    if limit is not None:
        chunks = chunks[:limit]
    st = preview_stats(len(raw), len(messages), chunks, config.SLOTS, use_domain=use_domain)

    print("\n=== dry-run: LLM 호출 없이 파싱·청킹까지만 수행 ===\n")
    print(f"발화        : {st['n_raw']:,}개 → 병합 후 {st['n_merged']:,}개")
    print(f"세션        : {st['n_sessions']}개 (공백 {config.SESSION_GAP_MINUTES}분 기준)")
    print(f"청크        : {st['n_chunks']}개 "
          f"(예산 {config.CHUNK_CHAR_BUDGET:,}자 / 최대 {config.CHUNK_MAX_MESSAGES}개)")
    if st["sizes"]:
        print(f"  메시지/청크: 최소 {min(st['sizes'])} · "
              f"중앙값 {int(statistics.median(st['sizes']))} · 최대 {max(st['sizes'])}")
    extra_desc = "도메인탐색+janitor" if use_domain else "janitor"
    print(f"예상 LLM 호출: {st['n_calls']}콜 "
          f"(분류 {st['n_chunks']} + {extra_desc} {st['n_calls'] - st['n_chunks']})")
    print(f"예상 토큰    : 입력 ~{st['tok_in']:,} / 출력 ~{st['tok_out']:,} (대략치)")

    if config.SLOTS:
        slot_desc = ", ".join(f"{s.name}({s.model}, p{s.priority})" for s in config.SLOTS)
        print(f"활성 슬롯    : {slot_desc}")
        print(f"예상 비용    : ~${st['cost']:.4f} (최상위 티어 평균 단가 기준)")
        if st["min_seconds"] is not None:
            print(f"예상 소요시간: 최소 ~{_fmt_dur(st['min_seconds'])} (레이트리밋 하한, LLM 응답시간 별도)")
        else:
            print("예상 소요시간: 레이트리밋 없음 (rpm=0) — LLM 응답시간에 좌우")
    else:
        print("활성 슬롯    : 없음 — 비용/시간 추정 생략 "
              "(.env 에 CEREBRAS_API_KEY 필요. dry-run 자체는 키 없이 동작)")
    print("\n실제 실행: 위 명령에서 --dry-run 제거 (테스트는 --limit N)")


def _warn_duplicate(src_hash: str) -> None:
    """같은 원본을 이미 처리한 완료 run 이 있으면 알려준다 (중복 실행 방지 힌트)."""
    runs_dir = outputs.RUNS_DIR
    if not runs_dir.exists():
        return
    for meta_path in sorted(runs_dir.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("source_sha256") == src_hash:
            print(f"⚠️ 같은 원본(sha256 일치)을 처리한 기존 실행이 있습니다: {meta_path.parent}")
            print("   (계속 진행합니다 — 결과 비교가 목적이 아니면 기존 결과를 재사용하세요)")
            return


def _add_domain_flags(p: argparse.ArgumentParser) -> None:
    """run/submit 공용 파이프라인 플래그."""
    p.add_argument(
        "--no-domain", action="store_true",
        help="도메인 탐색(LLM 1콜) 생략, 기본 예시 사용",
    )
    p.add_argument(
        "--room-desc", default="", metavar="TEXT",
        help='방에 대한 간략한 설명 (예: "동네 등산 동호회 모임방") — 도메인 탐색을 부트스트랩',
    )
    p.add_argument(
        "--fixed-taxonomy", action="store_true",
        help='도메인 탐색이 뽑은 category 목록을 고정 어휘로 잠금 (분류는 목록+"기타"에서만 선택)',
    )
    p.add_argument(
        "--anonymize", action="store_true",
        help="발신자를 '참가자N' 가명으로 치환 (본문 내 언급 포함, LLM에도 실명 미노출, 매핑 미저장)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kakaothread",
        description="카카오톡 오픈채팅 내보내기(.txt)를 주제별 스레드로 분리한다. "
                    "서브커맨드를 생략하면 run 으로 해석한다.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── run: 즉시 실행 ──────────────────────────────────────────────
    p_run = sub.add_parser("run", help="파일 하나를 즉시 분류 (기본 커맨드)")
    p_run.add_argument("input", nargs="?", default=None,
                       help="카카오톡 내보내기 .txt 파일 경로")
    p_run.add_argument(
        "-o", "--out-dir", default=None,
        help="산출물 폴더 (기본: data/runs/<타임스탬프>)",
    )
    p_run.add_argument(
        "--dry-run", action="store_true",
        help="LLM 호출 없이 파싱·청킹 통계와 예상 토큰/비용/시간만 출력",
    )
    p_run.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="앞 N개 청크만 분류 (프롬프트 튜닝·새 모델 스모크 테스트용 부분 실행)",
    )
    _add_domain_flags(p_run)
    p_run.add_argument(
        "--parallel", type=int, default=0, metavar="N",
        help="세션 병렬 실행 (동시 세션 N개). 체크포인트를 기록해 중단 시 이어서 실행 가능. "
             "0=순차(기본, LangGraph)",
    )
    p_run.add_argument(
        "--resume", metavar="RUN_DIR", default=None,
        help="중단된 병렬 실행을 해당 run 폴더의 체크포인트에서 이어서 실행",
    )
    p_run.add_argument(
        "--continue-from", metavar="PREV_RUN_DIR", default=None,
        help="같은 방을 다시 내보낸 파일에서 새 구간만 분류 (이전 run 의 배정을 재사용)",
    )

    # ── 잡 시스템 ───────────────────────────────────────────────────
    p_sub = sub.add_parser("submit", help="잡 큐에 등록 (방 1개 = job 1개, 재제출 시 증분)")
    p_sub.add_argument("input", help="카카오톡 내보내기 .txt 파일 경로")
    p_sub.add_argument("--job", default=None, metavar="NAME",
                       help="잡 이름 (기본: 파일명 — 같은 방은 같은 이름으로 재제출)")
    p_sub.add_argument("--concurrency", type=int, default=4, metavar="N",
                       help="worker 실행 시 동시 세션 수 (기본 4)")
    _add_domain_flags(p_sub)

    p_wk = sub.add_parser("worker", help="pending 잡을 제출 순서대로 실행 (단일 worker 전제)")
    p_wk.add_argument("--once", action="store_true", help="잡 1개만 처리하고 종료")

    sub.add_parser("jobs", help="잡 목록/상태 출력")

    p_cx = sub.add_parser("cancel", help="pending 잡 취소")
    p_cx.add_argument("job_id", help="취소할 잡 이름")
    return parser


def _cmd_run(args, parser: argparse.ArgumentParser) -> None:
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 은 1 이상이어야 합니다.")
    if args.fixed_taxonomy and args.no_domain:
        parser.error("--fixed-taxonomy 는 도메인 탐색이 필요합니다 (--no-domain 과 함께 쓸 수 없음).")
    if args.resume:
        # 원본 경로·범위·전처리 옵션은 전부 체크포인트에 저장돼 있으므로 재지정 불가.
        # (조용히 무시되면 --anonymize 등이 효과 없이 실명이 저장되는 사고로 이어짐)
        pipeline_flags = (args.no_domain or args.room_desc or args.fixed_taxonomy
                          or args.anonymize)
        if (args.input or args.dry_run or args.limit or args.out_dir
                or args.continue_from or pipeline_flags):
            parser.error("--resume 은 다른 옵션과 함께 쓸 수 없습니다 "
                         "(원본·범위·전처리 설정은 체크포인트에 저장돼 있음).")
    elif not args.input:
        parser.error("input 파일 경로가 필요합니다 (또는 --resume RUN_DIR).")
    if args.continue_from and (args.limit or args.dry_run):
        parser.error("--continue-from 은 --limit/--dry-run 과 함께 쓸 수 없습니다.")

    if args.dry_run:
        _dry_run(args.input, limit=args.limit, use_domain=not args.no_domain)
        return

    t0 = time.perf_counter()
    if args.resume:
        run_dir = Path(args.resume)
        ckpt = load_checkpoint(run_dir)
        source = ckpt["source"]
        src_hash = ckpt.get("source_sha256")
        if "prefix" in ckpt:  # 증분 실행의 체크포인트
            res = run_incremental(source, ckpt.get("prev_run", ""),
                                  concurrency=args.parallel or 4, run_dir=run_dir,
                                  resume_data=ckpt)
        else:
            res = run_parallel(source, options_from_checkpoint(ckpt),
                               concurrency=args.parallel or 4, run_dir=run_dir,
                               resume_data=ckpt)
    else:
        source = args.input
        src_hash = file_sha256(source)
        opts = PipelineOptions(max_chunks=args.limit, use_domain=not args.no_domain,
                               room_desc=args.room_desc, fixed_taxonomy=args.fixed_taxonomy,
                               anonymize=args.anonymize)
        if args.out_dir:
            run_dir = Path(args.out_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = new_run_dir()
        if args.continue_from:
            try:
                res = run_incremental(source, args.continue_from, opts,
                                      concurrency=args.parallel or 4, run_dir=run_dir)
            except NothingToUpdate as e:
                print(f"변경 없음: {e}")
                return
        else:
            _warn_duplicate(src_hash)
            if args.parallel > 0:
                res = run_parallel(source, opts, concurrency=args.parallel, run_dir=run_dir)
            else:
                res = run(source, opts)
    elapsed = time.perf_counter() - t0
    prev = args.continue_from or (args.resume and ckpt.get("prev_run"))
    note = f"증분 실행 — 이전 run({prev})의 배정을 재사용, 새 구간만 분류" if prev else None
    save_and_report(res.messages, res.threads, res.assignments, run_dir, source=source,
                    tracker=res.tracker, merge_map=res.merge_map, elapsed=elapsed,
                    chunks_done=res.chunks_done, chunks_total=res.chunks_total,
                    source_sha256=src_hash, anonymize=res.anonymize, note=note)
    delete_checkpoint(run_dir)  # 산출물 저장 성공 후에만 체크포인트 정리 (실패 시 재개 가능)
    print()
    res.tracker.report("전체")


def _cmd_submit(args, parser: argparse.ArgumentParser) -> None:
    if args.fixed_taxonomy and args.no_domain:
        parser.error("--fixed-taxonomy 는 도메인 탐색이 필요합니다 (--no-domain 과 함께 쓸 수 없음).")
    options = {
        "use_domain": not args.no_domain,
        "room_desc": args.room_desc,
        "fixed_taxonomy": args.fixed_taxonomy,
        "anonymize": args.anonymize,
        "concurrency": args.concurrency,
    }
    job = jobs.submit(args.input, job_id=args.job, options=options)
    print(f"잡 등록: {job.job_id} (pending) — 실행: kakaothread worker")


def _cmd_worker(args) -> None:
    n = jobs.worker(once=args.once)
    if n == 0:
        print("처리할 pending 잡이 없습니다.")
    else:
        print(f"\n잡 {n}개 처리 완료 — 상태: kakaothread jobs")


def _cmd_jobs() -> None:
    all_jobs = jobs.list_jobs()
    if not all_jobs:
        print("등록된 잡이 없습니다 — 등록: kakaothread submit <파일> [--job 이름]")
        return
    print(f"{'JOB':<20}{'STATUS':<11}{'제출시각':<20}RUN")
    print("-" * 78)
    for j in sorted(all_jobs, key=lambda j: j.submitted_at):
        run_info = j.last_run_dir or j.run_dir or "-"
        print(f"{j.job_id:<20}{j.status:<11}{j.submitted_at:<20}{run_info}")
        if j.error:
            print(f"{'':<20}└ 오류: {j.error}")


def _cmd_cancel(args) -> None:
    try:
        job = jobs.cancel(args.job_id)
    except FileNotFoundError:
        print(f"잡을 찾을 수 없습니다: {args.job_id}")
        raise SystemExit(1)
    print(f"잡 {job.job_id}: {job.status}")


def main() -> None:
    argv = sys.argv[1:]
    # 하위호환 shim: 서브커맨드를 생략하면 run 으로 해석
    # (kakaothread chat.txt / kakaothread --resume ... 형태 유지)
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["run", *argv]
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging()
    if args.cmd == "run":
        _cmd_run(args, parser)
    elif args.cmd == "submit":
        _cmd_submit(args, parser)
    elif args.cmd == "worker":
        _cmd_worker(args)
    elif args.cmd == "jobs":
        _cmd_jobs()
    elif args.cmd == "cancel":
        _cmd_cancel(args)


if __name__ == "__main__":
    main()
