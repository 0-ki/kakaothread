"""증분 처리 — 같은 방을 다시 내보낸 파일에서 '새 구간만' 분류한다.

카카오톡 재내보내기는 append-only 이므로, 이전 실행의 병합 메시지 목록은
새 파일의 병합 메시지 목록과 '공통 접두(prefix)'를 이룬다. 접두 구간은 이전
배정을 그대로 재사용하고, 그 뒤(새 메시지 + 경계에서 병합이 달라진 마지막
메시지)만 LLM 분류한다.

경계 처리:
- 병합 후 메시지 기준으로 (dt, sender, text) 가 일치하는 최장 접두 P 를 찾는다.
  이전 마지막 메시지가 새 연속발화와 합쳐져 내용이 달라졌다면 자연히 P 밖으로
  떨어져 재분류된다 (배정 하나 버리고 다시 얻는 것 — 안전한 방향).
- 새 구간의 시작이 이전 마지막 세션과 시간상 이어지면(SESSION_GAP 이내),
  이전 tail 세션에 배정됐던 스레드들을 open_threads 로 시드해 캐리를 잇는다.
- 새 스레드 id 는 이전 최대 id 다음부터 발급 → 충돌 없음.

전제: 익명화 여부는 이전 실행과 동일해야 한다 (meta.json 의 anonymize 를 따른다).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from . import config
from .chunking import make_session_chunks, split_sessions
from .cost import UsageTracker
from .llm_segment import Thread
from .parallel import (
    CHECKPOINT_NAME,
    _Ctx,
    _restore,
    _save_checkpoint,
    classify_parallel,
    consolidate_safe,
    file_sha256,
)
from .pipeline import (
    NOISE_THREAD_ID,
    IdAlloc,
    PipelineOptions,
    RunResult,
    domain_context,
)
from .preprocess import Message, load_messages, merge_consecutive, parse_kakao
from .preprocess import anonymize as anonymize_messages
from .provider_pool import ProviderPool

logger = logging.getLogger(__name__)


class NothingToUpdate(Exception):
    """새 파일에 이전 실행 이후 추가된 메시지가 없음."""


def common_prefix(old: list[Message], new: list[Message]) -> int:
    """(dt, sender, text) 가 일치하는 최장 공통 접두 길이."""
    n = 0
    for o, m in zip(old, new):
        if (o.dt, o.sender, o.text) != (m.dt, m.sender, m.text):
            break
        n += 1
    return n


def _load_prev(prev_run_dir: str | Path) -> tuple[list[Message], dict[int, Thread], dict[int, int], dict]:
    """이전 run 산출물 로드 → (메시지, 스레드, 배정, meta)."""
    prev = Path(prev_run_dir)
    old_msgs = load_messages(prev / "messages.jsonl")
    payload = json.loads((prev / "threads.json").read_text(encoding="utf-8"))
    all_threads = {
        t["thread_id"]: Thread(thread_id=t["thread_id"], category=t["category"],
                               topic=t["topic"], summary=t.get("summary", ""))
        for t in payload
    }
    assignments = {mid: t["thread_id"] for t in payload for mid in t["msg_ids"]}
    meta_path = prev / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return old_msgs, all_threads, assignments, meta


def _tail_session_threads(
    prefix_msgs: list[Message],
    assignments: dict[int, int],
    all_threads: dict[int, Thread],
) -> list[Thread]:
    """접두 구간의 마지막 세션에 등장한 스레드들 (증분 시작점의 open_threads 근사)."""
    sessions = split_sessions(prefix_msgs)
    if not sessions:
        return []
    tids = {assignments.get(m.msg_id) for m in sessions[-1]}
    tids.discard(None)
    tids.discard(NOISE_THREAD_ID)
    return [all_threads[t] for t in sorted(tids) if t in all_threads]


def run_incremental(
    path: str,
    prev_run_dir: str | Path,
    opts: PipelineOptions = PipelineOptions(),
    *,
    concurrency: int = 4,
    run_dir: Path | None = None,
    resume_data: dict | None = None,
    pool: ProviderPool | None = None,
) -> RunResult:
    """이전 run 을 이어받아 새 구간만 분류. 산출물은 전체(접두+새 구간) 기준으로 반환.

    opts.max_chunks 는 증분에서 지원하지 않는다 (전체 새 구간을 처리).
    """
    if opts.max_chunks is not None:
        raise ValueError("--limit 은 증분 처리(--continue-from)와 함께 쓸 수 없습니다.")

    pool = pool or ProviderPool(config.SLOTS)
    tracker = UsageTracker()

    if resume_data is not None:
        # 재개: 이전 상태는 전부 체크포인트에 있음 — prev run 재로드 불필요
        anonymize = resume_data.get("anonymize", False)
        prefix = resume_data["prefix"]
        messages = merge_consecutive(parse_kakao(path))
        if anonymize:
            messages = anonymize_messages(messages)
        src_hash = file_sha256(path)
        if resume_data.get("source_sha256") not in (None, src_hash):
            raise ValueError("원본 파일이 체크포인트 생성 시점과 다릅니다 (sha256 불일치).")
        ctx = _Ctx(pool=pool, tracker=tracker, examples="", alloc=IdAlloc(1),
                   sem=asyncio.Semaphore(max(1, concurrency)))
        _restore(ctx, resume_data)
        ctx.ckpt_meta = {k: resume_data[k] for k in
                         ("source", "source_sha256", "max_chunks", "anonymize", "prefix", "prev_run")
                         if k in resume_data}
    else:
        old_msgs, all_threads, old_assign, prev_meta = _load_prev(prev_run_dir)
        # 익명화는 이전 실행과 반드시 동일해야 접두가 맞는다 — meta 가 우선
        anonymize = prev_meta.get("anonymize", opts.anonymize)
        if anonymize != opts.anonymize:
            logger.info("익명화 설정을 이전 실행(meta.json)과 맞춤: anonymize=%s", anonymize)
        opts = replace(opts, anonymize=anonymize)

        messages = merge_consecutive(parse_kakao(path))
        if anonymize:
            messages = anonymize_messages(messages)

        prefix = common_prefix(old_msgs, messages)
        if prefix == len(messages):
            raise NothingToUpdate(
                f"새 메시지가 없습니다 (기존 {len(old_msgs)}개 전부 일치) — 기존 결과를 사용하세요.")
        if prefix < len(old_msgs):
            logger.warning(
                "이전 실행과 겹침 불일치: 접두 %d/%d — msg_id ≥ %d 의 이전 배정은 버리고 재분류",
                prefix, len(old_msgs), prefix)

        # 접두 배정 재사용. threads.json 에는 잡담(0)이 없으므로, 접두 구간에서
        # 실제 스레드에 배정되지 않았던 메시지는 명시적으로 잡담(0)으로 복원한다
        # (전체 재실행 결과와 잡담 수·배정 통계가 일치하도록).
        prefix_assign = {mid: tid for mid, tid in old_assign.items() if mid < prefix}
        for mid in range(prefix):
            prefix_assign.setdefault(mid, NOISE_THREAD_ID)

        ctx = _Ctx(
            pool=pool, tracker=tracker, examples="",
            alloc=IdAlloc(max(all_threads, default=0) + 1),
            sem=asyncio.Semaphore(max(1, concurrency)),
            all_threads=dict(all_threads),
            assignments=prefix_assign,
        )
        ctx.examples, ctx.vocab = domain_context(messages, pool, tracker, opts)

        # 시간상 이어지면 이전 tail 세션의 스레드를 캐리 시드로
        gap = timedelta(minutes=config.SESSION_GAP_MINUTES)
        if prefix > 0 and messages[prefix].dt - messages[prefix - 1].dt <= gap:
            seed = _tail_session_threads(messages[:prefix], ctx.assignments, ctx.all_threads)
            if seed:
                ctx.open_map[0] = seed
                logger.info("세션 연속 — 이전 스레드 %d개를 캐리 시드로 주입", len(seed))

        if run_dir is not None:
            ctx.ckpt_path = Path(run_dir) / CHECKPOINT_NAME
            ctx.ckpt_meta = {
                "source": str(path),
                "source_sha256": file_sha256(path),
                "max_chunks": None,
                "anonymize": anonymize,
                "prefix": prefix,               # --resume 시 증분 모드 식별자
                "prev_run": str(prev_run_dir),
            }
            _save_checkpoint(ctx)  # 첫 청크 전에 죽어도 시드 상태부터 재개 가능

    if resume_data is not None and run_dir is not None:
        ctx.ckpt_path = Path(run_dir) / CHECKPOINT_NAME

    new_part = messages[prefix:]
    chunks = make_session_chunks(new_part)
    logger.info("증분 처리: 전체 %d개 중 접두 %d개 재사용, 새 구간 %d개 → 청크 %d개",
                len(messages), prefix, len(new_part), len(chunks))

    classify_parallel(chunks, ctx)
    all_threads, merge_map = consolidate_safe(ctx.all_threads, pool, tracker)

    # 체크포인트 삭제는 호출부가 save_and_report 성공 후 delete_checkpoint 로 수행.

    logger.info("증분 분류 완료: 스레드 %d개, 토큰 in=%d out=%d, 비용 ~$%.4f",
                len(all_threads), tracker.tok_in, tracker.tok_out, tracker.cost)
    return RunResult(messages, all_threads, ctx.assignments, tracker, merge_map,
                     len(chunks), len(chunks), anonymize)
