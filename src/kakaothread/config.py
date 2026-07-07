"""중앙 설정 — LLM 슬롯(모델)/청킹 파라미터를 한 곳에서 관리.

멀티 프로바이더 라운드로빈:
  - Connection(계정) = base_url + api_key  (여러 슬롯이 공유)
  - Slot(엔드포인트)  = connection + model + 그 모델만의 rate limit / 우선순위 / 단가
라운드로빈·쿼터·쿨다운은 전부 Slot 단위로 provider_pool.py 가 처리한다.
모든 슬롯 설정은 .env 에서 조정 (코드 수정 불필요).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # ${VAR} 치환 지원 (python-dotenv 기본)

logger = logging.getLogger(__name__)


# ── LLM 슬롯 ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Connection:
    """base_url + api_key. 여러 슬롯이 공유하는 계정 단위."""
    name: str
    base_url: str  # 빈 문자열이면 OpenAI 기본 엔드포인트
    api_key: str


@dataclass(frozen=True)
class Slot:
    """라운드로빈/쿼터/우선순위의 단위 = (계정 × 모델)."""
    name: str
    conn: Connection
    model: str
    priority: int = 1       # 작을수록 먼저 사용 (같은 값이면 라운드로빈)
    rpm: int = 0            # 분당 요청 상한   (0 = 무제한)
    rpd: int = 0            # 일간 요청 상한   (0 = 무제한, 예: OpenRouter 무료 50/일)
    tpm: int = 0            # 분당 토큰 상한   (0 = 무제한)
    tph: int = 0            # 시간당 토큰 상한 (0 = 무제한)
    tpd: int = 0            # 일간 토큰 상한   (0 = 무제한)
    price_in: float = 0.0   # USD / 1M 입력 토큰
    price_out: float = 0.0  # USD / 1M 출력 토큰
    reasoning_effort: str = ""  # gpt-5/o 계열만: minimal|low|medium|high (빈값=미지정→gpt-5는 low 기본)


def _env(prefix: str, key: str, default: str = "") -> str:
    return os.getenv(f"{prefix}_{key}", default).strip()


def _load_connections() -> dict[str, Connection]:
    """LLM_CONN_<NAME>_BASE_URL / _API_KEY 를 훑어 계정 목록 구성."""
    conns: dict[str, Connection] = {}
    for k in os.environ:
        if not (k.startswith("LLM_CONN_") and k.endswith("_API_KEY")):
            continue
        name = k[len("LLM_CONN_"):-len("_API_KEY")]  # 예: CEREBRAS
        prefix = f"LLM_CONN_{name}"
        api_key = _env(prefix, "API_KEY")
        if not api_key:
            continue
        conns[name] = Connection(name=name, base_url=_env(prefix, "BASE_URL"), api_key=api_key)
    return conns


def _load_slots() -> list[Slot]:
    conns = _load_connections()
    slots: list[Slot] = []
    for raw in os.getenv("LLM_SLOTS", "").split(","):
        sname = raw.strip()
        if not sname:
            continue
        prefix = f"LLM_SLOT_{sname.upper()}"
        conn_name = _env(prefix, "CONN").upper()
        conn = conns.get(conn_name)
        if conn is None:
            logger.warning("슬롯 %s: 연결 '%s' 미정의 — 건너뜀", sname, conn_name)
            continue
        model = _env(prefix, "MODEL")
        if not model:
            logger.warning("슬롯 %s: MODEL 누락 — 건너뜀", sname)
            continue
        slots.append(Slot(
            name=sname,
            conn=conn,
            model=model,
            priority=int(_env(prefix, "PRIORITY", "1") or "1"),
            rpm=int(_env(prefix, "RPM", "0") or "0"),
            rpd=int(_env(prefix, "RPD", "0") or "0"),
            tpm=int(_env(prefix, "TPM", "0") or "0"),
            tph=int(_env(prefix, "TPH", "0") or "0"),
            tpd=int(_env(prefix, "TPD", "0") or "0"),
            price_in=float(_env(prefix, "PRICE_IN", "0") or "0"),
            price_out=float(_env(prefix, "PRICE_OUT", "0") or "0"),
            reasoning_effort=_env(prefix, "REASONING_EFFORT").lower(),
        ))
    if not slots:
        slots = _default_cerebras_slots()
        if slots:
            logger.info("LLM_SLOTS 미설정 — Cerebras 무료 3모델 기본 프리셋 사용: %s",
                        ", ".join(s.model for s in slots))
        else:
            logger.warning(
                "활성 LLM 슬롯이 없습니다 — .env 에 CEREBRAS_API_KEY 또는 LLM_SLOTS 를 설정하세요.")
    return slots


# ── 기본 프리셋: CEREBRAS_API_KEY 만 넣으면 동작 ────────────────────
# 무료 3모델을 같은 우선순위(=라운드로빈)로 돌린다. rpm=5 는 슬롯별 12초 간격
# 페이싱으로 지켜지므로(provider_pool), 셋이 번갈아 나가면 전체 ~4초에 1콜.
DEFAULT_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_CEREBRAS_MODELS = ("zai-glm-4.7", "gemma-4-31b", "gpt-oss-120b")
DEFAULT_CEREBRAS_LIMITS = {"rpm": 5, "tpm": 30_000, "tph": 1_000_000, "tpd": 1_000_000}


def _default_cerebras_slots() -> list[Slot]:
    key = os.getenv("CEREBRAS_API_KEY", "").strip() or _env("LLM_CONN_CEREBRAS", "API_KEY")
    if not key:
        return []
    conn = Connection(name="CEREBRAS", base_url=DEFAULT_CEREBRAS_BASE_URL, api_key=key)
    return [Slot(name=m, conn=conn, model=m, priority=1, **DEFAULT_CEREBRAS_LIMITS)
            for m in DEFAULT_CEREBRAS_MODELS]


SLOTS: list[Slot] = _load_slots()

# LLM 요청 타임아웃(초) — 무료 엔드포인트가 간혹 응답을 물고 있어 무한 대기 방지
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120") or "120")


# ── 청킹 ────────────────────────────────────────────────────────────
# 세션: 이 이상 시간 공백이면 대화 단절로 보고 분리 (경계)
SESSION_GAP_MINUTES = 180

# 청크 크기(중요): '컨텍스트 한계'가 아니라 '품질 한계'로 정한다.
#   청크가 커질수록 LLM이 동시에 추적할 인터리빙 스레드가 늘어 정확도가 먼저 무너진다.
#   → 컨텍스트 윈도우의 50%는 "넘지 말 것(안전 상한)"이지 목표가 아니다.
#   운영값은 훨씬 작게(수천 토큰) 두고 평가셋으로 튜닝한다.
CHUNK_TARGET_TOKENS = 3000
CHARS_PER_TOKEN = 2  # 한국어 대략치 (토큰 추정용, 프로바이더 무관)
CHUNK_CHAR_BUDGET = CHUNK_TARGET_TOKENS * CHARS_PER_TOKEN

# 짧은 메시지 폭주 대비: 토큰 예산과 별개로 '개수'로도 상한을 둔다.
# (disentanglement 품질은 배정 항목/스레드 개수에도 좌우되므로)
CHUNK_MAX_MESSAGES = 80
