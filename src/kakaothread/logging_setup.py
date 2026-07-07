"""로깅 설정.

- 콘솔: INFO 이상 (진행 마일스톤). 결과 표는 별도 print로 출력.
- 파일(logs/app.log): DEBUG 이상 (감사·추적용 상세 로그 누적).

진입점에서 setup_logging()을 한 번 호출하고, 각 모듈은
`logger = logging.getLogger(__name__)` 로 기록한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(
    console_level: int = logging.INFO, file_level: int = logging.DEBUG
) -> None:
    """루트 로거에 콘솔·파일 핸들러를 붙인다 (중복 호출 안전)."""
    root = logging.getLogger()
    if root.handlers:  # 이미 설정됨
        return

    LOG_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter(_FORMAT)
    root.setLevel(min(console_level, file_level))

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    # 서드파티 잡음 억제 (요청 모델은 우리 llm_segment 로그로 확인)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
