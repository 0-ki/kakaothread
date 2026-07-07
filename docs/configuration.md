# 고급 설정 — LLM 슬롯 직접 구성

기본값(`CEREBRAS_API_KEY` 한 줄)으로 충분하다면 이 문서는 필요 없습니다.
여러 프로바이더를 섞거나 유료 백업 티어를 두고 싶을 때만 읽으세요.

## 개념

- **Connection(계정)** = `base_url` + `api_key`. 여러 슬롯이 공유합니다.
- **Slot(엔드포인트)** = 계정 + 모델 + rate limit/우선순위/단가.

라운드로빈·쿼터·페일오버는 전부 **슬롯 단위**로 동작합니다.
`.env` 값만 바꾸면 되고 코드 수정은 필요 없습니다. OpenAI 호환(`base_url`)
엔드포인트면 무엇이든 연결할 수 있습니다 (OpenRouter, 로컬 vLLM/Ollama 등).

`LLM_SLOTS` 를 설정하면 기본 Cerebras 프리셋 대신 여기 나열한 슬롯만 사용합니다.

## 슬롯 파라미터

| 키 | 의미 |
|---|---|
| `PRIORITY` | 작을수록 먼저 사용. 같은 값이면 그 티어 안에서 라운드로빈 |
| `RPM` / `RPD` | 분당·일당 요청 상한. 0 = 무제한. RPM 은 `60/rpm` 초 간격 페이싱으로도 적용 |
| `TPM` / `TPH` / `TPD` | 분·시간·일당 토큰 상한. 0 = 무제한 |
| `PRICE_IN` / `PRICE_OUT` | USD / 1M 토큰 (무료면 0) — 비용 리포트에 사용 |
| `REASONING_EFFORT` | 추론 모델(gpt-5/o 계열)만: `minimal\|low\|medium\|high` |

## 예시 .env

```bash
## 활성 슬롯 목록 (여기 있는 것만 로드)
LLM_SLOTS=cere_glm,or_qwen,openai_nano

## ── 연결(계정) ──────────────────────────────────────────────
LLM_CONN_CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
LLM_CONN_CEREBRAS_API_KEY=csk-...

LLM_CONN_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_CONN_OPENROUTER_API_KEY=sk-or-...

LLM_CONN_OPENAI_BASE_URL=
LLM_CONN_OPENAI_API_KEY=sk-...

## ── 슬롯: Cerebras 무료 모델 (1순위) ────────────────────────
LLM_SLOT_CERE_GLM_CONN=CEREBRAS
LLM_SLOT_CERE_GLM_MODEL=zai-glm-4.7
LLM_SLOT_CERE_GLM_PRIORITY=1
LLM_SLOT_CERE_GLM_RPM=5
LLM_SLOT_CERE_GLM_TPM=30000
LLM_SLOT_CERE_GLM_TPH=1000000
LLM_SLOT_CERE_GLM_TPD=1000000

## ── 슬롯: OpenRouter 무료 백업 (3순위) ──────────────────────
LLM_SLOT_OR_QWEN_CONN=OPENROUTER
LLM_SLOT_OR_QWEN_MODEL=qwen/qwen3-next-80b-a3b-instruct:free
LLM_SLOT_OR_QWEN_PRIORITY=3
LLM_SLOT_OR_QWEN_RPM=20
LLM_SLOT_OR_QWEN_RPD=50

## ── 슬롯: OpenAI 유료 백업 (4순위 — 무료가 다 막히면 사용) ──
LLM_SLOT_OPENAI_NANO_CONN=OPENAI
LLM_SLOT_OPENAI_NANO_MODEL=gpt-5-nano
LLM_SLOT_OPENAI_NANO_PRIORITY=4
LLM_SLOT_OPENAI_NANO_RPM=5
LLM_SLOT_OPENAI_NANO_REASONING_EFFORT=low
LLM_SLOT_OPENAI_NANO_PRICE_IN=0.05
LLM_SLOT_OPENAI_NANO_PRICE_OUT=0.40
```

## 프로바이더별 주의사항

**OpenRouter 무료 한도는 '계정 단위'** 로 전 무료모델이 공유됩니다
(분당 ~20회 + 일 50회, 크레딧 $10 이상 구매 시 1000회). Cerebras 처럼
모델별로 늘지 않으므로 무료 슬롯 1개만 백업 티어에 두는 것을 권장합니다.
`RPD` 를 지정하면 소진 후 자동으로 다음 티어로 넘어갑니다.

이 도구는 structured output 을 기본 method(function calling)로 호출하므로
**모델이 `tools` 를 지원해야 합니다.** OpenRouter 무료 모델 참고:

- 추천: `qwen/qwen3-next-80b-a3b-instruct:free` (범용·한국어 강함),
  `google/gemma-4-26b-a4b-it:free` (출력 32k), `openai/gpt-oss-120b:free` (출력 131k)
- 주의: `google/gemma-4-31b-it:free` 는 무료판 출력이 8192로 묶여 큰 청크에서 잘릴 수 있음
- 사용 불가: `nousresearch/hermes-3-*:free` (tools/structured_outputs 미지원),
  코딩 특화 모델(`qwen/qwen3-coder:free` 등)은 대화 분류 품질이 떨어짐

**gpt-5/o 계열 추론 모델**은 기본 medium 추론이 켜져 느리고, 숨은 reasoning
토큰이 출력 토큰으로 잡혀 비용이 폭증합니다. 반대로 `minimal` 은 대화 분리에
필요한 추론까지 꺼서 품질이 무너집니다. 이 도구는 절충값 `low` 를 자동
적용하며(모델명에 `gpt-5` 포함 시), 필요하면 `REASONING_EFFORT` 로 재정의
하세요. 추론 슬롯에는 temperature 가 적용되지 않습니다.
