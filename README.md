# kakaothread

카카오톡 오픈채팅 로그를 **주제별 스레드로 분리**(conversation disentanglement)하는 CLI 도구.

> Disentangles interleaved Korean open-chat (KakaoTalk) logs into per-topic threads
> using LLM structured output, with a multi-provider slot pool and an HTML report.

<!-- 예시 스크린샷: report.html 캡처를 여기에 추가 -->
<!-- ![report](docs/images/report.png) -->

## 왜 필요한가

오픈채팅은 여러 사람이 여러 주제를 **동시에, 교차해서** 이야기합니다.
한 화면에 A 주제 질문, B 주제 답변, 잡담이 뒤섞여 있어 나중에 다시 읽거나
정보를 찾기가 어렵습니다. 이 도구는 내보낸 채팅 로그를 읽어 각 메시지를
`category(상위 범주) > topic(세부 주제)` 2단 계층의 스레드로 배정하고,
필터·타임라인이 있는 HTML 리포트를 만들어 줍니다.

## 어떻게 동작하는가 (원리)

1. **파싱·전처리** — 카카오톡 내보내기 txt를 파싱하고(PC 두 형식·모바일 자동
   감지), 봇/입퇴장 메시지를 제거하고, 같은 사람의 연속 발화를 병합합니다.
2. **세션·청크 분할** — 시간 공백으로 세션을 나누고, 품질이 유지되는
   크기(글자·개수 예산)로 청킹합니다.
3. **도메인 탐색 (LLM 1콜)** — 대화 샘플로 방의 성격을 파악해, 그 방에 맞는
   `category > topic` 예시를 분류 프롬프트에 주입합니다.
4. **LLM 분류** — 청크마다 structured output 으로 메시지별 스레드를 배정합니다.
   임베딩 클러스터링과 달리, 축약어·은어가 가리키는 **암묵적 주제**를 LLM의
   세상지식으로 잡아냅니다. 앞 청크의 스레드 목록을 이어받아 전체 로그에서
   스레드가 이어집니다.
5. **정리·리포트** — 흔들린 category 이름을 통일(LLM 1콜)하고,
   HTML 리포트와 JSON 산출물을 저장합니다.

LLM 은 특정 프로바이더에 묶여 있지 않습니다. **슬롯 풀**(우선순위 티어 +
라운드로빈 + 레이트리밋 페이싱 + 페일오버)로 OpenAI 호환 엔드포인트를
무엇이든 조합할 수 있고, 기본값은 Cerebras 무료 3모델입니다.

## 한계

- **분 단위 타임스탬프** — 카카오톡 내보내기는 초가 없어 시간 기반 로직이
  분 해상도에 갇힙니다.
- **LLM 판단 의존** — 경계가 모호한 메시지(짧은 반응, 중의적 발화)의 배정은
  모델에 따라 달라질 수 있습니다. 무료 모델 기준으로 튜닝되어 있습니다.
- **무료 티어 속도** — 기본 Cerebras 무료 슬롯은 분당 요청 제한 때문에
  큰 방(수천 발화)은 수십 분이 걸릴 수 있습니다. `--dry-run` 으로 예상
  시간을 먼저 확인하세요.
- **개인정보** — 채팅 로그는 개인정보입니다. `data/`, `logs/`, `.env` 는
  git 에서 제외되어 있지만, 결과물 공유 전 민감정보를 확인하세요.
  `--anonymize` 로 발신자를 가명 처리할 수 있습니다.

## 빠른 시작

필요한 것: [uv](https://docs.astral.sh/uv/), [Cerebras](https://cloud.cerebras.ai) 무료 API 키 1개.

```bash
# 1. 설치
git clone <repo-url> && cd kakaothread
uv sync

# 2. API 키 — .env 에 한 줄만 채우면 됩니다
cp .env.example .env        # CEREBRAS_API_KEY=csk-...

# 3. 카카오톡에서 채팅방을 "텍스트로 내보내기" 하여 .txt 파일을 받습니다.
#    받은 파일을 data/raw/ 폴더에 넣으세요 (이 폴더는 git 에서 제외됩니다).
uv run kakaothread data/raw/KakaoTalkChats.txt
```

> **내보낸 .txt 는 어디에 두나요?** 아무 곳이나 가능하며, 실행 시 그 **경로만
> 정확히 넘겨주면** 됩니다. 다만 개인정보 보호를 위해 저장소 안에 둘 거라면
> `data/raw/` 를 권장합니다 — 이 폴더는 `.gitignore` 로 제외되어 실수로 커밋·
> 공개되지 않습니다. (저장소 루트에 둔 `*.txt` 도 안전장치로 무시됩니다.)
>
> 카카오톡 내보내기 방법: **PC** 카카오톡은 채팅방 우측 상단 메뉴 →
> "대화 내용 → 대화 내보내기(txt)". **모바일**은 채팅방 메뉴 → 설정 →
> "대화 내용 내보내기 → 텍스트 메시지만". 두 포맷 모두 자동 인식합니다.

끝나면 `data/runs/<타임스탬프>/report.html` 을 브라우저로 열면 됩니다.

처음이라면 이렇게 감을 잡아 보세요:

```bash
uv run kakaothread chat.txt --dry-run    # LLM 호출 없이 통계·예상 비용/시간만
uv run kakaothread chat.txt --limit 5    # 앞 5청크만 빠르게 (스모크 테스트)
```

## 사용법

```bash
uv run kakaothread chat.txt                       # 전체 실행 (순차)
uv run kakaothread chat.txt --parallel 4          # 세션 병렬 + 체크포인트
uv run kakaothread --resume data/runs/<ts>        # 중단된 병렬 실행 이어서
uv run kakaothread new.txt --continue-from data/runs/<ts>  # 재내보내기 증분 처리
```

| 옵션 | 설명 |
|---|---|
| `--dry-run` | API 키 없이 동작. 청크 수·예상 토큰·비용·소요시간 미리보기 |
| `--limit N` | 앞 N청크만 처리. 산출물에 부분 실행임이 표시됨 |
| `--parallel N` | 세션 단위 병렬(동시 N세션) + 청크마다 체크포인트 → 중단 시 `--resume` |
| `--room-desc "…"` | 방 설명을 도메인 탐색에 주입 (예: "동네 등산 동호회 모임방") |
| `--fixed-taxonomy` | 탐색된 category 목록을 고정 어휘로 잠금 (+"기타") — 세분화 일관성 |
| `--anonymize` | 발신자를 참가자N 가명으로 치환 (본문 언급 포함, LLM에도 실명 미노출) |
| `--no-domain` | 도메인 탐색 생략. 새 모델 1콜 검증: `--limit 1 --no-domain` |
| `--continue-from <run>` | 같은 방을 다시 내보낸 파일에서 **새 구간만** 분류 (이전 배정 재사용) |

같은 원본을 다시 실행하면 sha256 기준으로 기존 run 을 알려줍니다.

### 잡 시스템 (방 1개 = job 1개)

여러 방을 반복 처리한다면 잡 큐가 편합니다. 같은 잡 이름으로 재제출하면
자동으로 증분 처리(새 구간만 분류)하고, 중단된 잡은 체크포인트에서 재개합니다.

```bash
uv run kakaothread submit chat.txt --job 등산모임   # 큐에 등록
uv run kakaothread worker                          # pending 잡 순차 실행
uv run kakaothread jobs                            # 목록/상태
uv run kakaothread cancel 등산모임                  # pending 취소
```

### 산출물

`data/runs/<타임스탬프>/` 에 저장됩니다:

| 파일 | 내용 |
|---|---|
| `report.html` | 스레드별/시간순 인터랙티브 리포트 (브라우저로 열기) |
| `threads.json` | 스레드 목록 (`category`, `topic`, `summary`, `msg_ids`) |
| `messages.jsonl` | 파싱·병합된 메시지 (msg_id ↔ 본문 매핑) |
| `category_merges.json` | 통일된 카테고리 병합 규칙 |
| `meta.json` | 실행 메타 (소스, 소요시간, 토큰·비용) |

리포트만 다시 만들려면:

```bash
python -m kakaothread.report data/runs/<ts>/threads.json data/runs/<ts>/messages.jsonl
```

### 다른 LLM 프로바이더 쓰기

OpenAI 호환(`base_url`) 엔드포인트면 무엇이든 연결할 수 있습니다 —
OpenRouter, OpenAI, 로컬 vLLM/Ollama 등. 우선순위 티어·레이트리밋·비용
추적까지 `.env` 로만 구성합니다. [docs/configuration.md](docs/configuration.md) 참고.

### 분류 품질 측정

gold 라벨을 직접 채운 뒤 채점할 수 있습니다:

```bash
python -m kakaothread.evaluate template data/runs/<ts>/messages.jsonl  # 라벨 템플릿 생성
python -m kakaothread.evaluate score gold_template.json data/runs/<ts>/threads.json
# → NMI · pairwise P/R/F1 · purity
```

## 모듈 구조

```
preprocess     카톡 txt 파싱 (PC/모바일 자동 감지, 봇/시스템 제거, 병합, 익명화)
chunking       세션 분할(시간 공백) + 글자·개수 예산 청킹
pipeline       러너 공용 코어 — 옵션/입력 준비/전역 id 발급/결과 통합/도메인 탐색
domain         샘플 대화(+방 설명)로 도메인 파악 → 맞춤 예시/고정 택소노미 (LLM 1콜)
segment_graph  순차 러너 (LangGraph 상태 루프)
parallel       세션 병렬 러너 (asyncio) + 체크포인트/재개
incremental    증분 러너 — 재내보내기의 공통 접두를 재사용, 새 구간만 분류
jobs           잡 스토어 (방 1개=job 1개, submit/worker/cancel, 재개·증분 자동 판단)
llm_segment    단일 청크 LLM 분류 (structured output, category > topic)
janitor        분류 후 흔들린 category 이름 통일 (LLM 1콜)
outputs        산출물 저장 (threads.json/messages.jsonl/report.html/meta.json)
report         HTML 리포트 렌더
evaluate       gold 라벨 채점 (NMI·pairwise F1·purity)
provider_pool  슬롯(계정×모델) 선택 — 티어/라운드로빈/페이싱/쿨다운
```

## 개발

```bash
uv sync --group dev
uv run pytest
```

## 개인정보 및 데이터 처리

이 도구는 **전적으로 사용자의 로컬 환경에서** 동작합니다.

- **로컬 처리** — 파싱·청킹·리포트 생성은 모두 사용자 컴퓨터에서 이루어집니다.
  입력 채팅 로그와 산출물(`data/`, `logs/`)은 로컬에만 저장되며, `.gitignore`
  로 저장소에서 제외됩니다.
- **BYOK (Bring Your Own Key)** — LLM 호출은 **사용자 본인이 발급·설정한**
  API 키로만 이루어집니다. 이 소프트웨어는 자체 서버·백엔드·텔레메트리를
  두지 않습니다.
- **외부 전송 범위** — 대화 데이터가 사용자 기기를 벗어나는 유일한 경우는,
  주제 분류를 위해 **사용자가 직접 선택한 LLM 프로바이더의 API**(예:
  Cerebras/OpenRouter/OpenAI 또는 사용자가 지정한 로컬·자체 호스팅
  엔드포인트)로 요청을 보낼 때뿐입니다. 그 외 어떤 제3자에게도 데이터를
  전송·수집·공유하지 않습니다. 해당 프로바이더의 데이터 취급 정책은 각
  프로바이더의 약관을 따르므로, 민감 데이터는 로컬/자체 호스팅 모델
  사용을 권장합니다.
- **학습·재사용 없음** — 이 프로젝트의 개발자 및 소프트웨어는 사용자의 채팅
  데이터를 수집·저장·모델 학습·기타 목적에 사용하지 **않습니다.** 개발자는
  사용자의 데이터에 접근할 수 없습니다.
- **가명 처리 옵션** — `--anonymize` 로 발신자명을 가명으로 치환해 LLM 에도
  실명이 전달되지 않도록 할 수 있습니다.

## 면책 조항 (Disclaimer)

> 아래는 법률 자문이 아니며, 이 소프트웨어 사용에 따른 책임 범위를 밝히기
> 위한 안내입니다.

- 이 소프트웨어는 [AGPL-3.0-or-later](LICENSE) 하에 **"있는 그대로(AS IS)",
  명시적·묵시적 어떠한 보증도 없이** 제공됩니다. 사용으로 인해 발생하는
  직접·간접 손해(데이터 손실, 유출, 보안 사고 등 포함)에 대해 개발자 및
  기여자는 법이 허용하는 최대 범위에서 **책임을 지지 않습니다.**
- **입력 데이터에 대한 책임은 사용자에게 있습니다.** 사용자는 자신이 처리하는
  채팅 로그를 수집·이용할 정당한 권한이 있는지, 그리고 관련 법령(예: 개인정보
  보호법 등)과 카카오톡을 포함한 서비스의 이용약관을 준수하는지 **스스로
  확인하고 책임져야 합니다.** 대화 상대의 동의 없이 개인정보를 처리·공개하는
  행위 등은 사용자 책임입니다.
- **API 사용·비용·보안** — 사용자가 설정한 LLM 프로바이더로의 데이터 전송,
  그에 따른 요금, API 키·로컬 데이터의 보안 관리(키 유출·기기 해킹 등 포함)는
  전적으로 사용자의 책임입니다.
- 이 도구는 카카오(주)와 아무런 제휴·후원·보증 관계가 없으며, "카카오톡"은
  해당 권리자의 상표입니다.

## 라이선스

[AGPL-3.0-or-later](LICENSE)
