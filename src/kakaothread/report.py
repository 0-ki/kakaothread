"""분류 결과(threads.json)를 메시지 본문과 매핑해 가독성 좋은 HTML 리포트로 렌더.

두 가지 뷰:
  - 스레드별: category > thread 계층으로 묶어 메시지 나열 (체크박스 필터 + 접기/펼치기)
  - 시간순:   원본 순서(msg_id) 그대로, 스레드 색으로 인터리빙 시각화

segment_graph 에서 인메모리로 호출하거나, 저장된 산출물로 단독 실행 가능:
    python -m kakaothread.report data/runs/<ts>/threads.json data/runs/<ts>/messages.jsonl
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path

from .preprocess import Message, load_messages, merge_consecutive, parse_kakao

logger = logging.getLogger(__name__)

NOISE_THREAD_ID = 0

# 카테고리별 색상 팔레트 (순환)
_PALETTE = [
    "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
    "#0d9488", "#be123c", "#7c2d12", "#1d4ed8", "#a21caf",
]
_NOISE_COLOR = "#9ca3af"


def _category_colors(categories: list[str]) -> dict[str, str]:
    return {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(sorted(categories))}


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _fmt_elapsed(sec: float) -> str:
    sec = int(round(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}시간 {m}분 {s}초"
    if m:
        return f"{m}분 {s}초"
    return f"{s}초"


def _runinfo_html(meta: dict) -> str:
    """실행 상세(토글 안 내용): 소스/시각/소요시간/토큰·비용/모델별 표."""
    rows: list[str] = []

    def row(label: str, value: str) -> str:
        return f'<div class="ri-row"><span class="ri-k">{label}</span><span class="ri-v">{value}</span></div>'

    if meta.get("source"):
        src = str(meta["source"])
        rows.append(row("소스 파일", f'<b>{_esc(Path(src).name)}</b>'))
        rows.append(row("경로", f'<span class="mono">{_esc(src)}</span>'))
    if meta.get("generated"):
        rows.append(row("생성 시각", _esc(meta["generated"])))
    if meta.get("partial"):
        rows.append(row("실행 범위", f"⚠️ {_esc(meta['partial'])}"))
    if meta.get("note"):
        rows.append(row("비고", _esc(meta["note"])))
    if meta.get("elapsed_seconds") is not None:
        rows.append(row("작업 소요시간", _fmt_elapsed(meta["elapsed_seconds"])))
    if meta.get("cost") is not None:
        rows.append(row(
            "토큰 (in/out)",
            f'{meta.get("tok_in", 0):,} / {meta.get("tok_out", 0):,} · 비용 ~${meta["cost"]:.4f}',
        ))

    model_table = ""
    stats = meta.get("model_stats")
    if stats:
        body = "".join(
            f'<tr><td>{_esc(s.get("model", ""))}</td>'
            f'<td class="num">{s.get("calls", 0):,}</td>'
            f'<td class="num">{s.get("tok_in", 0):,}</td>'
            f'<td class="num">{s.get("tok_out", 0):,}</td></tr>'
            for s in stats
        )
        model_table = (
            '<table class="ri-table"><thead><tr>'
            '<th>모델</th><th class="num">호출</th><th class="num">토큰 in</th><th class="num">토큰 out</th>'
            f'</tr></thead><tbody>{body}</tbody></table>'
        )
    elif meta.get("models"):
        model_table = row("모델", _esc(meta["models"]))

    return f'<div class="ri-rows">{"".join(rows)}</div>{model_table}'


def render(
    messages: list[Message],
    threads: list[dict],
    meta: dict | None = None,
) -> str:
    """messages + threads(payload: {thread_id,category,topic,summary,msg_ids}) -> HTML."""
    meta = meta or {}
    by_msg: dict[int, Message] = {m.msg_id: m for m in messages}

    assign: dict[int, tuple[int, str, str]] = {}
    for t in threads:
        for mid in t["msg_ids"]:
            assign[mid] = (t["thread_id"], t["category"], t["topic"])

    categories = sorted({t["category"] for t in threads})
    colors = _category_colors(categories)
    noise_ids = sorted(mid for mid in by_msg if mid not in assign)

    n_msgs = len(by_msg)
    n_assigned = len(assign)
    n_threads = len(threads)
    n_cats = len(categories)

    def color_of(mid: int) -> str:
        info = assign.get(mid)
        return colors[info[1]] if info else _NOISE_COLOR

    def fmt_time(m: Message) -> str:
        return m.dt.strftime("%m/%d %H:%M")

    def msg_row(mid: int, *, show_thread: bool = False) -> str:
        m = by_msg.get(mid)
        if m is None:
            return ""
        col = color_of(mid)
        badge = ""
        if show_thread:
            info = assign.get(mid)
            label = f"#{info[0]} {_esc(info[2])}" if info else "잡담"
            badge = f'<span class="tbadge" style="background:{col}">{label}</span>'
        return (
            f'<div class="msg" style="border-left-color:{col}">'
            f'<span class="mid">#{mid}</span>'
            f'<span class="time">{fmt_time(m)}</span>'
            f'{badge}'
            f'<span class="sender">{_esc(m.sender)}</span>'
            f'<span class="text">{_esc(m.text)}</span>'
            f"</div>"
        )

    def card(tid: str, topic: str, n: int, rows: str, summary: str = "") -> str:
        summary_html = f'<div class="summary">{_esc(summary)}</div>' if summary else ""
        return (
            f'<div class="card" data-tid="{tid}">'
            f'<div class="card-head" onclick="toggleCard(this)">'
            f'<span class="tid">#{tid}</span>'
            f'<span class="topic">{_esc(topic)}</span>'
            f'<span class="count">{n} msgs</span>'
            f'<span class="chev">▾</span></div>'
            f'{summary_html}'
            f'<div class="card-body">{rows}</div>'
            f"</div>"
        )

    # ── 스레드별 뷰 + 필터 항목 ──
    by_cat: dict[str, list[dict]] = {}
    for t in threads:
        by_cat.setdefault(t["category"], []).append(t)

    thread_view_parts: list[str] = []
    filter_groups: list[str] = []
    for cat in categories:
        col = colors[cat]
        cat_threads = sorted(by_cat[cat], key=lambda t: t["thread_id"])
        cat_msgs = sum(len(t["msg_ids"]) for t in cat_threads)
        cards = "".join(
            card(str(t["thread_id"]), t["topic"], len(t["msg_ids"]),
                 "".join(msg_row(mid) for mid in sorted(t["msg_ids"])), t.get("summary", ""))
            for t in cat_threads
        )
        thread_view_parts.append(
            f'<section class="cat" data-cat="{_esc(cat)}" style="--cat:{col}">'
            f'<h2><span class="dot"></span>{_esc(cat)}'
            f'<span class="cat-meta">{len(cat_threads)} threads · {cat_msgs} msgs</span></h2>'
            f'<div class="cards">{cards}</div>'
            f"</section>"
        )
        items = "".join(
            f'<label class="fitem"><input type="checkbox" checked data-tid="{t["thread_id"]}" '
            f'onchange="applyFilter()"><span class="fdot" style="background:{col}"></span>'
            f'#{t["thread_id"]} {_esc(t["topic"])} <span class="fn">{len(t["msg_ids"])}</span></label>'
            for t in cat_threads
        )
        filter_groups.append(
            f'<div class="fgroup"><div class="flabel" style="--cat:{col}">{_esc(cat)}</div>{items}</div>'
        )

    if noise_ids:
        rows = "".join(msg_row(mid) for mid in noise_ids)
        thread_view_parts.append(
            f'<section class="cat" data-cat="__noise__" style="--cat:{_NOISE_COLOR}">'
            f'<h2><span class="dot"></span>잡담 / 미분류'
            f'<span class="cat-meta">{len(noise_ids)} msgs</span></h2>'
            f'<div class="cards">{card("noise", "잡담 / 미분류", len(noise_ids), rows)}</div>'
            f"</section>"
        )
        filter_groups.append(
            f'<div class="fgroup"><div class="flabel" style="--cat:{_NOISE_COLOR}">잡담 / 미분류</div>'
            f'<label class="fitem"><input type="checkbox" checked data-tid="noise" onchange="applyFilter()">'
            f'<span class="fdot" style="background:{_NOISE_COLOR}"></span>잡담 / 미분류 '
            f'<span class="fn">{len(noise_ids)}</span></label></div>'
        )

    timeline = "".join(msg_row(mid, show_thread=True) for mid in sorted(by_msg))

    legend = "".join(
        f'<span class="leg"><span class="swatch" style="background:{colors[c]}"></span>{_esc(c)}</span>'
        for c in categories
    )
    runinfo = _runinfo_html(meta)
    filter_html = "".join(filter_groups)

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>대화 스레드 분류 리포트</title>
<style>
:root {{ --bg:#f8fafc; --fg:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#fff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Malgun Gothic",sans-serif;
       background:var(--bg); color:var(--fg); line-height:1.5; }}
header {{ position:sticky; top:0; z-index:10; background:#fff; border-bottom:1px solid var(--line);
          padding:16px 24px; box-shadow:0 1px 3px rgba(0,0,0,.04); max-height:100vh; overflow-y:auto; }}
header h1 {{ margin:0 0 8px; font-size:18px; }}
.stats {{ display:flex; gap:20px; flex-wrap:wrap; align-items:baseline; }}
.stat b {{ font-size:20px; }} .stat span {{ color:var(--muted); font-size:12px; margin-left:4px; }}
.controls {{ margin-top:12px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
.sep {{ width:1px; height:20px; background:var(--line); margin:0 4px; }}
.controls button {{ border:1px solid var(--line); background:#fff; padding:6px 14px; border-radius:8px;
                    cursor:pointer; font-size:13px; color:var(--fg); }}
.controls button:hover {{ background:#f1f5f9; }}
.controls button.active {{ background:var(--fg); color:#fff; border-color:var(--fg); }}
.legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-left:auto; }}
.leg {{ font-size:12px; color:var(--muted); display:inline-flex; align-items:center; gap:5px; }}
.swatch {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
#runinfo, #filter {{ display:none; margin-top:12px; padding:12px 14px; background:#f8fafc;
                     border:1px solid var(--line); border-radius:10px; }}
#runinfo.open, #filter.open {{ display:block; }}
.ri-rows {{ display:grid; grid-template-columns:auto 1fr; gap:2px 16px; font-size:12.5px; }}
.ri-k {{ color:var(--muted); }} .ri-v {{ color:var(--fg); }}
.mono {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px; word-break:break-all; }}
.ri-table {{ margin-top:10px; border-collapse:collapse; font-size:12px; width:100%; max-width:520px; }}
.ri-table th, .ri-table td {{ padding:4px 10px; border-bottom:1px solid var(--line); text-align:left; }}
.ri-table th {{ color:var(--muted); font-weight:600; }}
.ri-table .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
#filter .fbtns {{ margin-bottom:8px; display:flex; gap:8px; }}
.fgroup {{ margin:6px 0; }}
.flabel {{ font-size:12px; font-weight:700; color:var(--cat); margin-bottom:3px; }}
.fitem {{ display:inline-flex; align-items:center; gap:5px; font-size:12px; margin:2px 12px 2px 0;
          cursor:pointer; user-select:none; }}
.fdot {{ width:9px; height:9px; border-radius:2px; display:inline-block; }}
.fn {{ color:var(--muted); }}
main {{ max-width:1000px; margin:0 auto; padding:24px; }}
.cat {{ margin-bottom:28px; }}
.cat.hidden {{ display:none; }}
.cat h2 {{ font-size:16px; display:flex; align-items:center; gap:8px; margin:0 0 12px;
           padding-bottom:8px; border-bottom:2px solid var(--cat); }}
.cat .dot {{ width:12px; height:12px; border-radius:50%; background:var(--cat); }}
.cat-meta {{ font-size:12px; color:var(--muted); font-weight:400; margin-left:auto; }}
.cards {{ display:grid; gap:12px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
.card.hidden {{ display:none; }}
.card-head {{ display:flex; align-items:center; gap:10px; padding:10px 14px; background:#f1f5f9;
              border-bottom:1px solid var(--line); cursor:pointer; }}
.card-head:hover {{ background:#e9eef5; }}
.tid {{ font-weight:700; color:var(--cat,#334155); }}
.topic {{ font-weight:600; }}
.count {{ margin-left:auto; font-size:12px; color:var(--muted); }}
.chev {{ color:#94a3b8; transition:transform .15s; }}
.card.collapsed .chev {{ transform:rotate(-90deg); }}
.card.collapsed .card-body {{ display:none; }}
.card.collapsed .summary {{ border-bottom:none; }}
.summary {{ padding:8px 14px; font-size:13px; color:#475569; background:#fafbfc;
            border-bottom:1px solid var(--line); font-style:italic; }}
.summary::before {{ content:"💬 "; font-style:normal; }}
.card-body {{ padding:4px 0; }}
.msg {{ display:flex; align-items:baseline; gap:10px; padding:5px 14px; border-left:4px solid transparent;
        font-size:13.5px; }}
.msg:hover {{ background:#f8fafc; }}
.mid {{ color:#94a3b8; font-variant-numeric:tabular-nums; font-size:11.5px; min-width:44px; }}
.time {{ color:var(--muted); font-size:11.5px; font-variant-numeric:tabular-nums; min-width:76px; }}
.sender {{ font-weight:600; color:#334155; white-space:nowrap; }}
.text {{ flex:1; }}
.tbadge {{ color:#fff; font-size:11px; padding:1px 7px; border-radius:20px; white-space:nowrap; }}
#timeline {{ display:none; }}
#timeline.on {{ display:block; }}
#threads.off {{ display:none; }}
#timeline .msg {{ background:#fff; border-bottom:1px solid #f1f5f9; }}
</style></head>
<body>
<header>
  <h1>🧵 대화 스레드 분류 리포트</h1>
  <div class="stats">
    <div class="stat"><b>{n_msgs:,}</b><span>발화</span></div>
    <div class="stat"><b>{n_threads}</b><span>스레드</span></div>
    <div class="stat"><b>{n_cats}</b><span>카테고리</span></div>
    <div class="stat"><b>{n_assigned:,}</b><span>배정</span></div>
    <div class="stat"><b>{len(noise_ids):,}</b><span>잡담</span></div>
  </div>
  <div class="controls">
    <button id="btn-threads" class="active" onclick="showView('threads')">스레드별</button>
    <button id="btn-timeline" onclick="showView('timeline')">시간순</button>
    <div class="sep"></div>
    <button onclick="setAllCards(false)">전체 열기</button>
    <button onclick="setAllCards(true)">전체 닫기</button>
    <div class="sep"></div>
    <button id="btn-filter" onclick="togglePanel('filter', this)">필터 ▾</button>
    <button id="btn-runinfo" onclick="togglePanel('runinfo', this)">실행 정보 ▾</button>
    <div class="legend">{legend}</div>
  </div>
  <div id="runinfo">{runinfo}</div>
  <div id="filter">
    <div class="fbtns">
      <button onclick="setAllThreads(true)">전체 선택</button>
      <button onclick="setAllThreads(false)">전체 해제</button>
    </div>
    {filter_html}
  </div>
</header>
<main>
  <div id="threads">{''.join(thread_view_parts)}</div>
  <div id="timeline">{timeline}</div>
</main>
<script>
function showView(v) {{
  document.getElementById('threads').className = v==='threads' ? '' : 'off';
  document.getElementById('timeline').className = v==='timeline' ? 'on' : '';
  document.getElementById('btn-threads').className = v==='threads' ? 'active' : '';
  document.getElementById('btn-timeline').className = v==='timeline' ? 'active' : '';
}}
function togglePanel(id, btn) {{
  var open = document.getElementById(id).classList.toggle('open');
  if (btn) btn.classList.toggle('active', open);
}}
function toggleCard(head) {{ head.parentElement.classList.toggle('collapsed'); }}
function setAllCards(collapsed) {{
  document.querySelectorAll('#threads .card').forEach(function(c) {{
    c.classList.toggle('collapsed', collapsed);
  }});
}}
function setAllThreads(checked) {{
  document.querySelectorAll('#filter input[type=checkbox]').forEach(function(cb) {{ cb.checked = checked; }});
  applyFilter();
}}
function applyFilter() {{
  document.querySelectorAll('#filter input[type=checkbox]').forEach(function(cb) {{
    var card = document.querySelector('#threads .card[data-tid="' + cb.dataset.tid + '"]');
    if (card) card.classList.toggle('hidden', !cb.checked);
  }});
  document.querySelectorAll('#threads .cat').forEach(function(sec) {{
    var visible = sec.querySelectorAll('.card:not(.hidden)').length;
    sec.classList.toggle('hidden', visible === 0);
  }});
}}
</script>
</body></html>"""


def write_report(
    messages: list[Message],
    threads: list[dict],
    out_path: str | Path,
    meta: dict | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(messages, threads, meta), encoding="utf-8")
    logger.info("HTML 리포트 저장: %s", out_path)
    return out_path


def _load_threads(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys

    from .logging_setup import setup_logging

    setup_logging()
    if len(sys.argv) < 3:
        print("usage: python -m kakaothread.report <threads.json> <messages.jsonl | raw_kakao.txt> [out.html]")
        raise SystemExit(1)

    threads_path, msgs_path = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else str(Path(threads_path).with_name("report.html"))

    if msgs_path.endswith(".jsonl"):
        messages = load_messages(msgs_path)
    else:
        messages = merge_consecutive(parse_kakao(msgs_path))
    threads = _load_threads(threads_path)
    meta = {"source": msgs_path, "generated": "(재생성)"}
    write_report(messages, threads, out, meta)
    print(f"리포트 생성 완료 -> {out}")
