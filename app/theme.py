"""Visual system for the console.

Direction: a register tape. Receipts are printed documents whose whole design
problem is making columns of figures scan quickly under bad light, and that is
also this page's job. So the rules are borrowed from print, not from dashboard
convention:

  - Figures are the content. Every number is set in tabular monospace and
    right-aligned so digits stack in columns and magnitudes are comparable at a
    glance. Proportional numerals in a data table are a readability bug.
  - Hairlines, not boxes. Structure comes from 1px rules and whitespace rather
    than cards-within-cards.
  - Three signal colours, and they mean something specific: teal = gate passed,
    amber = watch, brick = blocked. Colour is never decorative here.

Fonts are self-hosted (base64, latin subset, ~108KB total) rather than pulled
from a CDN: no external dependency, no flash of unstyled text, and the app
works offline.
"""

from __future__ import annotations

import base64
from functools import cache
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"

# --- tokens ---------------------------------------------------------------
BG = "#0E1418"
SURFACE = "#151D23"
SURFACE_2 = "#1C262D"
RULE = "#27333B"
RULE_SOFT = "#1F2A31"
INK = "#E6EDF2"
INK_MUTE = "#8695A2"
INK_FAINT = "#5C6B77"

OK = "#4FA88B"
WATCH = "#D2A03C"
ALERT = "#C24B3E"

DISPLAY = "'Bricolage Grotesque'"
BODY = "'IBM Plex Sans'"
MONO = "'IBM Plex Mono'"


@cache
def _face(filename: str) -> str:
    data = (FONT_DIR / filename).read_bytes()
    return base64.b64encode(data).decode()


def _font_face(family: str, weight: int, filename: str) -> str:
    return (
        f"@font-face{{font-family:'{family}';font-style:normal;font-weight:{weight};"
        f"font-display:block;src:url(data:font/woff2;base64,{_face(filename)}) "
        "format('woff2');}"
    )


def css() -> str:
    faces = "".join(
        [
            _font_face("Bricolage Grotesque", 700, "bricolage-grotesque-latin-700-normal.woff2"),
            _font_face("IBM Plex Sans", 400, "ibm-plex-sans-latin-400-normal.woff2"),
            _font_face("IBM Plex Sans", 600, "ibm-plex-sans-latin-600-normal.woff2"),
            _font_face("IBM Plex Mono", 400, "ibm-plex-mono-latin-400-normal.woff2"),
            _font_face("IBM Plex Mono", 500, "ibm-plex-mono-latin-500-normal.woff2"),
        ]
    )
    return f"""<style>
{faces}

:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface-2:{SURFACE_2};
  --rule:{RULE}; --rule-soft:{RULE_SOFT};
  --ink:{INK}; --ink-mute:{INK_MUTE}; --ink-faint:{INK_FAINT};
  --ok:{OK}; --watch:{WATCH}; --alert:{ALERT};
}}

/* Strip Streamlit's own chrome so the page reads as a product, not a notebook */
#MainMenu, footer, header[data-testid="stHeader"] {{ display:none !important; }}
[data-testid="stDecoration"] {{ display:none !important; }}
[data-testid="stToolbar"] {{ display:none !important; }}

.stApp {{ background:var(--bg); }}
.block-container {{ padding-top:2.2rem; padding-bottom:4rem; max-width:1360px; }}

/* Scoped deliberately: a blanket `span, div` rule applies *directly* to those
   elements and so overrides any family inherited from an ancestor, no matter
   the specificity. That silently kills the monospace figures below. */
html, body, .stApp, .stMarkdown p, .stMarkdown li, label {{
  font-family:{BODY}, system-ui, sans-serif;
  color:var(--ink);
  -webkit-font-smoothing:antialiased;
}}

/* Figures: tabular monospace everywhere a number appears */
.mono, [data-testid="stDataFrame"] {{ font-family:{MONO}, ui-monospace, monospace;
  font-variant-numeric:tabular-nums; }}

.stApp h1, .stApp h2, .stApp h3 {{ font-family:{DISPLAY}, {BODY}, sans-serif !important;
  font-weight:700; letter-spacing:-0.02em; color:var(--ink); }}

/* ---- masthead ---- */
.rpg-head {{ display:flex; justify-content:space-between; align-items:flex-end;
  gap:2.5rem; border-bottom:1px solid var(--rule); padding-bottom:1.4rem;
  margin-bottom:.4rem; }}
.rpg-title {{ font-family:{DISPLAY}, sans-serif !important; font-weight:700; font-size:2.55rem;
  line-height:1.02; letter-spacing:-0.035em; margin:0; }}
.rpg-sub {{ font-family:{BODY}, sans-serif; color:var(--ink-mute); font-size:.9rem; max-width:52ch; margin-top:.55rem;
  line-height:1.5; }}

/* ---- the tape: a register total block, hairline rules, figures right-aligned ---- */
.tape {{ background:var(--surface); border:1px solid var(--rule); border-radius:3px;
  padding:.85rem 1rem .95rem; min-width:270px; position:relative; }}
.tape::before {{ content:""; position:absolute; left:0; right:0; top:-1px; height:3px;
  background:repeating-linear-gradient(90deg,var(--rule) 0 4px,transparent 4px 8px); }}
.tape-h {{ font-family:{MONO}, monospace; font-size:.62rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--ink-faint); margin-bottom:.6rem; }}
.tape-row {{ display:flex; justify-content:space-between; align-items:baseline;
  font-family:{MONO}, monospace; font-size:.78rem; padding:.2rem 0;
  border-bottom:1px dotted var(--rule-soft); }}
.tape-row:last-child {{ border-bottom:none; }}
.tape-row .k {{ font-family:{MONO}, monospace; color:var(--ink-mute); }}
.tape-row .v {{ font-family:{MONO}, monospace; font-variant-numeric:tabular-nums;
  font-weight:500; }}
.tape-total {{ border-top:1px solid var(--rule) !important; margin-top:.45rem;
  padding-top:.5rem; font-size:.85rem; }}

/* ---- stat row ---- */
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px; background:var(--rule); border:1px solid var(--rule);
  border-radius:3px; overflow:hidden; margin:1.5rem 0 .4rem; }}
.stat {{ background:var(--surface); padding:.9rem 1rem 1rem; }}
.stat-k {{ font-family:{MONO}, monospace; font-size:.6rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--ink-faint); }}
.stat-v {{ font-family:{MONO}, monospace; font-size:1.55rem; font-weight:500;
  font-variant-numeric:tabular-nums; margin-top:.35rem; line-height:1; }}
.stat-n {{ font-family:{BODY}, sans-serif; font-size:.7rem; color:var(--ink-mute);
  margin-top:.4rem; }}
.v-ok {{ color:var(--ok); }} .v-watch {{ color:var(--watch); }} .v-alert {{ color:var(--alert); }}

/* ---- pills ---- */
.pill {{ display:inline-flex; align-items:center; gap:.4rem; font-family:{MONO}, monospace;
  font-size:.66rem; letter-spacing:.1em; text-transform:uppercase;
  padding:.24rem .6rem; border-radius:2px; border:1px solid currentColor; }}
.pill.ok {{ color:var(--ok); }} .pill.watch {{ color:var(--watch); }}
.pill.alert {{ color:var(--alert); }}

/* ---- section headers: eyebrow encodes the layer, not decoration ---- */
.sec {{ margin:2.4rem 0 .9rem; padding-bottom:.5rem; border-bottom:1px solid var(--rule); }}
.sec-eyebrow {{ font-family:{MONO}, monospace; font-size:.6rem; letter-spacing:.2em;
  text-transform:uppercase; color:var(--ink-faint); }}
.sec-t {{ font-family:{DISPLAY}, sans-serif !important; font-weight:700; font-size:1.3rem;
  letter-spacing:-0.02em; margin-top:.25rem; }}
.note {{ font-family:{BODY}, sans-serif; color:var(--ink-mute); font-size:.82rem; line-height:1.6; margin-top:.7rem;
  max-width:78ch; border-left:2px solid var(--rule); padding-left:.85rem; }}
.note code {{ font-family:{MONO}, monospace; font-size:.78rem; color:var(--watch);
  background:none; }}

/* ---- tabs ---- */
[data-baseweb="tab-list"] {{ gap:0 !important; border-bottom:1px solid var(--rule);
  background:transparent; }}
[data-baseweb="tab"] {{ font-family:{MONO}, monospace !important; font-size:.72rem !important;
  letter-spacing:.11em; text-transform:uppercase; color:var(--ink-faint) !important;
  padding:.7rem 1.15rem !important; background:transparent !important; }}
[data-baseweb="tab"][aria-selected="true"] {{ color:var(--ink) !important;
  border-bottom:2px solid var(--watch) !important; }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ display:none !important; }}

/* ---- controls ---- */
section[data-testid="stSidebar"] {{ background:var(--surface); border-right:1px solid var(--rule); }}
section[data-testid="stSidebar"] .block-container {{ padding-top:2rem; }}
.stButton button {{ font-family:{MONO}, monospace; font-size:.72rem; letter-spacing:.1em;
  text-transform:uppercase; border-radius:2px; border:1px solid var(--rule);
  background:var(--surface-2); color:var(--ink); }}
.stButton button:hover {{ border-color:var(--watch); color:var(--watch); }}

[data-testid="stDataFrame"] {{ border:1px solid var(--rule); border-radius:3px; }}
[data-testid="stDataFrame"] * {{ font-family:{MONO}, monospace !important;
  font-variant-numeric:tabular-nums; font-size:.76rem !important; }}

@media (prefers-reduced-motion:reduce) {{
  *, *::before, *::after {{ animation:none !important; transition:none !important; }}
}}
</style>"""


# --- html fragments -------------------------------------------------------
def masthead(title: str, sub: str, tape_rows: list[tuple[str, str]], total: tuple[str, str]) -> str:
    rows = "".join(
        f'<div class="tape-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
        for k, v in tape_rows
    )
    return f"""<div class="rpg-head">
  <div><h1 class="rpg-title">{title}</h1><div class="rpg-sub">{sub}</div></div>
  <div class="tape"><div class="tape-h">Run summary</div>{rows}
    <div class="tape-row tape-total"><span class="k">{total[0]}</span>
    <span class="v">{total[1]}</span></div></div>
</div>"""


def stats(items: list[tuple[str, str, str, str]]) -> str:
    """items: (label, value, tone, note) where tone in {'', 'ok', 'watch', 'alert'}"""
    cells = "".join(
        f'<div class="stat"><div class="stat-k">{k}</div>'
        f'<div class="stat-v {"v-" + tone if tone else ""}">{v}</div>'
        f'<div class="stat-n">{note}</div></div>'
        for k, v, tone, note in items
    )
    return f'<div class="stats">{cells}</div>'


def section(eyebrow: str, title: str) -> str:
    return (
        f'<div class="sec"><div class="sec-eyebrow">{eyebrow}</div>'
        f'<div class="sec-t">{title}</div></div>'
    )


def note(text: str) -> str:
    return f'<div class="note">{text}</div>'


def pill(text: str, tone: str) -> str:
    return f'<span class="pill {tone}">{text}</span>'


# --- charts ---------------------------------------------------------------
def chart_theme() -> dict:
    """Altair config matching the tokens. Streamlit's default charts are an
    instant giveaway; these are set deliberately instead."""
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "font": "IBM Plex Sans",
            "axis": {
                "domainColor": RULE,
                "gridColor": RULE_SOFT,
                "tickColor": RULE,
                "labelColor": INK_MUTE,
                "titleColor": INK_FAINT,
                "labelFont": "IBM Plex Mono",
                "labelFontSize": 10,
                "titleFont": "IBM Plex Mono",
                "titleFontSize": 10,
                "titleFontWeight": "normal",
                "gridDash": [2, 3],
            },
            "legend": {
                "labelColor": INK_MUTE,
                "titleColor": INK_FAINT,
                "labelFont": "IBM Plex Mono",
                "labelFontSize": 10,
                "titleFont": "IBM Plex Mono",
                "titleFontSize": 9,
                "orient": "top",
                "direction": "horizontal",
                "symbolType": "stroke",
                "symbolStrokeWidth": 3,
            },
            "range": {"category": [OK, WATCH, ALERT, INK_MUTE]},
        }
    }
