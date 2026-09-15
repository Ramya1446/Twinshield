"""
Global styling: an ivory / navy / antique-gold console.

Design intent -- a light, "royal" palette rather than a dark cyber one:

    * ground   : warm ivory parchment, lit with faint gold and sapphire washes
    * ink      : deep navy for text, so the page reads like fine print
    * accent   : antique gold for anything the eye should land on first
    * support  : royal sapphire and amethyst for secondary data series

Every colour here is chosen to stay legible on a light ground -- the status
tones are deepened versions of emerald / amber / ruby so they still carry
meaning at small sizes without glowing.  Legacy names (CYAN, BLUE, GREEN, RED,
VIOLET) remain as aliases so all pages and charts inherit the palette.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
GOLD = "#A9822F"          # antique gold -- primary accent, readable on ivory
GOLD_DEEP = "#7E5F1E"
GOLD_BRIGHT = "#D9BC63"   # for fills and gradients only
GOLD_SOFT = "rgba(169,130,47,0.10)"
SAPPHIRE = "#2A4BA8"      # royal blue -- secondary accent
AMETHYST = "#6B4FBB"      # tertiary series
EMERALD = "#12775A"       # normal
AMBER = "#B07A0C"         # suspicious
RUBY = "#BE2F3C"          # attack

BG = "#F6F3EC"            # warm ivory ground
BG_RAISED = "#FFFDF8"
PANEL = "#FFFFFF"
INK = "#131C33"           # deep navy -- primary text
TEXT = INK
TEXT_DIM = "#2E3A54"
MUTED = "#65718B"
BORDER = "rgba(19,28,51,0.12)"
BORDER_GOLD = "rgba(169,130,47,0.34)"

# Backwards-compatible aliases used across the pages.
CYAN = GOLD
BLUE = SAPPHIRE
VIOLET = AMETHYST
GREEN = EMERALD
RED = RUBY

CLASS_COLORS = {
    "Normal": EMERALD,
    "DoS": RUBY,
    "Brake Tampering": GOLD,
    "Steering Tampering": AMETHYST,
    # Not a dataset class: what the anomaly gate can honestly say about traffic
    # that deviates from learnt-normal but carries no label the model was taught.
    "Unknown Anomaly": SAPPHIRE,
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_DIM, family="Inter, Segoe UI, sans-serif", size=12.5),
    margin=dict(l=8, r=8, t=44, b=8),
    xaxis=dict(gridcolor="rgba(19,28,51,0.075)",
               zerolinecolor="rgba(19,28,51,0.16)",
               linecolor="rgba(19,28,51,0.18)",
               tickfont=dict(color=MUTED, size=11.5)),
    yaxis=dict(gridcolor="rgba(19,28,51,0.075)",
               zerolinecolor="rgba(19,28,51,0.16)",
               linecolor="rgba(19,28,51,0.18)",
               tickfont=dict(color=MUTED, size=11.5)),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(color=MUTED, size=11.5)),
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=BORDER_GOLD,
                    font=dict(color=INK, size=12)),
    colorway=[GOLD, SAPPHIRE, AMETHYST, EMERALD, AMBER, RUBY],
)


def style_fig(fig, height: int = 300, title: str | None = None):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    if title:
        fig.update_layout(title=dict(
            text=title, font=dict(size=14, color=TEXT_DIM, family="Sora, Inter, sans-serif"),
            x=0, xanchor="left"))
    return fig


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ============================ ground ============================ */
.stApp {{
    background:
        radial-gradient(1150px 620px at 6% -8%,  rgba(169,130,47,0.11), transparent 60%),
        radial-gradient(950px 520px at 97% -4%,  rgba(42,75,168,0.075), transparent 58%),
        radial-gradient(780px 460px at 52% 105%, rgba(107,79,187,0.05), transparent 62%),
        {BG};
    color: {INK};
    font-family: 'Inter', 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}}
.block-container {{ padding-top: 1.9rem; padding-bottom: 4rem; max-width: 1340px; }}
#MainMenu, footer, .stDeployButton {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}

h1, h2, h3, h4 {{
    font-family: 'Sora', 'Inter', sans-serif;
    color: {INK}; font-weight: 600; letter-spacing: -0.022em;
}}
h2 {{ font-size: 1.72rem; }}
h3 {{ font-size: 1.16rem; }}
/* Scoped to markdown only -- a bare `span` rule would override button labels. */
.stMarkdown p, .stMarkdown li, .stMarkdown span {{ color: {TEXT_DIM}; }}
hr {{ border-color: {BORDER}; }}
a {{ color: {GOLD_DEEP}; text-decoration: none; font-weight: 500; }}
a:hover {{ color: {GOLD}; text-decoration: underline; }}
code {{ color: {GOLD_DEEP}; background: rgba(169,130,47,0.11); padding: .08rem .34rem;
        border-radius: 5px; font-size: .86em; }}

/* ============================ brand + nav ============================ */
.ts-brand {{
    font-family: 'Sora', sans-serif; font-size: 1.06rem; color: {INK};
    white-space: nowrap; letter-spacing: -0.015em;
}}
.ts-brand b {{ font-weight: 700; }}
.ts-brand .mark {{
    background: linear-gradient(96deg, {INK} 6%, {SAPPHIRE} 42%, {GOLD_DEEP} 92%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.ts-navrule {{
    height: 1px; margin: .7rem 0 1.75rem 0;
    background: linear-gradient(90deg, {BORDER_GOLD}, rgba(19,28,51,.10) 42%, transparent 78%);
}}
div[class*="st-key-nav_"] button {{
    background: transparent !important; border: 1px solid transparent !important;
    font-weight: 500 !important; font-size: .88rem !important;
    padding: .46rem .6rem !important; box-shadow: none !important;
    transition: color .18s ease, background .18s ease, border-color .18s ease;
}}
div[class*="st-key-nav_"] button, div[class*="st-key-nav_"] button * {{
    color: {MUTED} !important;
}}
div[class*="st-key-nav_"] button:hover {{ background: rgba(19,28,51,0.045) !important; }}
div[class*="st-key-nav_"] button:hover * {{ color: {INK} !important; }}
div[class*="st-key-nav_"] button[kind="primary"] {{
    background: {GOLD_SOFT} !important; border-color: {BORDER_GOLD} !important;
}}
div[class*="st-key-nav_"] button[kind="primary"],
div[class*="st-key-nav_"] button[kind="primary"] * {{
    color: {GOLD_DEEP} !important; font-weight: 700 !important;
}}

/* ============================ cards ============================ */
.ts-card {{
    background: linear-gradient(180deg, #FFFFFF, #FDFBF6);
    border: 1px solid {BORDER};
    border-radius: 18px; padding: 1.25rem 1.4rem; height: 100%;
    box-shadow: 0 1px 2px rgba(19,28,51,0.04), 0 10px 30px rgba(19,28,51,0.055);
    transition: border-color .22s ease, box-shadow .22s ease;
}}
.ts-card:hover {{
    border-color: {BORDER_GOLD};
    box-shadow: 0 1px 2px rgba(19,28,51,0.05), 0 14px 36px rgba(169,130,47,0.10);
}}
.ts-card h4 {{
    margin: 0 0 .4rem 0; font-size: 1.0rem; font-family: 'Sora', sans-serif;
    font-weight: 600; color: {INK};
}}
.ts-card p {{ margin: 0; color: {MUTED}; font-size: .885rem; line-height: 1.62; }}
.ts-card-icon {{ font-size: 1.42rem; line-height: 1; margin-bottom: .55rem; display: block; }}

.ts-kv {{ display: flex; justify-content: space-between; gap: 1rem; padding: .48rem 0;
          border-bottom: 1px solid rgba(19,28,51,0.075); font-size: .875rem; }}
.ts-kv:last-child {{ border-bottom: none; }}
.ts-kv span:first-child {{ color: {MUTED}; }}
.ts-kv span:last-child {{ color: {INK}; font-weight: 600; text-align: right; }}

/* ============================ hero ============================ */
.ts-hero {{ padding: 1.3rem 0 .5rem 0; }}
.ts-hero h1 {{
    font-family: 'Sora', sans-serif; font-size: 3.4rem; font-weight: 700;
    margin: 0; line-height: 1.04; letter-spacing: -0.035em;
    background: linear-gradient(97deg, {INK} 2%, #23407F 34%, {SAPPHIRE} 56%, {GOLD_DEEP} 96%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.ts-hero .sub {{
    font-size: 1.14rem; color: {INK}; margin-top: .7rem; font-weight: 500;
    font-family: 'Sora', sans-serif; letter-spacing: -0.012em;
}}
.ts-hero .desc {{
    font-size: .97rem; color: {MUTED}; margin-top: .6rem; max-width: 720px; line-height: 1.72;
}}
.ts-eyebrow {{
    display: inline-block; font-size: .68rem; letter-spacing: .2em; text-transform: uppercase;
    color: {GOLD_DEEP}; border: 1px solid {BORDER_GOLD}; background: {GOLD_SOFT};
    padding: .34rem .82rem; border-radius: 999px; font-weight: 700; margin-bottom: 1.05rem;
}}

/* ============================ pipeline ============================ */
.ts-pipe {{ display: flex; align-items: stretch; gap: .4rem; flex-wrap: nowrap;
            overflow-x: auto; padding: .35rem 0 .7rem 0; }}
.ts-stage {{
    flex: 1 1 0; min-width: 122px; text-align: center; border-radius: 15px;
    padding: .95rem .55rem; background: #FFFFFF;
    border: 1px solid {BORDER}; transition: all .26s ease;
    box-shadow: 0 1px 2px rgba(19,28,51,0.035);
}}
.ts-stage .ico {{ font-size: 1.38rem; display: block; margin-bottom: .38rem; }}
.ts-stage .nm {{
    font-size: .795rem; font-weight: 600; color: {INK}; letter-spacing: .015em;
    font-family: 'Sora', sans-serif; display: block;
}}
.ts-stage .ds {{ font-size: .705rem; color: {MUTED}; margin-top: .3rem; line-height: 1.42;
                 display: block; }}
.ts-stage.on {{
    background: linear-gradient(180deg, rgba(169,130,47,0.15), rgba(169,130,47,0.05));
    border-color: {BORDER_GOLD};
    box-shadow: 0 8px 24px rgba(169,130,47,0.16); transform: translateY(-2px);
}}
.ts-stage.on .nm {{ color: {GOLD_DEEP}; }}
.ts-stage.done {{ background: linear-gradient(180deg, rgba(42,75,168,0.055), #FFFFFF);
                  border-color: rgba(42,75,168,0.20); }}
.ts-arrow {{ display: flex; align-items: center; color: rgba(101,113,139,0.65);
             font-size: .95rem; flex: 0 0 auto; }}

/* ============================ status banner ============================ */
.ts-status {{
    border-radius: 18px; padding: 1.35rem 1.55rem; border: 1px solid;
    display: flex; align-items: center; gap: 1.2rem;
    box-shadow: 0 10px 30px rgba(19,28,51,0.06);
}}
.ts-status .big {{
    font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;
    font-family: 'Sora', sans-serif;
}}
.ts-status .meta {{ font-size: .855rem; color: {MUTED}; margin-top: .24rem; }}
.ts-ok    {{ background: linear-gradient(180deg, rgba(18,119,90,0.10), #FFFFFF);
             border-color: rgba(18,119,90,0.32); }}
.ts-ok .big    {{ color: {EMERALD}; }}
.ts-warn  {{ background: linear-gradient(180deg, rgba(176,122,12,0.11), #FFFFFF);
             border-color: rgba(176,122,12,0.36); }}
.ts-warn .big  {{ color: {AMBER}; }}
.ts-alert {{ background: linear-gradient(180deg, rgba(190,47,60,0.11), #FFFFFF);
             border-color: rgba(190,47,60,0.38); }}
.ts-alert .big {{ color: {RUBY}; }}
.ts-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: .5rem; }}
@keyframes tspulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: .3; }} }}
.ts-live {{ animation: tspulse 2.1s ease-in-out infinite; }}

/* ============================ CAN frame table ============================ */
.ts-frames {{
    font-family: 'JetBrains Mono', monospace; font-size: .782rem;
    border: 1px solid {BORDER}; border-radius: 14px; overflow: hidden;
    background: #FFFFFF; box-shadow: 0 1px 2px rgba(19,28,51,0.035);
}}
.ts-frame-head {{
    display: grid; grid-template-columns: 96px 104px 44px 1fr 140px; gap: .55rem;
    padding: .6rem .85rem; color: {MUTED}; font-size: .655rem; letter-spacing: .13em;
    text-transform: uppercase; font-weight: 700;
    background: #FAF7F0; border-bottom: 1px solid {BORDER};
    font-family: 'Inter', sans-serif;
}}
.ts-frame {{
    display: grid; grid-template-columns: 96px 104px 44px 1fr 140px; gap: .55rem;
    padding: .46rem .85rem; border-bottom: 1px solid rgba(19,28,51,0.06);
    align-items: center; transition: background .14s ease; color: {TEXT_DIM};
}}
.ts-frame:last-child {{ border-bottom: none; }}
.ts-frame:hover {{ background: rgba(169,130,47,0.06); }}
.ts-frame .pl {{ color: {MUTED}; letter-spacing: .055em; }}
.ts-badge {{
    padding: .18rem .52rem; border-radius: 6px; font-size: .655rem; font-weight: 700;
    letter-spacing: .05em; white-space: nowrap; font-family: 'Inter', sans-serif;
}}

/* ============================ metric tiles ============================ */
.ts-metric {{
    background: linear-gradient(180deg, #FFFFFF, #FDFBF6);
    border: 1px solid {BORDER}; border-radius: 16px; padding: 1.05rem 1.2rem; height: 100%;
    box-shadow: 0 1px 2px rgba(19,28,51,0.04), 0 8px 24px rgba(19,28,51,0.05);
}}
.ts-metric .lbl {{
    font-size: .665rem; color: {MUTED}; text-transform: uppercase;
    letter-spacing: .135em; font-weight: 700;
}}
.ts-metric .val {{
    font-size: 1.92rem; font-weight: 700; margin-top: .34rem; line-height: 1;
    font-family: 'Sora', sans-serif; letter-spacing: -0.028em; color: {INK};
}}
.ts-metric .sub {{ font-size: .745rem; color: {MUTED}; margin-top: .38rem; line-height: 1.45; }}
.ts-metric.hi {{
    border-color: {BORDER_GOLD};
    background: linear-gradient(180deg, rgba(169,130,47,0.12), #FFFDF6);
    box-shadow: 0 8px 26px rgba(169,130,47,0.14);
}}
.ts-metric.hi .val {{ color: {GOLD_DEEP}; }}
.ts-metric.hi .lbl {{ color: {GOLD_DEEP}; }}

/* ============================ buttons ============================ */
/* Streamlit nests the label in an inner <p>/<div>, so colours must be set on
   the descendants too or the surrounding text rules win and the label vanishes. */
.stButton > button {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 11px;
    font-weight: 600; padding: .56rem 1.1rem; transition: all .18s ease;
    font-family: 'Inter', sans-serif;
    box-shadow: 0 1px 2px rgba(19,28,51,0.04);
}}
.stButton > button, .stButton > button * {{ color: {TEXT_DIM} !important; }}
.stButton > button:hover {{
    background: #FFFDF6; border-color: {BORDER_GOLD};
    box-shadow: 0 4px 14px rgba(169,130,47,0.14);
}}
.stButton > button:hover * {{ color: {GOLD_DEEP} !important; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(96deg, {GOLD_BRIGHT}, #C0A24E 52%, {GOLD});
    border: 1px solid rgba(126,95,30,0.45); font-weight: 800; letter-spacing: .015em;
    box-shadow: 0 6px 20px rgba(169,130,47,0.30);
}}
.stButton > button[kind="primary"], .stButton > button[kind="primary"] * {{
    color: {INK} !important;
}}
.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.05); box-shadow: 0 8px 26px rgba(169,130,47,0.40);
}}
.stButton > button[kind="primary"]:hover * {{ color: {INK} !important; }}
.stDownloadButton > button {{
    background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 10px; font-size: .84rem;
}}
.stDownloadButton > button, .stDownloadButton > button * {{ color: {MUTED} !important; }}
.stDownloadButton > button:hover {{ border-color: {BORDER_GOLD}; }}
.stDownloadButton > button:hover * {{ color: {GOLD_DEEP} !important; }}

/* ============================ inputs / containers ============================ */
div[data-testid="stExpander"] {{
    border: 1px solid {BORDER}; border-radius: 15px; background: #FFFFFF;
    box-shadow: 0 1px 2px rgba(19,28,51,0.035);
}}
div[data-testid="stExpander"] summary {{ font-weight: 600; color: {TEXT_DIM}; font-size: .9rem; }}
div[data-testid="stExpander"] summary:hover {{ color: {GOLD_DEEP}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .35rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: {GOLD_DEEP} !important; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 13px; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
    background: #FFFFFF !important; border-color: {BORDER} !important;
    border-radius: 11px !important; transition: border-color .18s ease;
}}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"] > div:hover {{
    border-color: {BORDER_GOLD} !important;
}}
div[data-baseweb="popover"] li:hover {{ background: {GOLD_SOFT} !important; }}
div[role="radiogroup"] {{ gap: 1.15rem; }}
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, .stRadio label {{
    color: {MUTED} !important; font-size: .8rem !important; font-weight: 600 !important;
    letter-spacing: .04em;
}}

/* ============================ notes ============================ */
.ts-note {{
    font-size: .845rem; color: {MUTED}; border-left: 2px solid {BORDER_GOLD};
    padding: .22rem 0 .22rem .9rem; margin: .55rem 0; line-height: 1.68;
}}
.ts-flag {{
    background: linear-gradient(180deg, rgba(176,122,12,0.10), #FFFDF6);
    border: 1px solid rgba(176,122,12,0.32);
    border-radius: 15px; padding: 1rem 1.2rem; font-size: .875rem; line-height: 1.68;
    color: {TEXT_DIM};
}}
.ts-section {{ display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .1rem 0; }}
.ts-section .bar {{
    width: 3px; height: 17px; border-radius: 2px;
    background: linear-gradient(180deg, {GOLD_BRIGHT}, {GOLD_DEEP});
}}

/* ============================ model comparison board ============================ */
.ts-board {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
             gap: .7rem; margin: .2rem 0 .3rem 0; }}
@media (max-width: 1100px) {{ .ts-board {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.ts-model {{
    border: 1px solid {BORDER}; border-radius: 16px; padding: .95rem 1rem 0.85rem 1rem;
    background: linear-gradient(180deg, #FFFFFF, #FDFBF6); position: relative;
    box-shadow: 0 1px 2px rgba(19,28,51,0.04), 0 8px 22px rgba(19,28,51,0.05);
    transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    overflow: hidden;
}}
.ts-model .tag {{
    font-size: .58rem; letter-spacing: .16em; text-transform: uppercase;
    font-weight: 800; color: {MUTED};
}}
.ts-model .nm {{
    font-family: 'Sora', sans-serif; font-size: 1.02rem; font-weight: 700;
    color: {INK}; margin-top: .12rem; letter-spacing: -.02em;
}}
.ts-model .verdict {{
    margin-top: .7rem; font-family: 'Sora', sans-serif; font-weight: 800;
    font-size: .96rem; letter-spacing: -.01em; display: flex; align-items: center;
    gap: .4rem; line-height: 1.25;
}}
.ts-model .meta {{ font-size: .745rem; color: {MUTED}; margin-top: .3rem; line-height: 1.5;
                   min-height: 2.2em; }}
.ts-model .bar {{
    height: 5px; border-radius: 3px; background: rgba(19,28,51,.08);
    margin-top: .55rem; overflow: hidden;
}}
.ts-model .bar i {{ display: block; height: 100%; border-radius: 3px;
                    background: {MUTED}; transition: width .25s ease; }}
.ts-model .foot {{
    margin-top: .6rem; padding-top: .55rem; border-top: 1px solid rgba(19,28,51,.075);
    font-size: .715rem; color: {MUTED}; display: flex; justify-content: space-between;
    gap: .4rem; align-items: baseline;
}}
.ts-model .foot b {{ color: {INK}; font-weight: 700; }}

/* live state: this model is alarming on the frame being processed */
.ts-model.on {{ border-color: rgba(190,47,60,.45);
                background: linear-gradient(180deg, rgba(190,47,60,.09), #FFFFFF);
                box-shadow: 0 10px 28px rgba(190,47,60,.16); transform: translateY(-2px); }}
.ts-model.on .verdict {{ color: {RUBY}; }}
.ts-model.on .bar i {{ background: {RUBY}; }}
.ts-model.quiet .verdict {{ color: {EMERALD}; }}
.ts-model.quiet .bar i {{ background: {EMERALD}; }}

/* verdict against ground truth, once the attack segment has played */
.ts-model .mark {{
    position: absolute; top: .8rem; right: .85rem; font-size: .95rem; font-weight: 800;
}}
.ts-model.hit {{ border-color: rgba(18,119,90,.45); }}
.ts-model.hit .mark {{ color: {EMERALD}; }}
.ts-model.hit .verdict {{ color: {EMERALD}; }}
.ts-model.missed {{ border-color: rgba(190,47,60,.45); }}
.ts-model.missed .mark {{ color: {RUBY}; }}
/* A model that stayed quiet through the attack reads as a miss, not as calm:
   these two rules sit after .quiet so they win on a card that is not alarming. */
.ts-model.missed .verdict {{ color: {RUBY}; }}
.ts-model.missed .bar i {{ background: {RUBY}; }}
.ts-model.missed::after {{
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background: linear-gradient(180deg, rgba(190,47,60,.055), transparent 55%);
}}

/* the proposed model carries a gold rail so it reads apart from the baselines */
.ts-model.proposed {{ border-color: {BORDER_GOLD}; }}
.ts-model.proposed::before {{
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, {GOLD_BRIGHT}, {GOLD_DEEP});
}}
.ts-model.proposed .tag {{ color: {GOLD_DEEP}; }}

/* the gate readout, only on the proposed card */
.ts-gate {{
    margin-top: .55rem; font-size: .7rem; color: {MUTED};
    background: rgba(42,75,168,.055); border: 1px solid rgba(42,75,168,.16);
    border-radius: 9px; padding: .42rem .55rem; line-height: 1.5;
}}
.ts-gate b {{ color: {SAPPHIRE}; }}
.ts-gate.fired {{ background: rgba(42,75,168,.13); border-color: rgba(42,75,168,.38); }}

/* ============================ case picker + scoreboard ============================ */
.ts-case {{
    border: 1px solid {BORDER}; border-radius: 15px; padding: .85rem 1rem;
    background: #FFFFFF; font-size: .8rem; color: {MUTED}; line-height: 1.55;
}}
.ts-case.active {{ border-color: {BORDER_GOLD}; background: linear-gradient(180deg,
                   rgba(169,130,47,.09), #FFFDF6); }}
.ts-case .t {{ font-family: 'Sora', sans-serif; font-weight: 700; color: {INK};
               font-size: .88rem; display: block; margin-bottom: .2rem; }}
.ts-score {{
    border-radius: 16px; padding: 1.1rem 1.3rem; border: 1px solid;
    font-size: .92rem; line-height: 1.7;
}}
.ts-score .hl {{ font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.12rem;
                 display: block; margin-bottom: .3rem; }}
.ts-score.win {{ background: linear-gradient(180deg, rgba(18,119,90,.10), #FFFFFF);
                 border-color: rgba(18,119,90,.34); }}
.ts-score.win .hl {{ color: {EMERALD}; }}
.ts-score.flat {{ background: linear-gradient(180deg, rgba(176,122,12,.10), #FFFDF6);
                  border-color: rgba(176,122,12,.34); }}
.ts-score.flat .hl {{ color: {AMBER}; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status_badge(class_name: str) -> str:
    """Coloured pill for a predicted class."""
    colour = CLASS_COLORS.get(class_name, MUTED)
    icon = "●" if class_name == "Normal" else "▲"
    return (f'<span class="ts-badge" style="background:{colour}1A;color:{colour};'
            f'border:1px solid {colour}55">{icon} {class_name.upper()}</span>')
