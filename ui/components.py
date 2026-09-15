"""Reusable presentation blocks so every page speaks the same visual language."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from .theme import CLASS_COLORS, GREEN, MUTED, RED, TEXT

# (icon, name, one-line explanation) -- the story the judge reads in 10 seconds.
PIPELINE: List[Tuple[str, str, str]] = [
    ("🚗", "CAN Traffic", "Real frames from the vehicle bus"),
    ("🧠", "Digital Twin", "Predicts expected CAN behaviour"),
    ("⚡", "Residual", "Measures deviation from expected"),
    ("🔗", "Adaptive Fusion", "Combines evidence across ECU streams"),
    ("🤖", "CNN-BiLSTM", "Classifies normal and malicious traffic"),
    ("🚨", "Detection", "Raises the alert and its confidence"),
    ("🔍", "Explanation", "Shows why the model decided"),
]


def pipeline(active: Optional[int] = None, done: Optional[int] = None,
             stages: Sequence[Tuple[str, str, str]] = PIPELINE,
             compact: bool = False) -> None:
    """Horizontal pipeline. `active` highlights a stage, `done` marks completed ones."""
    parts = ['<div class="ts-pipe">']
    for i, (icon, name, desc) in enumerate(stages):
        cls = "ts-stage"
        if active is not None and i == active:
            cls += " on"
        elif done is not None and i < done:
            cls += " done"
        body = f'<span class="ico">{icon}</span><span class="nm">{name}</span>'
        if not compact:
            body += f'<span class="ds">{desc}</span>'
        parts.append(f'<div class="{cls}">{body}</div>')
        if i < len(stages) - 1:
            parts.append('<div class="ts-arrow">→</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def card(icon: str, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="ts-card"><span class="ts-card-icon">{icon}</span>'
        f'<h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)


def kv_card(title: str, rows: Sequence[Tuple[str, str]], icon: str = "") -> None:
    inner = "".join(f'<div class="ts-kv"><span>{k}</span><span>{v}</span></div>'
                    for k, v in rows)
    head = f'<span class="ts-card-icon">{icon}</span>' if icon else ""
    st.markdown(f'<div class="ts-card">{head}<h4>{title}</h4>{inner}</div>',
                unsafe_allow_html=True)


def metric_tile(label: str, value: str, sub: str = "", highlight: bool = False,
                colour: Optional[str] = None) -> None:
    cls = "ts-metric hi" if highlight else "ts-metric"
    style = f' style="color:{colour}"' if colour else ""
    st.markdown(
        f'<div class="{cls}"><div class="lbl">{label}</div>'
        f'<div class="val"{style}>{value}</div>'
        f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def status_banner(class_name: str, confidence: float, subtitle: str = "",
                  suspicious: bool = False) -> None:
    """Main attack-status banner driven by the model's actual output."""
    if class_name == "Normal" and not suspicious:
        tone, icon, headline = "ts-ok", "🟢", "SYSTEM NORMAL"
    elif suspicious and class_name == "Normal":
        tone, icon, headline = "ts-warn", "🟡", "SUSPICIOUS ACTIVITY"
    else:
        tone, icon, headline = "ts-alert", "🔴", "🚨 ATTACK DETECTED"

    colour = CLASS_COLORS.get(class_name, TEXT)
    detail = (f'Attack Type: <b style="color:{colour}">{class_name}</b> &nbsp;·&nbsp; '
              f'Confidence: <b>{confidence*100:.1f}%</b>')
    if class_name == "Normal":
        detail = (f'Predicted class: <b style="color:{GREEN}">Normal</b> &nbsp;·&nbsp; '
                  f'Confidence: <b>{confidence*100:.1f}%</b>')
    st.markdown(
        f'<div class="ts-status {tone}"><div style="font-size:2.1rem">{icon}</div>'
        f'<div><div class="big">{headline}</div>'
        f'<div class="meta">{detail}</div>'
        f'<div class="meta">{subtitle}</div></div></div>', unsafe_allow_html=True)


def frame_rows(table, limit: int = 14) -> None:
    """Monospaced CAN frame list -- real frames, real predictions."""
    head = ('<div class="ts-frame-head"><div>Time</div><div>CAN ID</div><div>DLC</div>'
            '<div>Payload</div><div>Status</div></div>')
    rows = []
    for _, r in table.tail(limit).iloc[::-1].iterrows():
        colour = CLASS_COLORS.get(r["Predicted"], MUTED)
        # Geometric marks rather than emoji: at this density emoji render
        # inconsistently across platforms and read as clutter.
        icon = "●" if r["Predicted"] == "Normal" else "▲"
        badge = (f'<span class="ts-badge" style="background:{colour}1F;color:{colour};'
                 f'border:1px solid {colour}4D">{icon} {r["Predicted"].upper()}</span>')
        rows.append(
            f'<div class="ts-frame"><div>{r["Time"]:.3f}</div>'
            f'<div style="color:{TEXT}">{r["CAN ID"]}</div><div>{r["DLC"]}</div>'
            f'<div class="pl">{r["Payload"]}</div><div>{badge}</div></div>')
    st.markdown(f'<div class="ts-frames">{head}{"".join(rows)}</div>',
                unsafe_allow_html=True)


def section(title: str, caption: str = "", small: bool = False) -> None:
    """Section heading with a gold rule, so the page has a visible hierarchy."""
    size = "0.98rem" if small else "1.16rem"
    bar_h = "13px" if small else "17px"
    st.markdown(
        f'<div class="ts-section"><span class="bar" style="height:{bar_h}"></span>'
        f'<h3 style="margin:0;font-size:{size}">{title}</h3></div>',
        unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="ts-note">{caption}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="ts-note">{text}</div>', unsafe_allow_html=True)


def flag(text: str) -> None:
    st.markdown(f'<div class="ts-flag">{text}</div>', unsafe_allow_html=True)
