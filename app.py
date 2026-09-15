"""
TwinShield-Fusion
A Digital Twin-Assisted Adaptive Multi-ECU Residual Fusion Framework for
Explainable Intrusion Detection in CAN Networks.

    streamlit run app.py
"""
from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="TwinShield-Fusion",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui.theme import inject  # noqa: E402

inject()

PAGES = [
    ("Overview", "🛡️", "pages/01_Overview.py"),
    ("Live Monitor", "🚗", "pages/02_Live_Monitor.py"),
    ("Results", "📊", "pages/03_Results.py"),
]

pages = [
    st.Page(path, title=title, icon=icon, default=(i == 0))
    for i, (title, icon, path) in enumerate(PAGES)
]

# Streamlit's own top navigation collapses every entry into an overflow menu at
# this viewport, so navigation is rendered explicitly below instead.  Keeping
# st.navigation (hidden) preserves real routing, URLs and page state.
nav = st.navigation(pages, position="hidden")


def navbar(active_title: str) -> None:
    brand, *slots = st.columns([2.2, 1, 1, 1, 1.1], vertical_alignment="center")
    with brand:
        st.markdown('<div class="ts-brand">🛡️ <b class="mark">TwinShield-Fusion</b></div>',
                    unsafe_allow_html=True)
    for col, (title, icon, path) in zip(slots, PAGES):
        with col:
            is_active = title == active_title
            # Key is a slug (no spaces) so the stylesheet can target these
            # buttons via the .st-key-nav_* container class.
            slug = title.lower().replace(" ", "_")
            if st.button(f"{icon}  {title}", key=f"nav_{slug}",
                         use_container_width=True,
                         type="primary" if is_active else "secondary"):
                if not is_active:
                    st.switch_page(path)
    st.markdown('<div class="ts-navrule"></div>', unsafe_allow_html=True)


navbar(nav.title)
nav.run()
