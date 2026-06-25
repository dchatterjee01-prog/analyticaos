# app.py
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
import streamlit as st
from config.settings import (
    APP_NAME, APP_VERSION, APP_TAGLINE,
    PRIMARY_COLOR, BACKGROUND_COLOR,
    SURFACE_COLOR, TEXT_COLOR, ACCENT_COLOR,
    BORDER_COLOR, MUTED_COLOR, SUCCESS_COLOR, SUCCESS_BG
)
# ── UPDATED IMPORTS ──────────────────────────────────────────────────────────
from pages import cleaning, eda, viz  
from pages import stats
from pages import questions
from pages import agents_ui
from pages import executive  # Added for Phase 10 Executive Consultant Engine
from pages import report  # Added for Phase 11 Report Generator
from pages import forecast
from pages import anomaly
from pages import nlpqa  # Added Step 9 - was missing despite module existing (Phase 11.5)
from pages import excel_intelligence_ui
from pages import sql_agent_ui
from pages import automl
from pages import experiments
from pages import causal
from pages import deep_learning
from pages import optimization
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Authentication gate (Phase 12, Step 7) ────────────────────────────────────
# Falls back to "no login required" if secrets.toml has no [auth] section,
# so local development without OIDC configured never crashes.
AUTH_ENABLED = "auth" in st.secrets

if AUTH_ENABLED and not st.user.is_logged_in:
    login_logo_svg = ('<svg width="48" height="48" viewBox="0 0 40 40">'
        '<rect x="7" y="22" width="6" height="12" rx="2" fill="#5C7290"/>'
        '<rect x="17" y="16" width="6" height="18" rx="2" fill="#2A5C8C"/>'
        '<rect x="27" y="10" width="6" height="24" rx="2" fill="#2A5C8C"/>'
        '<circle cx="30" cy="4" r="4" fill="#2E7D5B"/>'
        '</svg>')

    login_html = (
        '<div class="login-card">'
        '<div class="login-accent-bar"></div>'
        '<div class="login-top">'
        f'{login_logo_svg}'
        f'<div class="login-title">{APP_NAME}</div>'
        '<div class="login-byline">by Daipayan Chatterjee</div>'
        '<div class="login-desc">Your autonomous AI data scientist — clean, analyze, model, and report on any dataset automatically.</div>'
        '</div>'
        '<div class="login-divider"></div>'
        '<div class="login-features">'
        '<div class="feature-chip"><div class="feature-chip-icon">🧹</div><div class="feature-chip-label">Auto Cleaning</div></div>'
        '<div class="feature-chip"><div class="feature-chip-icon">🤖</div><div class="feature-chip-label">ML Models</div></div>'
        '<div class="feature-chip"><div class="feature-chip-icon">🏛️</div><div class="feature-chip-label">Executive Briefs</div></div>'
        '<div class="feature-chip"><div class="feature-chip-icon">📄</div><div class="feature-chip-label">Word Reports</div></div>'
        '</div>'
        '<div class="login-divider"></div>'
        '<div class="login-cta-label">Sign in to continue</div>'
        '</div>'
    )

    login_css = (
        f'<style>'
        f'.stApp {{ background-color: {BACKGROUND_COLOR}; }}'
        f'[data-testid="stSidebar"] {{ display: none; }}'
        f'.login-card {{ max-width: 420px; margin: 3rem auto 0 auto; background-color: {SURFACE_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 16px; overflow: hidden; text-align: center; }}'
        f'.login-accent-bar {{ height: 4px; background-color: {PRIMARY_COLOR}; }}'
        f'.login-top {{ padding: 2.2rem 2rem 1.4rem 2rem; }}'
        f'.login-title {{ font-family: "Georgia", serif; font-weight: 700; font-size: 1.65rem; color: {PRIMARY_COLOR}; margin-top: 0.9rem; }}'
        f'.login-byline {{ font-size: 0.85rem; color: {TEXT_COLOR}; font-weight: 600; margin-top: 0.3rem; }}'
        f'.login-desc {{ font-size: 0.78rem; color: {MUTED_COLOR}; margin-top: 0.8rem; line-height: 1.5; }}'
        f'.login-divider {{ height: 1px; background-color: {BORDER_COLOR}; margin: 0 1.6rem; }}'
        f'.login-features {{ padding: 1.2rem 1.6rem; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}'
        f'.feature-chip {{ background-color: {BACKGROUND_COLOR}; border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.6rem; }}'
        f'.feature-chip-icon {{ font-size: 1.1rem; }}'
        f'.feature-chip-label {{ font-size: 0.66rem; color: {MUTED_COLOR}; margin-top: 2px; }}'
        f'.login-cta-label {{ font-size: 0.72rem; color: {MUTED_COLOR}; padding: 1.4rem 0 0.4rem 0; }}'
        f'</style>'
    )

    st.markdown(login_css + login_html, unsafe_allow_html=True)

    _, center_col, _ = st.columns([1, 1, 1])
    with center_col:
        if st.button("🔵 Log in with Google", width='stretch', type="primary"):
            st.login()
    st.stop()

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  :root {{
    --primary:  {PRIMARY_COLOR};
    --bg:       {BACKGROUND_COLOR};
    --surface:  {SURFACE_COLOR};
    --text:     {TEXT_COLOR};
    --accent:   {ACCENT_COLOR};
  }}
  .stApp {{
    background-color: var(--bg);
    color: var(--text);
  }}
  [data-testid="stSidebar"] {{
    background-color: var(--surface);
    border-right: 1px solid #DCE4EC;
  }}
  .brand-card {{
    background-color: {SURFACE_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1rem;
  }}
  .brand-accent-bar {{ height: 3px; background-color: {PRIMARY_COLOR}; }}
  .brand-top {{ padding: 1.15rem 1.1rem 0.85rem 1.1rem; }}
  .brand-row {{ display: flex; align-items: center; gap: 11px; }}
  .brand-name {{ font-family: 'Georgia', serif; font-weight: 700; font-size: 1.28rem; color: {PRIMARY_COLOR}; line-height: 1; letter-spacing: 0.01em; }}
  .brand-version {{ font-size: 0.6rem; color: {MUTED_COLOR}; letter-spacing: 0.07em; font-weight: 600; }}
  .brand-author {{ font-size: 0.74rem; color: {TEXT_COLOR}; margin-top: 3px; font-weight: 500; }}
  .brand-tagline {{ font-size: 0.72rem; color: {MUTED_COLOR}; margin-top: 0.7rem; line-height: 1.45; }}
  .brand-divider {{ height: 1px; background-color: {BORDER_COLOR}; margin: 0 1.1rem; }}
  .status-row {{ padding: 0.75rem 1.1rem; display: flex; align-items: center; gap: 8px; }}
  .status-dot {{ width: 6px; height: 6px; border-radius: 50%; background-color: {SUCCESS_COLOR}; flex-shrink: 0; box-shadow: 0 0 0 3px {SUCCESS_BG}; }}
  .status-dot-off {{ width: 6px; height: 6px; border-radius: 50%; background-color: {MUTED_COLOR}; flex-shrink: 0; }}
  .status-filename {{ font-size: 0.76rem; color: {TEXT_COLOR}; font-weight: 500; }}
  .status-dims {{ font-size: 0.7rem; color: {MUTED_COLOR}; margin-left: auto; }}
  .status-empty {{ font-size: 0.76rem; color: {MUTED_COLOR}; }}
  /* ── Hide Streamlit's auto-generated pages/ navigation list ──────── */
  /* We use our own custom NAV router below — Streamlit's built-in     */
  /* multipage switcher (auto-created because the folder is literally */
  /* named "pages") is redundant and broken, since those files aren't */
  /* designed to run standalone. This hides it without touching any   */
  /* file names, imports, or routing logic.                           */
  [data-testid="stSidebarNav"] {{
    display: none;
  }}
  /* ── Tighten default Streamlit vertical spacing in sidebar nav ───── */
  [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.15rem;
  }}
  .user-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.1rem;
    margin-bottom: 0.6rem;
  }}
  .user-name {{
    font-size: 0.78rem;
    color: {TEXT_COLOR};
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  [data-testid="stSidebar"] .stButton button[kind="secondary"] {{
    font-size: 0.7rem !important;
    padding: 0.25rem 0.6rem !important;
    width: auto !important;
  }}
  .nav-group {{
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5C7290;
    padding: 1.1rem 0.4rem 0.4rem 0.4rem;
    margin-top: 0.2rem;
  }}
  /* ── Sidebar nav buttons ─────────────────────────────────────────── */
  [data-testid="stSidebar"] .stButton button {{
    width: 100%;
    text-align: left;
    background-color: transparent;
    color: {TEXT_COLOR};
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-size: 0.92rem;
    font-weight: 500;
    margin-bottom: 0.15rem;
    transition: all 0.15s ease;
    box-shadow: none;
  }}
  [data-testid="stSidebar"] .stButton button:hover {{
    background-color: #FFFFFF;
    border: 1px solid #DCE4EC;
    color: {PRIMARY_COLOR};
  }}
  [data-testid="stSidebar"] .stButton button:focus:not(:active) {{
    background-color: #FFFFFF;
    border: 1px solid #DCE4EC;
    color: {PRIMARY_COLOR};
  }}
  [data-testid="stSidebar"] .stButton button p {{
    font-size: 0.92rem;
    font-weight: 500;
    text-align: left;
  }}
  /* Active page button gets a distinct filled look */
  .nav-active button {{
    background-color: #FFFFFF !important;
    border: 1px solid {PRIMARY_COLOR} !important;
    color: {PRIMARY_COLOR} !important;
    font-weight: 700 !important;
  }}
  .hero {{
    padding: 3rem 2rem 2rem 2rem;
    text-align: center;
  }}
  .hero-title {{
    font-size: 2.8rem;
    font-weight: 700;
    color: {PRIMARY_COLOR};
    font-family: 'Georgia', serif;
    letter-spacing: 0.01em;
  }}
  .hero-byline {{
    font-size: 0.92rem;
    color: {PRIMARY_COLOR};
    font-weight: 600;
    margin-top: 0.5rem;
    letter-spacing: 0.01em;
  }}
  .hero-sub {{
    font-size: 1.05rem;
    color: #5C7290;
    margin-top: 0.6rem;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
  }}
  .feature-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    justify-content: center;
    margin-top: 2rem;
    padding: 0 1rem;
  }}
  .feature-card {{
    background: #FFFFFF;
    border: 1px solid #DCE4EC;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    width: 200px;
    text-align: center;
  }}
  .feature-icon {{
    font-size: 1.8rem;
  }}
  .feature-label {{
    font-size: 0.8rem;
    color: {TEXT_COLOR};
    margin-top: 0.5rem;
    font-weight: 500;
  }}
</style>
""", unsafe_allow_html=True)

# ── Navigation map ───────────────────────────────────────────────────────────
NAV = {
    "🏠 Home": ("home", None),
    "— DATA —": None,
    "📁 Upload Data":        ("upload",    None),
    "🗂️ Multi-Sheet Excel": ("excel_intelligence_ui", None),
    "🧹 Data Cleaning":      ("cleaning",  "df"),
    "— ANALYSIS —": None,
    "🔬 EDA":                ("eda",       "df"),
    "📊 Pivot Tables":       ("pivot",     "df"),
    "📈 Visualizations":     ("viz",       "df"),
    "📐 Statistical Engine": ("stats",     "df"),
    "🧪 A/B Testing":        ("experiments", "df"),
    "🔗 Causal Inference":   ("causal",      "df"),
    "🗄️ SQL Agent": ("sql_agent_ui", None),
    "— INTELLIGENCE —": None,
    "🤖 Machine Learning":   ("ml",        "df"),
    "🧬 AutoML Engine":      ("automl",    "df"),
    "⚡ Deep Learning":      ("deep_learning", "df"),
    "🧠 Auto Questions":    ("questions", "df"),
    "🤖 Multi-Agent System": ("agents_ui", "df"),
    "🔮 Forecasting":        ("forecast",  "df"),
    "🚨 Anomaly Detection":  ("anomaly",   "df"),
    "— DECISION INTELLIGENCE —": None,
    "💬 Ask Your Data":      ("nlpqa",     "df"),
    "🏛️ Executive Console":  ("executive", "df"),
    "📄 Report Generator":   ("report",    "df"),
    "🧮 Optimization Engine": ("optimization", None),
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand card with logo, author credit, and integrated dataset status.
    # NOTE: built as single-line concatenated strings (no leading
    # whitespace) deliberately — multi-line indented triple-quoted HTML
    # gets misread by Streamlit's markdown parser as a code block.
    LOGO_SVG = (
        '<svg width="34" height="34" viewBox="0 0 40 40">'
        '<rect x="7" y="22" width="6" height="12" rx="2" fill="#5C7290"/>'
        '<rect x="17" y="16" width="6" height="18" rx="2" fill="#2A5C8C"/>'
        '<rect x="27" y="10" width="6" height="24" rx="2" fill="#2A5C8C"/>'
        '<circle cx="30" cy="4" r="4" fill="#2E7D5B"/>'
        '</svg>'
    )

    if "df" in st.session_state and st.session_state["df"] is not None:
        fname = st.session_state.get("filename", "dataset")
        rows  = st.session_state["df"].shape[0]
        cols  = st.session_state["df"].shape[1]
        status_html = (
            '<div class="status-row">'
            '<span class="status-dot"></span>'
            f'<span class="status-filename">{fname}</span>'
            f'<span class="status-dims">{rows:,} × {cols}</span>'
            '</div>'
        )
    else:
        status_html = (
            '<div class="status-row">'
            '<span class="status-dot-off"></span>'
            '<span class="status-empty">No dataset loaded</span>'
            '</div>'
        )

    brand_html = (
        '<div class="brand-card">'
        '<div class="brand-accent-bar"></div>'
        '<div class="brand-top">'
        '<div class="brand-row">'
        f'{LOGO_SVG}'
        '<div>'
        '<div style="display:flex;align-items:baseline;gap:7px;">'
        f'<span class="brand-name">{APP_NAME}</span>'
        f'<span class="brand-version">v{APP_VERSION}</span>'
        '</div>'
        '<div class="brand-author">by Daipayan Chatterjee</div>'
        '</div>'
        '</div>'
        f'<div class="brand-tagline">{APP_TAGLINE}</div>'
        '</div>'
        '<div class="brand-divider"></div>'
        f'{status_html}'
        '</div>'
    )
    st.markdown(brand_html, unsafe_allow_html=True)

    # Logged-in user row + logout (only shown when auth is actually enabled)
    if AUTH_ENABLED:
        user_name = getattr(st.user, "name", None) or getattr(st.user, "email", "Signed in")
        ucol1, ucol2 = st.columns([3, 1])
        with ucol1:
            st.markdown(f'<div class="user-row"><span class="user-name">👤 {user_name}</span></div>', unsafe_allow_html=True)
        with ucol2:
            if st.button("Logout", key="logout_btn", type="secondary"):
                st.logout()

    st.markdown("")

    # Initialize current page in session state
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "🏠 Home"

    # Render nav items in order, with proper grouping and active highlight
    for key, val in NAV.items():
        if key.startswith("—"):
            # Section header
            label = key.strip("— ").strip()
            st.markdown(f'<div class="nav-group">{label}</div>', unsafe_allow_html=True)
        else:
            is_active = (st.session_state["current_page"] == key)
            wrapper_class = "nav-active" if is_active else "nav-inactive"
            st.markdown(f'<div class="{wrapper_class}">', unsafe_allow_html=True)
            if st.button(key, key=f"navbtn_{key}", width="stretch"):
                st.session_state["current_page"] = key
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    page = st.session_state["current_page"]

# ── Route to page ────────────────────────────────────────────────────────────
def needs_data():
    st.warning("⚠️ Please upload a dataset first via **📁 Upload Data**.")

if page == "🏠 Home":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">AnalyticaOS</div>
      <div class="hero-byline">Built by Daipayan Chatterjee</div>
      <div class="hero-sub">
        AnalyticaOS turns any dataset into boardroom-ready intelligence.
        Clean your data, explore it, test hypotheses, and train machine
        learning models — then move beyond description into action: optimize
        resource allocation, surface the levers that move your key metrics,
        and let autonomous AI agents synthesize it all into an executive
        briefing and a downloadable report.
      </div>
    </div>
    <div class="feature-grid">
      <div class="feature-card">
        <div class="feature-icon">🧹</div>
        <div class="feature-label">Auto Data Cleaning</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🔬</div>
        <div class="feature-label">Exploratory Analysis</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-label">Machine Learning</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🔮</div>
        <div class="feature-label">Forecasting</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🧮</div>
        <div class="feature-label">Optimization</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-label">Prescriptive Levers</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🏛️</div>
        <div class="feature-label">Executive Strategy</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "📁 Upload Data":
    from pages.upload import render
    render()

elif page == "🗂️ Multi-Sheet Excel":
       excel_intelligence_ui.show()

# ── ROUTING FOR DATA CLEANING ────────────────────────────────────────────────
elif page == "🧹 Data Cleaning":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        cleaning.show()

# ── ROUTING FOR EDA ──────────────────────────────────────────────────────────
elif page == "🔬 EDA":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        eda.show()

# ── ROUTING FOR VISUALIZATIONS ───────────────────────────────────────────────
elif page == "📈 Visualizations":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        viz.show()

# ── ROUTING FOR PIVOT TABLES ─────────────────────────────────────────────────
elif page == "📊 Pivot Tables":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        from pages.pivot import show
        show()

elif page == "📐 Statistical Engine":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        stats.show()

elif page == "🧪 A/B Testing":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        experiments.show()

elif page == "🔗 Causal Inference":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        causal.show()

elif page == "🗄️ SQL Agent":
       sql_agent_ui.show()

elif page == "🤖 Machine Learning":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        from pages.ml import show
        show()

elif page == "🧬 AutoML Engine":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        automl.show()

elif page == "⚡ Deep Learning":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        deep_learning.show()

elif page == "🧠 Auto Questions":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        questions.show()

elif page == "🤖 Multi-Agent System":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        agents_ui.show()

elif page == "🧮 Optimization Engine":
    optimization.show()


# ── ROUTING FOR EXECUTIVE CONSOLE (PHASE 10) ─────────────────────────────────
elif page == "🏛️ Executive Console":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        executive.show()

elif page == "📄 Report Generator":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        report.show()

elif page == "🔮 Forecasting":
       if "df" not in st.session_state or st.session_state["df"] is None:
           needs_data()
       else:
           forecast.show()

elif page == "🚨 Anomaly Detection":
       if "df" not in st.session_state or st.session_state["df"] is None:
           needs_data()
       else:
           anomaly.show()

elif page == "💬 Ask Your Data":
    if "df" not in st.session_state or st.session_state["df"] is None:
        needs_data()
    else:
        nlpqa.show()

else:
    _, requires = NAV.get(page, (None, None))
    if requires == "df" and (
        "df" not in st.session_state or
        st.session_state["df"] is None
    ):
        needs_data()
    else:
        st.title(page)
        st.info("This module is coming in a future phase.", icon="🔜")