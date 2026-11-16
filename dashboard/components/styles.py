from __future__ import annotations

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "color": "#E5E7EB",
            "family": "Inter, Segoe UI, sans-serif",
            "size": 12,
        },
        "title": {"font": {"color": "#F8FAFC", "size": 18}},
        "xaxis": {
            "gridcolor": "rgba(148,163,184,0.13)",
            "linecolor": "rgba(148,163,184,0.18)",
            "tickfont": {"color": "#A7B0C0"},
            "zerolinecolor": "rgba(148,163,184,0.15)",
        },
        "yaxis": {
            "gridcolor": "rgba(148,163,184,0.13)",
            "linecolor": "rgba(148,163,184,0.18)",
            "tickfont": {"color": "#A7B0C0"},
            "zerolinecolor": "rgba(148,163,184,0.15)",
        },
        "legend": {
            "bgcolor": "rgba(17,24,39,0.78)",
            "font": {"color": "#D1D5DB"},
        },
        "colorway": [
            "#A78BFA", "#38BDF8", "#34D399",
            "#FBBF24", "#FB7185", "#818CF8",
        ],
        "hoverlabel": {
            "bgcolor": "#1F2937",
            "bordercolor": "#8B5CF6",
            "font": {"color": "#F9FAFB"},
        },
        "margin": {"t": 62, "b": 38, "l": 30, "r": 24},
    }
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #090B14;
  --bg-soft: #0E1220;
  --surface: #141A2B;
  --surface-2: #1B2338;
  --surface-3: #222D45;
  --border: rgba(148,163,184,.18);
  --text: #F3F4F6;
  --muted: #A7B0C0;
  --muted-2: #7F8AA3;
  --violet: #A78BFA;
  --violet-strong: #8B5CF6;
  --cyan: #38BDF8;
  --green: #34D399;
  --amber: #FBBF24;
  --rose: #FB7185;
}

html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
}
.stApp {
  background:
    radial-gradient(ellipse at 82% 0%, rgba(139,92,246,.10), transparent 34%),
    radial-gradient(ellipse at 0% 28%, rgba(56,189,248,.055), transparent 28%),
    var(--bg);
  color: var(--text);
}
.main .block-container {
  max-width: 1580px;
  padding: 2.2rem 2.5rem 4rem;
}
[data-testid="stHeader"] {
  background: rgba(9,11,20,.88);
  backdrop-filter: blur(14px);
}
[data-testid="stToolbar"] { right: 1rem; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #101526 0%, #0B0F1C 100%);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.15rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
  color: #D2D8E5;
  font-size: .88rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--text);
  font-family: 'Manrope', sans-serif !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border);
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
  background: transparent;
  box-shadow: none;
  border: 0;
  padding: .2rem 0;
}

/* Typography */
h1, h2, h3, h4 {
  color: var(--text) !important;
  font-family: 'Manrope', 'DM Sans', sans-serif !important;
  letter-spacing: -.035em;
}
h1 { font-weight: 800 !important; }
h2 { font-weight: 750 !important; }
h3 { font-weight: 700 !important; }
p, li {
  color: #C1C9D8;
  line-height: 1.68;
}
small, .stCaption {
  color: var(--muted) !important;
}
hr {
  border-color: var(--border);
  margin: 1.35rem 0;
}

/* Metric cards */
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(31,41,65,.96), rgba(19,26,43,.98));
  border: 1px solid var(--border);
  border-radius: 15px;
  padding: 1.05rem 1.2rem;
  box-shadow: 0 12px 30px rgba(0,0,0,.16);
}
[data-testid="stMetricLabel"] {
  color: #AEB9CC !important;
  font-size: .74rem !important;
  font-weight: 700 !important;
  text-transform: uppercase;
  letter-spacing: .09em;
}
[data-testid="stMetricValue"] {
  color: #F8FAFC !important;
  font-family: 'Manrope', sans-serif !important;
  font-weight: 800 !important;
  letter-spacing: -.04em;
}
[data-testid="stMetricDelta"] { font-size: .8rem; }

/* Buttons and controls */
.stButton > button, .stDownloadButton > button, button[kind="secondary"] {
  border: 1px solid rgba(167,139,250,.48);
  border-radius: 10px;
  background: linear-gradient(135deg, #302653, #1E3550);
  color: #F5F3FF;
  font-weight: 700;
  transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--cyan);
  background: linear-gradient(135deg, #493477, #1E4960);
  color: white;
  transform: translateY(-1px);
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
  background: #151D30 !important;
  border-color: var(--border) !important;
  border-radius: 9px !important;
}
input, textarea { color: var(--text) !important; }
[data-baseweb="popover"], [data-baseweb="menu"] {
  background: #151D30 !important;
  border: 1px solid var(--border) !important;
}
[data-baseweb="menu"] li { color: var(--text) !important; }
[data-baseweb="tag"] { background: #40336A !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: rgba(20,26,43,.86);
  padding: 6px;
  border: 1px solid var(--border);
  border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted);
  border-radius: 8px;
  padding: 9px 17px;
  font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #493477, #24445E) !important;
  color: #FFFFFF !important;
}

/* Tables and dataframe surfaces */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  background: #111827;
}
[data-testid="stDataFrame"] * {
  font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
}
[data-testid="stDataFrame"] canvas {
  border-radius: 10px;
}
[data-testid="stTable"] table {
  border-collapse: separate;
  border-spacing: 0;
}
[data-testid="stTable"] th {
  background: #25213D !important;
  color: #EDE9FE !important;
  font-weight: 700 !important;
}
[data-testid="stTable"] td {
  background: #141A2B !important;
  color: #DCE3EE !important;
  border-color: var(--border) !important;
}
[data-testid="stAlert"] {
  border-radius: 12px;
  border: 1px solid var(--border);
}
details {
  background: rgba(20,26,43,.92);
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
summary { color: #E5E7EB !important; font-weight: 600; }

/* Shared page components */
.gw-page-header {
  padding: 0 0 1.25rem;
  margin-bottom: 1.55rem;
  border-bottom: 1px solid var(--border);
}
.gw-eyebrow {
  color: var(--cyan);
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .15em;
  text-transform: uppercase;
  margin-bottom: .5rem;
}
.gw-page-title {
  color: var(--text);
  font-family: 'Manrope', sans-serif;
  font-size: 2.15rem;
  line-height: 1.18;
  font-weight: 800;
  letter-spacing: -.05em;
}
.gw-page-subtitle {
  color: var(--muted);
  font-size: .94rem;
  margin-top: .5rem;
  max-width: 920px;
}
.section-header {
  color: #D8CCFF;
  font-family: 'Manrope', sans-serif;
  font-size: .78rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .13em;
  border-bottom: 1px solid var(--border);
  padding-bottom: .72rem;
  margin: 1.65rem 0 1.05rem;
}
.gw-card {
  background: linear-gradient(145deg, rgba(31,41,65,.96), rgba(19,26,43,.98));
  border: 1px solid var(--border);
  border-radius: 15px;
  padding: 1.2rem 1.3rem;
  margin-bottom: .85rem;
  box-shadow: 0 10px 26px rgba(0,0,0,.13);
}
.gw-card-title {
  color: var(--muted);
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .1em;
  font-weight: 800;
}
.gw-card-value {
  color: var(--text);
  font-family: 'Manrope', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  letter-spacing: -.045em;
  margin-top: .38rem;
}
.gw-card-sub { color: var(--muted); font-size: .82rem; margin-top: .4rem; }
.insight-card {
  background: linear-gradient(120deg, rgba(31,41,65,.92), rgba(20,26,43,.96));
  border: 1px solid var(--border);
  border-left: 3px solid var(--violet);
  border-radius: 12px;
  padding: 1.05rem 1.2rem;
  margin: .75rem 0;
}
.insight-card h4 { color: #E9DFFF; margin: 0 0 .4rem; }
.insight-card p { color: #C1C9D8; margin: 0; font-size: .9rem; }
.status-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 20px;
  font-size: .7rem;
  font-weight: 800;
  letter-spacing: .045em;
}
.status-ready { color: #6EE7B7; background: rgba(6,78,59,.38); border: 1px solid #276B58; }
.status-pending { color: #FCD34D; background: rgba(120,53,15,.28); border: 1px solid #8A642B; }
.status-demo { color: #7DD3FC; background: rgba(12,74,110,.3); border: 1px solid #2D6286; }
.status-blocked { color: #FDA4AF; background: rgba(136,19,55,.25); border: 1px solid #773944; }
.hero-title {
  color: var(--text);
  font-family: 'Manrope', sans-serif;
  font-size: clamp(2.25rem, 4vw, 3.5rem);
  line-height: 1.08;
  font-weight: 800;
  letter-spacing: -.06em;
}
.hero-subtitle { color: #B9C4D6; font-size: 1.02rem; max-width: 900px; line-height: 1.7; }
.nav-card {
  background: linear-gradient(145deg, #1D2740, #141A2B);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.2rem;
  min-height: 140px;
  transition: transform .18s ease, border-color .18s ease;
}
.nav-card:hover { transform: translateY(-2px); border-color: rgba(167,139,250,.55); }
.nav-card-title { color: var(--text); font-weight: 800; }
.nav-card-desc { color: var(--muted); font-size: .84rem; }
.demo-banner {
  background: rgba(120,53,15,.25);
  border: 1px solid var(--amber);
  border-radius: 10px;
  padding: 1rem;
  color: #FDE68A;
}
.phase-row {
  display: flex;
  gap: .6rem;
  padding: .55rem 0;
  border-bottom: 1px solid var(--border);
}
.phase-name { color: #D2D8E5; font-size: .85rem; }
.geo-scope-box {
  background: linear-gradient(145deg, #1D2740, #141A2B);
  border: 1px solid var(--border);
  border-radius: 13px;
  padding: 1.05rem;
}
@media (max-width: 800px) {
  .main .block-container { padding: 1rem 1rem 2rem; }
  .gw-page-title { font-size: 1.65rem; }
  .hero-title { font-size: 2.15rem; }
}
</style>
"""


def inject_theme() -> None:
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)


def demo_banner(message: str = "") -> None:
    import streamlit as st
    text = message or "Preview data is being used. Do not interpret it as a measured groundwater observation."
    st.markdown(f'<div class="demo-banner">{text}</div>', unsafe_allow_html=True)


def section_header(title: str) -> None:
    import html
    import streamlit as st
    st.markdown(
        f'<div class="section-header">{html.escape(str(title))}</div>',
        unsafe_allow_html=True,
    )


def status_pill(label: str, status: str = "pending") -> str:
    import html
    safe_status = status if status in {"ready", "pending", "demo", "blocked"} else "pending"
    return (
        f'<span class="status-pill status-{safe_status}">'
        f'{html.escape(str(label))}</span>'
    )


def insight_card(title: str, body: str) -> None:
    import html
    import streamlit as st
    st.markdown(
        f'<div class="insight-card"><h4>{html.escape(str(title))}</h4>'
        f'<p>{html.escape(str(body))}</p></div>',
        unsafe_allow_html=True,
    )
