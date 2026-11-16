from __future__ import annotations

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#0B1220",
        "plot_bgcolor": "#101B2D",
        "font": {"color": "#DCE7F5", "family": "Inter, Segoe UI, sans-serif", "size": 12},
        "title": {"font": {"color": "#F0F6FF", "size": 16}},
        "xaxis": {"gridcolor": "#24344A", "linecolor": "#24344A", "tickfont": {"color": "#91A5BF"}},
        "yaxis": {"gridcolor": "#24344A", "linecolor": "#24344A", "tickfont": {"color": "#91A5BF"}},
        "legend": {"bgcolor": "rgba(16,27,45,0.85)", "font": {"color": "#DCE7F5"}},
        "colorway": ["#36C5D9", "#6E9EFF", "#42D6A4", "#F5B95F", "#E87979", "#A78BFA"],
        "hoverlabel": {"bgcolor": "#17243A", "bordercolor": "#36C5D9", "font": {"color": "#F0F6FF"}},
        "margin": {"t": 48, "b": 36, "l": 48, "r": 24},
    }
}

_CSS = """
<style>
:root {
  --bg: #0B1220;
  --surface: #111C2E;
  --surface-2: #17243A;
  --border: #26364D;
  --text: #EAF1FA;
  --muted: #91A5BF;
  --accent: #36C5D9;
  --blue: #6E9EFF;
  --green: #42D6A4;
  --amber: #F5B95F;
}
.stApp { background:var(--bg); color:var(--text); }
.main .block-container {
  max-width: 1540px;
  padding: 2rem 2.4rem 3.5rem;
}
[data-testid="stHeader"] { background:rgba(11,18,32,.88); }
[data-testid="stSidebar"] {
  background:#0D1727;
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child { padding-top:1.1rem; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color:#C2D1E3; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:var(--text); }
h1,h2,h3 { color:var(--text) !important; letter-spacing:-.025em; }
p,li { color:#B9C9DC; line-height:1.65; }
small { color:var(--muted); }
hr { border-color:var(--border); margin:1.2rem 0; }

[data-testid="stMetric"] {
  background:linear-gradient(145deg,#17243A,#111C2E);
  border:1px solid var(--border);
  border-radius:13px;
  padding:1rem 1.15rem;
  box-shadow:0 5px 18px rgba(0,0,0,.13);
}
[data-testid="stMetricLabel"] {
  color:var(--muted) !important;
  font-size:.76rem !important;
  text-transform:uppercase;
  letter-spacing:.07em;
}
[data-testid="stMetricValue"] {
  color:var(--text) !important;
  font-weight:700;
}
[data-testid="stMetricDelta"] { font-size:.8rem; }

.stButton > button, .stDownloadButton > button {
  border:1px solid #347F91;
  border-radius:9px;
  background:#17384A;
  color:#DFFBFF;
  font-weight:600;
  transition:.18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color:var(--accent);
  background:#205268;
  color:white;
}
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
  background:#142137 !important;
  border-color:var(--border) !important;
  border-radius:8px !important;
}
input,textarea { color:var(--text) !important; }
[data-baseweb="popover"],[data-baseweb="menu"] {
  background:#142137 !important;
  border:1px solid var(--border) !important;
}
[data-baseweb="menu"] li { color:var(--text) !important; }
[data-baseweb="tag"] { background:#24435B !important; }

.stTabs [data-baseweb="tab-list"] {
  gap:5px;
  background:#101B2D;
  padding:5px;
  border:1px solid var(--border);
  border-radius:10px;
}
.stTabs [data-baseweb="tab"] {
  color:var(--muted);
  border-radius:7px;
  padding:8px 16px;
}
.stTabs [aria-selected="true"] {
  background:#21465A !important;
  color:#EAFBFF !important;
}
[data-testid="stDataFrame"] {
  border:1px solid var(--border);
  border-radius:10px;
}
[data-testid="stAlert"] { border-radius:10px; }
details {
  background:#111C2E;
  border:1px solid var(--border) !important;
  border-radius:10px !important;
}
summary { color:#C8D8EA !important; }

.gw-page-header {
  padding:0 0 1.15rem;
  margin-bottom:1.45rem;
  border-bottom:1px solid var(--border);
}
.gw-eyebrow {
  color:var(--accent);
  font-size:.72rem;
  font-weight:700;
  letter-spacing:.13em;
  text-transform:uppercase;
  margin-bottom:.45rem;
}
.gw-page-title {
  color:var(--text);
  font-size:2rem;
  line-height:1.2;
  font-weight:750;
  letter-spacing:-.04em;
}
.gw-page-subtitle {
  color:var(--muted);
  font-size:.92rem;
  margin-top:.45rem;
  max-width:900px;
}
.section-header {
  color:#BDEEF4;
  font-size:.82rem;
  font-weight:750;
  text-transform:uppercase;
  letter-spacing:.1em;
  border-bottom:1px solid var(--border);
  padding-bottom:.65rem;
  margin:1.5rem 0 1rem;
}
.gw-card {
  background:linear-gradient(145deg,#17243A,#111C2E);
  border:1px solid var(--border);
  border-radius:13px;
  padding:1.15rem 1.25rem;
  margin-bottom:.8rem;
}
.gw-card-title {
  color:var(--muted);
  font-size:.73rem;
  text-transform:uppercase;
  letter-spacing:.08em;
  font-weight:700;
}
.gw-card-value {
  color:var(--text);
  font-size:1.9rem;
  font-weight:750;
  margin-top:.35rem;
}
.gw-card-sub { color:var(--muted); font-size:.8rem; margin-top:.35rem; }
.insight-card {
  background:#111C2E;
  border:1px solid var(--border);
  border-left:3px solid var(--accent);
  border-radius:10px;
  padding:1rem 1.15rem;
  margin:.7rem 0;
}
.insight-card h4 { color:#C7F3F7; margin:0 0 .35rem; }
.insight-card p { color:#B9C9DC; margin:0; font-size:.88rem; }
.status-pill {
  display:inline-block;
  padding:3px 9px;
  border-radius:20px;
  font-size:.7rem;
  font-weight:700;
  letter-spacing:.04em;
}
.status-ready { color:#72E5BC; background:#12382F; border:1px solid #276B58; }
.status-pending { color:#F5C777; background:#3A2B17; border:1px solid #74552B; }
.status-demo { color:#83C8F5; background:#16334B; border:1px solid #2D6286; }
.status-blocked { color:#F29A9A; background:#401F28; border:1px solid #773944; }
.hero-title {
  color:var(--text);
  font-size:2.45rem;
  font-weight:800;
  letter-spacing:-.045em;
}
.hero-subtitle { color:var(--muted); font-size:1rem; max-width:850px; }
.nav-card {
  background:#111C2E;
  border:1px solid var(--border);
  border-radius:12px;
  padding:1.1rem;
  min-height:130px;
}
.nav-card-title { color:var(--text); font-weight:700; }
.nav-card-desc { color:var(--muted); font-size:.8rem; }
.demo-banner {
  background:#382B19;
  border:1px solid var(--amber);
  border-radius:9px;
  padding:1rem;
  color:#FFE1A8;
}
.phase-row {
  display:flex;
  gap:.6rem;
  padding:.5rem 0;
  border-bottom:1px solid var(--border);
}
.phase-name { color:#C2D1E3; font-size:.83rem; }
.geo-scope-box {
  background:#111C2E;
  border:1px solid var(--border);
  border-radius:12px;
  padding:1rem;
}
@media(max-width:800px) {
  .main .block-container { padding:1rem 1rem 2rem; }
  .gw-page-title { font-size:1.6rem; }
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
    st.markdown(f'<div class="section-header">{html.escape(str(title))}</div>', unsafe_allow_html=True)

def status_pill(label: str, status: str = "pending") -> str:
    import html
    safe_status = status if status in {"ready", "pending", "demo", "blocked"} else "pending"
    return f'<span class="status-pill status-{safe_status}">{html.escape(str(label))}</span>'

def insight_card(title: str, body: str) -> None:
    import html
    import streamlit as st
    st.markdown(
        f'<div class="insight-card"><h4>{html.escape(str(title))}</h4><p>{html.escape(str(body))}</p></div>',
        unsafe_allow_html=True,
    )
