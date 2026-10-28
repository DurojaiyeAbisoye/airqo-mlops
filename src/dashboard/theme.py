import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

FONT = "Inter, -apple-system, 'Segoe UI', Roboto, sans-serif"
INK = "#0F172A"
MUTED = "#64748B"
LINE = "#E2E8F0"

# roles: ACCENT = the model / the primary series, INK = actual observations,
# NEUTRAL = baselines and reference lines, GOOD / WARN / BAD = status only
ACCENT = "#2563EB"
NEUTRAL = "#94A3B8"
HINDCAST = "#C026D3"  # purple: past predictions on the AQI chart
GOOD = "#059669"
WARN = "#D97706"
BAD = "#DC2626"

# one colour per feature column, reused wherever that column appears.
# red and green are left out so they only ever mean bad / good
CATEGORICAL = [
    "#2563EB",  # blue
    "#7C3AED",  # violet
    "#0891B2",  # cyan
    "#DB2777",  # pink
    "#64748B",  # slate
    "#CA8A04",  # ochre
]

CSS = """
.block-container { padding-top: 2.25rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }

h1 { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; padding-bottom: 0.25rem; }
h2 { font-size: 1.2rem; font-weight: 600; letter-spacing: -0.005em; }
h3 { font-size: 1.05rem; font-weight: 600; }
[data-testid="stCaptionContainer"] { color: #64748B; }

button[data-baseweb="tab"] p { font-size: 0.95rem; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] p { font-weight: 600; }
[data-baseweb="tab-list"] { gap: 0.5rem; }
[data-baseweb="tab-highlight"] { height: 2px; }
[data-baseweb="tab-panel"] { padding-top: 1.25rem; }

[data-testid="stMetricLabel"] p {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #64748B;
}
[data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stMetricDelta"] { font-size: 0.8rem; }
"""


def _axis(grid: bool) -> dict:
    return dict(
        showgrid=grid,
        gridcolor=LINE,
        zeroline=False,
        showline=not grid,
        linecolor=LINE,
        ticks="outside" if not grid else "",
        tickcolor=LINE,
        tickfont=dict(size=11, color=MUTED),
        title=dict(font=dict(size=11, color=MUTED)),
    )


def _register_template() -> None:
    pio.templates["dashboard"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT, size=12, color=INK),
            colorway=CATEGORICAL,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(font=dict(size=13, color=INK), x=0, xanchor="left"),
            margin=dict(l=48, r=16, t=56, b=44),
            xaxis=_axis(grid=False),
            yaxis=_axis(grid=True),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color=MUTED),
            ),
            hovermode="x unified",
            hoverlabel=dict(
                font=dict(family=FONT, size=12),
                bgcolor="white",
                bordercolor=LINE,
            ),
            bargap=0.15,
        )
    )
    pio.templates.default = "dashboard"


def apply_theme() -> None:
    _register_template()
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def donut(
    labels,
    values,
    colors,
    title: str,
    center_value: str = "",
    center_label: str = "",
) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=list(labels),
            values=list(values),
            marker=dict(colors=list(colors), line=dict(color="white", width=2)),
            hole=0.62,
            sort=False,
            direction="clockwise",
            textinfo="none",
            hovertemplate="%{label}<br>%{value:,.0f} (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
        ),
    )
    if center_value:
        fig.add_annotation(
            text=center_value,
            x=0.5,
            y=0.53,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=22, color=INK),
        )
    if center_label:
        fig.add_annotation(
            text=center_label,
            x=0.5,
            y=0.44,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=11, color=MUTED),
        )
    return fig


def show(fig: go.Figure, height: int = 320) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(
        fig,
        theme=None,
        width="stretch",
        config={"displayModeBar": False},
    )
