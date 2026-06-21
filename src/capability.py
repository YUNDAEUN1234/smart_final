import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, boxcox
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.constants import get_cap_const


def normality_test(data: pd.Series) -> tuple[float, float, bool]:
    stat, p = shapiro(data.dropna())
    return stat, p, p >= 0.05


def boxcox_transform(data: pd.Series) -> tuple[np.ndarray, float, float, float]:
    arr = data.dropna().values
    transformed, lam = boxcox(arr)
    stat, p = shapiro(transformed)
    return transformed, lam, stat, p


def calc_unbiased_const(name: str, n: int) -> float:
    return get_cap_const(name, n)


def process_capability(
    df: pd.DataFrame, sg_col: str, val_col: str, LSL: float, USL: float
) -> dict:
    groups = df.groupby(sg_col)[val_col]
    mean_sg = groups.mean()
    std_sg = groups.std(ddof=1)
    n_sg = groups.count()

    x_bar = mean_sg.mean()
    sigma_hat = std_sg.mean()

    # within-subgroup sigma via c4 unbiasing
    n_mode = int(n_sg.mode().iloc[0]) if len(n_sg) > 0 else 1
    c4 = calc_unbiased_const("c4", n_mode)
    sigma_within = sigma_hat / c4 if c4 else sigma_hat

    # overall sigma
    sigma_overall_raw = df[val_col].std(ddof=1)
    c4_overall = calc_unbiased_const("c4", len(df))
    sigma_overall = sigma_overall_raw / c4_overall if c4_overall else sigma_overall_raw

    Cp = (USL - LSL) / (6 * sigma_within)
    Cpk = min((USL - x_bar) / (3 * sigma_within), (x_bar - LSL) / (3 * sigma_within))
    Pp = (USL - LSL) / (6 * sigma_overall)
    Ppk = min((USL - x_bar) / (3 * sigma_overall), (x_bar - LSL) / (3 * sigma_overall))

    return {
        "x_bar": x_bar,
        "sigma_within": sigma_within,
        "sigma_overall": sigma_overall,
        "Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk,
        "LSL": LSL, "USL": USL,
        "mean_sg": mean_sg, "std_sg": std_sg,
    }


def capability_grade(val: float) -> tuple[str, str]:
    if val >= 1.67:
        return "매우 우수", "#2ca02c"
    elif val >= 1.33:
        return "우수", "#1f77b4"
    elif val >= 1.0:
        return "보통", "#ff7f0e"
    else:
        return "미흡 (개선 필요)", "#d62728"


def plot_boxplot(df: pd.DataFrame, sg_col: str, val_col: str, LSL: float, USL: float) -> go.Figure:
    fig = px.box(
        df, x=sg_col, y=val_col,
        color=sg_col,
        title=f"{val_col.capitalize()} Boxplot by Subgroup",
        points="all",
    )
    fig.add_hline(y=LSL, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_hline(y=USL, line_dash="dash", line_color="red", annotation_text="USL")
    fig.update_layout(width=800, height=400, showlegend=False)
    return fig


def plot_histogram(df: pd.DataFrame, sg_col: str, val_col: str, LSL: float, USL: float) -> go.Figure:
    fig = px.histogram(
        df, x=val_col, color=sg_col,
        facet_row=sg_col,
        title=f"{val_col.capitalize()} Histogram",
        nbins=20, opacity=0.7,
    )
    fig.add_vline(x=LSL, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=USL, line_dash="dash", line_color="red", annotation_text="USL")
    fig.update_layout(width=800, height=500)
    return fig


def plot_qq(data: pd.Series, title: str = "Q-Q Plot") -> go.Figure:
    arr = data.dropna().values
    z = stats.zscore(arr)
    (x, y), _ = stats.probplot(z, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data"))
    lim = max(abs(min(x)), abs(max(x))) + 0.5
    fig.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                  line=dict(color="red", width=2))
    fig.update_layout(title=title,
                      xaxis_title="Theoretical Quantiles",
                      yaxis_title="Sample Quantiles",
                      width=400, height=400)
    return fig


def plot_process_capability(
    df: pd.DataFrame, sg_col: str, val_col: str, LSL: float, USL: float, cap: dict
) -> go.Figure:
    x_norm = np.linspace(min(df[val_col].min(), LSL) - 1,
                         max(df[val_col].max(), USL) + 1, 1000)
    y_norm = stats.norm.pdf(x_norm, loc=cap["x_bar"], scale=cap["sigma_within"])

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        shared_xaxes=True,
        subplot_titles=["Subgroup Boxplots", f"{val_col.capitalize()} Process Capability"],
    )

    sgs = df[sg_col].unique()
    colors = px.colors.qualitative.Plotly
    for i, sg in enumerate(sgs):
        d = df[df[sg_col] == sg][val_col]
        fig.add_trace(go.Box(x=d, name=str(sg), marker_color=colors[i % len(colors)],
                             orientation="h", showlegend=True), row=1, col=1)

    for i, sg in enumerate(sgs):
        d = df[df[sg_col] == sg][val_col]
        fig.add_trace(go.Histogram(x=d, name=str(sg), marker_color=colors[i % len(colors)],
                                   opacity=0.5, nbinsx=20, showlegend=False), row=2, col=1)

    fig.add_vline(x=LSL, line_dash="dash", line_color="red", annotation_text="LSL")
    fig.add_vline(x=USL, line_dash="dash", line_color="red", annotation_text="USL")

    annotation_text = (
        f"Cp={cap['Cp']:.4f}<br>Cpk={cap['Cpk']:.4f}<br>"
        f"Pp={cap['Pp']:.4f}<br>Ppk={cap['Ppk']:.4f}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.05,
        text=annotation_text, showarrow=False,
        bgcolor="white", bordercolor="black", align="right",
        font=dict(size=11),
    )
    fig.update_layout(
        width=800, height=500,
        title=f"{val_col.capitalize()} Process Capability Analysis",
        barmode="overlay",
    )
    return fig
