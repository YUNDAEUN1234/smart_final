import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, boxcox
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.constants import get_cap_const


def normality_test(data: pd.Series) -> tuple[float, float, bool]:
    arr = data.dropna()
    if len(arr) < 3:
        return 0.0, 0.0, False
    stat, p = shapiro(arr)
    return stat, p, p >= 0.05


def boxcox_transform(data: pd.Series) -> tuple[np.ndarray, float, float, float, float]:
    arr = data.dropna().values
    shift = 0.0
    if arr.min() <= 0:
        shift = abs(arr.min()) + 1e-6
        arr = arr + shift
    transformed, lam = boxcox(arr)
    stat, p = shapiro(transformed)
    return transformed, lam, stat, p, shift


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

    n_mode = int(n_sg.mode().iloc[0]) if len(n_sg) > 0 else 2

    if n_mode < 2:
        sorted_means = mean_sg.sort_index().values
        if len(sorted_means) >= 2:
            mr = np.abs(np.diff(sorted_means))
            mr_bar = mr.mean()
            d2 = calc_unbiased_const("d2", 2)
            d2 = d2 if d2 and not np.isnan(d2) else 1.128
            sigma_within = mr_bar / d2
        else:
            sigma_within = float("nan")
    else:
        c4 = calc_unbiased_const("c4", n_mode)
        sigma_within = sigma_hat / c4 if c4 and not np.isnan(c4) else sigma_hat

    sigma_overall_raw = df[val_col].std(ddof=1)
    n_total = max(len(df), 2)
    c4_overall = calc_unbiased_const("c4", n_total)
    sigma_overall = sigma_overall_raw / c4_overall if c4_overall and not np.isnan(c4_overall) else sigma_overall_raw

    _nan = float("nan")
    if sigma_within == 0 or np.isnan(sigma_within):
        Cp, Cpk = _nan, _nan
    else:
        Cp = (USL - LSL) / (6 * sigma_within)
        Cpk = min((USL - x_bar) / (3 * sigma_within), (x_bar - LSL) / (3 * sigma_within))

    if sigma_overall == 0 or np.isnan(sigma_overall):
        Pp, Ppk = _nan, _nan
    else:
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
    if np.isnan(val) or np.isinf(val):
        return "산출 불가", "#999999"
    if val >= 1.67:
        return "매우 우수", "#2ca02c"
    elif val >= 1.33:
        return "우수", "#1f77b4"
    elif val >= 1.0:
        return "보통", "#ff7f0e"
    else:
        return "미흡 (개선 필요)", "#d62728"


def fmt_cap(val: float) -> str:
    if np.isnan(val) or np.isinf(val):
        return "N/A"
    if abs(val) > 99.99:
        return "> 99"
    return f"{val:.4f}"


def _auto_nbins(n: int) -> int:
    return max(10, min(50, int(np.ceil(np.log2(n) + 1))))


def _smart_range(data_values, LSL, USL):
    """Axis range focused on data. Include spec limits only when close."""
    dmin, dmax = float(data_values.min()), float(data_values.max())
    spread = dmax - dmin
    if spread < 1e-10:
        spread = abs(dmin) * 0.1 or 1.0

    pad = spread * 0.3
    lo, hi = dmin - pad, dmax + pad

    lsl_dist = min(abs(dmin - LSL), abs(dmax - LSL))
    usl_dist = min(abs(dmin - USL), abs(dmax - USL))
    threshold = spread * 3

    lsl_in = lsl_dist < threshold
    usl_in = usl_dist < threshold

    if lsl_in:
        lo = min(lo, LSL - pad)
        hi = max(hi, LSL + pad)
    if usl_in:
        hi = max(hi, USL + pad)
        lo = min(lo, USL - pad)

    return lo, hi, lsl_in, usl_in


def _add_spec_lines_y(fig, LSL, USL, lo, hi, lsl_in, usl_in):
    if lsl_in:
        fig.add_hline(y=LSL, line_dash="dash", line_color="red",
                      annotation_text="LSL")
    else:
        arrow = "↓" if LSL < lo else "↑"
        y_pos = 0.02 if LSL < lo else 0.98
        fig.add_annotation(
            x=1.02, y=y_pos, xref="paper", yref="paper",
            text=f"{arrow} LSL={LSL}", showarrow=False,
            font=dict(color="red", size=11), xanchor="left",
        )
    if usl_in:
        fig.add_hline(y=USL, line_dash="dash", line_color="red",
                      annotation_text="USL")
    else:
        arrow = "↓" if USL < lo else "↑"
        y_pos = 0.06 if USL < lo else 0.94
        fig.add_annotation(
            x=1.02, y=y_pos, xref="paper", yref="paper",
            text=f"{arrow} USL={USL}", showarrow=False,
            font=dict(color="red", size=11), xanchor="left",
        )


def _add_spec_lines_x(fig, LSL, USL, lo, hi, lsl_in, usl_in):
    if lsl_in:
        fig.add_vline(x=LSL, line_dash="dash", line_color="red",
                      annotation_text="LSL")
    else:
        arrow = "←" if LSL < lo else "→"
        x_pos = 0.02 if LSL < lo else 0.98
        fig.add_annotation(
            x=x_pos, y=1.02, xref="paper", yref="paper",
            text=f"{arrow} LSL={LSL}", showarrow=False,
            font=dict(color="red", size=11), yanchor="bottom",
        )
    if usl_in:
        fig.add_vline(x=USL, line_dash="dash", line_color="red",
                      annotation_text="USL")
    else:
        arrow = "←" if USL < lo else "→"
        x_pos = 0.15 if USL < lo else 0.85
        fig.add_annotation(
            x=x_pos, y=1.02, xref="paper", yref="paper",
            text=f"{arrow} USL={USL}", showarrow=False,
            font=dict(color="red", size=11), yanchor="bottom",
        )


def plot_boxplot(df: pd.DataFrame, sg_col: str, val_col: str,
                 LSL: float, USL: float) -> go.Figure:
    fig = px.box(df, x=sg_col, y=val_col,
                 title=f"{val_col.capitalize()} Boxplot by Subgroup",
                 points="all")
    data_vals = df[val_col].dropna()
    lo, hi, lsl_in, usl_in = _smart_range(data_vals, LSL, USL)
    _add_spec_lines_y(fig, LSL, USL, lo, hi, lsl_in, usl_in)
    fig.update_yaxes(range=[lo, hi])
    fig.update_layout(width=800, height=400, showlegend=False)
    return fig


def plot_histogram(df: pd.DataFrame, sg_col: str, val_col: str,
                   LSL: float, USL: float) -> go.Figure:
    data_vals = df[val_col].dropna()
    lo, hi, lsl_in, usl_in = _smart_range(data_vals, LSL, USL)

    n_bins = _auto_nbins(len(data_vals))
    n_sg = df[sg_col].nunique()
    if n_sg <= 8:
        fig = px.histogram(df, x=val_col, color=sg_col,
                           title=f"{val_col.capitalize()} Histogram",
                           nbins=n_bins, opacity=0.7, barmode="overlay")
    else:
        fig = px.histogram(df, x=val_col,
                           title=f"{val_col.capitalize()} Histogram",
                           nbins=n_bins, opacity=0.7)

    _add_spec_lines_x(fig, LSL, USL, lo, hi, lsl_in, usl_in)
    fig.update_xaxes(range=[lo, hi])
    fig.update_layout(width=800, height=400)
    return fig


def plot_qq(data: pd.Series, title: str = "Q-Q Plot") -> go.Figure:
    arr = data.dropna().values
    if len(arr) < 3 or np.std(arr) == 0:
        fig = go.Figure()
        fig.update_layout(title=title + " (데이터 부족 또는 분산=0)",
                          width=400, height=400)
        return fig
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
    df: pd.DataFrame, sg_col: str, val_col: str,
    LSL: float, USL: float, cap: dict,
) -> go.Figure:
    col_data = df[val_col].dropna()
    lo, hi, lsl_in, usl_in = _smart_range(col_data, LSL, USL)

    sw = cap["sigma_within"]
    if np.isnan(sw) or sw == 0:
        sw = col_data.std(ddof=1) or 1.0
    x_curve = np.linspace(lo, hi, 500)
    y_curve = stats.norm.pdf(x_curve, loc=cap["x_bar"], scale=sw)

    fig = go.Figure()

    n_bins = _auto_nbins(len(col_data))
    fig.add_trace(go.Histogram(x=col_data, nbinsx=n_bins, opacity=0.7,
                               marker_color="#1f77b4", name="Data"))

    bin_width = (col_data.max() - col_data.min()) / n_bins
    if bin_width > 0:
        y_scaled = y_curve * len(col_data) * bin_width
        fig.add_trace(go.Scatter(x=x_curve, y=y_scaled, mode="lines",
                                 line=dict(color="darkblue", width=2),
                                 name="Normal fit"))

    _add_spec_lines_x(fig, LSL, USL, lo, hi, lsl_in, usl_in)

    fig.add_vline(x=cap["x_bar"], line_dash="dot", line_color="green",
                  annotation_text=f"x̄={cap['x_bar']:.4f}")

    fig.update_xaxes(range=[lo, hi])

    annotation_text = (
        f"Cp={fmt_cap(cap['Cp'])}<br>Cpk={fmt_cap(cap['Cpk'])}<br>"
        f"Pp={fmt_cap(cap['Pp'])}<br>Ppk={fmt_cap(cap['Ppk'])}"
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.98, y=0.05,
        text=annotation_text, showarrow=False,
        bgcolor="white", bordercolor="black", align="right",
        font=dict(size=11),
    )
    fig.update_layout(
        width=800, height=450,
        title=f"{val_col.capitalize()} Process Capability Analysis",
        barmode="overlay",
        xaxis_title=val_col,
        yaxis_title="Count",
    )
    return fig
