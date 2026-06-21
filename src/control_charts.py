import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.constants import get_spc_const


# ── Shewhart constants lookup ──────────────────────────────────────────────────

def _get(name: str, m: int) -> float:
    return get_spc_const(name, m)


# ── Variable (measurement) charts ─────────────────────────────────────────────

def xbar_r_chart(df: pd.DataFrame, sg_col: str, val_col: str) -> tuple[dict, dict]:
    sg = df.groupby(sg_col, sort=False)[val_col]
    means = sg.mean()
    ranges = sg.max() - sg.min()
    n = sg.count()
    m = int(n.mode().iloc[0])

    x_bar = means.mean()
    R_bar = ranges.mean()
    A2 = _get("A2", m)
    D3 = _get("D3", m)
    D4 = _get("D4", m)

    xbar_chart = pd.DataFrame({
        "point": means, "CL": x_bar,
        "LCL": x_bar - A2 * R_bar,
        "UCL": x_bar + A2 * R_bar,
    })
    r_chart = pd.DataFrame({
        "point": ranges, "CL": R_bar,
        "LCL": max(0.0, D3 * R_bar),
        "UCL": D4 * R_bar,
    })
    return xbar_chart.to_dict("list"), r_chart.to_dict("list")


def xbar_s_chart(df: pd.DataFrame, sg_col: str, val_col: str) -> tuple[dict, dict]:
    sg = df.groupby(sg_col, sort=False)[val_col]
    means = sg.mean()
    stds = sg.std(ddof=1)
    n = sg.count()
    m = int(n.mode().iloc[0])

    x_bar = means.mean()
    s_bar = stds.mean()
    A3 = _get("A3", m)
    B3 = _get("B3", m)
    B4 = _get("B4", m)

    xbar_chart = pd.DataFrame({
        "point": means, "CL": x_bar,
        "LCL": x_bar - A3 * s_bar,
        "UCL": x_bar + A3 * s_bar,
    })
    s_chart = pd.DataFrame({
        "point": stds, "CL": s_bar,
        "LCL": max(0.0, B3 * s_bar),
        "UCL": B4 * s_bar,
    })
    return xbar_chart.to_dict("list"), s_chart.to_dict("list")


def imr_chart(df: pd.DataFrame, sg_col: str, val_col: str, window: int = 2) -> tuple[dict, dict]:
    data = df.set_index(sg_col)[val_col]
    x_bar = data.mean()
    MR = data.rolling(window).apply(lambda x: x.max() - x.min(), raw=True)
    MR_bar = MR.mean()
    D3 = _get("D3", window)
    D4 = _get("D4", window)
    d2 = _get("d2", window)
    d2 = d2 if d2 else 1.128

    i_chart = pd.DataFrame({
        "point": data, "CL": x_bar,
        "LCL": x_bar - 3 * MR_bar / d2,
        "UCL": x_bar + 3 * MR_bar / d2,
    })
    mr_chart = pd.DataFrame({
        "point": MR, "CL": MR_bar,
        "LCL": max(0.0, D3 * MR_bar),
        "UCL": D4 * MR_bar,
    })
    return i_chart.to_dict("list"), mr_chart.to_dict("list")


# ── Attribute (count) charts ───────────────────────────────────────────────────

def np_chart(df: pd.DataFrame, sg_col: str, val_col: str) -> dict:
    sg = df.groupby(sg_col, sort=False)
    n_i = sg[val_col].count()
    np_i = sg[val_col].sum()
    np_bar = np_i.sum() / len(n_i)
    p_bar = np_i.sum() / n_i.sum()
    n_mode = int(n_i.mode().iloc[0])

    chart = pd.DataFrame({
        "point": np_i, "CL": np_bar,
        "LCL": max(0.0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar))),
        "UCL": np_bar + 3 * np.sqrt(np_bar * (1 - p_bar)),
    })
    return chart.to_dict("list")


def p_chart(df: pd.DataFrame, sg_col: str, val_col: str) -> dict:
    sg = df.groupby(sg_col, sort=False)
    n_i = sg[val_col].count()
    defects = sg[val_col].sum()
    p_i = defects / n_i
    p_bar = defects.sum() / n_i.sum()

    chart = pd.DataFrame({
        "point": p_i, "CL": p_bar,
        "LCL": (p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n_i)).clip(lower=0),
        "UCL": p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n_i),
    })
    return chart.to_dict("list")


def c_chart(df: pd.DataFrame, sg_col: str, val_col: str) -> dict:
    sg = df.groupby(sg_col, sort=False)
    c_i = sg[val_col].mean()
    c_bar = c_i.mean()

    chart = pd.DataFrame({
        "point": c_i, "CL": c_bar,
        "LCL": max(0.0, c_bar - 3 * np.sqrt(c_bar)),
        "UCL": c_bar + 3 * np.sqrt(c_bar),
    })
    return chart.to_dict("list")


def u_chart(df: pd.DataFrame, sg_col: str, val_col: str) -> dict:
    sg = df.groupby(sg_col, sort=False)
    n_i = sg[val_col].count()
    defects = sg[val_col].sum()
    u_i = defects / n_i
    u_bar = defects.sum() / n_i.sum()

    chart = pd.DataFrame({
        "point": u_i, "CL": u_bar,
        "LCL": (u_bar - 3 * np.sqrt(u_bar / n_i)).clip(lower=0),
        "UCL": u_bar + 3 * np.sqrt(u_bar / n_i),
    })
    return chart.to_dict("list")


# ── Anomaly detection ──────────────────────────────────────────────────────────

def detect_anomalies(chart: dict) -> list[int]:
    points = np.array(chart["point"])
    ucl = np.array(chart["UCL"])
    lcl = np.array(chart["LCL"])
    cl = np.array(chart["CL"])

    out = []
    for i, (p, u, l) in enumerate(zip(points, ucl, lcl)):
        if p > u or p < l:
            out.append(i)

    # Nelson rule 2: 9 consecutive on same side of CL
    sides = np.sign(points - cl)
    for i in range(8, len(sides)):
        window = sides[i-8:i+1]
        if np.all(window == 1) or np.all(window == -1):
            out.extend(range(i-8, i+1))

    # Nelson rule 3: 6 consecutive trending
    for i in range(5, len(points)):
        window = points[i-5:i+1]
        if np.all(np.diff(window) > 0) or np.all(np.diff(window) < 0):
            out.extend(range(i-5, i+1))

    return sorted(set(out))


def remove_anomalies(df: pd.DataFrame, sg_col: str, anomaly_indices: list[int]) -> pd.DataFrame:
    sgs = df[sg_col].unique()
    bad_sgs = [sgs[i] for i in anomaly_indices if i < len(sgs)]
    return df[~df[sg_col].isin(bad_sgs)].reset_index(drop=True)


# ── Plotting ───────────────────────────────────────────────────────────────────

def _single_control_chart(chart: dict, title: str, var_name: str = "value") -> go.Figure:
    idx = list(range(len(chart["point"])))
    anomalies = detect_anomalies(chart)

    colors = ["red" if i in anomalies else "steelblue" for i in idx]
    ucl_arr = chart["UCL"] if isinstance(chart["UCL"], list) else [chart["UCL"][0]] * len(idx)
    lcl_arr = chart["LCL"] if isinstance(chart["LCL"], list) else [chart["LCL"][0]] * len(idx)
    cl_arr = chart["CL"] if isinstance(chart["CL"], list) else [chart["CL"][0]] * len(idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=chart["point"], mode="lines+markers",
                             marker=dict(color=colors, size=8),
                             line=dict(color="steelblue"), name=var_name))
    fig.add_trace(go.Scatter(x=idx, y=cl_arr, mode="lines",
                             line=dict(color="green", dash="dashdot"), name="CL"))
    fig.add_trace(go.Scatter(x=idx, y=ucl_arr, mode="lines",
                             line=dict(color="magenta", dash="dot"), name="UCL"))
    fig.add_trace(go.Scatter(x=idx, y=lcl_arr, mode="lines",
                             line=dict(color="red", dash="dot"), name="LCL"))

    last = len(idx) - 1
    fig.add_annotation(x=last + 0.5, y=ucl_arr[-1], text=f"UCL={ucl_arr[-1]:.4f}",
                       showarrow=False, font=dict(color="magenta"))
    fig.add_annotation(x=last + 0.5, y=cl_arr[-1], text=f"CL={cl_arr[-1]:.4f}",
                       showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=last + 0.5, y=lcl_arr[-1], text=f"LCL={lcl_arr[-1]:.4f}",
                       showarrow=False, font=dict(color="red"))

    fig.update_layout(
        title=title,
        template="seaborn",
        width=800, height=350,
        showlegend=False,
        margin=dict(l=50, r=80, t=50, b=50),
    )
    return fig


def plot_variable_control_chart(
    charts: tuple[dict, dict], chart_type: str, var_name: str = "value"
) -> go.Figure:
    chart1, chart2 = charts
    if chart_type == "Xbar-R":
        t1, t2 = f"Xbar-R Control Chart of {var_name} (Xbar)", f"Xbar-R Control Chart of {var_name} (R)"
        y1, y2 = "Xbar", "R"
    elif chart_type == "Xbar-S":
        t1, t2 = f"Xbar-S Control Chart of {var_name} (Xbar)", f"Xbar-S Control Chart of {var_name} (S)"
        y1, y2 = "Xbar", "S"
    else:  # I-MR
        t1, t2 = f"I-MR Control Chart of {var_name} (I)", f"I-MR Control Chart of {var_name} (MR)"
        y1, y2 = "I", "MR"

    fig = make_subplots(rows=2, cols=1, subplot_titles=[t1, t2], shared_xaxes=True)
    for trace in _single_control_chart(chart1, t1, y1).data:
        fig.add_trace(trace, row=1, col=1)
    for trace in _single_control_chart(chart2, t2, y2).data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        title=f"Control Chart of {var_name}",
        template="seaborn", width=800, height=600,
        showlegend=False,
    )
    return fig


def recommend_chart(sg_size: int, data_type: str) -> str:
    if data_type == "계량형 (연속)":
        if sg_size == 1:
            return "I-MR"
        elif sg_size < 10:
            return "Xbar-R"
        else:
            return "Xbar-S"
    else:  # 계수형
        return "NP/P/C/U 중 선택"
