import pandas as pd
import numpy as np


def generate_sample_data(
    var_name: str,
    target: float,
    tolerance: float,
    sg_name: str,
    num_sg: int,
    sg_size: int,
    sg_std: float,
    mean_shift: tuple[float, float] = (0.0, 0.0),
    sg_size_variation: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, float, float]:
    np.random.seed(seed)
    LSL = target - tolerance
    USL = target + tolerance
    rows = []
    for i in range(num_sg):
        shift = np.random.uniform(mean_shift[0], mean_shift[1])
        n_i = sg_size + np.random.randint(-sg_size_variation, sg_size_variation + 1)
        n_i = max(1, n_i)
        vals = np.random.normal(loc=target + shift, scale=sg_std, size=n_i)
        for v in vals:
            rows.append({sg_name: f"{sg_name}_{i+1}", var_name: v})
    return pd.DataFrame(rows), LSL, USL


def parse_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def to_long(df: pd.DataFrame, sg_col: str, val_col: str) -> pd.DataFrame:
    """Already long — just rename if needed."""
    return df[[sg_col, val_col]].copy()


def to_wide(df: pd.DataFrame, sg_col: str, val_col: str) -> pd.DataFrame:
    df = df.copy()
    df["_idx"] = df.groupby(sg_col).cumcount()
    return df.pivot(index="_idx", columns=sg_col, values=val_col)
