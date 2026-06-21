import os
import pandas as pd
import numpy as np
from scipy.special import gamma

_DIR = os.path.dirname(os.path.dirname(__file__))
_CAP_CSV = os.path.join(_DIR, "data", "unbiased_capability_analysis.csv")
_SPC_CSV = os.path.join(_DIR, "data", "unbiased_control_chart.csv")

_cap_table: pd.DataFrame | None = None
_spc_table: pd.DataFrame | None = None


def _load_cap():
    global _cap_table
    if _cap_table is None:
        _cap_table = pd.read_csv(_CAP_CSV, encoding="utf-8-sig")
        _cap_table.columns = _cap_table.columns.str.strip()
    return _cap_table


def _load_spc():
    global _spc_table
    if _spc_table is None:
        _spc_table = pd.read_csv(_SPC_CSV, encoding="utf-8-sig")
        _spc_table.columns = _spc_table.columns.str.strip()
    return _spc_table


def _c4_approx(n: int) -> float:
    if n <= 1:
        return float("nan")
    return np.sqrt(2 / (n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))


def get_cap_const(name: str, n: int) -> float:
    table = _load_cap()
    row = table[table["N"] == n]
    if not row.empty and name in row.columns:
        return float(row[name].values[0])
    # fallback approximations for n > 50
    if name == "d2":
        return 3.4873 + 0.0258414 * n - 0.00000823 * n**2
    if name == "d3":
        return 0.88818 + 0.051871 * n + 0.00000506 * n**2 - 0.00000019 * n**3
    if name == "d4":
        return 2.88606 + 0.051313 * n - 0.00049243 * n**2 + 0.00000188 * n**3
    if name == "c4":
        return _c4_approx(n)
    return float("nan")


def get_spc_const(name: str, m: int) -> float:
    table = _load_spc()
    row = table[table["m"] == m]
    if not row.empty and name in row.columns:
        val = row[name].values[0]
        return float(val)
    # c4 fallback
    if name == "c4":
        return _c4_approx(m)
    return float("nan")
