"""Excel ingestion for the three replication workbooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trade_agsi.repo import resolve_path


@dataclass
class D1Bundle:
    treaty_catalog: pd.DataFrame
    provisions: pd.DataFrame
    expert_subset: pd.DataFrame


@dataclass
class D2Bundle:
    firm_year: pd.DataFrame
    jurisdiction_master: pd.DataFrame
    jurisdiction_year_omega: pd.DataFrame


@dataclass
class D3Bundle:
    nu_y_aggregate: pd.DataFrame
    policy_initiatives: pd.DataFrame | None = None


def load_d1_workbook(path: str | Path) -> D1Bundle:
    p = resolve_path(path)
    xl = pd.ExcelFile(p)
    return D1Bundle(
        treaty_catalog=pd.read_excel(xl, sheet_name="treaty_catalog"),
        provisions=pd.read_excel(xl, sheet_name="provisions"),
        expert_subset=pd.read_excel(xl, sheet_name="expert_subset_1500"),
    )


def load_d2_workbook(path: str | Path) -> D2Bundle:
    p = resolve_path(path)
    xl = pd.ExcelFile(p)
    return D2Bundle(
        firm_year=pd.read_excel(xl, sheet_name="firm_year_panel"),
        jurisdiction_master=pd.read_excel(xl, sheet_name="jurisdiction_master"),
        jurisdiction_year_omega=pd.read_excel(xl, sheet_name="jurisdiction_year_omega"),
    )


def load_d3_workbook(path: str | Path, include_policy_raw: bool = False) -> D3Bundle:
    p = resolve_path(path)
    xl = pd.ExcelFile(p)
    nu = pd.read_excel(xl, sheet_name="nu_Y_theme_aggregate")
    pol = pd.read_excel(xl, sheet_name="policy_initiatives_raw") if include_policy_raw else None
    return D3Bundle(nu_y_aggregate=nu, policy_initiatives=pol)


def assert_required_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}; found {list(df.columns)}")
