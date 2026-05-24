"""Phase K — синтетические признаки (отношения, log1p, Ro5-бинарники)."""
from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / np.maximum(den.astype(float), _EPS)


_INTERACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("MolLogP", "TPSA"),
    ("Chi1", "FractionCSP3"),
    ("MolLogP", "MolWt"),
    ("TPSA", "FractionCSP3"),
    ("BertzCT", "MolLogP"),
    ("HallKierAlpha", "TPSA"),
    ("LabuteASA", "MolLogP"),
    ("Chi0", "MolWt"),
    ("MolLogP", "FractionCSP3"),
    ("Chi1", "TPSA"),
)


def _peoe_polar_proxy(X: pd.DataFrame) -> pd.Series:
    """Сумма PEOE_VSA — proxy для PEOE_VSA_POL из PDF."""
    peoe = [c for c in X.columns if c.startswith("PEOE_VSA")]
    if not peoe:
        return pd.Series(1.0, index=X.index)
    return X[peoe].astype(float).sum(axis=1)


def engineer_features(
    X: pd.DataFrame,
    *,
    ratios: bool = True,
    ratios_v2: bool = False,
    log1p_cols: bool = False,
    binary_rules: bool = False,
    ro5_rules: bool = False,
    interactions: bool = False,
) -> pd.DataFrame:
    """Добавляет fe_* колонки к копии X (исходные 192 сохраняются)."""
    out = X.copy()

    if ratios:
        out["fe_logp_tpsa"] = _ratio(X["MolLogP"], X["TPSA"])
        out["fe_tpsa_molwt"] = _ratio(X["TPSA"], X["MolWt"])
        out["fe_bertz_tpsa"] = _ratio(X["BertzCT"], X["TPSA"])
        out["fe_fcsp3_tpsa"] = _ratio(X["FractionCSP3"], X["TPSA"])

    if ratios_v2:
        out["fe_chi0_logp"] = _ratio(X["Chi0"], X["MolLogP"])
        out["fe_complex_tpsa"] = _ratio(X["BertzCT"] + X["BalabanJ"], X["TPSA"])
        out["fe_rot_labute"] = _ratio(X["NumRotatableBonds"], X["LabuteASA"])
        charge_span = X["MaxAbsPartialCharge"] - X["MinAbsPartialCharge"]
        estate_span = X["MaxEStateIndex"] - X["MinEStateIndex"]
        out["fe_charge_idx"] = _ratio(charge_span, estate_span)
        out["fe_kappa3_peoe"] = _ratio(X["Kappa3"], _peoe_polar_proxy(X))

    if log1p_cols:
        for col in ("MolWt", "MolLogP", "BertzCT"):
            out[f"fe_log1p_{col}"] = np.log1p(np.clip(X[col].astype(float), 0, None))

    if binary_rules:
        out["fe_ro5_molwt"] = (X["MolWt"] < 500).astype(float)
        out["fe_high_tpsa"] = (X["TPSA"] > 140).astype(float)
        out["fe_high_fcsp3"] = (X["FractionCSP3"] > 0.4).astype(float)

    if ro5_rules:
        out["fe_ro5_molwt"] = (X["MolWt"] <= 500).astype(float)
        out["fe_ro5_logp"] = (X["MolLogP"] <= 5).astype(float)
        out["fe_ro5_hbd"] = (X["NumHDonors"] <= 5).astype(float)
        out["fe_ro5_hba"] = (X["NumHAcceptors"] <= 10).astype(float)
        out["fe_ro5_rot"] = (X["NumRotatableBonds"] <= 10).astype(float)
        out["fe_high_tpsa"] = (X["TPSA"] <= 140).astype(float)
        out["fe_ro5_fcsp3"] = (X["FractionCSP3"] >= 0.35).astype(float)
        out["fe_low_rings"] = (X["RingCount"] <= 5).astype(float)

    if interactions:
        for a, b in _INTERACTION_PAIRS:
            out[f"fe_{a}_x_{b}"] = X[a].astype(float) * X[b].astype(float)

    return out


def fe_column_names(
    *,
    ratios: bool = True,
    ratios_v2: bool = False,
    log1p_cols: bool = False,
    binary_rules: bool = False,
    ro5_rules: bool = False,
    interactions: bool = False,
) -> list[str]:
    cols: list[str] = []
    if ratios:
        cols += [
            "fe_logp_tpsa",
            "fe_tpsa_molwt",
            "fe_bertz_tpsa",
            "fe_fcsp3_tpsa",
        ]
    if ratios_v2:
        cols += [
            "fe_chi0_logp",
            "fe_complex_tpsa",
            "fe_rot_labute",
            "fe_charge_idx",
            "fe_kappa3_peoe",
        ]
    if log1p_cols:
        cols += [f"fe_log1p_{c}" for c in ("MolWt", "MolLogP", "BertzCT")]
    if binary_rules:
        cols += ["fe_ro5_molwt", "fe_high_tpsa", "fe_high_fcsp3"]
    if ro5_rules:
        cols += [
            "fe_ro5_molwt",
            "fe_ro5_logp",
            "fe_ro5_hbd",
            "fe_ro5_hba",
            "fe_ro5_rot",
            "fe_high_tpsa",
            "fe_ro5_fcsp3",
            "fe_low_rings",
        ]
    if interactions:
        cols += [f"fe_{a}_x_{b}" for a, b in _INTERACTION_PAIRS]
    return cols


def engineer_features_cc50(X: pd.DataFrame) -> pd.DataFrame:
    """CC50-ориентированные ratio fe (полярность / размер / HBD)."""
    out = X.copy()
    out["fe_cc50_tpsa_logp"] = _ratio(X["TPSA"], X["MolLogP"])
    out["fe_cc50_molwt_tpsa"] = _ratio(X["MolWt"], X["TPSA"])
    out["fe_cc50_logp_molwt"] = _ratio(X["MolLogP"], X["MolWt"])
    out["fe_cc50_donors_tpsa"] = _ratio(X["NumHDonors"], X["TPSA"])
    return out


def cc50_fe_column_names() -> list[str]:
    return [
        "fe_cc50_tpsa_logp",
        "fe_cc50_molwt_tpsa",
        "fe_cc50_logp_molwt",
        "fe_cc50_donors_tpsa",
    ]


PHYSCHEM_CC50_COLS = (
    "TPSA",
    "MolLogP",
    "MolWt",
    "NumHDonors",
    "NumHAcceptors",
    "HeavyAtomCount",
    "FractionCSP3",
    "RingCount",
    "NumRotatableBonds",
    "LabuteASA",
)


def build_physchem_trans_frame(
    X: pd.DataFrame,
    *,
    with_ic50_ratios: bool = True,
    with_cc50_ratios: bool = False,
) -> pd.DataFrame:
    """Компактный transductive input для CC50 (~10–14 cols)."""
    cols = [c for c in PHYSCHEM_CC50_COLS if c in X.columns]
    out = X[cols].astype(float).copy()
    if with_ic50_ratios:
        fe = engineer_features(X, ratios=True)
        for c in fe_column_names(ratios=True):
            out[c] = fe[c]
    if with_cc50_ratios:
        fe = engineer_features_cc50(X)
        for c in cc50_fe_column_names():
            out[c] = fe[c]
    return out
