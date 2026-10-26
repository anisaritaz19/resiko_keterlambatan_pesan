"""Pemuatan data mentah, rekayasa fitur, dan preprocessor scikit-learn."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (CATEGORICAL_FEATURES, CSV_ENCODING, FEATURES,
                    NUMERIC_FEATURES, RAW_COLUMNS, RAW_TO_FEATURE)


def _clean_name(name: str) -> str:
    return name.replace("\ufeff", "").replace("ï»¿", "").strip()


def load_raw(path) -> pd.DataFrame:
    """Baca hanya kolom yang dibutuhkan (hemat memori)."""
    header = pd.read_csv(path, nrows=0, encoding=CSV_ENCODING).columns
    cleaned = {col: _clean_name(col) for col in header}
    missing = set(RAW_COLUMNS) - set(cleaned.values())
    if missing:
        raise ValueError(f"Kolom wajib tidak ada di dataset: {sorted(missing)}")
    wanted = [orig for orig, c in cleaned.items() if c in RAW_COLUMNS]
    df = pd.read_csv(path, encoding=CSV_ENCODING, usecols=wanted)
    return df.rename(columns=cleaned)


def build_dataframe(raw: pd.DataFrame):
    """Ubah data mentah menjadi tabel fitur siap model.

    Mengembalikan (df, info). `info` mencatat baris yang dibuang.
    """
    df = raw.rename(columns=RAW_TO_FEATURE).copy()
    info = {"rows_raw": int(len(df))}

    # Target
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    bad_target = ~df["target"].isin([0, 1])
    info["rows_dropped_invalid_target"] = int(bad_target.sum())
    df = df.loc[~bad_target].copy()
    df["target"] = df["target"].astype(int)

    # Tanggal pesanan -> bulan, hari, jam
    raw_dates = df["order_date"]
    dates = pd.to_datetime(raw_dates, format="%m/%d/%Y %H:%M", errors="coerce")
    retry = dates.isna()
    if retry.any():
        dates.loc[retry] = pd.to_datetime(raw_dates[retry], format="mixed",
                                          errors="coerce")
    bad_date = dates.isna()
    info["rows_dropped_invalid_date"] = int(bad_date.sum())
    df = df.loc[~bad_date].copy()
    dates = dates.loc[~bad_date]
    df["order_month"] = dates.dt.month.astype(int).astype(str)
    df["order_weekday"] = dates.dt.dayofweek.astype(int).astype(str)
    df["order_hour"] = dates.dt.hour.astype(float)

    # Numerik
    for col in ["scheduled_days", "quantity", "discount_rate", "product_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Kategorikal: rapikan spasi, kosong -> NaN (diisi imputer di pipeline)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(object).map(
            lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace("", np.nan)

    df = df[FEATURES + ["target", "order_id"]].reset_index(drop=True)
    info["rows_used"] = int(len(df))
    return df, info


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                              ("cat", categorical, CATEGORICAL_FEATURES)])
