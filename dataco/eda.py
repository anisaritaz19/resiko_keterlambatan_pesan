"""Analisis data eksploratif (EDA): ringkasan statistik + grafik."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (EDA_DIR, EDA_SUMMARY_PATH, MONTH_NAMES,  # noqa: E402
                    NUMERIC_FEATURES, WEEKDAY_NAMES)

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False})
COLOR_LATE = "#d9534f"
COLOR_OK = "#4a90a4"


def _save(fig, name, figures, title, caption):
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(EDA_DIR / name)
    plt.close(fig)
    figures.append({"file": name, "title": title, "caption": caption})


def _rate_table(df, col, top=None):
    t = (df.groupby(col)["target"].agg(["mean", "count"])
         .rename(columns={"mean": "late_rate", "count": "n"})
         .sort_values("n", ascending=False))
    if top:
        t = t.head(top)
    return t


def _bar_rate(df, col, label, name, figures, top=None, order=None, labels=None):
    t = _rate_table(df, col, top)
    if order is not None:
        t = t.reindex([o for o in order if o in t.index])
    else:
        t = t.sort_values("late_rate")
    names = [labels.get(i, i) if labels else str(i) for i in t.index]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.38 * len(t) + 1)))
    ax.barh(names, t["late_rate"] * 100, color=COLOR_LATE)
    overall = df["target"].mean() * 100
    ax.axvline(overall, color="black", ls="--", lw=1,
               label=f"Rata-rata keseluruhan ({overall:.1f}%)")
    ax.set_xlabel("Persentase pengiriman terlambat (%)")
    ax.set_title(f"Tingkat keterlambatan menurut {label}")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, name, figures, f"Keterlambatan menurut {label}",
          f"Persentase baris dengan Late_delivery_risk = 1 pada tiap {label.lower()}. "
          "Garis putus-putus adalah rata-rata keseluruhan.")
    return t


def run_eda(df: pd.DataFrame, info: dict) -> dict:
    """Jalankan EDA pada tabel fitur hasil `build_dataframe`."""
    figures = []
    summary = {
        "n_rows": int(len(df)),
        "n_orders": int(df["order_id"].nunique()),
        "cleaning": info,
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_rows": int(df.drop(columns=["order_id"]).duplicated().sum()),
        "numeric_describe": json.loads(
            df[NUMERIC_FEATURES].describe().round(4).to_json()),
    }

    # Distribusi target
    counts = df["target"].value_counts().sort_index()
    summary["target_distribution"] = {
        "tidak_terlambat": int(counts.get(0, 0)),
        "terlambat": int(counts.get(1, 0)),
        "late_rate": float(df["target"].mean()),
    }
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["Tidak terlambat (0)", "Terlambat (1)"],
           [counts.get(0, 0), counts.get(1, 0)], color=[COLOR_OK, COLOR_LATE])
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Jumlah baris")
    ax.set_title("Distribusi target")
    _save(fig, "target_distribution.png", figures, "Distribusi target",
          "Jumlah baris untuk masing-masing kelas. Digunakan untuk menilai "
          "keseimbangan kelas.")

    # Tingkat keterlambatan per kategori
    rates = {}
    for col, label, kw in [
        ("shipping_mode", "mode pengiriman", {}),
        ("market", "pasar", {}),
        ("customer_segment", "segmen pelanggan", {}),
        ("payment_type", "tipe pembayaran", {}),
        ("department", "departemen produk", {"top": 15}),
        ("order_region", "wilayah pesanan", {"top": 15}),
        ("order_weekday", "hari pesanan",
         {"order": list(WEEKDAY_NAMES), "labels": WEEKDAY_NAMES}),
        ("order_month", "bulan pesanan",
         {"order": list(MONTH_NAMES), "labels": MONTH_NAMES}),
    ]:
        t = _bar_rate(df, col, label, f"late_by_{col}.png", figures, **kw)
        rates[col] = {str(k): {"late_rate": round(float(r.late_rate), 4),
                               "n": int(r.n)} for k, r in t.iterrows()}

    t = _rate_table(df, "scheduled_days").sort_index()
    rates["scheduled_days"] = {str(k): {"late_rate": round(float(r.late_rate), 4),
                                        "n": int(r.n)} for k, r in t.iterrows()}
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(t.index.astype(str), t["late_rate"] * 100, color=COLOR_LATE)
    ax.set_xlabel("Hari pengiriman terjadwal")
    ax.set_ylabel("Terlambat (%)")
    ax.set_title("Tingkat keterlambatan menurut hari terjadwal")
    _save(fig, "late_by_scheduled_days.png", figures,
          "Keterlambatan menurut hari terjadwal",
          "Persentase keterlambatan untuk tiap nilai hari pengiriman terjadwal.")
    summary["late_rate_by"] = rates

    # Histogram fitur numerik
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES):
        for val, color, lab in [(0, COLOR_OK, "Tidak terlambat"),
                                (1, COLOR_LATE, "Terlambat")]:
            ax.hist(df.loc[df["target"] == val, col].dropna(), bins=25,
                    alpha=0.6, color=color, label=lab)
        ax.set_title(col, fontsize=10)
    for ax in axes.ravel()[len(NUMERIC_FEATURES):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8)
    _save(fig, "numeric_histograms.png", figures, "Distribusi fitur numerik",
          "Histogram tiap fitur numerik, dibedakan menurut kelas target.")

    # Korelasi
    corr = df[NUMERIC_FEATURES + ["target"]].corr()
    summary["correlation_with_target"] = {
        k: (None if pd.isna(v) else round(float(v), 4))
        for k, v in corr["target"].drop("target").items()}
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Korelasi (Pearson) fitur numerik dan target")
    _save(fig, "correlation.png", figures, "Korelasi fitur numerik",
          "Korelasi Pearson antar fitur numerik dan target.")

    summary["figures"] = figures
    EDA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
