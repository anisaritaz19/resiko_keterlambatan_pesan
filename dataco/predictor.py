"""Pemuatan model, validasi input pengguna, dan prediksi."""
import json
import math
from functools import lru_cache

import joblib
import pandas as pd

from config import (CATEGORICAL_FEATURES, FEATURES, FIELD_LABELS, METADATA_PATH,
                    METRICS_PATH, MODEL_PATH, MONTH_NAMES, THRESHOLD,
                    WEEKDAY_NAMES)

# Batas keras (validasi) untuk input numerik. Kunci = nama kolom form.
NUMERIC_SPEC = {
    "scheduled_days": {"min": 0, "max": 60, "integer": True},
    "quantity": {"min": 1, "max": 1000, "integer": True},
    "discount_pct": {"min": 0, "max": 100, "integer": False},
    "product_price": {"min": 0, "max": 1_000_000, "integer": False},
    "order_hour": {"min": 0, "max": 23, "integer": True},
}


def artifacts_ready() -> bool:
    return all(p.exists() for p in (MODEL_PATH, METADATA_PATH, METRICS_PATH))


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _to_number(raw, label):
    text = str(raw).strip().replace(",", ".")
    if text == "":
        raise ValueError(f"{label} wajib diisi.")
    try:
        val = float(text)
    except ValueError:
        raise ValueError(f"{label} harus berupa angka.")
    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"{label} harus berupa angka yang valid.")
    return val


def validate_input(data: dict, metadata: dict):
    """Validasi input mentah. Mengembalikan (nilai_bersih, errors, warnings)."""
    clean, errors, warnings = {}, [], []

    for col in CATEGORICAL_FEATURES:
        label = FIELD_LABELS[col]
        val = str(data.get(col, "")).strip()
        if not val:
            errors.append(f"{label} wajib dipilih.")
        elif val not in metadata["categories"][col]:
            errors.append(f"{label}: nilai '{val}' tidak dikenal oleh model.")
        else:
            clean[col] = val

    for key, spec in NUMERIC_SPEC.items():
        label = FIELD_LABELS[key]
        try:
            val = _to_number(data.get(key, ""), label)
        except ValueError as e:
            errors.append(str(e))
            continue
        if spec["integer"] and val != int(val):
            errors.append(f"{label} harus bilangan bulat.")
            continue
        if not spec["min"] <= val <= spec["max"]:
            errors.append(f"{label} harus antara {spec['min']} dan {spec['max']}.")
            continue
        feat = "discount_rate" if key == "discount_pct" else key
        clean[feat] = (val / 100.0) if key == "discount_pct" else val
        rng = metadata["numeric_ranges"][feat]
        if not rng["min"] <= clean[feat] <= rng["max"]:
            warnings.append(
                f"{label} berada di luar rentang data latih "
                f"({rng['min']:g}-{rng['max']:g}); prediksi kurang andal.")
    return clean, errors, warnings


def _pretty(feature: str) -> str:
    if feature in FIELD_LABELS:
        return FIELD_LABELS[feature]
    for col in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if feature.startswith(col + "_"):
            val = feature[len(col) + 1:]
            if col == "order_month":
                val = MONTH_NAMES.get(val, val)
            elif col == "order_weekday":
                val = WEEKDAY_NAMES.get(val, val)
            return f"{FIELD_LABELS[col]} = {val}"
    return feature


def predict(clean: dict, top_k: int = 4) -> dict:
    """Prediksi satu pesanan dan jelaskan faktor yang paling berpengaruh."""
    model = load_model()
    row = pd.DataFrame([{c: clean[c] for c in FEATURES}])
    proba = float(model.predict_proba(row)[0, 1])

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    xt = prep.transform(row)
    xt = xt.toarray() if hasattr(xt, "toarray") else xt
    contrib = xt[0] * clf.coef_[0]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    order = sorted(range(len(contrib)), key=lambda i: abs(contrib[i]),
                   reverse=True)
    factors = [{"name": _pretty(names[i]), "effect": float(contrib[i]),
                "direction": "menaikkan" if contrib[i] > 0 else "menurunkan"}
               for i in order[:top_k] if abs(contrib[i]) > 1e-9]
    late = proba >= THRESHOLD
    return {"probability": proba, "probability_pct": round(proba * 100, 1),
            "late": bool(late),
            "label": "Berisiko TERLAMBAT" if late else "Diperkirakan TEPAT WAKTU",
            "threshold": THRESHOLD, "factors": factors}
