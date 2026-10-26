"""Pelatihan, evaluasi, dan penyimpanan model Logistic Regression."""
import json
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit,  # noqa: E402
                                     cross_validate)
from sklearn.pipeline import Pipeline  # noqa: E402

from config import (CATEGORICAL_FEATURES, CV_FOLDS, FEATURES, FIG_DIR,  # noqa: E402
                    METADATA_PATH, METRICS_PATH, MODEL_PATH, NUMERIC_FEATURES,
                    RANDOM_STATE, TEST_SIZE, THRESHOLD)
from dataco.preprocessing import make_preprocessor  # noqa: E402


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", make_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs",
                                   random_state=RANDOM_STATE)),
    ])


def split_data(df):
    """Split 80/20 per pesanan (Order Id) agar item dari pesanan yang sama
    tidak muncul sekaligus di data latih dan uji."""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)
    tr, te = next(gss.split(df[FEATURES], df["target"], groups=df["order_id"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], output_dict=True, zero_division=0,
            target_names=["tidak_terlambat", "terlambat"]),
    }
    out["roc_auc"] = float(roc_auc_score(y, proba)) if y.nunique() == 2 else None
    return out


def _plot_confusion(cm, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Tidak terlambat", "Terlambat"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion matrix (data uji)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_roc(model, X, y, auc, path):
    fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#d9534f", label=f"Logistic Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Acak")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Kurva ROC (data uji)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _coefficients(model, top=12):
    names = [n.split("__", 1)[1] for n in
             model.named_steps["prep"].get_feature_names_out()]
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    rows = [{"feature": names[i], "coef": float(coef[i]),
             "odds_ratio": float(np.exp(coef[i]))} for i in order]
    return rows[::-1][:top], rows[:top]  # (mendorong terlambat, menurunkan)


def _plot_coefficients(pos, neg, path):
    rows = pos[::-1] + neg[::-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh([r["feature"] for r in rows], [r["coef"] for r in rows],
            color=["#d9534f" if r["coef"] > 0 else "#4a90a4" for r in rows])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Koefisien (log-odds)")
    ax.set_title("Koefisien terbesar: menaikkan (merah) / menurunkan (biru) "
                 "risiko terlambat", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _metadata(df, info):
    categories = {}
    for col in CATEGORICAL_FEATURES:
        vals = df[col].dropna().unique().tolist()
        if col in ("order_month", "order_weekday"):
            vals = sorted(vals, key=int)
        else:
            vals = sorted(vals)
        categories[col] = vals
    numeric = {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                   "median": float(df[c].median())} for c in NUMERIC_FEATURES}
    return {"features": FEATURES, "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "categories": categories, "numeric_ranges": numeric,
            "late_rate_overall": float(df["target"].mean()),
            "cleaning": info}


def train_and_evaluate(df, info) -> dict:
    """Latih model, evaluasi pada data uji, simpan semua artefak."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = split_data(df)
    X_tr, y_tr = train_df[FEATURES], train_df["target"]
    X_te, y_te = test_df[FEATURES], test_df["target"]

    # Baseline: selalu menebak kelas mayoritas
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    baseline = evaluate(base, X_te, y_te)

    # Cross-validation (GroupKFold per pesanan) pada data latih saja
    cv = cross_validate(
        build_pipeline(), X_tr, y_tr, groups=train_df["order_id"],
        cv=GroupKFold(n_splits=CV_FOLDS),
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"], n_jobs=1)
    cv_summary = {k.replace("test_", ""): {"mean": float(np.mean(v)),
                                           "std": float(np.std(v))}
                  for k, v in cv.items() if k.startswith("test_")}

    model = build_pipeline().fit(X_tr, y_tr)
    train_m = evaluate(model, X_tr, y_tr)
    test_m = evaluate(model, X_te, y_te)
    pos, neg = _coefficients(model)

    _plot_confusion(test_m["confusion_matrix"], FIG_DIR / "confusion_matrix.png")
    if test_m["roc_auc"] is not None:
        _plot_roc(model, X_te, y_te, test_m["roc_auc"], FIG_DIR / "roc_curve.png")
    _plot_coefficients(pos, neg, FIG_DIR / "coefficients.png")

    joblib.dump(model, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(_metadata(df, info), indent=2,
                                        ensure_ascii=False))
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "LogisticRegression (lbfgs, max_iter=1000)",
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "split": {"method": "GroupShuffleSplit per Order Id",
                  "test_size": TEST_SIZE, "n_train": int(len(train_df)),
                  "n_test": int(len(test_df)),
                  "late_rate_train": float(y_tr.mean()),
                  "late_rate_test": float(y_te.mean())},
        "baseline_majority_test": baseline,
        "cross_validation_train": {"folds": CV_FOLDS,
                                   "method": "GroupKFold per Order Id",
                                   "scores": cv_summary},
        "train": train_m,
        "test": test_m,
        "top_coefficients_increase_late": pos,
        "top_coefficients_decrease_late": neg,
        "environment": {"python": platform.python_version(),
                        "scikit_learn": sklearn.__version__},
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics
