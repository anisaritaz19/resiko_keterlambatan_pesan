"""Aplikasi web Flask: prediksi keterlambatan pengiriman (Bahasa Indonesia)."""
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from config import (ARTIFACT_DIR, EDA_SUMMARY_PATH, FIELD_LABELS, METRICS_PATH,
                    MONTH_NAMES, WEEKDAY_NAMES)
from dataco import predictor

app = Flask(__name__)
app.json.ensure_ascii = False


def build_fields(metadata):
    """Susun definisi kolom form dari metadata hasil pelatihan."""
    labels = {"order_month": MONTH_NAMES, "order_weekday": WEEKDAY_NAMES}
    fields = []
    for col in ["shipping_mode", "scheduled_days", "market", "order_region",
                "customer_segment", "payment_type", "department",
                "order_month", "order_weekday", "order_hour", "quantity",
                "discount_pct", "product_price"]:
        if col in metadata["categories"]:
            opts = [(v, labels.get(col, {}).get(v, v))
                    for v in metadata["categories"][col]]
            fields.append({"key": col, "label": FIELD_LABELS[col],
                           "type": "select", "options": opts})
        else:
            spec = predictor.NUMERIC_SPEC[col]
            feat = "discount_rate" if col == "discount_pct" else col
            med = metadata["numeric_ranges"][feat]["median"]
            if col == "discount_pct":
                med *= 100
            fields.append({
                "key": col,
                "label": FIELD_LABELS[col] + (" (%)" if col == "discount_pct" else ""),
                "type": "number", "min": spec["min"], "max": spec["max"],
                "step": 1 if spec["integer"] else "any",
                "hint": f"median data latih: {med:g}"})
    return fields


def _context(**extra):
    ready = predictor.artifacts_ready()
    ctx = {"ready": ready, "fields": [], "values": {}, "errors": [],
           "warnings": [], "result": None}
    if ready:
        ctx["fields"] = build_fields(predictor.load_metadata())
    ctx.update(extra)
    return ctx


@app.get("/")
def index():
    return render_template("index.html", **_context())


@app.post("/prediksi")
def prediksi():
    if not predictor.artifacts_ready():
        return render_template("index.html", **_context()), 503
    clean, errors, warnings = predictor.validate_input(
        request.form, predictor.load_metadata())
    if errors:
        return render_template("index.html", **_context(
            values=request.form.to_dict(), errors=errors)), 400
    result = predictor.predict(clean)
    return render_template("index.html", **_context(
        values=request.form.to_dict(), warnings=warnings, result=result))


@app.post("/api/prediksi")
def api_prediksi():
    if not predictor.artifacts_ready():
        return jsonify(error="Model belum tersedia. Jalankan `python train.py`."), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Body harus berupa JSON object."), 400
    clean, errors, warnings = predictor.validate_input(
        payload, predictor.load_metadata())
    if errors:
        return jsonify(errors=errors), 400
    return jsonify(hasil=predictor.predict(clean), peringatan=warnings)


@app.get("/model")
def model_page():
    metrics = predictor.load_json(METRICS_PATH)
    return render_template("model.html", metrics=metrics)


@app.get("/eda")
def eda_page():
    summary = predictor.load_json(EDA_SUMMARY_PATH)
    return render_template("eda.html", summary=summary)


@app.get("/artifacts/<path:filename>")
def artifact_file(filename):
    if not filename.startswith(("eda/", "figures/")) or not filename.endswith(".png"):
        abort(404)
    return send_from_directory(ARTIFACT_DIR, filename)


@app.get("/health")
def health():
    return jsonify(status="ok", model_tersedia=predictor.artifacts_ready())


@app.template_filter("pct")
def pct(v, digits=2):
    return "-" if v is None else f"{v * 100:.{digits}f}%"


@app.template_filter("num")
def num(v):
    return f"{v:,}".replace(",", ".")


if __name__ == "__main__":
    app.run(debug=True)
