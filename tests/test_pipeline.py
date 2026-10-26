"""Uji asap (smoke test) alur lengkap: preprocessing -> EDA -> training -> Flask.

PENTING: berkas ini membuat data SINTETIS hanya untuk memastikan kode berjalan
tanpa error. Angka metrik dari data ini TIDAK bermakna dan ditulis ke folder
sementara, bukan ke folder `artifacts/` proyek. Hasil evaluasi nyata hanya
berasal dari `python train.py` dengan dataset DataCo asli.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = tempfile.TemporaryDirectory()
os.environ["ARTIFACT_DIR"] = str(Path(_TMP.name) / "artifacts")  # sebelum import config
CSV_PATH = Path(_TMP.name) / "DataCoSupplyChainDataset.csv"
os.environ["DATACO_CSV"] = str(CSV_PATH)

import config  # noqa: E402


def make_synthetic_csv(path, n_orders=3000, seed=0):
    """Data sintetis dengan skema kolom DataCo (BUKAN data asli)."""
    rng = np.random.default_rng(seed)
    rows = []
    modes = {"Standard Class": 4, "Second Class": 2, "First Class": 1, "Same Day": 0}
    for oid in range(1, n_orders + 1):
        mode = rng.choice(list(modes))
        sched = modes[mode]
        late = int(rng.random() < (0.9 if mode == "First Class" else 0.4))
        date = pd.Timestamp("2016-01-01") + pd.Timedelta(
            minutes=int(rng.integers(0, 60 * 24 * 700)))
        for _ in range(rng.integers(1, 4)):
            rows.append({
                "Type": rng.choice(["DEBIT", "TRANSFER", "CASH", "PAYMENT"]),
                "Days for shipment (scheduled)": sched,
                "Order Item Quantity": int(rng.integers(1, 6)),
                "Order Item Discount Rate": round(float(rng.uniform(0, 0.25)), 2),
                "Order Item Product Price": round(float(rng.uniform(10, 500)), 2),
                "Shipping Mode": mode,
                "Market": rng.choice(["LATAM", "Europe", "Pacific Asia", "USCA"]),
                "Order Region": rng.choice(["Western Europe", "Central America",
                                            "Southeast Asia", "Caribbean"]),
                "Customer Segment": rng.choice(["Consumer", "Corporate", "Home Office"]),
                "Department Name": rng.choice(["Fan Shop", "Apparel", "Golf", "Footwear"]),
                "order date (DateOrders)": date.strftime("%m/%d/%Y %H:%M"),
                "Late_delivery_risk": late,
                "Order Id": oid,
                "Customer City": "Pasig",  # kolom ekstra yang harus diabaikan
            })
    df = pd.DataFrame(rows)
    df.loc[df.sample(20, random_state=1).index, "Department Name"] = None
    df.to_csv(path, index=False, encoding="latin-1")
    return df


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        make_synthetic_csv(CSV_PATH)
        import train
        cls.code = train.main([])
        from app import app
        app.testing = True
        cls.client = app.test_client()

    def test_training_artifacts(self):
        self.assertEqual(self.code, 0)
        for p in (config.MODEL_PATH, config.METRICS_PATH, config.METADATA_PATH,
                  config.EDA_SUMMARY_PATH, config.FIG_DIR / "roc_curve.png",
                  config.EDA_DIR / "target_distribution.png"):
            self.assertTrue(p.exists(), p)
        m = json.loads(config.METRICS_PATH.read_text())
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertTrue(0 <= m["test"][k] <= 1)

    def test_no_group_leakage(self):
        from dataco.modeling import split_data
        from dataco.preprocessing import build_dataframe, load_raw
        df, _ = build_dataframe(load_raw(CSV_PATH))
        tr, te = split_data(df)
        self.assertFalse(set(tr["order_id"]) & set(te["order_id"]))

    def test_leaky_columns_not_used(self):
        for banned in ("Days for shipping (real)", "Delivery Status",
                       "shipping date (DateOrders)"):
            self.assertNotIn(banned, config.RAW_COLUMNS)

    def test_pages(self):
        for url in ("/", "/eda", "/model", "/health"):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        self.assertEqual(self.client.get("/artifacts/figures/roc_curve.png").status_code, 200)
        self.assertEqual(self.client.get("/artifacts/../config.py").status_code, 404)

    def _valid(self):
        meta = json.loads(config.METADATA_PATH.read_text())
        d = {c: meta["categories"][c][0] for c in config.CATEGORICAL_FEATURES}
        d.update(scheduled_days=2, quantity=1, discount_pct=5,
                 product_price=100, order_hour=14)
        return d

    def test_form_prediction(self):
        r = self.client.post("/prediksi", data=self._valid())
        self.assertEqual(r.status_code, 200)
        self.assertIn("peluang terlambat", r.get_data(as_text=True))

    def test_api_prediction(self):
        r = self.client.post("/api/prediksi", json=self._valid())
        self.assertEqual(r.status_code, 200)
        p = r.get_json()["hasil"]["probability"]
        self.assertTrue(0 <= p <= 1)

    def test_validation_errors(self):
        bad = self._valid()
        bad["order_hour"] = 99
        bad["shipping_mode"] = "Teleport"
        bad["quantity"] = "abc"
        r = self.client.post("/api/prediksi", json=bad)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(len(r.get_json()["errors"]), 3)
        self.assertEqual(self.client.post("/prediksi", data={}).status_code, 400)
        self.assertEqual(self.client.post("/api/prediksi", data="x").status_code, 400)


if __name__ == "__main__":
    unittest.main()
