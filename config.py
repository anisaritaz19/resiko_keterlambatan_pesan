"""Konfigurasi terpusat proyek prediksi keterlambatan pengiriman DataCo."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", BASE_DIR / "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "model_logreg.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
EDA_SUMMARY_PATH = ARTIFACT_DIR / "eda_summary.json"
EDA_DIR = ARTIFACT_DIR / "eda"
FIG_DIR = ARTIFACT_DIR / "figures"

# --- Dataset ---
DATASET_SLUG = "shashwatwork/dataco-smart-supply-chain-for-big-data-analysis"
DATA_FILE_NAME = "DataCoSupplyChainDataset.csv"
CSV_ENCODING = "latin-1"  # dataset DataCo tidak berformat UTF-8

# --- Target & pengelompokan ---
TARGET = "Late_delivery_risk"  # 1 = terlambat, 0 = tidak terlambat
GROUP_COL = "Order Id"         # satu pesanan bisa punya banyak baris item
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
THRESHOLD = 0.5

# Kolom mentah -> nama fitur internal.
# Kolom realisasi ("Days for shipping (real)", "Delivery Status",
# "shipping date") SENGAJA tidak dipakai: nilainya baru diketahui setelah
# barang dikirim sehingga menyebabkan kebocoran data (data leakage).
RAW_TO_FEATURE = {
    "Days for shipment (scheduled)": "scheduled_days",
    "Order Item Quantity": "quantity",
    "Order Item Discount Rate": "discount_rate",
    "Order Item Product Price": "product_price",
    "Shipping Mode": "shipping_mode",
    "Market": "market",
    "Order Region": "order_region",
    "Customer Segment": "customer_segment",
    "Type": "payment_type",
    "Department Name": "department",
    "order date (DateOrders)": "order_date",
    TARGET: "target",
    GROUP_COL: "order_id",
}
RAW_COLUMNS = list(RAW_TO_FEATURE.keys())

NUMERIC_FEATURES = ["scheduled_days", "quantity", "discount_rate",
                    "product_price", "order_hour"]
CATEGORICAL_FEATURES = ["shipping_mode", "market", "order_region",
                        "customer_segment", "payment_type", "department",
                        "order_month", "order_weekday"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FIELD_LABELS = {
    "shipping_mode": "Mode pengiriman",
    "scheduled_days": "Hari pengiriman terjadwal",
    "market": "Pasar",
    "order_region": "Wilayah pesanan",
    "customer_segment": "Segmen pelanggan",
    "payment_type": "Tipe pembayaran",
    "department": "Departemen produk",
    "order_month": "Bulan pesanan",
    "order_weekday": "Hari pesanan",
    "order_hour": "Jam pesanan (0-23)",
    "quantity": "Jumlah item",
    "discount_rate": "Diskon",
    "discount_pct": "Diskon",
    "product_price": "Harga produk",
}
MONTH_NAMES = {"1": "Januari", "2": "Februari", "3": "Maret", "4": "April",
               "5": "Mei", "6": "Juni", "7": "Juli", "8": "Agustus",
               "9": "September", "10": "Oktober", "11": "November",
               "12": "Desember"}
WEEKDAY_NAMES = {"0": "Senin", "1": "Selasa", "2": "Rabu", "3": "Kamis",
                 "4": "Jumat", "5": "Sabtu", "6": "Minggu"}
