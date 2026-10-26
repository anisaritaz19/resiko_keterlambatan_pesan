# Prediksi Keterlambatan Pengiriman E-commerce (Logistic Regression)

Proyek end-to-end untuk memprediksi apakah sebuah pesanan **berisiko terlambat dikirim**,
memakai dataset **DataCo Smart Supply Chain** dari Kaggle. Mencakup pengunduhan data otomatis,
preprocessing, EDA, pelatihan dan evaluasi Logistic Regression, penyimpanan model dengan Joblib,
aplikasi web Flask berbahasa Indonesia, dan konfigurasi deployment ke Render.

> **Catatan jujur tentang modul TA-03.** Isi modul TA-03 tidak tersedia bagi pembuatnya, jadi
> struktur proyek mengikuti alur baku pembelajaran mesin untuk klasifikasi (data → EDA →
> preprocessing → model → evaluasi → deployment). Jika modul Anda mewajibkan langkah atau format
> laporan tertentu, cocokkan dengan bagian-bagian di bawah ini dan sesuaikan.

> **Tentang hasil evaluasi.** README ini sengaja **tidak memuat angka akurasi/F1/AUC**. Semua
> metrik dihitung saat `python train.py` dijalankan pada dataset asli dan disimpan di
> `artifacts/metrics.json`, lalu ditampilkan di halaman **Evaluasi Model**. ZIP ini dibuat di
> lingkungan tanpa akses ke Kaggle, sehingga **model belum dilatih di dalam ZIP**.

## Struktur proyek

```
app.py                  Aplikasi Flask (halaman, form, API JSON, /health)
train.py                CLI: unduh data -> EDA -> latih -> evaluasi -> simpan
config.py               Konfigurasi: kolom, fitur, path, parameter
dataco/
  data.py               Unduh dataset via kagglehub
  preprocessing.py      Pemuatan data, rekayasa fitur, ColumnTransformer
  eda.py                Ringkasan statistik + grafik EDA
  modeling.py           Logistic Regression, CV, evaluasi, simpan Joblib
  predictor.py          Validasi input + prediksi + penjelasan faktor
scripts/download_data.py  Unduh dataset saja
templates/, static/     Antarmuka (Bahasa Indonesia)
tests/test_pipeline.py  Uji asap (memakai data sintetis, lihat bagian Pengujian)
artifacts/              Keluaran training: model .joblib, metrics.json, grafik
render.yaml             Blueprint deployment Render
requirements.txt        Dependensi (versi dipin)
```

## Menjalankan secara lokal

Butuh Python 3.11+ (diuji dengan 3.12) dan koneksi internet untuk mengunduh dataset.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py                    # unduh dataset, EDA, latih, evaluasi, simpan model
python app.py                      # buka http://127.0.0.1:5000
```

Dataset diunduh dengan:

```python
import kagglehub
path = kagglehub.dataset_download("shashwatwork/dataco-smart-supply-chain-for-big-data-analysis")
```

(Skrip `python scripts/download_data.py` hanya mengunduh dan mencetak lokasi berkasnya.)
Jika Kaggle meminta kredensial di mesin Anda, atur `KAGGLE_USERNAME`/`KAGGLE_KEY` atau
`~/.kaggle/kaggle.json`. Jika sudah punya CSV-nya, lewati unduhan:

```bash
DATACO_CSV=/path/ke/DataCoSupplyChainDataset.csv python train.py
```

Keluaran `train.py` (folder `artifacts/`): `model_logreg.joblib`, `metrics.json`, `metadata.json`,
`eda_summary.json`, serta grafik di `artifacts/eda/` dan `artifacts/figures/`.

## Alur data dan model

**Target:** `Late_delivery_risk` (1 = terlambat, 0 = tidak).

**Fitur** (hanya informasi yang sudah diketahui saat pesanan dibuat):
hari pengiriman terjadwal, mode pengiriman, pasar, wilayah pesanan, segmen pelanggan, tipe
pembayaran, departemen produk, jumlah item, diskon, harga produk, serta bulan/hari/jam pesanan
(diturunkan dari `order date (DateOrders)`).

**Pencegahan kebocoran data.** Kolom `Days for shipping (real)`, `Delivery Status`, dan
`shipping date (DateOrders)` **tidak dipakai** karena baru diketahui setelah barang dikirim.
Memakainya membuat skor tampak bagus tetapi model tak berguna untuk pesanan baru.

**Pembagian data.** 80/20 dengan `GroupShuffleSplit` per `Order Id` (seed 42). Dataset berisi
beberapa baris item per pesanan; split acak biasa akan menaruh item dari pesanan yang sama di
latih dan uji sekaligus sehingga skor terlalu optimistis. Cross-validation memakai `GroupKFold`.

**Preprocessing** (di dalam `Pipeline`, di-fit hanya pada data latih):
imputasi median + `StandardScaler` untuk numerik; imputasi modus + `OneHotEncoder(handle_unknown="ignore")`
untuk kategorikal.

**Model:** `LogisticRegression(solver="lbfgs", max_iter=1000)`, ambang keputusan 0,5.

**Evaluasi** (data uji, juga CV 5-fold dan data latih sebagai pemeriksaan overfitting):
akurasi, presisi, recall, F1, ROC-AUC, confusion matrix, kurva ROC, koefisien model/odds ratio,
dan **baseline kelas mayoritas** sebagai pembanding. Presisi/recall/F1 untuk kelas "terlambat".

**Penyimpanan:** seluruh pipeline (preprocessor + model) disimpan dengan `joblib.dump`, sehingga
aplikasi memuat satu berkas dan menerapkan preprocessing yang identik saat inferensi.

## Aplikasi web

| Rute | Fungsi |
|---|---|
| `/` | Form prediksi satu pesanan + hasil dan faktor paling berpengaruh |
| `/eda` | Ringkasan dan grafik EDA hasil training |
| `/model` | Metrik evaluasi, confusion matrix, ROC, koefisien |
| `POST /api/prediksi` | API JSON (kunci sama seperti nama kolom form) |
| `/health` | Health check untuk Render |

Opsi dropdown diambil dari `metadata.json` (nilai yang benar-benar ada di data latih), dan input
divalidasi (rentang angka, nilai kategori). Nilai numerik di luar rentang data latih memunculkan
peringatan.

Contoh API:

```bash
curl -X POST http://127.0.0.1:5000/api/prediksi -H "Content-Type: application/json" -d '{
  "shipping_mode": "Standard Class", "scheduled_days": 4, "market": "LATAM",
  "order_region": "Central America", "customer_segment": "Consumer",
  "payment_type": "DEBIT", "department": "Fan Shop", "order_month": "3",
  "order_weekday": "2", "order_hour": 14, "quantity": 1,
  "discount_pct": 5, "product_price": 100}'
```

Nilai kategori harus sama dengan yang ada di `artifacts/metadata.json`; nilai tak dikenal ditolak
dengan HTTP 400.

## Deployment ke Render

Ada dua cara menyediakan model di Render. **Cara A lebih disarankan** karena build lebih cepat dan
tidak bergantung pada Kaggle saat deploy.

**Cara A: latih lokal, commit artefak.**
1. Jalankan `python train.py` di komputer Anda.
2. Commit seluruh isi `artifacts/` (model terkompresi berukuran kecil, bukan dataset).
3. Push ke GitHub/GitLab. Di Render: **New → Blueprint** (membaca `render.yaml`), atau
   **New → Web Service** dengan pengaturan di bawah.
4. Saat build, `python train.py --if-missing` mendeteksi artefak dan melewati pelatihan.

**Cara B: latih saat build.** Push tanpa folder `artifacts/` terisi. Build akan mengunduh dataset via
`kagglehub` dan melatih model. Ini butuh akses internet saat build dan memori build yang memadai
untuk memuat CSV (~180 ribu baris; hanya 13 kolom yang dibaca untuk menghemat memori). Jika gagal,
pakai Cara A.

Pengaturan Web Service manual:

| Kolom | Nilai |
|---|---|
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt && python train.py --if-missing` |
| Start Command | `gunicorn app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT` |
| Health Check Path | `/health` |
| Env var | `PYTHON_VERSION` = `3.12.3` |

Pada paket gratis, layanan "tidur" saat tidak dipakai sehingga permintaan pertama bisa lambat.
Versi library dipin di `requirements.txt` agar berkas `.joblib` kompatibel (scikit-learn sensitif
terhadap versi); jika Anda mengubah versi, latih ulang model.

## Pengujian

```bash
python -m unittest discover -s tests -v
```

Tes ini menjalankan seluruh alur (preprocessing, EDA, training, semua halaman Flask, API,
validasi, pengecekan tidak ada kebocoran group dan kolom bocor tidak dipakai).
**Data yang dipakai adalah data SINTETIS** dengan skema kolom DataCo, dibuat dalam tes itu sendiri
hanya untuk memastikan kode berjalan. Artefaknya ditulis ke folder sementara dan **tidak boleh**
dipakai sebagai hasil penelitian; metrik dari tes ini tidak bermakna.

## Keterbatasan

- Model linear: tidak menangkap interaksi non-linear antarfitur. Logistic Regression dipilih sesuai
  tujuan tugas dan karena koefisiennya mudah ditafsirkan.
- Prediksi hanya sebaik informasi pra-pengiriman yang tersedia; hasilnya estimasi probabilistik.
- Dataset bersifat historis dan sudah dikurasi; performa pada kondisi logistik nyata bisa berbeda.
- Pembagian latih/uji dilakukan acak per pesanan, bukan berdasarkan waktu.
