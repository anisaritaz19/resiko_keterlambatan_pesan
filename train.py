"""Skrip pelatihan end-to-end: unduh data -> EDA -> latih -> evaluasi -> simpan.

Contoh:
    python train.py                 # unduh via kagglehub lalu latih
    python train.py --if-missing    # lewati bila model sudah ada (dipakai Render)
    DATACO_CSV=/path/DataCoSupplyChainDataset.csv python train.py
"""
import argparse
import sys

from config import ARTIFACT_DIR, METADATA_PATH, METRICS_PATH, MODEL_PATH
from dataco.data import get_dataset_path
from dataco.eda import run_eda
from dataco.modeling import train_and_evaluate
from dataco.preprocessing import build_dataframe, load_raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--if-missing", action="store_true",
                    help="lewati pelatihan jika artefak model sudah lengkap")
    ap.add_argument("--skip-eda", action="store_true", help="lewati EDA")
    args = ap.parse_args(argv)

    if args.if_missing and all(p.exists() for p in
                               (MODEL_PATH, METADATA_PATH, METRICS_PATH)):
        print(f"[train] Artefak sudah ada di {ARTIFACT_DIR}, pelatihan dilewati.")
        return 0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print("[train] Mengambil dataset ...")
    try:
        path = get_dataset_path()
    except Exception as exc:  # jaringan mati, kredensial salah, dsb.
        print(f"[train] GAGAL mengambil dataset: {type(exc).__name__}: {exc}\n"
              "        Pastikan ada koneksi internet, atau unduh dataset secara manual "
              "dari Kaggle lalu jalankan:\n"
              "        DATACO_CSV=/path/ke/DataCoSupplyChainDataset.csv python train.py",
              file=sys.stderr)
        return 1
    print(f"[train] Membaca {path}")
    raw = load_raw(path)
    df, info = build_dataframe(raw)
    del raw
    print(f"[train] Baris dipakai: {info['rows_used']:,} "
          f"(mentah {info['rows_raw']:,})")

    if not args.skip_eda:
        print("[train] Menjalankan EDA ...")
        run_eda(df, info)

    print("[train] Melatih dan mengevaluasi model ...")
    m = train_and_evaluate(df, info)
    t = m["test"]
    print("[train] Hasil pada data uji (dihitung dari dataset yang dimuat):")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"        {k:<10}: {t[k]:.4f}")
    print(f"[train] Model disimpan di {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
