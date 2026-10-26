"""Unduh dataset DataCo dan cetak lokasi berkasnya.

Pemakaian:  python scripts/download_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataco.data import get_dataset_path  # noqa: E402

if __name__ == "__main__":
    path = get_dataset_path()
    print(f"Dataset siap: {path}")
    print(f"Ukuran berkas: {path.stat().st_size / 1e6:.1f} MB")
