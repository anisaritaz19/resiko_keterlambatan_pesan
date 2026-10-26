"""Pengunduhan dataset DataCo melalui kagglehub."""
import os
from pathlib import Path

from config import DATA_FILE_NAME, DATASET_SLUG


def find_csv(root: Path) -> Path:
    """Cari berkas CSV utama di dalam folder hasil unduhan."""
    target = DATA_FILE_NAME.lower()
    for p in Path(root).rglob("*.csv"):
        if p.name.lower() == target:
            return p
    found = [p.name for p in Path(root).rglob("*.csv")]
    raise FileNotFoundError(
        f"{DATA_FILE_NAME} tidak ditemukan di {root}. CSV yang ada: {found}")


def get_dataset_path() -> Path:
    """Kembalikan path CSV DataCo.

    Jika variabel lingkungan DATACO_CSV diisi, berkas lokal itu dipakai.
    Jika tidak, dataset diunduh dengan kagglehub.dataset_download.
    """
    override = os.environ.get("DATACO_CSV")
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"DATACO_CSV menunjuk ke berkas yang tidak ada: {p}")
        return p
    import kagglehub  # diimpor di sini agar modul lain tidak butuh jaringan

    folder = kagglehub.dataset_download(DATASET_SLUG)
    return find_csv(Path(folder))
