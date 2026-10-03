"""
prepare_dataset.py
Merapikan foto mentah dari HP menjadi struktur dataset_raw/ sesuai slide 19:

    dataset_raw/
    |-- metadata.csv
    |-- mouse/mouse_20261003_normal_001.jpg
    `-- kotak_kacamata/kotak_kacamata_20261003_normal_001.jpg

Yang dilakukan:
  1. membaca orientasi EXIF (supaya foto tidak miring/terbalik)
  2. mengecilkan sisi terpanjang ke --max-side piksel (default 1024)
  3. mengganti nama file: kelas_tanggal_kondisi_nomor.jpg
  4. membuat metadata.csv: nama_file, kelas, tanggal, kondisi_cahaya
     (+ kolom tambahan: waktu, sesi, nama_asli untuk pelacakan asal data)
     "sesi" dideteksi otomatis: jeda lebih dari --jeda-sesi detik (default 600)
     antar foto berurutan dianggap awal sesi pengambilan baru.

Pemakaian:
  python prepare_dataset.py --src foto_mentah --dst dataset_raw --cahaya normal

--src berisi satu subfolder per kelas (nama folder = nama kelas).
Nama file asli harus berformat IMG_YYYYMMDD_HHMMSS.jpg (format kamera HP);
kalau tidak cocok, waktu diambil dari waktu modifikasi file.
"""
import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

PATTERN = re.compile(r"(\d{8})_(\d{6})")


def slug(nama):
    return re.sub(r"[^a-z0-9]+", "_", nama.lower()).strip("_")


def waktu_foto(path):
    m = PATTERN.search(path.stem)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder berisi subfolder per kelas")
    ap.add_argument("--dst", default="dataset_raw")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--cahaya", default="normal",
                    help="label kondisi_cahaya untuk semua foto (default)")
    ap.add_argument("--cahaya-sesi", action="append", default=[],
                    metavar="NOMOR=LABEL",
                    help="label cahaya khusus per sesi, mis. --cahaya-sesi 2=bervariasi")
    ap.add_argument("--jeda-sesi", type=int, default=600,
                    help="jeda (detik) yang menandai sesi baru")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    cahaya_sesi = {}
    for item in args.cahaya_sesi:
        nomor, _, label = item.partition("=")
        cahaya_sesi[int(nomor)] = label

    # kumpulkan semua foto dari semua kelas, lalu tentukan sesi dari jeda waktu
    semua = []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        for f in folder.iterdir():
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                semua.append((folder, f, waktu_foto(f)))
    semua.sort(key=lambda x: x[2])
    sesi_foto, sesi, sebelumnya = {}, 1, None
    for _, f, t in semua:
        if sebelumnya is not None and (t - sebelumnya).total_seconds() > args.jeda_sesi:
            sesi += 1
        sesi_foto[f] = sesi
        sebelumnya = t

    baris = []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        kelas = slug(folder.name)
        fotos = sorted(
            [f for (fo, f, _) in semua if fo == folder], key=waktu_foto)
        (dst / kelas).mkdir()
        for i, f in enumerate(fotos, 1):
            t = waktu_foto(f)
            nomor_sesi = sesi_foto[f]
            cahaya = cahaya_sesi.get(nomor_sesi, args.cahaya)
            nama = f"{kelas}_{t:%Y%m%d}_{slug(cahaya)}_{i:03d}.jpg"
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            im.thumbnail((args.max_side, args.max_side), Image.LANCZOS)
            im.save(dst / kelas / nama, quality=92)
            baris.append({
                "nama_file": f"{kelas}/{nama}",
                "kelas": kelas,
                "tanggal": f"{t:%Y-%m-%d}",
                "kondisi_cahaya": cahaya,
                "waktu": f"{t:%H:%M:%S}",
                "sesi": nomor_sesi,
                "nama_asli": f.name,
            })
        print(f"{kelas}: {len(fotos)} foto")
    print(f"jumlah sesi terdeteksi: {sesi}")

    with open(dst / "metadata.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)
    print(f"Selesai: {len(baris)} foto -> {dst}/ (metadata.csv dibuat)")


if __name__ == "__main__":
    main()
