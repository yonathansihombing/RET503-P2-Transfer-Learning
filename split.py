"""
split.py
Membagi dataset_raw/ menjadi dataset/train dan dataset/val.

Pembagian BUKAN acak (slide 22 "Waspadai data leakage"): foto yang diambil
beruntun posisinya hampir sama, jadi kalau diacak, foto kembar masuk ke train
DAN val dan akurasi val terlihat terlalu bagus.

Aturan (kolom "sesi" di metadata.csv dibuat oleh prepare_dataset.py):
  - Kalau ada lebih dari satu sesi pengambilan: SESI TERAKHIR dijadikan val,
    sesi sebelumnya jadi train. Ini pemisahan yang dianjurkan slide 22.
  - Kalau hanya satu sesi: foto diurutkan menurut waktu lalu 20% foto
    paling akhir per kelas jadi val (pengurangan risiko, bukan pemisahan penuh).

Pemakaian:
  python split.py                  # otomatis seperti aturan di atas
  python split.py --val-sesi 2     # paksa sesi nomor 2 sebagai val
  python split.py --val-frac 0.25  # hanya dipakai kalau cuma ada satu sesi
"""
import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset_raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-sesi", type=int, default=None,
                    help="nomor sesi yang dijadikan val (default: sesi terakhir)")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    with open(raw / "metadata.csv", encoding="utf-8") as fh:
        baris = list(csv.DictReader(fh))

    per_kelas = defaultdict(list)
    for b in baris:
        per_kelas[b["kelas"]].append(b)

    ada_sesi = "sesi" in baris[0]
    daftar_sesi = sorted({int(b["sesi"]) for b in baris}) if ada_sesi else [1]
    sesi_val = args.val_sesi
    if sesi_val is None and len(daftar_sesi) > 1:
        sesi_val = daftar_sesi[-1]
    if sesi_val is not None:
        print(f"sesi terdeteksi: {daftar_sesi} -> sesi {sesi_val} dijadikan val")
    else:
        print("hanya satu sesi -> val = blok foto terakhir menurut waktu")

    if out.exists():
        shutil.rmtree(out)

    catatan = []
    for kelas, daftar in sorted(per_kelas.items()):
        daftar.sort(key=lambda b: (b["tanggal"], b["waktu"]))
        if sesi_val is not None:
            val = [b for b in daftar if int(b["sesi"]) == sesi_val]
            train = [b for b in daftar if int(b["sesi"]) != sesi_val]
        else:
            n_val = max(1, round(len(daftar) * args.val_frac))
            train, val = daftar[:-n_val], daftar[-n_val:]
        for nama, bagian in (("train", train), ("val", val)):
            (out / nama / kelas).mkdir(parents=True, exist_ok=True)
            for b in bagian:
                shutil.copy2(raw / b["nama_file"], out / nama / kelas)
                catatan.append({"nama_file": b["nama_file"], "kelas": kelas,
                                "split": nama})
        print(f"{kelas}: train={len(train)}  val={len(val)}")

    with open(out / "split.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["nama_file", "kelas", "split"])
        w.writeheader()
        w.writerows(catatan)
    print(f"Selesai -> {out}/train , {out}/val , {out}/split.csv")


if __name__ == "__main__":
    main()
