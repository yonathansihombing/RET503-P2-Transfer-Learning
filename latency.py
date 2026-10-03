"""
latency.py - ukur latensi inferensi ResNet-18 vs MobileNetV3-Small (slide 20, langkah 4).

Yang diukur: waktu satu kali forward pass untuk 1 gambar 224x224 (batch 1),
setelah pemanasan. Bobot acak dipakai karena kecepatan tidak bergantung pada
nilai bobot (jadi tidak perlu internet). Waktu preprocess (resize + normalisasi)
diukur terpisah memakai satu foto dari dataset.

Pemakaian:
  python latency.py
  python latency.py --runs 200 --warmup 20

Hasil disimpan ke results/latensi.csv.

PENTING: angka dari laptop hanya sebagai acuan. Anggaran latensi (slide 16:
15 FPS = sekitar 67 ms per frame untuk SELURUH pipeline) harus diukur di
komputer yang benar-benar dipakai robot (misalnya Raspberry Pi atau Jetson).
Jalankan skrip ini di perangkat itu untuk angka yang sah.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

KANDIDAT = {
    "ResNet-18": lambda: models.resnet18(weights=None),
    "MobileNetV3-Small": lambda: models.mobilenet_v3_small(weights=None),
}


def sinkron(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def ukur_model(model, device, runs, warmup):
    model = model.to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    waktu = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sinkron(device)
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            sinkron(device)
            waktu.append((time.perf_counter() - t0) * 1000)
    waktu.sort()
    return {
        "rata2_ms": statistics.mean(waktu),
        "median_ms": statistics.median(waktu),
        "p95_ms": waktu[int(0.95 * (len(waktu) - 1))],
        "fps": 1000 / statistics.mean(waktu),
    }


def ukur_preprocess(runs, contoh):
    if contoh is None:
        return None
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    img = Image.open(contoh).convert("RGB")
    waktu = []
    for _ in range(runs):
        t0 = time.perf_counter()
        tf(img)
        waktu.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(waktu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--data", default="dataset_raw")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | threads CPU: {torch.get_num_threads()}")

    hasil = []
    for nama, buat in KANDIDAT.items():
        model = buat()
        n_param = sum(p.numel() for p in model.parameters()) / 1e6
        r = ukur_model(model, device, args.runs, args.warmup)
        r.update({"model": nama, "parameter_juta": round(n_param, 2),
                  "device": str(device)})
        hasil.append(r)

    contoh = next(iter(sorted(Path(args.data).glob("*/*.jpg"))), None)
    pre = ukur_preprocess(args.runs, contoh)

    print(f"\n{'model':<20}{'param (juta)':>13}{'rata2 (ms)':>12}{'p95 (ms)':>10}{'FPS':>8}")
    for r in hasil:
        print(f"{r['model']:<20}{r['parameter_juta']:>13.2f}{r['rata2_ms']:>12.1f}"
              f"{r['p95_ms']:>10.1f}{r['fps']:>8.1f}")
    if pre is not None:
        print(f"\nPreprocess (resize+normalisasi, 1 foto): {pre:.1f} ms")
    print("Target slide 16: 15 FPS = sekitar 67 ms per frame untuk seluruh pipeline.")

    out = Path(args.results)
    out.mkdir(exist_ok=True)
    with open(out / "latensi.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "parameter_juta", "rata2_ms", "median_ms", "p95_ms",
                    "fps", "device", "preprocess_ms"])
        for r in hasil:
            w.writerow([r["model"], r["parameter_juta"], round(r["rata2_ms"], 2),
                        round(r["median_ms"], 2), round(r["p95_ms"], 2),
                        round(r["fps"], 1), r["device"],
                        round(pre, 2) if pre is not None else ""])
    print(f"Disimpan: {out / 'latensi.csv'}")


if __name__ == "__main__":
    main()
