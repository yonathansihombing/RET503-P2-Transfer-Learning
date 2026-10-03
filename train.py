"""
train.py - Praktikum P2 RET503: transfer learning ResNet-18 dengan 3 mode.

  mode      bobot awal   yang dilatih      learning rate
  feature   ImageNet     fc saja           1e-3
  partial   ImageNet     layer4 + fc       1e-4 (layer4) / 1e-3 (fc)
  scratch   acak         semua             1e-3

Data : dataset/train/<kelas>/*.jpg dan dataset/val/<kelas>/*.jpg (hasil split.py)
Latih: 10 epoch, Adam, CosineAnnealingLR, augmentasi RandomResizedCrop +
       HorizontalFlip + ColorJitter (slide 21).

Pemakaian:
  python train.py                  # jalankan ketiga mode berurutan + tabel + grafik
  python train.py --mode feature   # satu mode saja
  python train.py --epochs 10 --batch-size 16

Keluaran di folder results/:
  results/<mode>/history.csv   loss dan akurasi tiap epoch
  results/<mode>/summary.json  akurasi val terbaik, waktu latih, epoch >= 90%
  results/<mode>/best.pt       bobot terbaik (berdasarkan akurasi val)
  results/tabel_hasil.csv      ringkasan 3 mode (untuk README)
  results/akurasi_per_epoch.png grafik akurasi val per epoch (jika matplotlib ada)

Catatan: mode feature dan partial mengunduh bobot ImageNet saat pertama kali
(butuh internet, sekitar 45 MB).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MEAN = [0.485, 0.456, 0.406]   # statistik ImageNet (slide 13)
STD = [0.229, 0.224, 0.225]
MODES = ["feature", "partial", "scratch"]

# Bagian backbone yang DIBEKUKAN per mode. Lapisan beku harus tetap eval()
# agar statistik BatchNorm milik ImageNet tidak berubah oleh batch kecil
# (slide 11).
BEKU = {
    "feature": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "partial": ["conv1", "bn1", "layer1", "layer2", "layer3"],
    "scratch": [],
}


def make_loaders(data_dir, batch_size):
    # scale minimal 0.5 supaya potongan acak tidak membuang objek yang kecil.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # Val: resize langsung ke 224x224 tanpa CenterCrop, supaya objek yang
    # posisinya di tepi foto (slide 19) tidak terpotong.
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", train_tf)  # urutan RGB
    val_ds = datasets.ImageFolder(data_dir / "val", val_tf)
    assert train_ds.classes == val_ds.classes, "kelas train dan val tidak sama"
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, train_ds.classes


def build_model(mode, num_classes):
    if mode == "scratch":
        m = models.resnet18(weights=None)
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if mode in ("feature", "partial"):
        for p in m.parameters():
            p.requires_grad = False
    if mode == "partial":
        for p in m.layer4.parameters():
            p.requires_grad = True

    # Head baru: dibuat setelah pembekuan, jadi otomatis requires_grad=True.
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_optimizer(m, mode):
    if mode == "partial":
        return torch.optim.Adam([
            {"params": m.layer4.parameters(), "lr": 1e-4},
            {"params": m.fc.parameters(), "lr": 1e-3},
        ])
    params = [p for p in m.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=1e-3)


def set_train_mode(m, mode):
    m.train()
    for nama in BEKU[mode]:
        getattr(m, nama).eval()


def run_epoch(model, loader, device, criterion, mode, optimizer=None):
    training = optimizer is not None
    if training:
        set_train_mode(model, mode)
    else:
        model.eval()

    total_loss, benar, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            benar += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, benar / n


def train_mode(mode, args, device, results_dir):
    torch.manual_seed(args.seed)
    train_dl, val_dl, classes = make_loaders(args.data, args.batch_size)
    model = build_model(mode, len(classes)).to(device)
    optimizer = build_optimizer(model, mode)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    n_latih = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== mode: {mode} | parameter yang dilatih: {n_latih:,} ===")

    out_dir = results_dir / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    riwayat = []
    terbaik, epoch_90 = 0.0, None
    t_mulai = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, device, criterion, mode, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, device, criterion, mode)
        scheduler.step()
        riwayat.append({"epoch": ep, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 4),
                        "val_acc": round(va_acc, 4),
                        "detik_sejak_mulai": round(time.time() - t_mulai, 1)})
        print(f"epoch {ep:2d}/{args.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_acc > terbaik:
            terbaik = va_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
        if epoch_90 is None and va_acc >= 0.9:
            epoch_90 = ep
    waktu = time.time() - t_mulai

    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(riwayat[0].keys()))
        w.writeheader()
        w.writerows(riwayat)

    ringkas = {"mode": mode, "val_acc_terbaik": round(terbaik, 4),
               "waktu_latih_detik": round(waktu, 1), "epoch_val_acc_90": epoch_90,
               "parameter_dilatih": n_latih, "device": str(device),
               "epochs": args.epochs, "batch_size": args.batch_size}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(ringkas, fh, indent=2)
    return riwayat, ringkas, classes


def buat_tabel_dan_grafik(semua, results_dir):
    with open(results_dir / "tabel_hasil.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "val_acc_terbaik", "waktu_latih_detik", "epoch_val_acc_90"])
        for mode, (_, r) in semua.items():
            w.writerow([mode, r["val_acc_terbaik"], r["waktu_latih_detik"],
                        r["epoch_val_acc_90"] if r["epoch_val_acc_90"] else "belum tercapai"])

    print("\n=== Ringkasan ===")
    print(f"{'mode':<9}{'val acc terbaik':>16}{'waktu latih (s)':>17}{'epoch >=90%':>13}")
    for mode, (_, r) in semua.items():
        e = r["epoch_val_acc_90"] if r["epoch_val_acc_90"] else "-"
        print(f"{mode:<9}{r['val_acc_terbaik']:>16.3f}{r['waktu_latih_detik']:>17.1f}{str(e):>13}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib belum terpasang -> grafik dilewati "
              "(python -m pip install matplotlib)")
        return
    plt.figure(figsize=(7, 4))
    for mode, (riwayat, _) in semua.items():
        plt.plot([h["epoch"] for h in riwayat], [h["val_acc"] for h in riwayat],
                 marker="o", label=mode)
    plt.xlabel("epoch")
    plt.ylabel("akurasi val")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Akurasi validasi per epoch (ResNet-18)")
    plt.tight_layout()
    plt.savefig(results_dir / "akurasi_per_epoch.png", dpi=150)
    print(f"Grafik disimpan: {results_dir / 'akurasi_per_epoch.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES + ["all"], default="all")
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    results_dir = Path(args.results)
    results_dir.mkdir(exist_ok=True)

    modes = MODES if args.mode == "all" else [args.mode]
    semua, classes = {}, None
    for mode in modes:
        riwayat, ringkas, classes = train_mode(mode, args, device, results_dir)
        semua[mode] = (riwayat, ringkas)

    with open(results_dir / "classes.json", "w", encoding="utf-8") as fh:
        json.dump(classes, fh)
    buat_tabel_dan_grafik(semua, results_dir)


if __name__ == "__main__":
    main()
