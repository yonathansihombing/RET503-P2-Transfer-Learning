# RET503 P2: Transfer Learning ResNet-18 untuk Klasifikasi Objek

Praktikum Pertemuan 3 (Transfer Learning dan Fine-Tuning Model Visi), Politeknik Negeri Batam.
Nama:Yonatan Andreas
NIM:4222411028
PBL:Robot Manucfaturing System

## 1. Tujuan

Membandingkan tiga pendekatan melatih ResNet-18 untuk mengenali dua objek dari foto, lalu mengukur latensi dua model kandidat untuk robot.

## 2. Data

| Kelas | Jumlah foto | Sesi 1 (train) | Sesi 2 (val) |
|---|---|---|---|
| mouse | 50 | 25 | 25 |
| kotak_kacamata | 50 | 25 | 25 |

- Kamera: kamera HP, resolusi asli 4160×3120, dikecilkan ke sisi terpanjang 1024 px.
- Sesi 1 diambil 19:52-20:00 dan sesi 2 diambil 20:26-20:32 pada 3 Oktober 2026. Sesi 2 dijadikan data validasi supaya train dan val tidak berisi foto yang diambil beruntun (slide 22, data leakage).
- Variasi: jarak, posisi di citra, orientasi, bayangan dan sorotan cahaya, objek pengganggu (kabel, kunci), dan tangan. Latar selalu lantai yang sama.
- `dataset_raw/metadata.csv` mencatat nama file, kelas, tanggal, kondisi cahaya, waktu, sesi, dan nama file asli.

## 3. Metode

| Mode | Bobot awal | Yang dilatih | Learning rate |
|---|---|---|---|
| feature | ImageNet | fc saja | 1e-3 |
| partial | ImageNet | layer4 + fc | 1e-4 (layer4) / 1e-3 (fc) |
| scratch | acak | semua | 1e-3 |

Pengaturan: 10 epoch, batch 16, Adam, CosineAnnealingLR, seed 42, CPU. Augmentasi train: RandomResizedCrop (scale 0,5-1,0), HorizontalFlip, ColorJitter. Lapisan yang dibekukan tetap dalam mode `eval()` agar statistik BatchNorm ImageNet tidak berubah.

## 4. Hipotesis

Hipotesis dari materi praktikum (slide 21): mode feature dan partial akan jauh lebih baik daripada scratch pada data kecil, baik dari akurasi maupun kecepatan mencapai akurasi 90%. 

## 5. Hasil

| Mode | Akurasi val terbaik | Waktu latih (detik) | Epoch pertama val acc ≥ 90% |
|---|---|---|---|
| feature | 0,980 | 38,1 | 3 |
| partial | 1,000 | 43,9 | 1 |
| scratch | 0,940 | 77,6 | 7 |

Grafik akurasi val per epoch: `results/akurasi_per_epoch.png`. Riwayat lengkap: `results/<mode>/history.csv`.

### Latensi (CPU laptop, 6 thread, batch 1, input 224×224)

| Model | Parameter (juta) | Rata-rata (ms) | p95 (ms) | FPS |
|---|---|---|---|---|
| ResNet-18 | 11,69 | 33,0 | 38,3 | 30,3 |
| MobileNetV3-Small | 2,54 | 10,7 | 13,7 | 93,6 |

Preprocess (resize + normalisasi, 1 foto): 5,1 ms. Detail di `results/latensi.csv`.
<img width="1050" height="600" alt="akurasi_per_epoch" src="https://github.com/user-attachments/assets/34208623-1401-411d-82e9-8447064ce166" />


## 6. Analisis

1. **Transfer learning lebih cepat konvergen.** Mode feature dan partial mencapai akurasi val ≥ 90% pada epoch 3 dan 1, sedangkan scratch baru di epoch 7. Waktu latih scratch (77,6 s) sekitar dua kali mode lain.
2. **Selisih akurasi kecil.** Val berisi 50 foto, jadi selisih 0,98 vs 1,00 hanya satu foto. Dari satu kali percobaan, feature dan partial tidak dapat dibedakan secara meyakinkan. Scratch lebih rendah (0,94) dan lebih tidak stabil: val loss sempat tinggi di epoch awal (35,4 pada epoch 1) dan akurasi val tertahan di 0,50 pada beberapa epoch pertama.
3. **Tugas ini relatif mudah.** Kedua kelas berbeda jelas (mouse putih-biru, kotak kacamata hitam) dan latarnya sama. Akurasi val 100% pada mode partial sejak epoch awal tidak berarti model akan sempurna di lapangan (lihat bagian keterbatasan).
4. **Latensi.** Keduanya memenuhi anggaran 67 ms per frame (15 FPS) pada laptop. ResNet-18 + preprocess sekitar 38 ms, MobileNetV3-Small + preprocess sekitar 16 ms. Anggaran lengkap juga mencakup akuisisi, ROS2, dan postprocess, serta harus diukur di komputer robot.


## 7. Keterbatasan

- Data hanya 2 kelas dengan 50 foto per kelas; val 50 foto sehingga satu foto salah menurunkan akurasi 2%.
- Semua foto diambil di latar dan lokasi yang sama dalam satu hari, jadi performa di lingkungan lain belum diketahui.
- Foto dari kamera HP dan belum dikoreksi distorsi dengan `calib.npz`.
- Satu kali percobaan (seed 42), tanpa pengulangan, sehingga variasi antar-run tidak diukur.
- Latensi diukur di CPU laptop, bukan di komputer robot.

## 8. Menjalankan ulang

```
python -m pip install torch torchvision matplotlib pillow
python prepare_dataset.py --src <folder_foto_mentah> --dst dataset_raw --cahaya normal --cahaya-sesi 2=bervariasi
python split.py
python train.py
python latency.py
```

## 9. Struktur repo

```
README.md
dokumen_desain.pdf
dataset_raw/            foto + metadata.csv
results/                tabel_hasil.csv, akurasi_per_epoch.png, latensi.csv, history tiap mode
bukti/                  tangkapan layar terminal saat train.py dan latency.py dijalankan
prepare_dataset.py  split.py  train.py  latency.py
```
