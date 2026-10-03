# ResNet-18 Feature Mode — Klasifikasi Botol vs Box

Project ini memproses dataset `botol` dan `box` menggunakan **ResNet-18 mode feature**, bukan YOLO.

## Konsep metode

ResNet-18 digunakan sebagai **frozen feature extractor**. Layer klasifikasi terakhir (`fc`) dilepas sehingga setiap citra diubah menjadi vektor fitur 512 dimensi. Vektor fitur tersebut kemudian dipakai oleh classifier Linear sederhana untuk klasifikasi `botol` dan `box`.

TorchVision menyediakan bobot ResNet-18 pretrained ImageNet dan preprocessing standar berupa resize 256, center crop 224, serta normalisasi ImageNet.

## Struktur

```text
.
├── dataset_raw/
│   ├── botol/
│   └── box/
├── dataset/
│   ├── train/
│   │   ├── botol/
│   │   └── box/
│   └── val/
│       ├── botol/
│       └── box/
├── features/
├── results/
├── scripts/
│   ├── prepare_dataset.py
│   ├── make_metadata.py
│   ├── extract_features.py
│   ├── train_feature_classifier.py
│   └── benchmark_latency.py
├── metadata.csv
└── requirements.txt
```

## Dataset

Dataset asli yang diberikan berisi 50 citra unik per kelas. Folder `dataset_raw` mempertahankan citra asli.

Agar memenuhi kebutuhan **lebih dari 50 citra/kelas**, folder `dataset` dibuat dengan pembagian train/validation dan augmentasi hanya pada data train:
- 40 citra asli/kelas untuk train.
- 10 citra asli/kelas untuk validation.
- Setiap citra train dibuat menjadi 2 variasi augmentasi tambahan.
- Total train = 120 citra/kelas.
- Total validation = 10 citra/kelas.
- Total dataset = 130 citra/kelas.

Augmentasi tidak diterapkan pada validation sehingga evaluasi tidak mencampurkan variasi augmentasi ke validation.

## Menjalankan di Ubuntu

### 1. Buat virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pastikan dataset siap

```bash
python scripts/prepare_dataset.py
python scripts/make_metadata.py
```

### 3. Ekstraksi feature ResNet-18

```bash
python scripts/extract_features.py
```

Pada eksekusi pertama, TorchVision akan mengunduh bobot pretrained ResNet-18 jika belum ada di cache. Feature yang dihasilkan berukuran 512 dimensi.

```bash
python scripts/extract_features.py --batch-size 16 --workers 2
```

### 4. Training classifier dan grafik akurasi

```bash
python scripts/train_feature_classifier.py
```

Output:
- `results/accuracy_per_epoch.csv`
- `results/accuracy_per_epoch.png`
- `results/classifier.pt`

### 5. Pengukuran latensi

```bash
python scripts/benchmark_latency.py --runs 100
```

Output:
- `results/latency_resnet18_feature.csv`

Latensi dilaporkan dalam milidetik per citra, beserta median, p95, minimum, maksimum, dan FPS ekuivalen.

## Catatan penting tentang akurasi

Akurasi per epoch **tidak ditulis sebagai angka contoh** di repository ini. Angka tersebut harus berasal dari eksekusi aktual di komputer Ubuntu yang digunakan untuk eksperimen. Ini penting agar laporan tidak menggunakan hasil eksperimen yang dibuat-buat.

## Catatan reproduksibilitas

Gunakan seed 42 pada pembentukan dataset dan classifier. Simpan informasi hardware (CPU/GPU), versi Python, PyTorch, dan torchvision ketika melakukan eksperimen.

## Referensi metode

TorchVision mendokumentasikan ResNet-18 pretrained ImageNet dan preprocessing yang sesuai. ResNet-18 memiliki feature keluaran 512 dimensi sebelum layer `fc` pada implementasi standar.

