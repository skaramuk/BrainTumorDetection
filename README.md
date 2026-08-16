# 🧠 Beyin Tümörü MRI Sınıflandırma Sistemi

Transfer learning (ResNet18 + ImageNet) kullanarak beyin MRI görüntülerini 4 farklı sınıfa ayıran bir derin öğrenme projesi.

> ⚠️ **UYARI**: Bu proje yalnızca **eğitim ve araştırma amaçlıdır**. Tıbbi tanı, teşhis veya tedavi amacıyla kullanılamaz. Tıbbi değerlendirme için mutlaka uzman bir hekime başvurun.

---

## 📌 Projenin Amacı

MRI görüntüsünü analiz ederek aşağıdaki 4 sınıftan birine sınıflandıran, transfer learning tabanlı, düzenli ve yeniden kullanılabilir bir görüntü sınıflandırma sistemi geliştirmek.

**Sınıflar:**

| Sınıf | Açıklama |
|---|---|
| `glioma` | Glioma tümörü |
| `meningioma` | Meningiom tümörü |
| `notumor` | Tümör yok (normal) |
| `pituitary` | Hipofiz tümörü |

---

## 📊 Veri Seti Açıklaması

Veri seti 6507 adet beyin MRI görüntüsü içermektedir ve 3 ayrıma bölünmüştür:

| Sınıf | Train | Val | Test | Toplam |
|---|---|---|---|---|
| glioma | 1258 | 139 | 382 | 1779 |
| meningioma | 1237 | 137 | 390 | 1764 |
| notumor | 820 | 91 | 303 | 1214 |
| pituitary | 1215 | 135 | 400 | 1750 |
| **Toplam** | **4530** | **502** | **1475** | **6507** |

---

## 📁 Klasör Yapısı

```
BrainTumorDetection/
│
├── main.py              # Ana çalıştırma dosyası
├── train.py             # Eğitim ve doğrulama döngüsü
├── evaluate.py          # Test seti değerlendirmesi
├── predict.py           # Tek görüntü tahmini
├── dataset.py           # Veri yükleme ve ön-işleme
├── model.py             # ResNet18 model tanımı
├── config.py            # Konfigürasyon ve hyperparameter'lar
├── utils.py             # Yardımcı fonksiyonlar
├── requirements.txt     # Bağımlılıklar
├── README.md            # Bu dosya
│
├── data/                # Veri seti (Git'e dahil edilmez)
│   ├── train/           # Eğitim verileri
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   ├── val/             # Doğrulama verileri
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/            # Test verileri
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
│
├── models/              # Kaydedilmiş model ağırlıkları
│   └── best_model.pth
│
└── outputs/             # Çıktılar
    ├── figures/         # Grafikler ve görseller
    │   ├── training_history.png
    │   ├── confusion_matrix.png
    │   ├── class_distribution.png
    │   └── sample_images.png
    └── reports/         # Raporlar
        ├── evaluation_results.txt
        └── evaluation_results.csv
```

---

## ⚙️ Kurulum

### 1. Python ortamı oluşturun (önerilen: 3.9+)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 2. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

> **Not**: GPU kullanmak için PyTorch'un CUDA sürümünü yükleyin:
> [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 📦 Gerekli Kütüphaneler

| Kütüphane | Versiyon | Açıklama |
|---|---|---|
| `torch` | ≥ 2.0.0 | PyTorch derin öğrenme framework'ü |
| `torchvision` | ≥ 0.15.0 | Görüntü işleme ve önceden eğitilmiş modeller |
| `numpy` | ≥ 1.24.0 | Sayısal hesaplamalar |
| `matplotlib` | ≥ 3.7.0 | Grafik çizme |
| `seaborn` | ≥ 0.12.0 | İstatistiksel görselleştirme |
| `scikit-learn` | ≥ 1.2.0 | Metrik hesaplama |
| `Pillow` | ≥ 9.4.0 | Görüntü yükleme |
| `tqdm` | ≥ 4.65.0 | İlerleme çubuğu |

---

## 🚀 Model Eğitimi

Tüm pipeline'ı başlatmak için:

```bash
python main.py
```

Bu komut sırasıyla:
1. Random seed ayarlar
2. Bozuk görüntü kontrolü yapar
3. Veri setini yükler ve sınıf dağılımını gösterir
4. Duplicate ve veri sızıntısı kontrolü yapar
5. ResNet18 modelini oluşturur
6. Modeli eğitir (early stopping ile)
7. Eğitim grafiklerini kaydeder
8. Test seti üzerinde nihai değerlendirme yapar

### Hyperparameter'ları Değiştirme

`config.py` dosyasını düzenleyerek tüm ayarları değiştirebilirsiniz:

```python
BATCH_SIZE = 32        # Batch boyutu
LEARNING_RATE = 1e-4   # Öğrenme oranı
EPOCHS = 30            # Maksimum epoch sayısı
RANDOM_SEED = 42       # Tekrar üretilebilirlik
```

---

## 📈 Model Değerlendirme

Eğitim tamamlandıktan sonra modeli test seti üzerinde ayrıca değerlendirmek için:

```python
from evaluate import evaluate_model
from dataset import load_datasets, create_dataloaders

_, _, test_dataset = load_datasets()
_, _, test_loader = create_dataloaders(test_dataset, test_dataset, test_dataset)
metrics = evaluate_model(test_loader)
```

---

## 🔍 Tek Görüntü Tahmini

Eğitilmiş modeli kullanarak tek bir MRI görüntüsünü sınıflandırmak için:

```bash
python predict.py --image path/to/mri_image.jpg
```

Farklı bir model dosyası kullanmak için:

```bash
python predict.py --image path/to/mri_image.jpg --model models/best_model.pth
```

**Çıktı örneği:**
```
  Tahmin Edilen Sınıf : glioma
  Güven Skoru         : 0.9523 (95.23%)

  Sınıf Olasılıkları:
  glioma         : 0.9523 (95.23%) ████████████████████████████ ◄
  meningioma     : 0.0312 ( 3.12%) █
  pituitary      : 0.0102 ( 1.02%)
  notumor        : 0.0063 ( 0.63%)
```

---

## 📊 Kullanılan Metrikler

| Metrik | Açıklama |
|---|---|
| **Accuracy** | Genel doğruluk oranı |
| **Precision** | Her sınıf için kesinlik |
| **Recall** | Her sınıf için duyarlılık |
| **F1-Score (Macro)** | Sınıfların ağırlıksız ortalaması |
| **F1-Score (Weighted)** | Sınıf sayılarına göre ağırlıklı ortalama |
| **Classification Report** | sklearn detaylı sınıflandırma raporu |
| **Confusion Matrix** | Karışıklık matrisi (görsel + sayısal) |

---

## 📉 Elde Edilen Sonuçlar

Eğitim tamamlandıktan sonra sonuçlar aşağıdaki dosyalarda bulunabilir:

- **Grafikler**: `outputs/figures/` dizini altında
  - `training_history.png` — Train/Val loss ve accuracy grafikleri
  - `confusion_matrix.png` — Karışıklık matrisi görseli
  - `class_distribution.png` — Sınıf dağılımı çubuk grafiği
  - `sample_images.png` — Her sınıftan örnek MRI görüntüleri

- **Raporlar**: `outputs/reports/` dizini altında
  - `evaluation_results.txt` — Detaylı metin raporu
  - `evaluation_results.csv` — CSV formatında metrikler

---

## ⚠️ Projenin Sınırlılıkları

1. **Veri seti boyutu**: ~6500 görüntü nispeten küçük bir veri setidir. Daha büyük veri setleri ile daha iyi sonuçlar elde edilebilir.
2. **Sınıf dengesizliği**: `notumor` sınıfı diğer sınıflardan daha az görüntüye sahiptir.
3. **Tek model**: Yalnızca ResNet18 kullanılmıştır. ResNet50, EfficientNet gibi farklı mimariler denenebilir.
4. **2D görüntü**: 3D hacimsel MRI verileri değil, 2D kesit görüntüleri kullanılmaktadır.
5. **Augmentation**: Tıbbi görüntü özelliklerine uygun olmak adına yalnızca hafif augmentation uygulanmıştır.
6. **Genelleme**: Model, yalnızca bu veri setinin dağılımı üzerinde eğitilmiştir. Farklı hastanelerin veya cihazların verileri üzerinde performansı değişebilir.

---

## 🚨 Tıbbi Uyarı

> **Bu proje yalnızca eğitim ve araştırma amaçlıdır.**
>
> Bu model, tıbbi tanı, teşhis veya tedavi kararları için **kesinlikle kullanılamaz**.
> Beyin tümörü tanısı yalnızca uzman radyologlar ve nörologlar tarafından,
> kapsamlı klinik değerlendirme ile konulabilir.
>
> Sağlık sorunlarınız için mutlaka bir sağlık profesyoneline danışın.

---

## 🛠️ Teknik Detaylar

| Özellik | Değer |
|---|---|
| Model | ResNet18 (ImageNet pretrained) |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |
| Scheduler | ReduceLROnPlateau |
| Görüntü Boyutu | 224 × 224 |
| Normalizasyon | ImageNet mean/std |
| Early Stopping | Patience = 7 |
| Framework | PyTorch |
