"""
config.py — Proje konfigürasyon dosyası.

Tüm hyperparameter'lar, veri yolları ve eğitim ayarları burada
tanımlanır. Değişiklik yapmak için yalnızca bu dosyayı düzenlemeniz
yeterlidir.
"""

import os
import torch

# =============================================================================
# Proje Kök Dizini
# =============================================================================
# Bu dosyanın bulunduğu dizin proje kökü olarak kabul edilir.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Veri Yolları
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

# =============================================================================
# Çıktı Yolları
# =============================================================================
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")

# =============================================================================
# Sınıf Bilgileri
# =============================================================================
# Sınıf isimleri — ImageFolder alfabetik sıraya göre indeksler.
# Bu sıranın doğru olduğu dataset.py içinde doğrulanır.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# Görüntü Ayarları
# =============================================================================
IMAGE_SIZE = 224  # ResNet18 giriş boyutu

# ImageNet normalizasyon değerleri
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Eğitim Hyperparameter'ları
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30
WEIGHT_DECAY = 1e-4  # AdamW regularization

# =============================================================================
# Early Stopping
# =============================================================================
EARLY_STOPPING_PATIENCE = 7  # Kaç epoch iyileşme olmazsa dur
EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum iyileşme eşiği

# =============================================================================
# Learning Rate Scheduler
# =============================================================================
# ReduceLROnPlateau ayarları
SCHEDULER_FACTOR = 0.5  # LR'yi yarıya düşür
SCHEDULER_PATIENCE = 3  # Kaç epoch bekle
SCHEDULER_MIN_LR = 1e-7  # Minimum learning rate

# =============================================================================
# Augmentation Ayarları (Hafif — Tıbbi görüntüler için uygun)
# =============================================================================
ROTATION_DEGREES = 10  # Maksimum dönüş açısı
AFFINE_TRANSLATE = (0.05, 0.05)  # Yatay/dikey öteleme oranı
HORIZONTAL_FLIP_P = 0.5  # Yatay çevirme olasılığı

# =============================================================================
# DataLoader Ayarları
# =============================================================================
# Windows + PyCharm ortamında multiprocessing sorunlarını önlemek için
# num_workers=0 kullanılır. Linux/Mac'te artırılabilir.
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

# =============================================================================
# Tekrar Üretilebilirlik
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Cihaz Seçimi
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# API / Sunum Katmanı Ayarları (backend + frontend tarafından kullanılır)
# =============================================================================
# Tahmin güven eşiği — bu değerin altındaki tahminler "Review Required" olarak işaretlenir.
CONFIDENCE_THRESHOLD = 0.80
# İlk iki sınıf arasındaki minimum olasılık farkı eşiği.
MARGIN_THRESHOLD = 0.20

# Analiz geçmişinin kalıcı olarak saklandığı SQLite veritabanı yolu.
DB_PATH = os.path.join(PROJECT_ROOT, "outputs", "history.db")
