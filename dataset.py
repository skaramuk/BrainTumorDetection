"""
dataset.py — Veri seti yükleme ve ön-işleme modülü.

Transform tanımları, ImageFolder yükleme, DataLoader oluşturma,
bozuk görüntü kontrolü, duplicate tespiti ve veri sızıntısı
doğrulaması bu modülde yapılır.
"""

import os
import hashlib
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

import config


# =============================================================================
# Transform Tanımları
# =============================================================================

def get_train_transform() -> transforms.Compose:
    """
    Eğitim verisi için transform zinciri oluşturur.
    Tıbbi görüntüler için hafif augmentation uygulanır:
      - RandomHorizontalFlip
      - Küçük açılı RandomRotation (10°)
      - Hafif RandomAffine (%5 öteleme)
    Agresif dönüşümler (vertical flip, 90/180° rotasyon) kullanılmaz.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_P),
        transforms.RandomRotation(degrees=config.ROTATION_DEGREES),
        transforms.RandomAffine(
            degrees=0,
            translate=config.AFFINE_TRANSLATE,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Validation ve test verisi için transform zinciri oluşturur.
    Yalnızca yeniden boyutlandırma ve normalizasyon uygulanır.
    Augmentation uygulanmaz.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


# =============================================================================
# Veri Seti Yükleme
# =============================================================================

def load_datasets() -> Tuple[datasets.ImageFolder,
                               datasets.ImageFolder,
                               datasets.ImageFolder]:
    """
    Train, validation ve test veri setlerini ImageFolder ile yükler.

    Returns:
        (train_dataset, val_dataset, test_dataset) tuple'ı.

    Raises:
        FileNotFoundError: Veri dizini bulunamazsa.
        ValueError: Beklenen sınıflar veri setinde yoksa.
    """
    # Dizinlerin varlığını kontrol et
    for name, path in [("Train", config.TRAIN_DIR),
                       ("Val", config.VAL_DIR),
                       ("Test", config.TEST_DIR)]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{name} dizini bulunamadı: {path}")

    # Veri setlerini yükle
    train_dataset = datasets.ImageFolder(
        root=config.TRAIN_DIR,
        transform=get_train_transform(),
    )
    val_dataset = datasets.ImageFolder(
        root=config.VAL_DIR,
        transform=get_eval_transform(),
    )
    test_dataset = datasets.ImageFolder(
        root=config.TEST_DIR,
        transform=get_eval_transform(),
    )

    # Sınıf sırasını doğrula
    _validate_classes(train_dataset, "Train")
    _validate_classes(val_dataset, "Val")
    _validate_classes(test_dataset, "Test")

    print(f"[DATA] Train : {len(train_dataset)} görüntü")
    print(f"[DATA] Val   : {len(val_dataset)} görüntü")
    print(f"[DATA] Test  : {len(test_dataset)} görüntü")

    return train_dataset, val_dataset, test_dataset


def _validate_classes(dataset: datasets.ImageFolder, split_name: str) -> None:
    """
    ImageFolder tarafından algılanan sınıfların config'deki sınıflarla
    eşleştiğini doğrular.

    Args:
        dataset: ImageFolder veri seti.
        split_name: Ayrım adı (Train/Val/Test) — hata mesajı için.

    Raises:
        ValueError: Sınıf uyumsuzluğu varsa.
    """
    detected_classes = dataset.classes
    expected_classes = config.CLASS_NAMES

    if detected_classes != expected_classes:
        raise ValueError(
            f"[{split_name}] Sınıf uyumsuzluğu!\n"
            f"  Beklenen : {expected_classes}\n"
            f"  Algılanan: {detected_classes}"
        )


# =============================================================================
# DataLoader Oluşturma
# =============================================================================

def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
    batch_size: int = config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train, validation ve test DataLoader'larını oluşturur.

    Args:
        train_dataset: Eğitim veri seti.
        val_dataset: Doğrulama veri seti.
        test_dataset: Test veri seti.
        batch_size: Batch boyutu.

    Returns:
        (train_loader, val_loader, test_loader) tuple'ı.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=False,
    )

    print(f"[DATA] DataLoader'lar oluşturuldu (batch_size={batch_size}, "
          f"num_workers={config.NUM_WORKERS})")

    return train_loader, val_loader, test_loader


# =============================================================================
# Sınıf Dağılımı
# =============================================================================

def get_class_counts(dataset: datasets.ImageFolder) -> Dict[str, int]:
    """
    Veri setindeki her sınıfın görüntü sayısını döndürür.

    Args:
        dataset: ImageFolder veri seti.

    Returns:
        Sınıf adı → görüntü sayısı sözlüğü.
    """
    label_counts = Counter([label for _, label in dataset.samples])
    class_counts = {}
    for cls_name in dataset.classes:
        cls_idx = dataset.class_to_idx[cls_name]
        class_counts[cls_name] = label_counts.get(cls_idx, 0)
    return class_counts


def print_class_counts(datasets_dict: Dict[str, datasets.ImageFolder]) -> Dict[str, Dict[str, int]]:
    """
    Tüm veri ayrımları için sınıf sayılarını yazdırır.

    Args:
        datasets_dict: {'train': dataset, 'val': dataset, 'test': dataset}

    Returns:
        Her ayrım için sınıf sayıları sözlüğü.
    """
    all_counts = {}
    for split_name, dataset in datasets_dict.items():
        counts = get_class_counts(dataset)
        all_counts[split_name] = counts
        print(f"\n[DATA] {split_name.upper()} Sınıf Dağılımı:")
        for cls_name, count in counts.items():
            print(f"  {cls_name:15s}: {count:5d}")
        print(f"  {'TOPLAM':15s}: {sum(counts.values()):5d}")

    return all_counts


# =============================================================================
# Bozuk Görüntü Kontrolü
# =============================================================================

def check_corrupted_images(data_dir: str) -> List[str]:
    """
    Belirtilen dizindeki bozuk veya okunamayan görüntüleri tespit eder.

    Args:
        data_dir: Kontrol edilecek üst dizin (data/).

    Returns:
        Bozuk dosya yollarının listesi.
    """
    corrupted = []
    total_checked = 0

    for root, _, files in os.walk(data_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            total_checked += 1
            try:
                with Image.open(fpath) as img:
                    img.verify()  # Dosyanın geçerli bir görüntü olduğunu doğrula
            except Exception as e:
                corrupted.append(fpath)
                print(f"  [!] Bozuk goruntu: {fpath} - {e}")

    if corrupted:
        print(f"[CHECK] {len(corrupted)} bozuk görüntü tespit edildi "
              f"(toplam {total_checked} dosya kontrol edildi).")
    else:
        print(f"[CHECK] Bozuk goruntu yok (OK) ({total_checked} dosya kontrol edildi).")

    return corrupted


# =============================================================================
# Duplicate (Hash) Kontrolü
# =============================================================================

def _compute_file_hash(file_path: str, algorithm: str = "md5") -> str:
    """
    Bir dosyanın hash değerini hesaplar.

    Args:
        file_path: Dosya yolu.
        algorithm: Hash algoritması (varsayılan: md5).

    Returns:
        Hex formatında hash string'i.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_duplicates_between_splits(
    train_dir: str = config.TRAIN_DIR,
    val_dir: str = config.VAL_DIR,
    test_dir: str = config.TEST_DIR,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Train, validation ve test setleri arasında aynı dosya olup
    olmadığını MD5 hash kontrolüyle inceler.

    Args:
        train_dir: Eğitim dizini.
        val_dir: Doğrulama dizini.
        test_dir: Test dizini.

    Returns:
        Duplicate çiftlerini içeren sözlük.
    """
    def _get_hashes(directory: str) -> Dict[str, str]:
        """Dizindeki tüm dosyaların hash'lerini hesaplar."""
        hashes = {}
        for root, _, files in os.walk(directory):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    h = _compute_file_hash(fpath)
                    hashes[fpath] = h
                except Exception:
                    pass
        return hashes

    print("[CHECK] Ayrımlar arası duplicate kontrolü yapılıyor (MD5)...")

    train_hashes = _get_hashes(train_dir)
    val_hashes = _get_hashes(val_dir)
    test_hashes = _get_hashes(test_dir)

    duplicates = {
        "train_val": [],
        "train_test": [],
        "val_test": [],
    }

    # Hash → dosya yolu eşlemesi oluştur
    train_hash_to_path = {}
    for path, h in train_hashes.items():
        train_hash_to_path.setdefault(h, []).append(path)

    val_hash_to_path = {}
    for path, h in val_hashes.items():
        val_hash_to_path.setdefault(h, []).append(path)

    # Train-Val kontrolü
    common_train_val = set(train_hashes.values()) & set(val_hashes.values())
    for h in common_train_val:
        for tp in train_hash_to_path.get(h, []):
            for vp in val_hash_to_path.get(h, []):
                duplicates["train_val"].append((tp, vp))

    # Train-Test kontrolü
    test_hash_set = set(test_hashes.values())
    common_train_test = set(train_hashes.values()) & test_hash_set
    test_hash_to_path = {}
    for path, h in test_hashes.items():
        test_hash_to_path.setdefault(h, []).append(path)

    for h in common_train_test:
        for tp in train_hash_to_path.get(h, []):
            for tsp in test_hash_to_path.get(h, []):
                duplicates["train_test"].append((tp, tsp))

    # Val-Test kontrolü
    common_val_test = set(val_hashes.values()) & test_hash_set
    for h in common_val_test:
        for vp in val_hash_to_path.get(h, []):
            for tsp in test_hash_to_path.get(h, []):
                duplicates["val_test"].append((vp, tsp))

    # Sonuçları yazdır
    total_dups = sum(len(v) for v in duplicates.values())
    if total_dups == 0:
        print("[CHECK] Ayrimlar arasi duplicate bulunamadi (OK)")
    else:
        print(f"[CHECK] UYARI: {total_dups} duplicate cifti tespit edildi:")
        for pair_name, pairs in duplicates.items():
            if pairs:
                print(f"  {pair_name}: {len(pairs)} çift")
                for p1, p2 in pairs[:5]:  # İlk 5'ini göster
                    print(f"    {p1}")
                    print(f"    <-> {p2}")

    return duplicates


def verify_no_data_leakage(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
) -> bool:
    """
    Train, validation ve test setleri arasında dosya yolu bazında
    veri sızıntısı olmadığını doğrular.

    Args:
        train_dataset: Eğitim veri seti.
        val_dataset: Doğrulama veri seti.
        test_dataset: Test veri seti.

    Returns:
        True ise sızıntı yok, False ise sızıntı var.
    """
    # Dosya adlarını al (tam yol yerine sadece dosya adı)
    train_files = {os.path.basename(s[0]) for s in train_dataset.samples}
    val_files = {os.path.basename(s[0]) for s in val_dataset.samples}
    test_files = {os.path.basename(s[0]) for s in test_dataset.samples}

    leaks = {
        "train ∩ val": train_files & val_files,
        "train ∩ test": train_files & test_files,
        "val ∩ test": val_files & test_files,
    }

    has_leak = False
    for pair, common in leaks.items():
        if common:
            has_leak = True
            print(f"[CHECK] UYARI: Dosya adi cakismasi ({pair}): {len(common)} dosya")
            for f in list(common)[:5]:
                print(f"    {f}")

    if not has_leak:
        print("[CHECK] Dosya adi bazinda veri sizintisi yok (OK)")

    return not has_leak
