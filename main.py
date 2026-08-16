"""
main.py -- Ana calistirma dosyasi.

Tum egitim pipeline'ini yonetir:
  1. Seed ayarlama ve cikti dizinlerini olusturma
  2. Veri seti kontrolleri (sinif dogrulama, bozuk goruntu, duplicate, sizinti)
  3. Veri yukleme ve DataLoader olusturma
  4. Model olusturma
  5. Egitim (train + validation)
  6. Egitim grafiklerini kaydetme
  7. Test seti degerlendirmesi
  8. Sonuc ozeti

Kullanim:
    python main.py
"""

import os
import sys
import time

# Windows console'da UTF-8 karakter desteği
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import config
import utils
import dataset
from model import create_model, get_model_summary
from train import train_model
from evaluate import evaluate_model


def main():
    """Ana eğitim ve değerlendirme pipeline'ı."""

    total_start = time.time()

    print("=" * 70)
    print("  BEYIN TÜMÖRÜ MRI SINIFLANDIRMA SİSTEMİ")
    print("  Transfer Learning -- ResNet18 + ImageNet")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Seed ve dizinler
    # -----------------------------------------------------------------
    print("\n[1/8] Seed ayarlanıyor ve dizinler oluşturuluyor...")
    utils.set_seed(config.RANDOM_SEED)
    utils.ensure_dirs()
    print(f"[INFO] Cihaz: {config.DEVICE}")

    # -----------------------------------------------------------------
    # 2. Bozuk görüntü kontrolü
    # -----------------------------------------------------------------
    print("\n[2/8] Bozuk görüntüler kontrol ediliyor...")
    corrupted = dataset.check_corrupted_images(config.DATA_DIR)
    if corrupted:
        print(f"[UYARI] {len(corrupted)} bozuk görüntü bulundu. "
              f"Bu görüntüler yükleme sırasında atlanabilir.")
        # Bozuk dosya listesini kaydet
        corrupted_path = os.path.join(config.REPORTS_DIR, "corrupted_images.txt")
        with open(corrupted_path, "w", encoding="utf-8") as f:
            for fpath in corrupted:
                f.write(fpath + "\n")
        print(f"[SAVE] Bozuk görüntü listesi: {corrupted_path}")

    # -----------------------------------------------------------------
    # 3. Veri setini yükle
    # -----------------------------------------------------------------
    print("\n[3/8] Veri seti yükleniyor...")
    train_dataset, val_dataset, test_dataset = dataset.load_datasets()

    # Sınıf dağılımını yazdır ve kaydet
    data_counts = dataset.print_class_counts({
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    })

    # Sınıf dağılımı grafiği
    utils.plot_class_distribution(data_counts)

    # Örnek görüntüler
    print("\n[INFO] Örnek görüntüler oluşturuluyor...")
    utils.plot_sample_images(train_dataset, config.CLASS_NAMES)

    # -----------------------------------------------------------------
    # 4. Duplicate ve veri sızıntısı kontrolü
    # -----------------------------------------------------------------
    print("\n[4/8] Duplicate ve veri sızıntısı kontrolü...")
    dataset.check_duplicates_between_splits()
    dataset.verify_no_data_leakage(train_dataset, val_dataset, test_dataset)

    # -----------------------------------------------------------------
    # 5. DataLoader oluştur
    # -----------------------------------------------------------------
    print("\n[5/8] DataLoader'lar oluşturuluyor...")
    train_loader, val_loader, test_loader = dataset.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )

    # -----------------------------------------------------------------
    # 6. Model oluştur
    # -----------------------------------------------------------------
    print("\n[6/8] Model oluşturuluyor...")
    model = create_model(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False,
    )
    get_model_summary(model)

    # -----------------------------------------------------------------
    # 7. Eğitim
    # -----------------------------------------------------------------
    print("\n[7/8] Eğitim başlıyor...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=config.DEVICE,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        save_path=config.BEST_MODEL_PATH,
        class_names=config.CLASS_NAMES,
    )

    # Eğitim grafiklerini kaydet
    utils.plot_training_history(history)

    # -----------------------------------------------------------------
    # 8. Test seti değerlendirmesi
    # -----------------------------------------------------------------
    print("\n[8/8] Test seti değerlendirmesi...")
    metrics = evaluate_model(
        test_loader=test_loader,
        model_path=config.BEST_MODEL_PATH,
        device=config.DEVICE,
        class_names=config.CLASS_NAMES,
    )

    # -----------------------------------------------------------------
    # Sonuç özeti
    # -----------------------------------------------------------------
    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("  SONUÇ ÖZETİ")
    print("=" * 70)
    print(f"  Toplam süre         : {total_time:.1f}s ({total_time / 60:.1f} dakika)")
    print(f"  Cihaz               : {config.DEVICE}")
    print(f"  Test Accuracy       : {metrics['accuracy']:.4f} "
          f"({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Test F1 (Macro)     : {metrics['f1_macro']:.4f}")
    print(f"  Test F1 (Weighted)  : {metrics['f1_weighted']:.4f}")
    print(f"  Model kaydedildi    : {config.BEST_MODEL_PATH}")
    print(f"  Grafikler           : {config.FIGURES_DIR}")
    print(f"  Raporlar            : {config.REPORTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
