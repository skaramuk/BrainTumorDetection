"""tests/test_dataset.py — dataset.py testleri (sentetik tmp_path klasor agaci ile)."""

import shutil

import torch

import config
import dataset


def test_train_transform_output_shape(sample_image):
    tensor = dataset.get_train_transform()(sample_image)
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_eval_transform_output_shape(sample_image):
    tensor = dataset.get_eval_transform()(sample_image)
    assert tensor.shape == (3, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_load_datasets_and_class_counts(synthetic_dataset_dir, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_DIR", str(synthetic_dataset_dir / "train"))
    monkeypatch.setattr(config, "VAL_DIR", str(synthetic_dataset_dir / "val"))
    monkeypatch.setattr(config, "TEST_DIR", str(synthetic_dataset_dir / "test"))

    train_ds, val_ds, test_ds = dataset.load_datasets()

    assert len(train_ds) == len(config.CLASS_NAMES) * 2
    assert train_ds.classes == config.CLASS_NAMES

    counts = dataset.get_class_counts(train_ds)
    assert all(count == 2 for count in counts.values())


def test_check_corrupted_images_clean(synthetic_dataset_dir):
    corrupted = dataset.check_corrupted_images(str(synthetic_dataset_dir))
    assert corrupted == []


def test_check_corrupted_images_detects_bad_file(synthetic_dataset_dir):
    bad_path = synthetic_dataset_dir / "train" / config.CLASS_NAMES[0] / "bozuk.jpg"
    bad_path.write_bytes(b"bu gecerli bir goruntu degil")

    corrupted = dataset.check_corrupted_images(str(synthetic_dataset_dir))

    assert str(bad_path) in corrupted


def test_check_duplicates_between_splits_detects_pair(synthetic_dataset_dir):
    cls = config.CLASS_NAMES[0]
    train_dir = synthetic_dataset_dir / "train"
    val_dir = synthetic_dataset_dir / "val"
    test_dir = synthetic_dataset_dir / "test"

    original = train_dir / cls / f"train_{cls}_0.jpg"
    duplicate = val_dir / cls / "kopya.jpg"
    shutil.copyfile(original, duplicate)

    duplicates = dataset.check_duplicates_between_splits(
        train_dir=str(train_dir), val_dir=str(val_dir), test_dir=str(test_dir)
    )

    pairs = duplicates["train_val"]
    assert any(str(original) in pair and str(duplicate) in pair for pair in pairs)


def test_verify_no_data_leakage_clean(synthetic_dataset_dir, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_DIR", str(synthetic_dataset_dir / "train"))
    monkeypatch.setattr(config, "VAL_DIR", str(synthetic_dataset_dir / "val"))
    monkeypatch.setattr(config, "TEST_DIR", str(synthetic_dataset_dir / "test"))
    train_ds, val_ds, test_ds = dataset.load_datasets()

    assert dataset.verify_no_data_leakage(train_ds, val_ds, test_ds) is True


def test_verify_no_data_leakage_detects_filename_collision(synthetic_dataset_dir, monkeypatch):
    cls = config.CLASS_NAMES[0]
    # val'e, train'deki bir dosyayla AYNI ada sahip yeni bir dosya ekle
    colliding_name = f"train_{cls}_0.jpg"
    shutil.copyfile(
        synthetic_dataset_dir / "train" / cls / colliding_name,
        synthetic_dataset_dir / "val" / cls / colliding_name,
    )

    monkeypatch.setattr(config, "TRAIN_DIR", str(synthetic_dataset_dir / "train"))
    monkeypatch.setattr(config, "VAL_DIR", str(synthetic_dataset_dir / "val"))
    monkeypatch.setattr(config, "TEST_DIR", str(synthetic_dataset_dir / "test"))
    train_ds, val_ds, test_ds = dataset.load_datasets()

    assert dataset.verify_no_data_leakage(train_ds, val_ds, test_ds) is False
