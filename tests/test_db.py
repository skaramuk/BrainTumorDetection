"""tests/test_db.py — backend/db.py testleri (gecici sqlite dosyasina karsi)."""

from backend import db


SAMPLE_PROBS = [
    {"class_name": "glioma", "probability": 0.1},
    {"class_name": "meningioma", "probability": 0.2},
    {"class_name": "notumor", "probability": 0.6},
    {"class_name": "pituitary", "probability": 0.1},
]


def test_insert_and_get_history(temp_db):
    row = db.insert_analysis(
        filename="ornek.jpg",
        predicted_class="notumor",
        confidence=0.6,
        top2_class="meningioma",
        top2_probability=0.2,
        margin=0.4,
        status="High Confidence",
        probabilities=SAMPLE_PROBS,
    )
    assert row["id"] == 1
    assert row["created_at"]

    history = db.get_history()
    assert len(history) == 1
    assert history[0]["filename"] == "ornek.jpg"
    assert history[0]["predicted_class"] == "notumor"
    # probabilities_json ham (JSON'a cevrilmemis) bir string olarak donuyor -- sozlesme bu
    assert isinstance(history[0]["probabilities_json"], str)


def test_get_history_orders_newest_first(temp_db):
    db.insert_analysis("a.jpg", "glioma", 0.9, "meningioma", 0.05, 0.85, "High Confidence", SAMPLE_PROBS)
    db.insert_analysis("b.jpg", "pituitary", 0.9, "notumor", 0.05, 0.85, "High Confidence", SAMPLE_PROBS)

    history = db.get_history()
    assert [h["filename"] for h in history] == ["b.jpg", "a.jpg"]


def test_get_history_limit(temp_db):
    for i in range(5):
        db.insert_analysis(f"img_{i}.jpg", "glioma", 0.9, "meningioma", 0.05, 0.85, "High Confidence", SAMPLE_PROBS)

    assert len(db.get_history(limit=2)) == 2
    assert len(db.get_history()) == 5


def test_clear_history(temp_db):
    db.insert_analysis("a.jpg", "glioma", 0.9, "meningioma", 0.05, 0.85, "High Confidence", SAMPLE_PROBS)
    db.insert_analysis("b.jpg", "glioma", 0.9, "meningioma", 0.05, 0.85, "High Confidence", SAMPLE_PROBS)

    deleted = db.clear_history()

    assert deleted == 2
    assert db.get_history() == []
