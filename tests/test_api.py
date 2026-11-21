from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_genres():
    res = client.get("/genres")
    assert res.status_code == 200
    assert "genres" in res.json()


def test_predict():
    payload = {
        "release_year": 2023,
        "genre": "drama",
        "director": "",
        "imdb_rating_count": 1000,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "estimated_rating" in data
    assert "high_rating_probability" in data
