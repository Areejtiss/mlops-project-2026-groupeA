from fastapi.testclient import TestClient

from crediscore.api.main import app

client = TestClient(app)


def test_health_returns_200() -> None:
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_healthy_status() -> None:
    response = client.get("/health")

    assert response.json() == {"status": "healthy"}
