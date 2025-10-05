"""
Contract tests for Curiosity Mission API (Spec 029 / Spec 002 curiosity engine integration)

TDD RED phase: these tests should fail against placeholder endpoints.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client() -> TestClient:
    from fastapi import FastAPI
    from backend.src.api.routes.curiosity import router as curiosity_router

    app = FastAPI()
    app.include_router(curiosity_router, prefix="/api/v1")
    return TestClient(app)


def sample_payload():  # pragma: no cover - helper
    return {
        "user_id": "user-123",
        "mission_title": "Investigate quantum neural networks",
        "mission_description": "Understand prediction-error spikes around quantum neural networks.",
        "primary_curiosity_type": "epistemic",
        "research_questions": [
            "What are the latest breakthroughs in quantum neural networks?",
            "Which datasets are required to reproduce key experiments?"
        ]
    }


class TestCuriosityMissionAPI:
    """Contract tests covering CRUD behaviour for curiosity missions."""

    def test_create_curiosity_mission(self, client: TestClient):
        payload = sample_payload()

        response = client.post("/api/v1/curiosity/missions", json=payload)
        assert response.status_code == 201

        data = response.json()
        assert data["mission_id"].startswith("cm_") or data["mission_id"], "Mission ID must be returned"
        assert data["mission_status"] == "forming"
        assert data["primary_curiosity_type"] == payload["primary_curiosity_type"]
        assert data["research_questions"] == payload["research_questions"]

    def test_list_curiosity_missions_returns_created_mission(self, client: TestClient):
        payload = sample_payload()
        client.post("/api/v1/curiosity/missions", json=payload)

        response = client.get("/api/v1/curiosity/missions")
        assert response.status_code == 200

        missions = response.json().get("missions", [])
        assert missions, "Mission list should include newly created mission"
        assert missions[0]["mission_title"] == payload["mission_title"]

    def test_get_curiosity_mission_by_id(self, client: TestClient):
        payload = sample_payload()
        creation_resp = client.post("/api/v1/curiosity/missions", json=payload)
        mission_id = creation_resp.json()["mission_id"]

        response = client.get(f"/api/v1/curiosity/missions/{mission_id}")
        assert response.status_code == 200

        mission = response.json()
        assert mission["mission_id"] == mission_id
        assert mission["mission_title"] == payload["mission_title"]

    def test_update_curiosity_mission_status(self, client: TestClient):
        payload = sample_payload()
        creation_resp = client.post("/api/v1/curiosity/missions", json=payload)
        mission_id = creation_resp.json()["mission_id"]

        update_resp = client.patch(
            f"/api/v1/curiosity/missions/{mission_id}",
            json={"mission_status": "exploring"}
        )
        assert update_resp.status_code == 200
        data = update_resp.json()
        assert data["mission_status"] == "exploring"

    def test_create_mission_requires_fields(self, client: TestClient):
        response = client.post("/api/v1/curiosity/missions", json={})
        assert response.status_code in (400, 422), "Missing required fields should be rejected"
