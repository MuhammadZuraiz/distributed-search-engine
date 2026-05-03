"""Integration tests for Flask API endpoints."""
import sys, os
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create a Flask test client."""
    import search.app as app_module
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def test_home_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_search_returns_results(client):
    resp = client.get("/?q=distributed")
    assert resp.status_code == 200
    assert b"distributed" in resp.data.lower()


def test_api_stats(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "total_documents" in data
    assert "unique_terms"    in data
    assert data["total_documents"] > 0
    assert data["unique_terms"]    > 0


def test_api_suggest(client):
    resp = client.get("/api/suggest?prefix=dist")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)


def test_api_suggest_short_prefix(client):
    resp = client.get("/api/suggest?prefix=a")
    data = resp.get_json()
    assert data["suggestions"] == []


def test_api_search_json(client):
    resp = client.get("/api/search?q=python&limit=5")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "results" in data
    assert "count"   in data
    assert len(data["results"]) <= 5


def test_api_document_valid(client):
    resp = client.get("/api/document/0")
    assert resp.status_code in [200, 404]   # 404 ok if doc 0 doesn't exist
    if resp.status_code == 200:
        data = resp.get_json()
        assert "url"   in data
        assert "title" in data


def test_api_document_invalid(client):
    resp = client.get("/api/document/999999")
    assert resp.status_code == 404


def test_api_spell(client):
    resp = client.get("/api/spell?w=distribted")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "correction" in data


def test_api_expand(client):
    resp = client.get("/api/expand?q=distributed")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "expansions" in data


def test_analytics_page(client):
    resp = client.get("/analytics")
    assert resp.status_code == 200


def test_api_analytics(client):
    resp = client.get("/api/analytics")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "total_queries"  in data
    assert "unique_queries" in data
    assert "top_queries"    in data