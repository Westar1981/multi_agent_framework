"""Unit tests for pointcut API routes."""

import pytest
from fastapi.testclient import TestClient
from multi_agent_framework.web.app import app
from multi_agent_framework.core.pointcuts.manager import PointcutType, PatternType

client = TestClient(app)

def test_create_pointcut():
    """Test creating a pointcut via API."""
    response = client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION,
        "pattern_type": PatternType.REGEX,
        "metadata": {"description": "Test pointcut"}
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pattern"] == "test_.*"
    assert data["type"] == PointcutType.METHOD_EXECUTION
    assert data["pattern_type"] == PatternType.REGEX
    assert data["enabled"] is True
    assert data["metadata"]["description"] == "Test pointcut"

def test_create_wildcard_pointcut():
    """Test creating a wildcard pointcut."""
    response = client.post("/api/pointcuts/", json={
        "pattern": "test_*.py",
        "type": PointcutType.METHOD_EXECUTION,
        "pattern_type": PatternType.WILDCARD,
        "metadata": {"description": "Wildcard test"}
    })
    assert response.status_code == 200
    data = response.json()
    assert data["pattern"] == "test_*.py"
    assert data["pattern_type"] == PatternType.WILDCARD

def test_create_composite_pointcut():
    """Test creating a composite pointcut."""
    # Create two pointcuts to combine
    p1 = client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION
    }).json()
    
    p2 = client.post("/api/pointcuts/", json={
        "pattern": ".*_test",
        "type": PointcutType.METHOD_EXECUTION
    }).json()
    
    # Create composite
    response = client.post("/api/pointcuts/composite", json={
        "operator": "AND",
        "pointcut_ids": [p1["id"], p2["id"]]
    })
    assert response.status_code == 200
    data = response.json()
    assert "composite" in data["id"]

def test_composite_matching():
    """Test matching with composite pointcuts."""
    # Create pointcuts
    p1 = client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION
    }).json()
    
    p2 = client.post("/api/pointcuts/", json={
        "pattern": ".*_test",
        "type": PointcutType.METHOD_EXECUTION
    }).json()
    
    # Create AND composite
    and_composite = client.post("/api/pointcuts/composite", json={
        "operator": "AND",
        "pointcut_ids": [p1["id"], p2["id"]]
    }).json()
    
    # Test matching
    response = client.post(f"/api/pointcuts/{and_composite['id']}/check", json={
        "target": "test_something_test"
    })
    assert response.status_code == 200
    assert response.json()["matches"] is True
    
    response = client.post(f"/api/pointcuts/{and_composite['id']}/check", json={
        "target": "test_only"
    })
    assert response.status_code == 200
    assert response.json()["matches"] is False

def test_get_match_stats():
    """Test retrieving match statistics."""
    # Create a pointcut
    response = client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION
    })
    pointcut_id = response.json()["id"]
    
    # Perform some matches
    client.post(f"/api/pointcuts/{pointcut_id}/check", json={
        "target": "test_match"
    })
    client.post(f"/api/pointcuts/{pointcut_id}/check", json={
        "target": "no_match"
    })
    
    # Get stats
    response = client.get(f"/api/pointcuts/{pointcut_id}/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats["checks"] == 2
    assert stats["matches"] == 1

def test_create_invalid_pointcut():
    """Test creating an invalid pointcut."""
    response = client.post("/api/pointcuts/", json={
        "pattern": "[invalid",
        "type": PointcutType.METHOD_EXECUTION
    })
    assert response.status_code == 400

def test_get_pointcuts():
    """Test getting all pointcuts."""
    # Create a test pointcut first
    client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION
    })
    
    response = client.get("/api/pointcuts/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert isinstance(data, list)

def test_get_pointcut():
    """Test getting a specific pointcut."""
    # Create a test pointcut
    create_response = client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION
    })
    pointcut_id = create_response.json()["id"]
    
    response = client.get(f"/api/pointcuts/{pointcut_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == pointcut_id
    assert data["pattern"] == "test_.*"

def test_get_nonexistent_pointcut():
    """Test getting a nonexistent pointcut."""
    response = client.get("/api/pointcuts/nonexistent")
    assert response.status_code == 404

def test_delete_pointcut():
    """Test deleting a pointcut."""
    # Create a test pointcut
    create_response = client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION
    })
    pointcut_id = create_response.json()["id"]
    
    response = client.delete(f"/api/pointcuts/{pointcut_id}")
    assert response.status_code == 200
    
    # Verify deletion
    get_response = client.get(f"/api/pointcuts/{pointcut_id}")
    assert get_response.status_code == 404

def test_enable_disable_pointcut():
    """Test enabling and disabling a pointcut."""
    # Create a test pointcut
    create_response = client.post("/api/pointcuts/", json={
        "pattern": "test_.*",
        "type": PointcutType.METHOD_EXECUTION
    })
    pointcut_id = create_response.json()["id"]
    
    # Disable pointcut
    response = client.post(f"/api/pointcuts/{pointcut_id}/disable")
    assert response.status_code == 200
    
    # Verify disabled
    get_response = client.get(f"/api/pointcuts/{pointcut_id}")
    assert get_response.json()["enabled"] is False
    
    # Enable pointcut
    response = client.post(f"/api/pointcuts/{pointcut_id}/enable")
    assert response.status_code == 200
    
    # Verify enabled
    get_response = client.get(f"/api/pointcuts/{pointcut_id}")
    assert get_response.json()["enabled"] is True

def test_enable_disable_nonexistent_pointcut():
    """Test enabling/disabling a nonexistent pointcut."""
    response = client.post("/api/pointcuts/nonexistent/enable")
    assert response.status_code == 404
    
    response = client.post("/api/pointcuts/nonexistent/disable")
    assert response.status_code == 404 