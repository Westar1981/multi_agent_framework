"""Tests for pointcut dashboard."""

import io
import json
import pytest
from unittest.mock import patch, MagicMock
from flask.testing import FlaskClient

from ..visualization.pointcut_dashboard import (
    app, pointcut_manager, PointcutType,
    PatternValidator, PatternSuggestion, ValidationResult
)

@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def validator():
    """Create test pattern validator."""
    return PatternValidator()

def test_pattern_validation(validator):
    """Test pattern validation."""
    # Test valid pattern
    result = validator.validate_pattern(r"get\w+")
    assert result.is_valid
    assert not result.errors
    assert any(s.pattern == r"get\w+" for s in result.suggestions)
    
    # Test invalid pattern
    result = validator.validate_pattern("[invalid")
    assert not result.is_valid
    assert result.errors
    assert "Invalid regex pattern" in result.errors[0]
    
    # Test empty pattern
    result = validator.validate_pattern("")
    assert not result.is_valid
    assert "Pattern cannot be empty" in result.errors
    
    # Test too broad pattern
    result = validator.validate_pattern(".*")
    assert result.is_valid
    assert "Pattern matches everything" in result.warnings
    
    # Test single character pattern
    result = validator.validate_pattern("a")
    assert result.is_valid
    assert "Single character pattern" in result.warnings
    
    # Test too many wildcards
    result = validator.validate_pattern("a*b*c*d*")
    assert result.is_valid
    assert "Pattern may be too permissive" in result.warnings

def test_pattern_similarity(validator):
    """Test pattern similarity calculation."""
    # Test exact match
    assert validator._pattern_similarity("test", "test") == 1.0
    
    # Test completely different
    assert validator._pattern_similarity("abc", "xyz") == 0.0
    
    # Test similar patterns
    similarity = validator._pattern_similarity("getUser", "get\\w+")
    assert 0.5 < similarity < 1.0
    
    # Test empty patterns
    assert validator._pattern_similarity("", "") == 1.0
    assert validator._pattern_similarity("test", "") == 0.0

def test_validate_endpoint(client):
    """Test pattern validation endpoint."""
    # Test valid pattern
    response = client.post(
        '/api/validate',
        data=json.dumps({'pattern': r'get\w+'}),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['is_valid']
    assert not data['errors']
    assert any(s['pattern'] == r'get\w+' for s in data['suggestions'])
    
    # Test invalid pattern
    response = client.post(
        '/api/validate',
        data=json.dumps({'pattern': '[invalid'}),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert not data['is_valid']
    assert data['errors']
    
    # Test missing pattern
    response = client.post(
        '/api/validate',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400

def test_suggestions_endpoint(client):
    """Test pattern suggestions endpoint."""
    response = client.get('/api/suggestions')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'suggestions' in data
    assert len(data['suggestions']) > 0
    
    suggestion = data['suggestions'][0]
    assert all(key in suggestion for key in ['pattern', 'description', 'examples', 'category'])

def test_add_pointcut_with_validation(client):
    """Test adding pointcut with validation."""
    # Test valid pattern
    data = {
        'pattern': r'get\w+',
        'type': 'METHOD_EXECUTION',
        'metadata': {'scope': 'public'}
    }
    response = client.post(
        '/api/pointcuts',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert 'validation' in response_data
    assert 'warnings' in response_data['validation']
    
    # Test invalid pattern
    data['pattern'] = '[invalid'
    response = client.post(
        '/api/pointcuts',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 400
    response_data = json.loads(response.data)
    assert 'validation' in response_data
    assert 'errors' in response_data['validation']
    
    # Test pattern with warnings
    data['pattern'] = '.*'
    response = client.post(
        '/api/pointcuts',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert 'validation' in response_data
    assert 'warnings' in response_data['validation']
    assert any('matches everything' in w for w in response_data['validation']['warnings'])

def test_common_patterns():
    """Test common pattern suggestions."""
    validator = PatternValidator()
    
    # Test getter pattern
    result = validator.validate_pattern('getUserData')
    assert any(s.pattern == r'get\w+' for s in result.suggestions)
    
    # Test setter pattern
    result = validator.validate_pattern('setConfig')
    assert any(s.pattern == r'set\w+' for s in result.suggestions)
    
    # Test service pattern
    result = validator.validate_pattern('UserService')
    assert any(s.pattern == r'\w+Service' for s in result.suggestions)
    
    # Test controller pattern
    result = validator.validate_pattern('HomeController')
    assert any(s.pattern == r'\w+Controller' for s in result.suggestions)
    
    # Test test pattern
    result = validator.validate_pattern('testLogin')
    assert any(s.pattern == r'test\w+' for s in result.suggestions)
    
    # Test event handler pattern
    result = validator.validate_pattern('onClick')
    assert any(s.pattern == r'on\w+' for s in result.suggestions)

def test_index(client):
    """Test index page."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Pointcut Manager Dashboard' in response.data

def test_list_pointcuts(client):
    """Test listing pointcuts."""
    # Add test pointcut
    pointcut_id = pointcut_manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    
    response = client.get('/api/pointcuts')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'pointcuts' in data
    assert len(data['pointcuts']) == 1
    assert data['pointcuts'][0]['id'] == pointcut_id
    
    # Cleanup
    pointcut_manager.remove_pointcut(pointcut_id)

def test_add_pointcut(client):
    """Test adding pointcut."""
    data = {
        'pattern': 'test_.*',
        'type': 'METHOD_EXECUTION',
        'metadata': {'scope': 'public'}
    }
    
    response = client.post(
        '/api/pointcuts',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 200
    
    response_data = json.loads(response.data)
    assert 'id' in response_data
    
    # Verify pointcut was added
    pointcut = pointcut_manager.get_pointcut(response_data['id'])
    assert pointcut.pattern == data['pattern']
    assert pointcut.metadata == data['metadata']
    
    # Cleanup
    pointcut_manager.remove_pointcut(response_data['id'])

def test_add_pointcut_invalid(client):
    """Test adding invalid pointcut."""
    # Missing required fields
    response = client.post(
        '/api/pointcuts',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    # Invalid type
    data = {
        'pattern': 'test_.*',
        'type': 'INVALID_TYPE'
    }
    response = client.post(
        '/api/pointcuts',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 400

def test_remove_pointcut(client):
    """Test removing pointcut."""
    # Add test pointcut
    pointcut_id = pointcut_manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    
    response = client.delete(f'/api/pointcuts/{pointcut_id}')
    assert response.status_code == 200
    
    # Verify pointcut was removed
    with pytest.raises(KeyError):
        pointcut_manager.get_pointcut(pointcut_id)

def test_remove_nonexistent_pointcut(client):
    """Test removing non-existent pointcut."""
    response = client.delete('/api/pointcuts/nonexistent')
    assert response.status_code == 404

def test_enable_disable_pointcut(client):
    """Test enabling and disabling pointcut."""
    # Add test pointcut
    pointcut_id = pointcut_manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    
    # Test disable
    response = client.post(f'/api/pointcuts/{pointcut_id}/disable')
    assert response.status_code == 200
    assert not pointcut_manager.get_pointcut(pointcut_id).enabled
    
    # Test enable
    response = client.post(f'/api/pointcuts/{pointcut_id}/enable')
    assert response.status_code == 200
    assert pointcut_manager.get_pointcut(pointcut_id).enabled
    
    # Cleanup
    pointcut_manager.remove_pointcut(pointcut_id)

def test_check_matches(client):
    """Test checking matches."""
    # Add test pointcut
    pointcut_id = pointcut_manager.add_pointcut(
        pattern="test_.*",
        pointcut_type=PointcutType.METHOD_EXECUTION
    )
    
    # Test matching target
    data = {
        'target': 'test_method',
        'context': {'module': 'test'}
    }
    response = client.post(
        '/api/check',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 200
    
    response_data = json.loads(response.data)
    assert len(response_data['matches']) == 1
    assert response_data['matches'][0]['pattern'] == 'test_.*'
    
    # Test non-matching target
    data['target'] = 'other_method'
    response = client.post(
        '/api/check',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 200
    assert len(json.loads(response.data)['matches']) == 0
    
    # Cleanup
    pointcut_manager.remove_pointcut(pointcut_id)

def test_check_matches_invalid(client):
    """Test checking matches with invalid data."""
    # Missing target
    response = client.post(
        '/api/check',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    # Invalid JSON in context
    data = {
        'target': 'test_method',
        'context': 'invalid'
    }
    response = client.post(
        '/api/check',
        data=json.dumps(data),
        content_type='application/json'
    )
    assert response.status_code == 200  # Should handle invalid context gracefully

def test_export_pointcuts(client):
    """Test exporting pointcuts."""
    # Add test pointcuts
    pointcut_ids = []
    for i in range(3):
        pointcut_id = pointcut_manager.add_pointcut(
            pattern=f"test_{i}_.*",
            pointcut_type=PointcutType.METHOD_EXECUTION,
            metadata={'index': i}
        )
        pointcut_ids.append(pointcut_id)
    
    response = client.get('/api/pointcuts/export')
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/json'
    assert response.headers['Content-Disposition'].startswith('attachment')
    
    data = json.loads(response.data)
    assert 'pointcuts' in data
    assert len(data['pointcuts']) == 3
    
    # Cleanup
    for pointcut_id in pointcut_ids:
        pointcut_manager.remove_pointcut(pointcut_id)

def test_import_pointcuts(client):
    """Test importing pointcuts."""
    # Create test file
    data = {
        'pointcuts': [
            {
                'pattern': 'test_1_.*',
                'type': 'METHOD_EXECUTION',
                'metadata': {'index': 1}
            },
            {
                'pattern': 'test_2_.*',
                'type': 'METHOD_CALL',
                'metadata': {'index': 2}
            }
        ]
    }
    
    file = (io.BytesIO(json.dumps(data).encode()), 'pointcuts.json')
    
    response = client.post(
        '/api/pointcuts/import',
        data={'file': file},
        content_type='multipart/form-data'
    )
    assert response.status_code == 200
    
    response_data = json.loads(response.data)
    assert len(response_data['imported']) == 2
    
    # Cleanup
    for pointcut_id in response_data['imported']:
        pointcut_manager.remove_pointcut(pointcut_id)

def test_bulk_operations(client):
    """Test bulk operations."""
    # Add test pointcuts
    pointcut_ids = []
    for i in range(3):
        pointcut_id = pointcut_manager.add_pointcut(
            pattern=f"test_{i}_.*",
            pointcut_type=PointcutType.METHOD_EXECUTION
        )
        pointcut_ids.append(pointcut_id)
    
    # Test bulk enable
    response = client.post('/api/pointcuts/bulk/enable')
    assert response.status_code == 200
    for pointcut_id in pointcut_ids:
        assert pointcut_manager.get_pointcut(pointcut_id).enabled
    
    # Test bulk disable
    response = client.post('/api/pointcuts/bulk/disable')
    assert response.status_code == 200
    for pointcut_id in pointcut_ids:
        assert not pointcut_manager.get_pointcut(pointcut_id).enabled
    
    # Test bulk delete
    response = client.post(
        '/api/pointcuts/bulk/delete',
        data=json.dumps({'pointcuts': pointcut_ids}),
        content_type='application/json'
    )
    assert response.status_code == 200
    
    response_data = json.loads(response.data)
    assert len(response_data['deleted']) == 3
    
    # Verify all pointcuts were deleted
    for pointcut_id in pointcut_ids:
        with pytest.raises(KeyError):
            pointcut_manager.get_pointcut(pointcut_id)

def test_bulk_operations_invalid(client):
    """Test bulk operations with invalid data."""
    # Missing pointcuts for bulk delete
    response = client.post(
        '/api/pointcuts/bulk/delete',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    
    # Invalid pointcut IDs for bulk delete
    response = client.post(
        '/api/pointcuts/bulk/delete',
        data=json.dumps({'pointcuts': ['nonexistent']}),
        content_type='application/json'
    )
    assert response.status_code == 200  # Should handle invalid IDs gracefully