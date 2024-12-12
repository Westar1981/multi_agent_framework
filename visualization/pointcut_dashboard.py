"""Web-based dashboard for pointcut visualization and management."""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO

from ..core.pointcut_manager import PointcutManager, PointcutType, Pointcut, PointcutMatch

logger = logging.getLogger(__name__)

@dataclass
class PatternSuggestion:
    """Suggestion for pointcut pattern."""
    pattern: str
    description: str
    examples: List[str]
    category: str

@dataclass
class ValidationResult:
    """Result of pattern validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[PatternSuggestion]

class PatternValidator:
    """Validates and suggests improvements for pointcut patterns."""
    
    COMMON_PATTERNS: List[PatternSuggestion] = [
        PatternSuggestion(
            pattern=r"get\w+",
            description="Matches getter methods",
            examples=["getUser", "getValue", "getConfig"],
            category="methods"
        ),
        PatternSuggestion(
            pattern=r"set\w+",
            description="Matches setter methods",
            examples=["setUser", "setValue", "setConfig"],
            category="methods"
        ),
        PatternSuggestion(
            pattern=r"\w+Service",
            description="Matches service classes",
            examples=["UserService", "AuthService", "DataService"],
            category="classes"
        ),
        PatternSuggestion(
            pattern=r"\w+Controller",
            description="Matches controller classes",
            examples=["UserController", "ApiController", "HomeController"],
            category="classes"
        ),
        PatternSuggestion(
            pattern=r"test\w+",
            description="Matches test methods",
            examples=["testLogin", "testApi", "testDatabase"],
            category="tests"
        ),
        PatternSuggestion(
            pattern=r"on\w+",
            description="Matches event handlers",
            examples=["onClick", "onSubmit", "onLoad"],
            category="events"
        )
    ]
    
    @classmethod
    def validate_pattern(cls, pattern: str) -> ValidationResult:
        """Validate a pointcut pattern.
        
        Args:
            pattern: Pattern to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[PatternSuggestion] = []
        
        # Check for basic regex validity
        try:
            re.compile(pattern)
        except re.error as e:
            errors.append(f"Invalid regex pattern: {str(e)}")
            return ValidationResult(False, errors, warnings, suggestions)
            
        # Check for common issues
        if pattern == ".*":
            warnings.append("Pattern matches everything, consider being more specific")
            
        if not pattern.strip():
            errors.append("Pattern cannot be empty")
            
        if len(pattern) == 1:
            warnings.append("Single character pattern may be too broad")
            
        if pattern.count("*") > 3:
            warnings.append("Pattern may be too permissive with many wildcards")
            
        # Find similar patterns
        for suggestion in cls.COMMON_PATTERNS:
            if (pattern in suggestion.pattern or 
                suggestion.pattern in pattern or
                cls._pattern_similarity(pattern, suggestion.pattern) > 0.7):
                suggestions.append(suggestion)
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    @staticmethod
    def _pattern_similarity(pattern1: str, pattern2: str) -> float:
        """Calculate similarity between two patterns.
        
        Args:
            pattern1: First pattern
            pattern2: Second pattern
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple Levenshtein distance-based similarity
        def levenshtein(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if not s2:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
                
            return previous_row[-1]
            
        max_len = max(len(pattern1), len(pattern2))
        if max_len == 0:
            return 1.0
            
        distance = levenshtein(pattern1, pattern2)
        return 1 - (distance / max_len)

app = Flask(__name__)
pointcut_manager = PointcutManager()
pattern_validator = PatternValidator()

@app.route('/')
def index() -> str:
    """Render main dashboard page."""
    return render_template('pointcut_dashboard.html')

@app.route('/api/pointcuts', methods=['GET'])
def list_pointcuts() -> Any:
    """List all pointcuts."""
    pointcuts = []
    for pointcut_id, pointcut in pointcut_manager.pointcuts.items():
        pointcuts.append({
            'id': pointcut_id,
            'pattern': pointcut.pattern,
            'type': pointcut.pointcut_type.value,
            'enabled': pointcut.enabled,
            'metadata': pointcut.metadata
        })
    return jsonify({'pointcuts': pointcuts})

@app.route('/api/validate', methods=['POST'])
def validate_pattern() -> Any:
    """Validate a pointcut pattern."""
    try:
        data = request.get_json()
        if not data or 'pattern' not in data:
            return jsonify({'error': 'Missing pattern'}), 400
            
        result = pattern_validator.validate_pattern(data['pattern'])
        return jsonify({
            'is_valid': result.is_valid,
            'errors': result.errors,
            'warnings': result.warnings,
            'suggestions': [
                {
                    'pattern': s.pattern,
                    'description': s.description,
                    'examples': s.examples,
                    'category': s.category
                }
                for s in result.suggestions
            ]
        })
        
    except Exception as e:
        logger.error(f"Error validating pattern: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions() -> Any:
    """Get all pattern suggestions."""
    return jsonify({
        'suggestions': [
            {
                'pattern': s.pattern,
                'description': s.description,
                'examples': s.examples,
                'category': s.category
            }
            for s in pattern_validator.COMMON_PATTERNS
        ]
    })

@app.route('/api/pointcuts', methods=['POST'])
def add_pointcut() -> Any:
    """Add a new pointcut."""
    try:
        data = request.get_json()
        if not data or 'pattern' not in data or 'type' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Validate pattern first
        validation = pattern_validator.validate_pattern(data['pattern'])
        if not validation.is_valid:
            return jsonify({
                'error': 'Invalid pattern',
                'validation': {
                    'errors': validation.errors,
                    'warnings': validation.warnings
                }
            }), 400
            
        pointcut_type = data['type'].upper()
        if not hasattr(PointcutType, pointcut_type):
            return jsonify({'error': 'Invalid pointcut type'}), 400
            
        pointcut_id = pointcut_manager.add_pointcut(
            pattern=data['pattern'],
            pointcut_type=getattr(PointcutType, pointcut_type),
            metadata=data.get('metadata', {})
        )
        
        return jsonify({
            'id': pointcut_id,
            'message': 'Pointcut added successfully',
            'validation': {
                'warnings': validation.warnings,
                'suggestions': [
                    {
                        'pattern': s.pattern,
                        'description': s.description
                    }
                    for s in validation.suggestions
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error adding pointcut: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/<pointcut_id>', methods=['DELETE'])
def remove_pointcut(pointcut_id: str):
    """Remove a pointcut."""
    try:
        pointcut_manager.remove_pointcut(pointcut_id)
        return jsonify({'message': 'Pointcut removed successfully'})
    except KeyError:
        return jsonify({'error': 'Pointcut not found'}), 404
    except Exception as e:
        logger.error(f"Error removing pointcut: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/<pointcut_id>/enable', methods=['POST'])
def enable_pointcut(pointcut_id: str):
    """Enable a pointcut."""
    try:
        pointcut_manager.enable_pointcut(pointcut_id)
        return jsonify({'message': 'Pointcut enabled'})
    except KeyError:
        return jsonify({'error': 'Pointcut not found'}), 404
    except Exception as e:
        logger.error(f"Error enabling pointcut: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/<pointcut_id>/disable', methods=['POST'])
def disable_pointcut(pointcut_id: str):
    """Disable a pointcut."""
    try:
        pointcut_manager.disable_pointcut(pointcut_id)
        return jsonify({'message': 'Pointcut disabled'})
    except KeyError:
        return jsonify({'error': 'Pointcut not found'}), 404
    except Exception as e:
        logger.error(f"Error disabling pointcut: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check', methods=['POST'])
def check_matches():
    """Check for pointcut matches."""
    try:
        data = request.get_json()
        if not data or 'target' not in data:
            return jsonify({'error': 'Missing target'}), 400
            
        matches = pointcut_manager.check_matches(
            data['target'],
            context=data.get('context', {})
        )
        
        results = []
        for match in matches:
            results.append({
                'pattern': match.pointcut.pattern,
                'type': match.pointcut.pointcut_type.value,
                'groups': list(match.match_groups),
                'context': match.context
            })
            
        return jsonify({
            'target': data['target'],
            'matches': results
        })
        
    except Exception as e:
        logger.error(f"Error checking matches: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/export', methods=['GET'])
def export_pointcuts():
    """Export pointcuts as JSON file."""
    try:
        pointcuts = []
        for pointcut_id, pointcut in pointcut_manager.pointcuts.items():
            pointcuts.append({
                'id': pointcut_id,
                'pattern': pointcut.pattern,
                'type': pointcut.pointcut_type.value,
                'enabled': pointcut.enabled,
                'metadata': pointcut.metadata
            })
            
        data = json.dumps({'pointcuts': pointcuts}, indent=2)
        buffer = BytesIO(data.encode())
        
        return send_file(
            buffer,
            mimetype='application/json',
            as_attachment=True,
            download_name='pointcuts.json'
        )
        
    except Exception as e:
        logger.error(f"Error exporting pointcuts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/import', methods=['POST'])
def import_pointcuts():
    """Import pointcuts from JSON file."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if not file.filename.endswith('.json'):
            return jsonify({'error': 'Invalid file type'}), 400
            
        data = json.loads(file.read())
        if 'pointcuts' not in data:
            return jsonify({'error': 'Invalid file format'}), 400
            
        imported = []
        for pointcut in data['pointcuts']:
            try:
                pointcut_id = pointcut_manager.add_pointcut(
                    pattern=pointcut['pattern'],
                    pointcut_type=getattr(PointcutType, pointcut['type'].upper()),
                    metadata=pointcut.get('metadata', {})
                )
                imported.append(pointcut_id)
            except Exception as e:
                logger.warning(f"Error importing pointcut: {str(e)}")
                
        return jsonify({
            'message': f'Imported {len(imported)} pointcuts',
            'imported': imported
        })
        
    except Exception as e:
        logger.error(f"Error importing pointcuts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/bulk/enable', methods=['POST'])
def enable_all_pointcuts():
    """Enable all pointcuts."""
    try:
        for pointcut_id in pointcut_manager.pointcuts:
            pointcut_manager.enable_pointcut(pointcut_id)
        return jsonify({'message': 'All pointcuts enabled'})
    except Exception as e:
        logger.error(f"Error enabling all pointcuts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/bulk/disable', methods=['POST'])
def disable_all_pointcuts():
    """Disable all pointcuts."""
    try:
        for pointcut_id in pointcut_manager.pointcuts:
            pointcut_manager.disable_pointcut(pointcut_id)
        return jsonify({'message': 'All pointcuts disabled'})
    except Exception as e:
        logger.error(f"Error disabling all pointcuts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/pointcuts/bulk/delete', methods=['POST'])
def delete_selected_pointcuts():
    """Delete selected pointcuts."""
    try:
        data = request.get_json()
        if not data or 'pointcuts' not in data:
            return jsonify({'error': 'No pointcuts specified'}), 400
            
        deleted = []
        for pointcut_id in data['pointcuts']:
            try:
                pointcut_manager.remove_pointcut(pointcut_id)
                deleted.append(pointcut_id)
            except Exception as e:
                logger.warning(f"Error deleting pointcut {pointcut_id}: {str(e)}")
                
        return jsonify({
            'message': f'Deleted {len(deleted)} pointcuts',
            'deleted': deleted
        })
        
    except Exception as e:
        logger.error(f"Error deleting pointcuts: {str(e)}")
        return jsonify({'error': str(e)}), 500

def run_dashboard(host: str = 'localhost', port: int = 5000, debug: bool = False):
    """Run the dashboard server."""
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        raise

if __name__ == '__main__':
    run_dashboard(debug=True) 