"""API routes for pointcut management."""

from typing import Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...core.pointcuts.manager import (
    PointcutManager, PointcutType, PatternType, 
    ValidationResult, PatternSuggestion
)

router = APIRouter(prefix="/api/pointcuts")
pointcut_manager = PointcutManager()

class PointcutCreate(BaseModel):
    """Request model for creating a pointcut."""
    pattern: Union[str, Dict]
    type: PointcutType
    pattern_type: PatternType = PatternType.REGEX
    metadata: Optional[Dict[str, str]] = None

class CompositeCreate(BaseModel):
    """Request model for creating a composite pointcut."""
    operator: str
    pointcut_ids: List[str]

class PointcutResponse(BaseModel):
    """Response model for pointcut data."""
    id: str
    pattern: Union[str, Dict]
    type: PointcutType
    pattern_type: Optional[PatternType]
    enabled: bool
    metadata: Dict[str, str]
    warnings: Optional[List[str]] = None
    suggestions: Optional[List[Dict[str, str]]] = None

class MatchRequest(BaseModel):
    """Request model for checking matches."""
    target: str
    context: Optional[Dict[str, str]] = None

class BatchMatchRequest(BaseModel):
    """Request model for batch matching."""
    targets: List[str]
    context: Optional[Dict[str, str]] = None
    parallel: bool = False

class MatchResponse(BaseModel):
    """Response model for match results."""
    matches: bool
    groups: List[List[str]]
    context: Dict[str, str]

class BatchMatchResponse(BaseModel):
    """Response model for batch match results."""
    matches: Dict[str, List[Dict[str, str]]]
    stats: Dict[str, Dict[str, int]]
    duration: float

class ValidationRequest(BaseModel):
    """Request model for pattern validation."""
    pattern: Union[str, Dict]
    pattern_type: PatternType

class ValidationResponse(BaseModel):
    """Response model for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[Dict[str, str]]

class StatsResponse(BaseModel):
    """Response model for match statistics."""
    checks: int
    matches: int

@router.post("/validate", response_model=ValidationResponse)
async def validate_pattern(request: ValidationRequest):
    """Validate a pattern before creating a pointcut."""
    try:
        result = pointcut_manager.validate_pattern(
            pattern=request.pattern,
            pattern_type=request.pattern_type
        )
        return ValidationResponse(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            suggestions=[{
                "original": s.original,
                "suggested": s.suggested,
                "reason": s.reason,
                "confidence": s.confidence
            } for s in result.suggestions]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/check", response_model=BatchMatchResponse)
async def batch_check_matches(request: BatchMatchRequest):
    """Check multiple targets against pointcuts."""
    try:
        result = pointcut_manager.batch_check_matches(
            targets=request.targets,
            context=request.context,
            parallel=request.parallel
        )
        return BatchMatchResponse(
            matches={
                target: [{
                    "pointcut_id": pid,
                    "groups": [list(g) for g in match.match_groups],
                    "context": match.context
                } for match in matches]
                for target, matches in result.matches.items()
            },
            stats=result.stats,
            duration=result.duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/add", response_model=List[str])
async def batch_add_pointcuts(pointcuts: List[PointcutCreate]):
    """Add multiple pointcuts in a batch."""
    try:
        specs = [{
            "pattern": p.pattern,
            "type": p.type,
            "pattern_type": p.pattern_type,
            "metadata": p.metadata
        } for p in pointcuts]
        return pointcut_manager.batch_add_pointcuts(specs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/remove", response_model=List[str])
async def batch_remove_pointcuts(pointcut_ids: List[str]):
    """Remove multiple pointcuts in a batch."""
    try:
        return pointcut_manager.batch_remove_pointcuts(pointcut_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/enable", response_model=List[str])
async def batch_enable_pointcuts(pointcut_ids: List[str]):
    """Enable multiple pointcuts in a batch."""
    try:
        return pointcut_manager.batch_enable_pointcuts(pointcut_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch/disable", response_model=List[str])
async def batch_disable_pointcuts(pointcut_ids: List[str]):
    """Disable multiple pointcuts in a batch."""
    try:
        return pointcut_manager.batch_disable_pointcuts(pointcut_ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import csv
import json
from io import StringIO
from datetime import datetime
from fastapi.responses import StreamingResponse, FileResponse

@router.get("/analytics/export", response_class=StreamingResponse)
async def export_analytics(format: str = "csv"):
    """Export analytics data in CSV or JSON format."""
    try:
        report = pointcut_manager.get_performance_report()
        
        if format.lower() == "csv":
            return _export_csv(report)
        elif format.lower() == "json":
            return _export_json(report)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Use 'csv' or 'json'."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _export_csv(report: Dict[str, Any]) -> StreamingResponse:
    """Export analytics data as CSV."""
    output = StringIO()
    writer = csv.writer(output)
    
    # Write global stats
    writer.writerow(["Global Statistics"])
    writer.writerow(["Metric", "Value"])
    for metric, value in report["global_stats"].items():
        writer.writerow([metric, f"{value:.4f}"])
    writer.writerow([])
    
    # Write pattern stats
    writer.writerow(["Pattern Statistics"])
    writer.writerow([
        "Pattern ID",
        "Total Checks",
        "Total Matches",
        "Match Rate",
        "Avg Match Time",
        "Peak Match Time",
        "Complexity Score",
        "P50 Response",
        "P90 Response",
        "P95 Response",
        "P99 Response"
    ])
    
    for pattern_id, stats in report["pattern_stats"].items():
        writer.writerow([
            pattern_id,
            stats["total_checks"],
            stats["total_matches"],
            f"{stats['match_rate']:.2%}",
            f"{stats['avg_match_time']:.4f}ms",
            f"{stats['peak_match_time']:.4f}ms",
            f"{stats.get('complexity_score', 'N/A')}",
            f"{stats['percentiles'].get('p50', 'N/A')}ms",
            f"{stats['percentiles'].get('p90', 'N/A')}ms",
            f"{stats['percentiles'].get('p95', 'N/A')}ms",
            f"{stats['percentiles'].get('p99', 'N/A')}ms"
        ])
    writer.writerow([])
    
    # Write hourly trends
    writer.writerow(["Hourly Trends"])
    writer.writerow(["Pattern ID", "Hour", "Count", "Matches", "Avg Time"])
    for pattern_id, trends in report["hourly_trends"].items():
        for hour, stats in trends.items():
            writer.writerow([
                pattern_id,
                hour,
                stats["count"],
                stats["matches"],
                f"{stats['avg_time']:.4f}ms"
            ])
    writer.writerow([])
    
    # Write error analysis
    writer.writerow(["Error Analysis"])
    writer.writerow(["Error Type", "Count"])
    for error, count in report["error_analysis"].items():
        writer.writerow([error, count])
    writer.writerow([])
    
    # Write recommendations
    writer.writerow(["Recommendations"])
    writer.writerow(["Pattern ID", "Type", "Priority", "Message", "Suggestion"])
    for rec in report["recommendations"]:
        writer.writerow([
            rec["pattern_id"],
            rec["type"],
            rec["priority"],
            rec["message"],
            rec["suggestion"]
        ])
    
    # Prepare response
    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=pointcut_analytics_{timestamp}.csv"
        }
    )

def _export_json(report: Dict[str, Any]) -> StreamingResponse:
    """Export analytics data as JSON."""
    # Add metadata
    export_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        },
        "report": report
    }
    
    # Convert to JSON with pretty printing
    json_data = json.dumps(export_data, indent=2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return StreamingResponse(
        iter([json_data]),
        media_type="application/json",
        headers={
            "Content-Disposition": f"attachment; filename=pointcut_analytics_{timestamp}.json"
        }
    )

@router.get("/analytics/{pattern_id}/history", response_class=FileResponse)
async def export_pattern_history(pattern_id: str, format: str = "csv"):
    """Export detailed history for a specific pattern."""
    try:
        analytics = pointcut_manager.get_analytics(pattern_id)
        if not analytics:
            raise HTTPException(
                status_code=404,
                detail=f"No analytics found for pattern {pattern_id}"
            )
            
        if format.lower() == "csv":
            return _export_pattern_history_csv(pattern_id, analytics)
        elif format.lower() == "json":
            return _export_pattern_history_json(pattern_id, analytics)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {format}. Use 'csv' or 'json'."
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _export_pattern_history_csv(pattern_id: str, analytics: PatternAnalytics) -> StreamingResponse:
    """Export pattern history as CSV."""
    output = StringIO()
    writer = csv.writer(output)
    
    # Write pattern info
    writer.writerow(["Pattern Information"])
    writer.writerow(["ID", pattern_id])
    writer.writerow(["Type", analytics.pattern_type])
    writer.writerow(["Total Checks", analytics.total_checks])
    writer.writerow(["Total Matches", analytics.total_matches])
    writer.writerow([])
    
    # Write match history
    writer.writerow(["Match History"])
    writer.writerow(["Timestamp", "Matched", "Response Time"])
    for timestamp, matched in analytics.match_history:
        writer.writerow([
            datetime.fromtimestamp(timestamp).isoformat(),
            matched,
            f"{analytics.performance_stats.get('response_time', 'N/A')}ms"
        ])
    writer.writerow([])
    
    # Write hourly stats
    writer.writerow(["Hourly Statistics"])
    writer.writerow(["Hour", "Count", "Matches", "Avg Time"])
    for hour, stats in analytics.hourly_stats.items():
        writer.writerow([
            hour,
            stats["count"],
            stats["matches"],
            f"{stats['avg_time']:.4f}ms"
        ])
    writer.writerow([])
    
    # Write complexity analysis
    if analytics.complexity:
        writer.writerow(["Complexity Analysis"])
        writer.writerow(["Score", analytics.complexity.score])
        writer.writerow(["Factors"])
        for factor, value in analytics.complexity.factors.items():
            writer.writerow([factor, f"{value:.4f}"])
        writer.writerow(["Suggestions"])
        for suggestion in analytics.complexity.suggestions:
            writer.writerow([suggestion])
    
    # Prepare response
    output.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": 
                f"attachment; filename=pattern_{pattern_id}_history_{timestamp}.csv"
        }
    )

def _export_pattern_history_json(pattern_id: str, analytics: PatternAnalytics) -> StreamingResponse:
    """Export pattern history as JSON."""
    export_data = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "pattern_id": pattern_id,
            "version": "1.0"
        },
        "pattern_info": {
            "type": analytics.pattern_type,
            "total_checks": analytics.total_checks,
            "total_matches": analytics.total_matches
        },
        "match_history": [
            {
                "timestamp": datetime.fromtimestamp(ts).isoformat(),
                "matched": matched,
                "response_time": analytics.performance_stats.get("response_time", "N/A")
            }
            for ts, matched in analytics.match_history
        ],
        "hourly_stats": analytics.hourly_stats,
        "complexity_analysis": None
    }
    
    if analytics.complexity:
        export_data["complexity_analysis"] = {
            "score": analytics.complexity.score,
            "factors": analytics.complexity.factors,
            "suggestions": analytics.complexity.suggestions
        }
    
    # Convert to JSON with pretty printing
    json_data = json.dumps(export_data, indent=2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return StreamingResponse(
        iter([json_data]),
        media_type="application/json",
        headers={
            "Content-Disposition": 
                f"attachment; filename=pattern_{pattern_id}_history_{timestamp}.json"
        }
    )

# ... (rest of the code remains unchanged) ... 