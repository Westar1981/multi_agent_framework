"""Pointcut specification and management for AOP integration."""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Set, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class PointcutType(Enum):
    """Types of pointcuts supported."""
    METHOD_EXECUTION = "execution"
    METHOD_CALL = "call"
    FIELD_GET = "get"
    FIELD_SET = "set"
    INITIALIZATION = "initialization"
    EXCEPTION = "exception"

@dataclass
class Pointcut:
    """Represents a pointcut specification."""
    pattern: str
    pointcut_type: PointcutType
    regex: Pattern = field(init=False)
    metadata: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        """Compile regex pattern after initialization."""
        try:
            self.regex = re.compile(self.pattern)
        except re.error as e:
            logger.error(f"Invalid regex pattern '{self.pattern}': {str(e)}")
            # Fall back to exact string matching
            self.regex = re.compile(re.escape(self.pattern))

@dataclass
class PointcutMatch:
    """Represents a successful pointcut match."""
    pointcut: Pointcut
    target_name: str
    match_groups: Tuple[str, ...] = field(default_factory=tuple)
    context: Dict[str, str] = field(default_factory=dict)

class PointcutManager:
    """Manages pointcut specifications and matching."""
    
    def __init__(self):
        """Initialize pointcut manager."""
        self.pointcuts: Dict[str, Pointcut] = {}
        self.active_matches: Set[PointcutMatch] = set()
        self._match_cache: Dict[str, List[PointcutMatch]] = {}
        
    def add_pointcut(self, pattern: str, pointcut_type: PointcutType, 
                    metadata: Optional[Dict[str, str]] = None) -> str:
        """Add a new pointcut specification.
        
        Args:
            pattern: Regex pattern for the pointcut
            pointcut_type: Type of pointcut
            metadata: Optional metadata for the pointcut
            
        Returns:
            str: ID of the created pointcut
            
        Raises:
            ValueError: If pattern is invalid
        """
        try:
            pointcut_id = f"{pointcut_type.value}_{len(self.pointcuts)}"
            pointcut = Pointcut(
                pattern=pattern,
                pointcut_type=pointcut_type,
                metadata=metadata or {}
            )
            self.pointcuts[pointcut_id] = pointcut
            self._match_cache.clear()  # Invalidate cache
            logger.info(f"Added pointcut {pointcut_id}: {pattern}")
            return pointcut_id
            
        except Exception as e:
            logger.error(f"Error adding pointcut: {str(e)}")
            raise ValueError(f"Invalid pointcut specification: {str(e)}")
            
    def remove_pointcut(self, pointcut_id: str) -> None:
        """Remove a pointcut by ID.
        
        Args:
            pointcut_id: ID of pointcut to remove
            
        Raises:
            KeyError: If pointcut_id not found
        """
        if pointcut_id not in self.pointcuts:
            raise KeyError(f"Pointcut {pointcut_id} not found")
            
        del self.pointcuts[pointcut_id]
        self._match_cache.clear()
        logger.info(f"Removed pointcut {pointcut_id}")
        
    def check_matches(self, target_name: str, context: Optional[Dict[str, str]] = None) -> List[PointcutMatch]:
        """Check if target matches any pointcuts.
        
        Args:
            target_name: Name to check against pointcuts
            context: Optional context information
            
        Returns:
            List[PointcutMatch]: List of matching pointcuts
        """
        # Check cache first
        cache_key = f"{target_name}_{str(context)}"
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]
            
        matches = []
        for pointcut_id, pointcut in self.pointcuts.items():
            if not pointcut.enabled:
                continue
                
            match = pointcut.regex.search(target_name)
            if match:
                match_obj = PointcutMatch(
                    pointcut=pointcut,
                    target_name=target_name,
                    match_groups=match.groups(),
                    context=context or {}
                )
                matches.append(match_obj)
                self.active_matches.add(match_obj)
                
        self._match_cache[cache_key] = matches
        return matches
        
    def get_active_matches(self) -> Set[PointcutMatch]:
        """Get all active pointcut matches.
        
        Returns:
            Set[PointcutMatch]: Set of active matches
        """
        return self.active_matches
        
    def clear_matches(self) -> None:
        """Clear all active matches."""
        self.active_matches.clear()
        self._match_cache.clear()
        
    def enable_pointcut(self, pointcut_id: str) -> None:
        """Enable a pointcut.
        
        Args:
            pointcut_id: ID of pointcut to enable
            
        Raises:
            KeyError: If pointcut_id not found
        """
        if pointcut_id not in self.pointcuts:
            raise KeyError(f"Pointcut {pointcut_id} not found")
            
        self.pointcuts[pointcut_id].enabled = True
        self._match_cache.clear()
        
    def disable_pointcut(self, pointcut_id: str) -> None:
        """Disable a pointcut.
        
        Args:
            pointcut_id: ID of pointcut to disable
            
        Raises:
            KeyError: If pointcut_id not found
        """
        if pointcut_id not in self.pointcuts:
            raise KeyError(f"Pointcut {pointcut_id} not found")
            
        self.pointcuts[pointcut_id].enabled = False
        self._match_cache.clear()
        
    def get_pointcut(self, pointcut_id: str) -> Pointcut:
        """Get pointcut by ID.
        
        Args:
            pointcut_id: ID of pointcut to retrieve
            
        Returns:
            Pointcut: The requested pointcut
            
        Raises:
            KeyError: If pointcut_id not found
        """
        if pointcut_id not in self.pointcuts:
            raise KeyError(f"Pointcut {pointcut_id} not found")
            
        return self.pointcuts[pointcut_id] 