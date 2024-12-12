"""Core pointcut management functionality."""

import re
import ast
import time
import logging
import inspect
from typing import Dict, List, Optional, Set, Tuple, Union, Any, DefaultDict
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)

class PointcutType(str, Enum):
    """Types of pointcuts supported."""
    METHOD_EXECUTION = "execution"
    METHOD_CALL = "call"
    FIELD_GET = "get"
    FIELD_SET = "set"
    INITIALIZATION = "initialization"
    EXCEPTION = "exception"

class PatternType(str, Enum):
    """Types of pattern matching supported."""
    REGEX = "regex"
    WILDCARD = "wildcard"
    SEMANTIC = "semantic"

class SemanticRule(str, Enum):
    """Semantic rules for AST matching."""
    IS_METHOD = "is_method"
    IS_CLASS = "is_class"
    HAS_DECORATOR = "has_decorator"
    RETURNS_TYPE = "returns_type"
    TAKES_ARGUMENT = "takes_argument"
    CALLS_FUNCTION = "calls_function"
    ACCESSES_ATTRIBUTE = "accesses_attribute"
    RAISES_EXCEPTION = "raises_exception"

class MatchStrategy(ABC):
    """Abstract base class for pattern matching strategies."""
    @abstractmethod
    def matches(self, target: str) -> Tuple[bool, Tuple[str, ...]]:
        """Check if target matches the pattern."""
        pass

class RegexStrategy(MatchStrategy):
    """Regex-based pattern matching."""
    def __init__(self, pattern: str):
        self.regex = re.compile(pattern)
        
    def matches(self, target: str) -> Tuple[bool, Tuple[str, ...]]:
        match = self.regex.search(target)
        return bool(match), match.groups() if match else ()

class WildcardStrategy(MatchStrategy):
    """Wildcard-based pattern matching."""
    def __init__(self, pattern: str):
        self.regex = re.compile(
            pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
        )
        
    def matches(self, target: str) -> Tuple[bool, Tuple[str, ...]]:
        match = self.regex.search(target)
        return bool(match), match.groups() if match else ()

class SemanticStrategy(MatchStrategy):
    """Semantic-based pattern matching using AST analysis."""
    def __init__(self, pattern: Dict[str, Any]):
        self.rules = pattern
        
    def matches(self, target: str) -> Tuple[bool, Tuple[str, ...]]:
        try:
            tree = ast.parse(target)
            matches = []
            
            for node in ast.walk(tree):
                if self._matches_rules(node):
                    matches.append(self._extract_info(node))
                    
            return bool(matches), tuple(matches)
        except SyntaxError:
            return False, ()
            
    def _matches_rules(self, node: ast.AST) -> bool:
        """Check if AST node matches semantic rules."""
        for rule, value in self.rules.items():
            if rule == SemanticRule.IS_METHOD:
                if not isinstance(node, ast.FunctionDef):
                    return False
            elif rule == SemanticRule.IS_CLASS:
                if not isinstance(node, ast.ClassDef):
                    return False
            elif rule == SemanticRule.HAS_DECORATOR:
                if not (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and 
                       any(self._matches_decorator(d, value) for d in node.decorator_list)):
                    return False
            elif rule == SemanticRule.RETURNS_TYPE:
                if not (isinstance(node, ast.FunctionDef) and 
                       self._matches_return_type(node, value)):
                    return False
            elif rule == SemanticRule.TAKES_ARGUMENT:
                if not (isinstance(node, ast.FunctionDef) and 
                       self._matches_argument(node, value)):
                    return False
            elif rule == SemanticRule.CALLS_FUNCTION:
                if not self._has_function_call(node, value):
                    return False
            elif rule == SemanticRule.ACCESSES_ATTRIBUTE:
                if not self._has_attribute_access(node, value):
                    return False
            elif rule == SemanticRule.RAISES_EXCEPTION:
                if not self._has_exception(node, value):
                    return False
        return True
        
    def _matches_decorator(self, decorator: ast.AST, pattern: str) -> bool:
        """Check if decorator matches pattern."""
        if isinstance(decorator, ast.Name):
            return decorator.id == pattern
        elif isinstance(decorator, ast.Call):
            return decorator.func.id == pattern if isinstance(decorator.func, ast.Name) else False
        return False
        
    def _matches_return_type(self, node: ast.FunctionDef, type_name: str) -> bool:
        """Check if function returns specified type."""
        return_annotation = node.returns
        if isinstance(return_annotation, ast.Name):
            return return_annotation.id == type_name
        return False
        
    def _matches_argument(self, node: ast.FunctionDef, arg_pattern: Dict[str, str]) -> bool:
        """Check if function has matching argument."""
        for arg in node.args.args:
            if arg.arg == arg_pattern.get("name"):
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    if arg.annotation.id == arg_pattern.get("type"):
                        return True
        return False
        
    def _has_function_call(self, node: ast.AST, func_name: str) -> bool:
        """Check if node contains call to specified function."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == func_name:
                    return True
        return False
        
    def _has_attribute_access(self, node: ast.AST, attr_name: str) -> bool:
        """Check if node accesses specified attribute."""
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == attr_name:
                return True
        return False
        
    def _has_exception(self, node: ast.AST, exception_type: str) -> bool:
        """Check if node raises specified exception."""
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Name) and child.exc.id == exception_type:
                    return True
        return False
        
    def _extract_info(self, node: ast.AST) -> str:
        """Extract relevant information from matching node."""
        if isinstance(node, ast.FunctionDef):
            return f"function:{node.name}"
        elif isinstance(node, ast.ClassDef):
            return f"class:{node.name}"
        return str(node)

@dataclass
class Pointcut:
    """Represents a pointcut specification."""
    pattern: Union[str, Dict[str, Any]]
    pointcut_type: PointcutType
    pattern_type: PatternType = PatternType.REGEX
    metadata: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    _matcher: MatchStrategy = field(init=False)
    
    def __post_init__(self):
        """Initialize pattern matcher."""
        try:
            if self.pattern_type == PatternType.REGEX:
                self._matcher = RegexStrategy(self.pattern)
            elif self.pattern_type == PatternType.WILDCARD:
                self._matcher = WildcardStrategy(self.pattern)
            elif self.pattern_type == PatternType.SEMANTIC:
                self._matcher = SemanticStrategy(self.pattern)
            else:
                raise ValueError(f"Unsupported pattern type: {self.pattern_type}")
        except Exception as e:
            logger.error(f"Error initializing pattern matcher: {str(e)}")
            # Fall back to exact string matching
            self._matcher = RegexStrategy(re.escape(str(self.pattern)))

    def matches(self, target: str) -> Tuple[bool, Tuple[str, ...]]:
        """Check if target matches this pointcut."""
        return self._matcher.matches(target) 

@dataclass
class BatchMatchResult:
    """Results from a batch match operation."""
    matches: Dict[str, List[PointcutMatch]]
    stats: Dict[str, Dict[str, int]]
    duration: float

@dataclass
class PatternSuggestion:
    """Suggested pattern improvement."""
    original: str
    suggested: str
    reason: str
    confidence: float

@dataclass
class ValidationResult:
    """Result of pattern validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[PatternSuggestion]

class PatternValidator:
    """Validates and suggests improvements for patterns."""
    
    def __init__(self):
        """Initialize pattern validator."""
        self.common_patterns = {
            "method_name": r"^[a-z_][a-z0-9_]*$",
            "class_name": r"^[A-Z][a-zA-Z0-9]*$",
            "module_path": r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$",
            "decorator": r"^@[a-zA-Z_][a-zA-Z0-9_]*$"
        }
        
        self.semantic_templates = {
            "method": {
                SemanticRule.IS_METHOD: True,
                SemanticRule.RETURNS_TYPE: "Any"
            },
            "class": {
                SemanticRule.IS_CLASS: True
            },
            "decorated_method": {
                SemanticRule.IS_METHOD: True,
                SemanticRule.HAS_DECORATOR: "property"
            }
        }
        
    def validate_pattern(self, pattern: Union[str, Dict[str, Any]], 
                        pattern_type: PatternType) -> ValidationResult:
        """Validate a pattern and suggest improvements.
        
        Args:
            pattern: Pattern to validate
            pattern_type: Type of pattern
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []
        
        if pattern_type == PatternType.REGEX:
            return self._validate_regex(pattern)
        elif pattern_type == PatternType.WILDCARD:
            return self._validate_wildcard(pattern)
        elif pattern_type == PatternType.SEMANTIC:
            return self._validate_semantic(pattern)
        else:
            errors.append(f"Unsupported pattern type: {pattern_type}")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
        
    def _validate_regex(self, pattern: str) -> ValidationResult:
        """Validate regex pattern."""
        errors = []
        warnings = []
        suggestions = []
        
        try:
            re.compile(pattern)
        except re.error as e:
            errors.append(f"Invalid regex pattern: {str(e)}")
            return ValidationResult(False, errors, [], [])
            
        # Check for common issues
        if pattern.startswith(".*"):
            warnings.append("Pattern starts with '.*', which may be inefficient")
            suggestions.append(PatternSuggestion(
                original=pattern,
                suggested=pattern[2:],
                reason="Remove leading wildcard for better performance",
                confidence=0.8
            ))
            
        if ".*.*" in pattern:
            warnings.append("Pattern contains consecutive wildcards")
            suggestions.append(PatternSuggestion(
                original=pattern,
                suggested=pattern.replace(".*.*", ".*"),
                reason="Combine consecutive wildcards",
                confidence=0.9
            ))
            
        # Suggest common patterns
        for name, common_pattern in self.common_patterns.items():
            if re.match(common_pattern, pattern):
                suggestions.append(PatternSuggestion(
                    original=pattern,
                    suggested=f"(?:{common_pattern})",
                    reason=f"Use predefined {name} pattern",
                    confidence=0.7
                ))
                
        return ValidationResult(True, errors, warnings, suggestions)
        
    def _validate_wildcard(self, pattern: str) -> ValidationResult:
        """Validate wildcard pattern."""
        errors = []
        warnings = []
        suggestions = []
        
        # Check for unescaped special characters
        special_chars = ".+{}[]()^$"
        for char in special_chars:
            if char in pattern and f"\\{char}" not in pattern:
                warnings.append(f"Unescaped special character: {char}")
                suggestions.append(PatternSuggestion(
                    original=pattern,
                    suggested=pattern.replace(char, f"\\{char}"),
                    reason=f"Escape special character {char}",
                    confidence=0.9
                ))
                
        # Check for redundant wildcards
        if "**" in pattern:
            warnings.append("Redundant wildcards")
            suggestions.append(PatternSuggestion(
                original=pattern,
                suggested=pattern.replace("**", "*"),
                reason="Single wildcard is sufficient",
                confidence=0.8
            ))
            
        return ValidationResult(True, errors, warnings, suggestions)
        
    def _validate_semantic(self, pattern: Dict[str, Any]) -> ValidationResult:
        """Validate semantic pattern."""
        errors = []
        warnings = []
        suggestions = []
        
        # Validate rules
        for rule in pattern:
            if not isinstance(rule, SemanticRule):
                errors.append(f"Invalid semantic rule: {rule}")
                continue
                
        # Check for missing complementary rules
        if SemanticRule.IS_METHOD in pattern:
            if SemanticRule.RETURNS_TYPE not in pattern:
                warnings.append("Method pattern missing return type")
                
        if SemanticRule.TAKES_ARGUMENT in pattern:
            if SemanticRule.IS_METHOD not in pattern:
                warnings.append("Argument pattern should be combined with IS_METHOD")
                
        # Suggest template patterns
        for template_name, template in self.semantic_templates.items():
            if all(rule in pattern for rule in template):
                suggestions.append(PatternSuggestion(
                    original=str(pattern),
                    suggested=str(template),
                    reason=f"Use {template_name} template pattern",
                    confidence=0.7
                ))
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )

@dataclass
class CachedPattern:
    """Represents a cached pattern with TTL."""
    pattern: Union[str, Dict[str, Any]]
    matcher: MatchStrategy
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    hits: int = 0
    misses: int = 0

class PatternIndex:
    """Indexes patterns for faster matching."""
    
    def __init__(self):
        """Initialize pattern index."""
        self.prefix_index: DefaultDict[str, Set[str]] = defaultdict(set)
        self.suffix_index: DefaultDict[str, Set[str]] = defaultdict(set)
        self.length_index: DefaultDict[int, Set[str]] = defaultdict(set)
        self.type_index: DefaultDict[str, Set[str]] = defaultdict(set)
        
    def add_pattern(self, pattern_id: str, pattern: str, pattern_type: str):
        """Add pattern to indexes."""
        if isinstance(pattern, str):
            # Index by prefix (first 3 chars)
            if len(pattern) >= 3:
                self.prefix_index[pattern[:3]].add(pattern_id)
            
            # Index by suffix (last 3 chars)
            if len(pattern) >= 3:
                self.suffix_index[pattern[-3:]].add(pattern_id)
            
            # Index by length
            self.length_index[len(pattern)].add(pattern_id)
            
        # Index by type
        self.type_index[pattern_type].add(pattern_id)
        
    def remove_pattern(self, pattern_id: str, pattern: str, pattern_type: str):
        """Remove pattern from indexes."""
        if isinstance(pattern, str):
            if len(pattern) >= 3:
                self.prefix_index[pattern[:3]].discard(pattern_id)
                self.suffix_index[pattern[-3:]].discard(pattern_id)
            self.length_index[len(pattern)].discard(pattern_id)
        self.type_index[pattern_type].discard(pattern_id)
        
    def find_candidates(self, target: str, pattern_type: Optional[str] = None) -> Set[str]:
        """Find candidate patterns that might match target."""
        candidates = set()
        
        if pattern_type:
            candidates.update(self.type_index[pattern_type])
        
        if isinstance(target, str) and len(target) >= 3:
            # Add patterns with matching prefix
            candidates.update(self.prefix_index[target[:3]])
            
            # Add patterns with matching suffix
            candidates.update(self.suffix_index[target[-3:]])
            
            # Add patterns with similar length (±2)
            target_len = len(target)
            for length in range(target_len - 2, target_len + 3):
                candidates.update(self.length_index[length])
                
        return candidates

@dataclass
class PatternComplexity:
    """Analyzes and scores pattern complexity."""
    pattern: Union[str, Dict[str, Any]]
    pattern_type: PatternType
    score: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)

    def analyze(self) -> None:
        """Analyze pattern complexity."""
        if self.pattern_type == PatternType.REGEX:
            self._analyze_regex()
        elif self.pattern_type == PatternType.WILDCARD:
            self._analyze_wildcard()
        elif self.pattern_type == PatternType.SEMANTIC:
            self._analyze_semantic()
            
    def _analyze_regex(self) -> None:
        """Analyze regex pattern complexity."""
        pattern = str(self.pattern)
        factors = {}
        
        # Length factor
        length = len(pattern)
        factors['length'] = min(length / 50, 1.0)  # Cap at length 50
        
        # Special character density
        special_chars = set('.*+?{}[]()^$|\\')
        special_count = sum(1 for c in pattern if c in special_chars)
        factors['special_density'] = special_count / max(length, 1)
        
        # Backtracking risk
        backtrack_patterns = [r'\.*\.*', r'\+*\+*', r'\{.*\}.*\{.*\}']
        backtrack_count = sum(len(re.findall(p, pattern)) for p in backtrack_patterns)
        factors['backtracking'] = min(backtrack_count / 3, 1.0)
        
        # Capture group complexity
        capture_count = pattern.count('(') - pattern.count('(?:')
        factors['capture_groups'] = min(capture_count / 5, 1.0)
        
        # Calculate overall score (0-10 scale)
        weights = {
            'length': 0.2,
            'special_density': 0.3,
            'backtracking': 0.3,
            'capture_groups': 0.2
        }
        self.score = sum(score * weights[factor] for factor, score in factors.items()) * 10
        self.factors = factors
        
        # Generate suggestions
        if factors['backtracking'] > 0.5:
            self.suggestions.append(
                "Pattern may cause excessive backtracking. Consider using atomic groups (?>) "
                "or possessive quantifiers (++, *+)."
            )
        if factors['special_density'] > 0.3:
            self.suggestions.append(
                "High density of special characters. Consider simplifying or using named groups "
                "for better readability."
            )
            
    def _analyze_wildcard(self) -> None:
        """Analyze wildcard pattern complexity."""
        pattern = str(self.pattern)
        factors = {}
        
        # Length factor
        length = len(pattern)
        factors['length'] = min(length / 30, 1.0)
        
        # Wildcard density
        wildcard_chars = set('*?')
        wildcard_count = sum(1 for c in pattern if c in wildcard_chars)
        factors['wildcard_density'] = wildcard_count / max(length, 1)
        
        # Calculate overall score
        weights = {
            'length': 0.4,
            'wildcard_density': 0.6
        }
        self.score = sum(score * weights[factor] for factor, score in factors.items()) * 10
        self.factors = factors
        
        if factors['wildcard_density'] > 0.4:
            self.suggestions.append(
                "High wildcard usage may impact performance. Consider using more specific patterns."
            )
            
    def _analyze_semantic(self) -> None:
        """Analyze semantic pattern complexity."""
        if not isinstance(self.pattern, dict):
            return
            
        factors = {}
        
        # Rule count
        rule_count = len(self.pattern)
        factors['rule_count'] = min(rule_count / 5, 1.0)
        
        # Rule complexity
        complex_rules = {
            SemanticRule.TAKES_ARGUMENT,
            SemanticRule.CALLS_FUNCTION,
            SemanticRule.RAISES_EXCEPTION
        }
        complex_count = sum(1 for rule in self.pattern if rule in complex_rules)
        factors['rule_complexity'] = complex_count / max(rule_count, 1)
        
        # Calculate overall score
        weights = {
            'rule_count': 0.4,
            'rule_complexity': 0.6
        }
        self.score = sum(score * weights[factor] for factor, score in factors.items()) * 10
        self.factors = factors
        
        if factors['rule_complexity'] > 0.6:
            self.suggestions.append(
                "Complex semantic rules may impact performance. Consider breaking down into "
                "simpler patterns or using caching."
            )

@dataclass
class PatternAnalytics:
    """Analytics data for a pattern."""
    pattern_id: str
    pattern: Union[str, Dict[str, Any]]
    pattern_type: PatternType
    total_checks: int = 0
    total_matches: int = 0
    avg_match_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    match_history: List[Tuple[float, bool]] = field(default_factory=list)
    performance_stats: Dict[str, float] = field(default_factory=dict)
    complexity: Optional[PatternComplexity] = None
    
    # Enhanced metrics
    peak_match_time: float = 0.0
    match_time_variance: float = 0.0
    match_time_percentiles: Dict[str, float] = field(default_factory=dict)
    hourly_stats: Dict[int, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    cache_stats: Dict[str, int] = field(default_factory=lambda: {"hits": 0, "misses": 0})
    error_stats: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

class AnalyticsManager:
    """Manages pattern analytics and usage statistics."""
    
    def __init__(self, history_size: int = 1000):
        """Initialize analytics manager."""
        self.analytics: Dict[str, PatternAnalytics] = {}
        self.history_size = history_size
        self.global_stats: Dict[str, float] = {
            "total_checks": 0,
            "total_matches": 0,
            "avg_match_time": 0.0,
            "cache_hit_rate": 0.0,
            "peak_memory_usage": 0.0,
            "pattern_complexity_avg": 0.0
        }
        
    def record_match_attempt(self, pattern_id: str, pattern: Union[str, Dict[str, Any]],
                           pattern_type: PatternType, matched: bool, duration: float,
                           memory_usage: Optional[float] = None, error: Optional[str] = None):
        """Record a pattern match attempt with enhanced metrics."""
        if pattern_id not in self.analytics:
            analytics = PatternAnalytics(
                pattern_id=pattern_id,
                pattern=pattern,
                pattern_type=pattern_type
            )
            # Calculate pattern complexity
            complexity = PatternComplexity(pattern, pattern_type)
            complexity.analyze()
            analytics.complexity = complexity
            self.analytics[pattern_id] = analytics
        else:
        analytics = self.analytics[pattern_id]
            
        # Update basic stats
        analytics.total_checks += 1
        if matched:
            analytics.total_matches += 1
            
        # Update timing stats
        analytics.avg_match_time = (
            (analytics.avg_match_time * (analytics.total_checks - 1) + duration) /
            analytics.total_checks
        )
        analytics.peak_match_time = max(analytics.peak_match_time, duration)
        
        # Update match history
        current_time = time.time()
        analytics.match_history.append((current_time, matched))
        if len(analytics.match_history) > self.history_size:
            analytics.match_history = analytics.match_history[-self.history_size:]
            
        # Update hourly stats
        hour = time.localtime(current_time).tm_hour
        hour_stats = analytics.hourly_stats[hour]
        hour_stats["count"] = hour_stats.get("count", 0) + 1
        hour_stats["matches"] = hour_stats.get("matches", 0) + (1 if matched else 0)
        hour_stats["avg_time"] = (
            (hour_stats.get("avg_time", 0) * (hour_stats["count"] - 1) + duration) /
            hour_stats["count"]
        )
        
        # Update error stats if applicable
        if error:
            analytics.error_stats[error] += 1
            
        # Update memory stats if provided
        if memory_usage is not None:
            analytics.performance_stats["memory_usage"] = memory_usage
            self.global_stats["peak_memory_usage"] = max(
                self.global_stats["peak_memory_usage"],
                memory_usage
            )
        
        # Update global stats
        self.global_stats["total_checks"] += 1
        if matched:
            self.global_stats["total_matches"] += 1
        self.global_stats["avg_match_time"] = (
            (self.global_stats["avg_match_time"] * 
             (self.global_stats["total_checks"] - 1) + duration) /
            self.global_stats["total_checks"]
        )
        
        # Update pattern complexity average
        total_complexity = sum(
            a.complexity.score for a in self.analytics.values() 
            if a.complexity is not None
        )
        self.global_stats["pattern_complexity_avg"] = (
            total_complexity / len(self.analytics)
        )
        
    def calculate_percentiles(self, pattern_id: str) -> None:
        """Calculate match time percentiles for a pattern."""
        analytics = self.analytics.get(pattern_id)
        if not analytics or not analytics.match_history:
            return
            
        times = [t for t, _ in analytics.match_history]
        times.sort()
        n = len(times)
        
        analytics.match_time_percentiles = {
            "p50": times[n // 2],
            "p90": times[int(n * 0.9)],
            "p95": times[int(n * 0.95)],
            "p99": times[int(n * 0.99)]
        }
        
        # Calculate variance
        mean = sum(times) / n
        variance = sum((t - mean) ** 2 for t in times) / n
        analytics.match_time_variance = variance
        
    def get_pattern_analytics(self, pattern_id: str) -> Optional[PatternAnalytics]:
        """Get analytics for a specific pattern."""
        analytics = self.analytics.get(pattern_id)
        if analytics:
            self.calculate_percentiles(pattern_id)
        return analytics
        
    def get_global_stats(self) -> Dict[str, float]:
        """Get global statistics."""
        return self.global_stats
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""
        report = {
            "global_stats": self.global_stats,
            "pattern_stats": {},
            "recommendations": [],
            "complexity_analysis": {},
            "hourly_trends": defaultdict(dict),
            "error_analysis": defaultdict(int)
        }
        
        # Analyze each pattern
        for pattern_id, analytics in self.analytics.items():
            # Basic stats
            pattern_stats = {
                "total_checks": analytics.total_checks,
                "total_matches": analytics.total_matches,
                "match_rate": (analytics.total_matches / analytics.total_checks 
                             if analytics.total_checks > 0 else 0),
                "avg_match_time": analytics.avg_match_time,
                "peak_match_time": analytics.peak_match_time,
                "last_used": analytics.last_used
            }
            
            # Add percentiles
            self.calculate_percentiles(pattern_id)
            pattern_stats.update({
                "percentiles": analytics.match_time_percentiles,
                "variance": analytics.match_time_variance
            })
            
            # Add complexity score
            if analytics.complexity:
                pattern_stats["complexity_score"] = analytics.complexity.score
                pattern_stats["complexity_factors"] = analytics.complexity.factors
            
            report["pattern_stats"][pattern_id] = pattern_stats
            
            # Add hourly trends
            report["hourly_trends"][pattern_id] = analytics.hourly_stats
            
            # Add error stats
            for error, count in analytics.error_stats.items():
                report["error_analysis"][error] += count
            
            # Generate recommendations
            self._add_recommendations(report, pattern_id, pattern_stats, analytics)
            
        return report
        
    def _add_recommendations(self, report: Dict[str, Any], pattern_id: str,
                           stats: Dict[str, Any], analytics: PatternAnalytics) -> None:
        """Add recommendations based on pattern analysis."""
        # Check match rate
        if stats["match_rate"] < 0.01:
                report["recommendations"].append({
                    "pattern_id": pattern_id,
                    "type": "low_match_rate",
                "message": f"Pattern has very low match rate ({stats['match_rate']:.2%})",
                "suggestion": "Consider refining or removing this pattern",
                "priority": "high"
                })
                
        # Check performance
        if stats["avg_match_time"] > self.global_stats["avg_match_time"] * 2:
                report["recommendations"].append({
                    "pattern_id": pattern_id,
                    "type": "slow_matching",
                    "message": "Pattern matching is significantly slower than average",
                "suggestion": "Optimize pattern or use simpler matching strategy",
                "priority": "high"
            })
            
        # Check complexity
        if analytics.complexity and analytics.complexity.score > 7:
            report["recommendations"].append({
                "pattern_id": pattern_id,
                "type": "high_complexity",
                "message": f"Pattern complexity score is high ({analytics.complexity.score:.1f}/10)",
                "suggestions": analytics.complexity.suggestions,
                "priority": "medium"
            })
            
        # Check variance
        if analytics.match_time_variance > stats["avg_match_time"] * 2:
            report["recommendations"].append({
                "pattern_id": pattern_id,
                "type": "high_variance",
                "message": "Pattern has high performance variance",
                "suggestion": "Consider optimizing for more consistent performance",
                "priority": "medium"
            })
            
        # Check usage patterns
        if (time.time() - stats["last_used"]) > 86400:  # 24 hours
                report["recommendations"].append({
                    "pattern_id": pattern_id,
                    "type": "inactive",
                    "message": "Pattern has not been used in the last 24 hours",
                "suggestion": "Consider disabling or removing if no longer needed",
                "priority": "low"
            })
            
        # Check error rates
        total_errors = sum(analytics.error_stats.values())
        if total_errors > 0:
            error_rate = total_errors / analytics.total_checks
            if error_rate > 0.05:  # 5% error rate threshold
                report["recommendations"].append({
                    "pattern_id": pattern_id,
                    "type": "high_error_rate",
                    "message": f"Pattern has high error rate ({error_rate:.2%})",
                    "suggestion": "Review pattern for potential issues",
                    "priority": "high"
                })

class PointcutManager:
    """Manages pointcut specifications and matching."""
    
    def __init__(self, cache_ttl: int = 3600, max_cache_size: int = 1000):
        """Initialize pointcut manager."""
        self.pointcuts: Dict[str, Union[Pointcut, CompositePointcut]] = {}
        self.active_matches: Set[PointcutMatch] = set()
        self._match_cache: Dict[str, List[PointcutMatch]] = {}
        self._match_stats: Dict[str, Dict[str, int]] = {}
        self._batch_size: int = 100
        self._validator = PatternValidator()
        
        # Pattern caching
        self._pattern_cache: Dict[str, CachedPattern] = {}
        self._cache_ttl = cache_ttl
        self._max_cache_size = max_cache_size
        
        # Pattern indexing
        self._pattern_index = PatternIndex()
        
        # Analytics
        self._analytics = AnalyticsManager()
        
    def _get_cached_matcher(self, pattern: Union[str, Dict[str, Any]], 
                           pattern_type: PatternType) -> MatchStrategy:
        """Get or create cached pattern matcher."""
        cache_key = f"{pattern}:{pattern_type}"
        now = time.time()
        
        # Check cache and TTL
        if cache_key in self._pattern_cache:
            cached = self._pattern_cache[cache_key]
            if now - cached.created_at <= self._cache_ttl:
                cached.hits += 1
                cached.last_used = now
                return cached.matcher
            else:
                # Expired
                cached.misses += 1
                del self._pattern_cache[cache_key]
                
        # Create new matcher
        if pattern_type == PatternType.REGEX:
            matcher = RegexStrategy(pattern)
        elif pattern_type == PatternType.WILDCARD:
            matcher = WildcardStrategy(pattern)
        elif pattern_type == PatternType.SEMANTIC:
            matcher = SemanticStrategy(pattern)
        else:
            raise ValueError(f"Unsupported pattern type: {pattern_type}")
            
        # Add to cache
        self._pattern_cache[cache_key] = CachedPattern(
            pattern=pattern,
            matcher=matcher
        )
        
        # Cleanup cache if needed
        if len(self._pattern_cache) > self._max_cache_size:
            self._cleanup_cache()
            
        return matcher
        
    def _cleanup_cache(self):
        """Remove least recently used patterns from cache."""
        # Sort by last used time
        sorted_patterns = sorted(
            self._pattern_cache.items(),
            key=lambda x: x[1].last_used
        )
        
        # Remove oldest 10%
        num_to_remove = max(1, len(self._pattern_cache) // 10)
        for key, _ in sorted_patterns[:num_to_remove]:
            del self._pattern_cache[key]
        
    def validate_pattern(self, pattern: Union[str, Dict[str, Any]], 
                        pattern_type: PatternType) -> ValidationResult:
        """Validate a pattern before adding it."""
        return self._validator.validate_pattern(pattern, pattern_type)
        
    def add_pointcut(self, pattern: Union[str, Dict[str, Any]], 
                    pointcut_type: PointcutType,
                    pattern_type: PatternType = PatternType.REGEX,
                    metadata: Optional[Dict[str, str]] = None) -> str:
        """Add a new pointcut with validation and indexing."""
        # Validate pattern first
        validation = self.validate_pattern(pattern, pattern_type)
        if not validation.is_valid:
            raise ValueError(f"Invalid pattern: {', '.join(validation.errors)}")
            
        # Add warnings to metadata
        metadata = metadata or {}
        if validation.warnings:
            metadata["warnings"] = ", ".join(validation.warnings)
            
        # Add suggestions to metadata
        if validation.suggestions:
            metadata["suggestions"] = ", ".join(
                f"{s.suggested} ({s.reason})" for s in validation.suggestions
            )
            
        # Create pointcut
        pointcut_id = f"{pointcut_type}_{len(self.pointcuts)}"
        pointcut = Pointcut(
            pattern=pattern,
            pointcut_type=pointcut_type,
            pattern_type=pattern_type,
            metadata=metadata
        )
        
        # Add to indexes
        self._pattern_index.add_pattern(
            pattern_id=pointcut_id,
            pattern=pattern if isinstance(pattern, str) else str(pattern),
            pattern_type=pattern_type
        )
        
        self.pointcuts[pointcut_id] = pointcut
        self._match_stats[pointcut_id] = {"checks": 0, "matches": 0}
        self._match_cache.clear()
        
        return pointcut_id
        
    def remove_pointcut(self, pointcut_id: str) -> None:
        """Remove a pointcut and its indexes."""
        if pointcut_id not in self.pointcuts:
            raise KeyError(f"Pointcut {pointcut_id} not found")
            
        pointcut = self.pointcuts[pointcut_id]
        pattern = pointcut.pattern if isinstance(pointcut.pattern, str) else str(pointcut.pattern)
        
        # Remove from indexes
        self._pattern_index.remove_pattern(
            pattern_id=pointcut_id,
            pattern=pattern,
            pattern_type=pointcut.pattern_type
        )
        
        del self.pointcuts[pointcut_id]
        del self._match_stats[pointcut_id]
        self._match_cache.clear()
        
    def check_matches(self, target: str, context: Optional[Dict[str, str]] = None) -> List[PointcutMatch]:
        """Check if target matches any pointcuts using indexes."""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{target}_{str(context)}"
        if cache_key in self._match_cache:
            return self._match_cache[cache_key]
            
        matches = []
        # Get candidate patterns using indexes
        candidates = self._pattern_index.find_candidates(target)
        
        for pointcut_id in candidates:
            pointcut = self.pointcuts[pointcut_id]
            if not isinstance(pointcut, Pointcut) or not pointcut.enabled:
                continue
                
            match_start = time.time()
            matched, groups = pointcut.matches(target)
            match_duration = time.time() - match_start
            
            # Record analytics
            self._analytics.record_match_attempt(
                pattern_id=pointcut_id,
                pattern=pointcut.pattern,
                pattern_type=pointcut.pattern_type,
                matched=matched,
                duration=match_duration
            )
            
            if matched:
                self._match_stats[pointcut_id]["matches"] += 1
                match_obj = PointcutMatch(
                    pointcut=pointcut,
                    target_name=target,
                    match_groups=[groups] if isinstance(pointcut, Pointcut) else groups,
                    context=context or {}
                )
                matches.append(match_obj)
                self.active_matches.add(match_obj)
                
        self._match_cache[cache_key] = matches
        return matches
        
    def set_batch_size(self, size: int) -> None:
        """Set the batch size for batch operations."""
        self._batch_size = size
        
    def batch_check_matches(self, targets: List[str], 
                          context: Optional[Dict[str, str]] = None,
                          parallel: bool = False) -> BatchMatchResult:
        """Check multiple targets against all pointcuts.
        
        Args:
            targets: List of targets to check
            context: Optional context information
            parallel: Whether to process in parallel
            
        Returns:
            BatchMatchResult containing matches and statistics
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.time()
        results: Dict[str, List[PointcutMatch]] = {}
        stats: Dict[str, Dict[str, int]] = {
            pid: {"checks": 0, "matches": 0} for pid in self.pointcuts
        }
        
        def process_batch(batch: List[str]) -> Dict[str, List[PointcutMatch]]:
            batch_results = {}
            for target in batch:
                matches = self.check_matches(target, context)
                if matches:
                    batch_results[target] = matches
            return batch_results
            
        # Split targets into batches
        batches = [targets[i:i + self._batch_size] 
                  for i in range(0, len(targets), self._batch_size)]
        
        if parallel and len(batches) > 1:
            with ThreadPoolExecutor() as executor:
                future_to_batch = {
                    executor.submit(process_batch, batch): batch 
                    for batch in batches
                }
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    results.update(batch_results)
        else:
            for batch in batches:
                batch_results = process_batch(batch)
                results.update(batch_results)
        
        # Update statistics
        for matches in results.values():
            for match in matches:
                if isinstance(match.pointcut, Pointcut):
                    pointcut_id = next(pid for pid, p in self.pointcuts.items() 
                                    if p is match.pointcut)
                    stats[pointcut_id]["matches"] += 1
                    
        # Record total checks
        total_targets = len(targets)
        for pointcut_id in self.pointcuts:
            stats[pointcut_id]["checks"] += total_targets
            
        return BatchMatchResult(
            matches=results,
            stats=stats,
            duration=time.time() - start_time
        )
        
    def batch_add_pointcuts(self, pointcuts: List[Dict[str, Any]]) -> List[str]:
        """Add multiple pointcuts in a batch.
        
        Args:
            pointcuts: List of pointcut specifications
            
        Returns:
            List of created pointcut IDs
        """
        pointcut_ids = []
        for spec in pointcuts:
            try:
                pointcut_id = self.add_pointcut(
                    pattern=spec["pattern"],
                    pointcut_type=spec["type"],
                    pattern_type=spec.get("pattern_type", PatternType.REGEX),
                    metadata=spec.get("metadata", {})
                )
                pointcut_ids.append(pointcut_id)
            except Exception as e:
                logger.error(f"Error adding pointcut: {str(e)}")
                # Continue with remaining pointcuts
                continue
        return pointcut_ids
        
    def batch_remove_pointcuts(self, pointcut_ids: List[str]) -> List[str]:
        """Remove multiple pointcuts in a batch.
        
        Args:
            pointcut_ids: List of pointcut IDs to remove
            
        Returns:
            List of successfully removed pointcut IDs
        """
        removed_ids = []
        for pid in pointcut_ids:
            try:
                self.remove_pointcut(pid)
                removed_ids.append(pid)
            except Exception as e:
                logger.error(f"Error removing pointcut {pid}: {str(e)}")
                continue
        return removed_ids
        
    def batch_enable_pointcuts(self, pointcut_ids: List[str]) -> List[str]:
        """Enable multiple pointcuts in a batch."""
        enabled_ids = []
        for pid in pointcut_ids:
            try:
                self.enable_pointcut(pid)
                enabled_ids.append(pid)
            except Exception as e:
                logger.error(f"Error enabling pointcut {pid}: {str(e)}")
                continue
        return enabled_ids
        
    def batch_disable_pointcuts(self, pointcut_ids: List[str]) -> List[str]:
        """Disable multiple pointcuts in a batch."""
        disabled_ids = []
        for pid in pointcut_ids:
            try:
                self.disable_pointcut(pid)
                disabled_ids.append(pid)
            except Exception as e:
                logger.error(f"Error disabling pointcut {pid}: {str(e)}")
                continue
        return disabled_ids
        
    def get_analytics(self, pattern_id: Optional[str] = None) -> Union[PatternAnalytics, Dict[str, float]]:
        """Get analytics data."""
        if pattern_id:
            return self._analytics.get_pattern_analytics(pattern_id)
        return self._analytics.get_global_stats()
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        return self._analytics.get_performance_report()

@dataclass
class PointcutMatch:
    """Represents a match result from a pointcut."""
    pointcut: 'Pointcut'
    target_name: str
    match_groups: List[Tuple[str, ...]]
    context: Dict[str, str]
    timestamp: float = field(default_factory=time.time)

@dataclass
class CompositePointcut:
    """Represents a composite pointcut combining multiple pointcuts."""
    operator: str  # 'AND', 'OR', 'NOT'
    pointcuts: List[Union['Pointcut', 'CompositePointcut']]
    metadata: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    
    def matches(self, target: str) -> Tuple[bool, List[Tuple[str, ...]]]:
        """Check if target matches this composite pointcut."""
        results = [p.matches(target) for p in self.pointcuts]
        groups = [g for _, g in results if g]
        
        if self.operator == 'AND':
            return all(m for m, _ in results), groups
        elif self.operator == 'OR':
            return any(m for m, _ in results), groups
        elif self.operator == 'NOT':
            # NOT only makes sense with a single pointcut
            matched, _ = results[0]
            return not matched, []
        else:
            raise ValueError(f"Invalid operator: {self.operator}")