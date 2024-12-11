from typing import Dict, Any
from .base_agent import BaseAgent, Message
from loguru import logger
import ast
import asyncio

class CodeAnalyzer(BaseAgent):
    """Agent specialized in analyzing code structure and quality."""

    def __init__(self, agent_id: str = None):
        super().__init__(agent_id=agent_id, name="CodeAnalyzer")
        self.analysis_results = {}

    async def process_message(self, message: Message):
        """Process incoming messages."""
        if message.message_type == "analyze_code":
            await self.analyze_code(message.content.get("code", ""), message.sender)
        elif message.message_type == "get_analysis":
            await self.send_analysis_results(message.sender)
        else:
            logger.warning(f"Unknown message type received: {message.message_type}")

    async def handle_task(self, task: Dict[str, Any]):
        """Handle specific analysis tasks."""
        if task.get("type") == "analyze":
            code = task.get("code", "")
            await self.analyze_code(code, task.get("requester"))

    async def analyze_code(self, code: str, requester: str):
        """Analyze the provided code and store results."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Perform various analyses
            analysis = {
                "complexity": self._analyze_complexity(tree),
                "structure": self._analyze_structure(tree),
                "metrics": self._calculate_metrics(tree)
            }
            
            self.analysis_results[requester] = analysis
            
            # Send back the results
            await self.send_message(
                receiver=requester,
                content={"analysis": analysis},
                message_type="analysis_complete"
            )
            
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {str(e)}")
            await self.send_message(
                receiver=requester,
                content={"error": f"Syntax error: {str(e)}"},
                message_type="analysis_error"
            )
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            await self.send_message(
                receiver=requester,
                content={"error": f"Analysis error: {str(e)}"},
                message_type="analysis_error"
            )

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity."""
        complexity = {
            "cyclomatic_complexity": 0,
            "depth": 0,
            "num_functions": 0
        }
        
        for node in ast.walk(tree):
            # Count control flow statements for cyclomatic complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity["cyclomatic_complexity"] += 1
            # Count function definitions
            elif isinstance(node, ast.FunctionDef):
                complexity["num_functions"] += 1
        
        return complexity

    def _analyze_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code structure."""
        structure = {
            "imports": [],
            "classes": [],
            "functions": [],
            "global_variables": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                structure["imports"].extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                structure["imports"].append(f"{node.module}.{node.names[0].name}")
            elif isinstance(node, ast.ClassDef):
                structure["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                structure["functions"].append(node.name)
            elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                structure["global_variables"].append(node.targets[0].id)
        
        return structure

    def _calculate_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate various code metrics."""
        metrics = {
            "loc": len(ast.unparse(tree).splitlines()),
            "num_classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            "num_functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            "num_imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
        }
        return metrics

    async def send_analysis_results(self, requester: str):
        """Send analysis results to the requester."""
        if requester in self.analysis_results:
            await self.send_message(
                receiver=requester,
                content={"analysis": self.analysis_results[requester]},
                message_type="analysis_results"
            )
        else:
            await self.send_message(
                receiver=requester,
                content={"error": "No analysis results found"},
                message_type="analysis_error"
            )
