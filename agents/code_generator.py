from typing import Dict, Any, List
from .base_agent import BaseAgent, Message
from loguru import logger
import ast
import astor
import re
from contextlib import contextmanager

class CodeGenerator(BaseAgent):
    """Agent specialized in generating and modifying code with AOP patterns."""

    def __init__(self, agent_id: str = None):
        super().__init__(agent_id=agent_id, name="CodeGenerator")
        self.aspects: List[Dict[str, Any]] = []
        self.generated_code = {}

    async def process_message(self, message: Message):
        """Process incoming messages."""
        if message.message_type == "generate_code":
            await self.generate_code(message.content, message.sender)
        elif message.message_type == "add_aspect":
            await self.add_aspect(message.content, message.sender)
        elif message.message_type == "get_generated_code":
            await self.send_generated_code(message.sender)
        else:
            logger.warning(f"Unknown message type received: {message.message_type}")

    async def handle_task(self, task: Dict[str, Any]):
        """Handle specific generation tasks."""
        task_type = task.get("type")
        if task_type == "generate":
            await self.generate_code(task, task.get("requester"))
        elif task_type == "add_aspect":
            await self.add_aspect(task, task.get("requester"))

    async def add_aspect(self, content: Dict[str, Any], requester: str):
        """Add an aspect to be applied during code generation."""
        try:
            aspect = {
                "pointcut": content.get("pointcut"),
                "advice_type": content.get("advice_type"),
                "advice_code": content.get("advice_code")
            }
            self.aspects.append(aspect)
            await self.send_message(
                receiver=requester,
                content={"status": "Aspect added successfully"},
                message_type="aspect_added"
            )
        except Exception as e:
            logger.error(f"Error adding aspect: {str(e)}")
            await self.send_message(
                receiver=requester,
                content={"error": f"Failed to add aspect: {str(e)}"},
                message_type="aspect_error"
            )

    async def generate_code(self, content: Dict[str, Any], requester: str):
        """Generate code with applied aspects."""
        try:
            code = content.get("code", "")
            tree = ast.parse(code)
            modified_tree = self._apply_aspects(tree)
            
            generated_code = astor.to_source(modified_tree)
            self.generated_code[requester] = generated_code
            
            await self.send_message(
                receiver=requester,
                content={"code": generated_code},
                message_type="generation_complete"
            )
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            await self.send_message(
                receiver=requester,
                content={"error": f"Generation error: {str(e)}"},
                message_type="generation_error"
            )

    def _apply_aspects(self, tree: ast.AST) -> ast.AST:
        """Apply registered aspects to the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for aspect in self.aspects:
                    if self._matches_pointcut(node.name, aspect["pointcut"]):
                        node = self._apply_advice(node, aspect)
        return tree

    def _matches_pointcut(self, func_name: str, pointcut: str) -> bool:
        """Check if function name matches pointcut pattern."""
        pattern = pointcut.replace("*", ".*")
        return bool(re.match(pattern, func_name))

    def _apply_advice(self, func_node: ast.FunctionDef, aspect: Dict[str, Any]) -> ast.FunctionDef:
        """Apply advice to function node."""
        advice_type = aspect["advice_type"]
        advice_code = aspect["advice_code"]
        
        # Parse advice code
        advice_tree = ast.parse(advice_code)
        advice_body = advice_tree.body
        
        if advice_type == "before":
            func_node.body = advice_body + func_node.body
        elif advice_type == "after":
            func_node.body = func_node.body + advice_body
        elif advice_type == "around":
            # Create a with statement wrapping the original body
            with_node = ast.With(
                items=[ast.withitem(
                    context_expr=ast.Name(id='contextmanager', ctx=ast.Load()),
                    optional_vars=None
                )],
                body=func_node.body
            )
            func_node.body = [with_node]
        
        return func_node

    async def send_generated_code(self, requester: str):
        """Send generated code to the requester."""
        if requester in self.generated_code:
            await self.send_message(
                receiver=requester,
                content={"code": self.generated_code[requester]},
                message_type="generated_code"
            )
        else:
            await self.send_message(
                receiver=requester,
                content={"error": "No generated code found"},
                message_type="generation_error"
            )
