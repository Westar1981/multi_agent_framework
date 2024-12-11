from typing import Dict, Any, List
from .base_agent import BaseAgent, Message
from loguru import logger
import ast
import re
import astor
from contextlib import contextmanager

class Advice:
    """Base class for all advice types."""
    def __init__(self, func):
        self.func = func
        self.__name__ = self.__class__.__name__

    @contextmanager
    def __call__(self, func_name):
        yield

class BeforeAdvice(Advice):
    """Advice that executes before the target function."""
    @contextmanager
    def __call__(self, func_name):
        self.func()
        yield

class AfterAdvice(Advice):
    """Advice that executes after the target function."""
    @contextmanager
    def __call__(self, func_name):
        yield
        self.func()

class AroundAdvice(Advice):
    """Advice that executes around the target function."""
    @contextmanager
    def __call__(self, func_name):
        self.func(before=True)
        yield
        self.func(before=False)

class Aspect:
    """Represents an aspect with pointcut and advice."""
    def __init__(self, pointcut: str, advice: Advice):
        self.pointcut = pointcut
        self.advice = advice
        self.pointcut_regex = re.compile(pointcut.replace("*", ".*"))

    def matches(self, target: str) -> bool:
        """Check if the target matches this aspect's pointcut."""
        return bool(self.pointcut_regex.match(target))

class AOPWeaver:
    """Handles the weaving of aspects into code."""
    def __init__(self):
        self.aspects: List[Aspect] = []

    def register_aspect(self, aspect: Aspect):
        """Register a new aspect."""
        self.aspects.append(aspect)
        logger.info(f"Registered aspect with pointcut: {aspect.pointcut}")

    def _apply_advice(self, func_node: ast.FunctionDef, advice: Advice):
        """Apply advice to a function node."""
        existing_with_node = None
        existing_with_item_index = None

        # Look for existing advice
        for i, node in enumerate(func_node.body):
            if isinstance(node, ast.With):
                for item in node.items:
                    try:
                        if item.context_expr.func.id == advice.__name__:
                            existing_with_node = node
                            existing_with_item_index = i
                            break
                    except AttributeError:
                        continue

        # Create new with node
        new_with_node = ast.With(
            items=[ast.withitem(
                context_expr=ast.Call(
                    func=ast.Name(id=advice.__name__),
                    args=[ast.Name(id=func_node.name)],
                    keywords=[]
                ),
                optional_vars=None
            )],
            body=existing_with_node.body if existing_with_node else func_node.body
        )

        if existing_with_node:
            func_node.body[existing_with_item_index] = new_with_node
        else:
            func_node.body = [new_with_node]

    def weave(self, tree: ast.AST) -> ast.AST:
        """Weave all registered aspects into the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for aspect in self.aspects:
                    if aspect.matches(node.name):
                        self._apply_advice(node, aspect.advice)
        return tree

class CodeTransformerAgent(BaseAgent):
    """Agent specialized in code transformation using AOP."""
    
    def __init__(self, agent_id: str = None):
        super().__init__(agent_id=agent_id, name="CodeTransformer")
        self.weaver = AOPWeaver()
        self.transformed_code = {}

    async def process_message(self, message: Message):
        """Process incoming messages."""
        if message.message_type == "transform_code":
            await self.transform_code(
                message.content.get("code", ""),
                message.content.get("aspects", []),
                message.sender
            )
        elif message.message_type == "get_transformed_code":
            await self.send_transformed_code(message.sender)
        else:
            logger.warning(f"Unknown message type received: {message.message_type}")

    async def handle_task(self, task: Dict[str, Any]):
        """Handle specific transformation tasks."""
        if task.get("type") == "transform":
            await self.transform_code(
                task.get("code", ""),
                task.get("aspects", []),
                task.get("requester")
            )

    async def transform_code(self, code: str, aspects: List[Dict[str, Any]], requester: str):
        """Transform code using the provided aspects."""
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Register aspects
            for aspect_data in aspects:
                advice_type = aspect_data.get("advice_type", "before")
                advice_class = {
                    "before": BeforeAdvice,
                    "after": AfterAdvice,
                    "around": AroundAdvice
                }.get(advice_type)
                
                if advice_class:
                    aspect = Aspect(
                        aspect_data["pointcut"],
                        advice_class(eval(aspect_data["advice_code"]))
                    )
                    self.weaver.register_aspect(aspect)
            
            # Apply transformations
            transformed_tree = self.weaver.weave(tree)
            
            # Convert back to source code
            transformed_code = astor.to_source(transformed_tree)
            self.transformed_code[requester] = transformed_code
            
            # Send back the results
            await self.send_message(
                receiver=requester,
                content={"transformed_code": transformed_code},
                message_type="transformation_complete"
            )
            
        except Exception as e:
            logger.error(f"Error transforming code: {str(e)}")
            await self.send_message(
                receiver=requester,
                content={"error": f"Transformation error: {str(e)}"},
                message_type="transformation_error"
            )

    async def send_transformed_code(self, requester: str):
        """Send transformed code to the requester."""
        if requester in self.transformed_code:
            await self.send_message(
                receiver=requester,
                content={"transformed_code": self.transformed_code[requester]},
                message_type="transformed_code"
            )
        else:
            await self.send_message(
                receiver=requester,
                content={"error": "No transformed code found"},
                message_type="transformation_error"
            ) 