"""
Command completion system for the REPL.
"""

from typing import Dict, List, Iterable, Optional
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
import re

from ..agents.base_agent import AgentCapability
from ..core.task_allocation import TaskPriority

class CommandCompleter(Completer):
    """Intelligent command completer for the REPL."""
    
    def __init__(self, repl_instance):
        self.repl = repl_instance
        self.commands = {
            'create_agent': self._complete_create_agent,
            'list_agents': lambda _: [],
            'create_task': self._complete_create_task,
            'get_metrics': self._complete_agent_name,
            'get_insights': self._complete_agent_name,
            'dump': self._complete_dump,
            'help': lambda _: []
        }
        
    def get_completions(
        self,
        document: Document,
        complete_event=None
    ) -> Iterable[Completion]:
        """Get command completions."""
        text = document.text
        
        # Complete command names
        if not any(cmd in text for cmd in self.commands):
            word = text.split()[-1] if text else ""
            for cmd in self.commands:
                if cmd.startswith(word):
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display_meta=self._get_command_meta(cmd)
                    )
            return
            
        # Complete command arguments
        for cmd, completer in self.commands.items():
            if cmd in text:
                yield from completer(text)
                
    def _complete_create_agent(self, text: str) -> Iterable[Completion]:
        """Complete create_agent command arguments."""
        if '"' not in text or text.count('"') == 1:
            # Complete agent name
            return []  # Free-form input
            
        if '[' not in text:
            # Start capabilities list
            yield Completion(
                '["',
                display='[capabilities]',
                display_meta='Start capabilities list'
            )
            return
            
        if text.count('"') % 2 == 1:
            # Complete capability name
            word = text.split('"')[-1]
            for cap in AgentCapability:
                if cap.name.startswith(word):
                    yield Completion(
                        cap.name + '"]' if text.count('"') > 2 else cap.name + '", ',
                        start_position=-len(word),
                        display_meta='Capability'
                    )
                    
    def _complete_create_task(self, text: str) -> Iterable[Completion]:
        """Complete create_task command arguments."""
        parts = text.split(',')
        
        if len(parts) == 1:
            # Task name - free-form input
            return []
            
        if len(parts) == 2:
            # Priority
            word = parts[1].strip().strip('"\'')
            for priority in TaskPriority:
                if priority.name.startswith(word):
                    yield Completion(
                        f'"{priority.name}"',
                        start_position=-len(word) if word else 0,
                        display_meta='Priority'
                    )
            return
            
        if len(parts) == 3:
            # Capabilities
            if '[' not in parts[2]:
                yield Completion(
                    '[',
                    display='[capabilities]',
                    display_meta='Start capabilities list'
                )
                return
                
            word = parts[2].split('"')[-1] if '"' in parts[2] else ""
            for cap in AgentCapability:
                if cap.name.startswith(word):
                    yield Completion(
                        cap.name + '"]' if ']' not in parts[2] else cap.name + '", ',
                        start_position=-len(word),
                        display_meta='Capability'
                    )
                    
    def _complete_agent_name(self, text: str) -> Iterable[Completion]:
        """Complete agent name arguments."""
        if '(' not in text:
            return []
            
        word = text.split('(')[1].strip().strip('"\'')
        for agent_id in self.repl.agents:
            if agent_id.startswith(word):
                yield Completion(
                    f'"{agent_id}")',
                    start_position=-len(word) if word else 0,
                    display_meta='Agent ID'
                )
                
    def _complete_dump(self, text: str) -> Iterable[Completion]:
        """Complete dump command arguments."""
        if '(' not in text:
            return []
            
        targets = ['agents', 'tasks', 'learning']
        word = text.split('(')[1].strip().strip('"\'')
        
        for target in targets:
            if target.startswith(word):
                yield Completion(
                    f'"{target}")',
                    start_position=-len(word) if word else 0,
                    display_meta='Dump target'
                )
                
    def _get_command_meta(self, command: str) -> str:
        """Get command metadata for display."""
        meta = {
            'create_agent': 'Create a new agent with capabilities',
            'list_agents': 'List all registered agents',
            'create_task': 'Create a new task for allocation',
            'get_metrics': 'Get agent performance metrics',
            'get_insights': 'Get agent learning insights',
            'dump': 'Dump system state information',
            'help': 'Show help information'
        }
        return meta.get(command, '') 