"""
Interactive REPL shell for the multi-agent framework.
"""

import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from pygments.lexers.python import PythonLexer
from .command_completer import CommandCompleter
from .code_sandbox import CodeSandbox, SandboxConfig

class InteractiveShell:
    """Interactive shell for the multi-agent framework."""
    
    def __init__(self):
        self.history = InMemoryHistory()
        self.style = Style.from_dict({
            'prompt': 'ansicyan bold',
            'continuation': 'ansiblue',
            'error': 'ansired bold'
        })
        
        self.session = PromptSession(
            history=self.history,
            lexer=PygmentsLexer(PythonLexer),
            style=self.style,
            multiline=True,
            prompt_continuation=self.continuation_prompt
        )
        
        self.command_completer = CommandCompleter()
        self.sandbox = CodeSandbox(SandboxConfig(
            max_memory_mb=512,
            max_time_seconds=10,
            max_cpu_time_seconds=5
        ))
        
    def continuation_prompt(self, width, line_number, is_soft_wrap):
        """Format the continuation prompt for multi-line input."""
        return HTML('... ')
        
    async def run_code(self, code: str, context: dict = None) -> dict:
        """Run code in sandbox."""
        return await self.sandbox.run_code(code, context)
        
    def format_result(self, result: dict) -> str:
        """Format execution result for display."""
        if result['success']:
            output = []
            if result['output']:
                output.append(result['output'].rstrip())
            if result['locals']:
                # Format any new/modified variables
                vars_output = []
                for name, value in result['locals'].items():
                    if not name.startswith('_'):
                        vars_output.append(f"{name} = {repr(value)}")
                if vars_output:
                    output.append("\nVariables:")
                    output.extend(vars_output)
            return "\n".join(output)
        else:
            error = result['error']
            return HTML(
                f'<ansired>Error: {error["type"]}\n'
                f'{error["message"]}\n\n'
                f'{error["traceback"]}</ansired>'
            )
            
    async def run(self):
        """Run the interactive shell."""
        print(HTML(
            '<ansigreen>Multi-Agent Framework Interactive Shell</ansigreen>\n'
            '<ansiblue>Type "exit" to quit, "help" for commands</ansiblue>'
        ))
        
        context = {}  # Shared context between commands
        
        while True:
            try:
                # Get input with command completion and syntax highlighting
                code = await self.session.prompt_async(
                    HTML('<ansicyan>>> </ansicyan>'),
                    completer=self.command_completer,
                    enable_suspend=True,
                    key_bindings=None,  # Use default key bindings for multi-line
                    bottom_toolbar=self.get_toolbar_tokens
                )
                
                if not code.strip():
                    continue
                    
                if code.strip() == 'exit':
                    break
                    
                if code.strip() == 'help':
                    print(self.get_help())
                    continue
                    
                # Execute code in sandbox
                result = await self.run_code(code, context)
                
                # Update shared context with new variables
                if result['success']:
                    context.update(result['locals'])
                    
                # Display result
                output = self.format_result(result)
                if output:
                    print(output)
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                print(HTML(f'<ansired>Shell error: {str(e)}</ansired>'))
                
        # Cleanup
        self.sandbox.cleanup()
        print(HTML('\n<ansigreen>Goodbye!</ansigreen>'))
        
    def get_toolbar_tokens(self):
        """Get the bottom toolbar tokens."""
        return HTML(
            '<ansigray>'
            'Multi-line: Alt+Enter | '
            'History: Up/Down | '
            'Clear: Ctrl+C | '
            'Exit: Ctrl+D'
            '</ansigray>'
        )
        
    def get_help(self) -> str:
        """Get help text."""
        return HTML("""
<ansigreen>Available Commands:</ansigreen>
<ansiblue>- exit:</ansiblue> Exit the shell
<ansiblue>- help:</ansiblue> Show this help message

<ansigreen>Python code can be executed directly.</ansigreen>
<ansigreen>The following modules are available by default:</ansigreen>
<ansiblue>- asyncio</ansiblue>
<ansiblue>- datetime</ansiblue>
<ansiblue>- json</ansiblue>
<ansiblue>- math</ansiblue>
<ansiblue>- random</ansiblue>
<ansiblue>- typing</ansiblue>
<ansiblue>- collections</ansiblue>
<ansiblue>- itertools</ansiblue>
<ansiblue>- functools</ansiblue>

<ansiyellow>Variables persist between commands in the same session.</ansiyellow>
<ansired>Memory and execution time are limited for security.</ansired>

<ansigreen>Key Bindings:</ansigreen>
<ansiblue>- Alt+Enter:</ansiblue> Submit multi-line code
<ansiblue>- Up/Down:</ansiblue> Navigate history
<ansiblue>- Ctrl+C:</ansiblue> Clear current input
<ansiblue>- Ctrl+D:</ansiblue> Exit shell
""")

async def main():
    """Run the interactive shell."""
    shell = InteractiveShell()
    await shell.run()
    
if __name__ == "__main__":
    asyncio.run(main()) 