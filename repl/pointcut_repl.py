"""REPL interface for pointcut specification and management."""

import cmd
import logging
import shlex
from typing import List, Optional

from ..core.pointcut_manager import PointcutManager, PointcutType

logger = logging.getLogger(__name__)

class PointcutREPL(cmd.Cmd):
    """Interactive REPL for pointcut management."""
    
    intro = "Welcome to the Pointcut Manager REPL. Type help or ? to list commands.\n"
    prompt = "(pointcut) "
    
    def __init__(self):
        """Initialize REPL."""
        super().__init__()
        self.manager = PointcutManager()
        
    def do_add(self, arg: str) -> None:
        """Add a new pointcut.
        
        Usage: add <type> <pattern> [metadata key=value...]
        Example: add execution get_* scope=public type=method
        """
        try:
            args = shlex.split(arg)
            if len(args) < 2:
                print("Error: Must provide pointcut type and pattern")
                return
                
            pointcut_type = args[0].upper()
            if not hasattr(PointcutType, pointcut_type):
                print(f"Error: Invalid pointcut type. Valid types: {[t.name for t in PointcutType]}")
                return
                
            pattern = args[1]
            metadata = {}
            
            # Parse metadata key=value pairs
            for meta_arg in args[2:]:
                try:
                    key, value = meta_arg.split('=')
                    metadata[key] = value
                except ValueError:
                    print(f"Warning: Skipping invalid metadata format: {meta_arg}")
                    
            pointcut_id = self.manager.add_pointcut(
                pattern=pattern,
                pointcut_type=getattr(PointcutType, pointcut_type),
                metadata=metadata
            )
            print(f"Added pointcut {pointcut_id}")
            
        except Exception as e:
            print(f"Error adding pointcut: {str(e)}")
            
    def do_remove(self, arg: str) -> None:
        """Remove a pointcut by ID.
        
        Usage: remove <pointcut_id>
        Example: remove execution_0
        """
        try:
            pointcut_id = arg.strip()
            if not pointcut_id:
                print("Error: Must provide pointcut ID")
                return
                
            self.manager.remove_pointcut(pointcut_id)
            print(f"Removed pointcut {pointcut_id}")
            
        except Exception as e:
            print(f"Error removing pointcut: {str(e)}")
            
    def do_list(self, arg: str) -> None:
        """List all pointcuts.
        
        Usage: list [enabled|disabled]
        Example: list enabled
        """
        filter_enabled = None
        if arg.strip().lower() == 'enabled':
            filter_enabled = True
        elif arg.strip().lower() == 'disabled':
            filter_enabled = False
            
        print("\nActive Pointcuts:")
        for pointcut_id, pointcut in self.manager.pointcuts.items():
            if filter_enabled is not None and pointcut.enabled != filter_enabled:
                continue
                
            status = "enabled" if pointcut.enabled else "disabled"
            print(f"{pointcut_id}: {pointcut.pattern} ({status})")
            if pointcut.metadata:
                print(f"  Metadata: {pointcut.metadata}")
        print()
        
    def do_check(self, arg: str) -> None:
        """Check if a target matches any pointcuts.
        
        Usage: check <target_name>
        Example: check get_user_data
        """
        target = arg.strip()
        if not target:
            print("Error: Must provide target name")
            return
            
        matches = self.manager.check_matches(target)
        if matches:
            print(f"\nMatches for '{target}':")
            for match in matches:
                print(f"- {match.pointcut.pattern} ({match.pointcut.pointcut_type.value})")
                if match.match_groups:
                    print(f"  Groups: {match.match_groups}")
        else:
            print(f"\nNo matches found for '{target}'")
        print()
        
    def do_enable(self, arg: str) -> None:
        """Enable a pointcut.
        
        Usage: enable <pointcut_id>
        Example: enable execution_0
        """
        try:
            pointcut_id = arg.strip()
            if not pointcut_id:
                print("Error: Must provide pointcut ID")
                return
                
            self.manager.enable_pointcut(pointcut_id)
            print(f"Enabled pointcut {pointcut_id}")
            
        except Exception as e:
            print(f"Error enabling pointcut: {str(e)}")
            
    def do_disable(self, arg: str) -> None:
        """Disable a pointcut.
        
        Usage: disable <pointcut_id>
        Example: disable execution_0
        """
        try:
            pointcut_id = arg.strip()
            if not pointcut_id:
                print("Error: Must provide pointcut ID")
                return
                
            self.manager.disable_pointcut(pointcut_id)
            print(f"Disabled pointcut {pointcut_id}")
            
        except Exception as e:
            print(f"Error disabling pointcut: {str(e)}")
            
    def do_clear(self, arg: str) -> None:
        """Clear all active matches.
        
        Usage: clear
        """
        self.manager.clear_matches()
        print("Cleared all active matches")
        
    def do_quit(self, arg: str) -> bool:
        """Exit the REPL.
        
        Usage: quit
        """
        print("\nGoodbye!")
        return True
        
    def do_EOF(self, arg: str) -> bool:
        """Handle EOF (Ctrl+D)."""
        return self.do_quit(arg)
        
def main():
    """Run the pointcut REPL."""
    try:
        repl = PointcutREPL()
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        
if __name__ == "__main__":
    main() 