"""
Knowledge Tree Visualization Component.
Provides interactive visualization of the neural-symbolic knowledge network.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import json

class KnowledgeTreeVisualizer:
    """Interactive visualization of the neural-symbolic knowledge tree."""
    
    def __init__(self):
        self.nodes = {}  # Knowledge nodes
        self.connections = []  # Node connections
        self.active_paths = set()  # Currently active learning paths
        self.color_scheme = {
            'root': '#FFFFFF',  # White for core knowledge
            'symbolic': '#FFB347',  # Orange for symbolic reasoning
            'neural': '#98FB98',  # Green for neural patterns
            'hybrid': '#87CEEB',  # Sky blue for hybrid insights
            'community': '#DDA0DD'  # Plum for community knowledge
        }
        
    def add_knowledge_node(self, 
                          node_id: str,
                          node_type: str,
                          content: Any,
                          parent_id: Optional[str] = None):
        """Add a new knowledge node to the tree."""
        self.nodes[node_id] = {
            'type': node_type,
            'content': content,
            'children': [],
            'metrics': {
                'confidence': 0.0,
                'usage_count': 0,
                'last_accessed': datetime.now()
            },
            'visual': {
                'color': self.color_scheme[node_type],
                'size': 1.0,  # Will scale with importance
                'position': self._calculate_position(parent_id)
            }
        }
        
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id]['children'].append(node_id)
            self.connections.append((parent_id, node_id))
            
    def update_node_metrics(self,
                          node_id: str,
                          confidence: Optional[float] = None,
                          usage_count: Optional[int] = None):
        """Update node metrics and adjust visualization."""
        if node_id not in self.nodes:
            return
            
        node = self.nodes[node_id]
        
        if confidence is not None:
            node['metrics']['confidence'] = confidence
            # Adjust node size based on confidence
            node['visual']['size'] = 0.5 + confidence * 2
            
        if usage_count is not None:
            node['metrics']['usage_count'] = usage_count
            # Brighten color based on usage
            base_color = self.color_scheme[node['type']]
            node['visual']['color'] = self._adjust_color_brightness(
                base_color,
                min(1.0, usage_count / 100)  # Cap at 100 uses
            )
            
        node['metrics']['last_accessed'] = datetime.now()
        
    def highlight_active_path(self, node_ids: List[str]):
        """Highlight an active learning or reasoning path."""
        self.active_paths.clear()
        for i in range(len(node_ids) - 1):
            self.active_paths.add((node_ids[i], node_ids[i + 1]))
            
    def get_visualization_data(self) -> Dict[str, Any]:
        """Get current visualization state."""
        return {
            'nodes': [
                {
                    'id': node_id,
                    **node['visual'],
                    'metrics': node['metrics'],
                    'type': node['type']
                }
                for node_id, node in self.nodes.items()
            ],
            'connections': [
                {
                    'source': src,
                    'target': dst,
                    'active': (src, dst) in self.active_paths,
                    'width': 2 if (src, dst) in self.active_paths else 1
                }
                for src, dst in self.connections
            ],
            'color_scheme': self.color_scheme
        }
        
    def _calculate_position(self, parent_id: Optional[str]) -> Tuple[float, float]:
        """Calculate optimal node position in visualization."""
        if not parent_id:
            return (0.5, 0.1)  # Root position
            
        parent = self.nodes.get(parent_id)
        if not parent:
            return (0.5, 0.5)
            
        siblings = len(parent['children'])
        if siblings == 0:
            angle = 0
        else:
            angle = (len(parent['children']) * 30) % 360
            
        parent_x, parent_y = parent['visual']['position']
        radius = 0.2  # Distance from parent
        
        x = parent_x + radius * np.cos(np.radians(angle))
        y = parent_y + radius * np.sin(np.radians(angle))
        
        return (x, y)
        
    def _adjust_color_brightness(self, hex_color: str, factor: float) -> str:
        """Adjust color brightness based on usage."""
        # Convert hex to RGB
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        
        # Adjust brightness
        adjusted = tuple(min(255, int(c * (1 + factor))) for c in rgb)
        
        # Convert back to hex
        return f'#{adjusted[0]:02x}{adjusted[1]:02x}{adjusted[2]:02x}' 