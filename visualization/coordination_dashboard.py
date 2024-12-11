"""
Dashboard for visualizing agent coordination metrics and collaboration patterns.
"""

import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import networkx as nx
import psutil
from pathlib import Path

from ..core.agent_coordination import AgentCoordinator, TaskPriority, CoordinationStrategy
from ..core.self_analysis import SelfAnalysis

class CoordinationDashboard:
    """Interactive dashboard for monitoring agent coordination."""
    
    def __init__(self, coordinator: AgentCoordinator, analyzer: SelfAnalysis, 
                 update_interval: int = 5000):
        """Initialize dashboard with coordinator and analyzer."""
        self.app = dash.Dash(__name__, 
                           assets_folder=str(Path(__file__).parent / 'static'))
        self.coordinator = coordinator
        self.analyzer = analyzer
        self.update_interval = update_interval
        self.setup_layout()
        
    def setup_layout(self):
        """Setup dashboard layout with multiple visualization components."""
        self.app.layout = html.Div([
            # ... (rest of the layout remains the same)
        ])
        
        self.setup_callbacks()
        
    def setup_callbacks(self):
        """Setup dashboard callback functions."""
        @self.app.callback(
            [Output('realtime-metrics', 'figure'),
             Output('collaboration-network', 'figure'),
             Output('performance-history', 'figure'),
             Output('task-distribution', 'figure'),
             Output('health-indicators', 'children')],
            [Input('metric-update', 'n_intervals'),
             Input('performance-metric-selector', 'value')]
        )
        def update_dashboard(n_intervals: int, selected_metric: str):
            """Update all dashboard components."""
            return (
                self.create_realtime_metrics(),
                self.create_collaboration_network(),
                self.create_performance_history(selected_metric),
                self.create_task_distribution(),
                self.create_health_indicators()
            )
            
    def get_queue_size(self) -> int:
        """Get current message queue size."""
        return len(self.coordinator.message_queue)
        
    def get_active_collaborations(self) -> Dict[str, int]:
        """Get current active collaborations."""
        collaborations = {}
        for group in self.coordinator.collaboration_groups:
            task_type = group.task_type
            collaborations[task_type] = collaborations.get(task_type, 0) + 1
        return collaborations
        
    def get_error_rate_history(self) -> Dict[str, List]:
        """Get error rate history."""
        metrics = self.analyzer.get_metric_summary()
        timestamps = [
            datetime.now() - timedelta(minutes=i) 
            for i in range(len(self.analyzer.metric_collector.metrics_history))
        ]
        return {
            'timestamp': timestamps,
            'error_rate': [m.error_rate for m in self.analyzer.metric_collector.metrics_history]
        }
        
    def get_response_times(self) -> List[float]:
        """Get recent response times."""
        return [
            m.response_time 
            for m in self.analyzer.metric_collector.metrics_history
        ]
        
    def get_agent_data(self) -> Dict[str, Dict]:
        """Get agent performance data."""
        agent_data = {}
        for name, agent in self.coordinator.agents.items():
            capabilities = self.coordinator.agent_capabilities.get(name)
            performance = np.mean(self.coordinator.performance_history.get(name, [0.8]))
            agent_data[name] = {
                'performance': performance,
                'expertise': capabilities.expertise_level if capabilities else 0.5
            }
        return agent_data
        
    def get_collaboration_data(self) -> List[Dict]:
        """Get collaboration relationship data."""
        collaborations = []
        for group in self.coordinator.collaboration_groups:
            # Add primary to supporting connections
            for supporting in group.supporting_agents:
                collaborations.append({
                    'primary': group.primary_agent,
                    'secondary': supporting,
                    'strength': np.mean(group.performance_history) if group.performance_history else 0.7
                })
        return collaborations
        
    def get_performance_history(self, metric: str) -> Dict[str, Dict]:
        """Get performance history for specified metric."""
        history = {}
        metrics_history = self.analyzer.metric_collector.metrics_history
        timestamps = [
            datetime.now() - timedelta(minutes=i) 
            for i in range(len(metrics_history))
        ]
        
        for agent in self.coordinator.agents:
            metric_values = []
            for m in metrics_history:
                if metric == 'response_time':
                    metric_values.append(m.response_time)
                elif metric == 'throughput':
                    metric_values.append(m.throughput)
                elif metric == 'error_rate':
                    metric_values.append(m.error_rate)
                elif metric == 'resource_usage':
                    metric_values.append(m.resource_usage)
                    
            history[agent] = {
                'timestamp': timestamps,
                metric: metric_values
            }
            
        return history
        
    def get_task_distribution(self) -> Dict[str, List]:
        """Get task distribution data."""
        task_counts = {}
        for agent in self.coordinator.agents.values():
            capabilities = self.coordinator.agent_capabilities.get(agent.name)
            if capabilities:
                for task in capabilities.supported_tasks:
                    parts = task.split('_')
                    category = parts[0].title()
                    subcategory = '_'.join(parts[1:]).title()
                    
                    if category not in task_counts:
                        task_counts[category] = {'total': 0, 'subtasks': {}}
                    if subcategory not in task_counts[category]['subtasks']:
                        task_counts[category]['subtasks'][subcategory] = 0
                        
                    task_counts[category]['total'] += 1
                    task_counts[category]['subtasks'][subcategory] += 1
                    
        # Convert to sunburst format
        ids = ['Tasks']
        labels = ['All Tasks']
        parents = ['']
        values = [sum(cat['total'] for cat in task_counts.values())]
        
        for category, data in task_counts.items():
            ids.append(category)
            labels.append(category)
            parents.append('Tasks')
            values.append(data['total'])
            
            for subcategory, count in data['subtasks'].items():
                ids.append(f"{category}_{subcategory}")
                labels.append(subcategory)
                parents.append(category)
                values.append(count)
                
        return {
            'ids': ids,
            'labels': labels,
            'parents': parents,
            'values': values
        }
        
    def get_system_health(self) -> Dict[str, Dict]:
        """Get system health metrics."""
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        
        return {
            'CPU Usage': {
                'value': cpu_percent,
                'status': 'good' if cpu_percent < 70 else 'warning' if cpu_percent < 90 else 'critical',
                'description': 'System CPU utilization'
            },
            'Memory Usage': {
                'value': memory_info.rss / (1024 * 1024),  # Convert to MB
                'status': 'good' if memory_info.rss < 500*1024*1024 else 'warning',
                'description': 'System memory utilization'
            },
            'Queue Health': {
                'value': 100 * (1 - len(self.coordinator.message_queue)/1000),  # Assume 1000 is max healthy size
                'status': 'good' if len(self.coordinator.message_queue) < 500 else 'warning',
                'description': 'Message queue health'
            },
            'Error Rate': {
                'value': self.coordinator.metrics['error_rate'] * 100,
                'status': 'good' if self.coordinator.metrics['error_rate'] < 0.05 else 'critical',
                'description': 'System error rate'
            }
        }
        
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port) 