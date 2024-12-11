"""
Dashboard for monitoring task allocation and agent performance.
"""

import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from ..core.task_allocation import TaskAllocationManager, TaskPriority
from ..core.memory_manager import MemoryManager

class TaskAllocationDashboard:
    """Interactive dashboard for monitoring task allocation system."""
    
    def __init__(
        self,
        task_manager: TaskAllocationManager,
        memory_manager: MemoryManager,
        update_interval: int = 5000  # milliseconds
    ):
        self.task_manager = task_manager
        self.memory_manager = memory_manager
        self.update_interval = update_interval
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Set up dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Task Allocation Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Agent Workload Section
            html.Div([
                html.H3("Agent Workloads"),
                dcc.Graph(id='workload-chart'),
                dcc.Interval(
                    id='workload-interval',
                    interval=self.update_interval
                )
            ], style={'marginBottom': 40}),
            
            # Task Status Section
            html.Div([
                html.H3("Task Status"),
                dcc.Graph(id='task-status-chart'),
                dcc.Interval(
                    id='task-status-interval',
                    interval=self.update_interval
                )
            ], style={'marginBottom': 40}),
            
            # Performance Metrics Section
            html.Div([
                html.H3("Agent Performance"),
                dcc.Graph(id='performance-chart'),
                dcc.Interval(
                    id='performance-interval',
                    interval=self.update_interval
                )
            ], style={'marginBottom': 40}),
            
            # Memory Usage Section
            html.Div([
                html.H3("Memory Usage"),
                dcc.Graph(id='memory-chart'),
                dcc.Interval(
                    id='memory-interval',
                    interval=self.update_interval
                )
            ])
        ], style={'padding': 20})
        
    def _setup_callbacks(self):
        """Set up dashboard callbacks."""
        
        @self.app.callback(
            Output('workload-chart', 'figure'),
            Input('workload-interval', 'n_intervals')
        )
        def update_workload_chart(_):
            """Update agent workload chart."""
            workloads = self.task_manager.agent_workloads
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(workloads.keys()),
                    y=list(workloads.values()),
                    marker_color=['red' if w > 0.8 else 'blue' 
                                for w in workloads.values()]
                )
            ])
            
            fig.update_layout(
                title="Current Agent Workloads",
                xaxis_title="Agent ID",
                yaxis_title="Workload",
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        @self.app.callback(
            Output('task-status-chart', 'figure'),
            Input('task-status-interval', 'n_intervals')
        )
        def update_task_status_chart(_):
            """Update task status chart."""
            tasks = self.task_manager.tasks
            
            # Count tasks by priority and status
            status_counts = {
                priority: {
                    'Pending': 0,
                    'In Progress': 0,
                    'Completed': 0
                }
                for priority in TaskPriority
            }
            
            for task in tasks.values():
                priority = task.priority
                if task.progress >= 1.0:
                    status = 'Completed'
                elif task.assigned_agent:
                    status = 'In Progress'
                else:
                    status = 'Pending'
                status_counts[priority][status] += 1
                
            # Create stacked bar chart
            fig = go.Figure(data=[
                go.Bar(
                    name='Pending',
                    x=[p.name for p in TaskPriority],
                    y=[status_counts[p]['Pending'] for p in TaskPriority],
                    marker_color='red'
                ),
                go.Bar(
                    name='In Progress',
                    x=[p.name for p in TaskPriority],
                    y=[status_counts[p]['In Progress'] for p in TaskPriority],
                    marker_color='yellow'
                ),
                go.Bar(
                    name='Completed',
                    x=[p.name for p in TaskPriority],
                    y=[status_counts[p]['Completed'] for p in TaskPriority],
                    marker_color='green'
                )
            ])
            
            fig.update_layout(
                title="Task Status by Priority",
                xaxis_title="Priority",
                yaxis_title="Number of Tasks",
                barmode='stack'
            )
            
            return fig
            
        @self.app.callback(
            Output('performance-chart', 'figure'),
            Input('performance-interval', 'n_intervals')
        )
        def update_performance_chart(_):
            """Update agent performance chart."""
            performance_history = self.task_manager.performance_history
            
            fig = go.Figure()
            
            for agent_id, history in performance_history.items():
                if not history:
                    continue
                    
                # Calculate moving average
                window_size = min(10, len(history))
                moving_avg = pd.Series(history).rolling(window=window_size).mean()
                
                fig.add_trace(go.Scatter(
                    y=moving_avg,
                    mode='lines+markers',
                    name=agent_id
                ))
                
            fig.update_layout(
                title="Agent Performance History (Moving Average)",
                xaxis_title="Task Completion Index",
                yaxis_title="Performance Score",
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        @self.app.callback(
            Output('memory-chart', 'figure'),
            Input('memory-interval', 'n_intervals')
        )
        def update_memory_chart(_):
            """Update memory usage chart."""
            metrics = self.memory_manager.get_metrics()
            
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=("Memory Usage", "Cache Statistics"))
            
            # Memory pressure gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=metrics['memory_pressure'] * 100,
                    title={'text': "Memory Pressure (%)"},
                    gauge={'axis': {'range': [0, 100]},
                          'threshold': {
                              'line': {'color': "red", 'width': 4},
                              'thickness': 0.75,
                              'value': 80
                          }},
                    domain={'row': 0, 'column': 0}
                )
            )
            
            # Cache statistics
            fig.add_trace(
                go.Bar(
                    x=['Cache Size', 'Hit Rate', 'Miss Rate'],
                    y=[
                        metrics['cache_size'],
                        metrics['hit_rate'] * 100,
                        (1 - metrics['hit_rate']) * 100
                    ],
                    marker_color=['blue', 'green', 'red']
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                title_text="Memory and Cache Metrics"
            )
            
            return fig
            
    def run(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port) 