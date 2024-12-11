from typing import Dict, Any, List
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import json
import os
from loguru import logger

class SystemVisualizer:
    """Visualizes system metrics and agent interactions."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        self.metrics_history = []
        self.interaction_graph = nx.DiGraph()
        os.makedirs(output_dir, exist_ok=True)
        
    def update_metrics(self, system_state: Dict[str, Any]):
        """Update metrics history with new system state."""
        self.metrics_history.append({
            "timestamp": datetime.now().isoformat(),
            "state": system_state
        })
        
    def update_interaction_graph(self, source: str, target: str, interaction_type: str):
        """Update the agent interaction graph."""
        if not self.interaction_graph.has_edge(source, target):
            self.interaction_graph.add_edge(source, target, interactions=[])
        self.interaction_graph[source][target]["interactions"].append({
            "type": interaction_type,
            "timestamp": datetime.now().isoformat()
        })
        
    def plot_performance_metrics(self):
        """Plot performance metrics over time."""
        try:
            if not self.metrics_history:
                return
                
            plt.figure(figsize=(12, 8))
            timestamps = [m["timestamp"] for m in self.metrics_history]
            
            # Plot response times
            response_times = []
            error_rates = []
            for metrics in self.metrics_history:
                agents = metrics["state"].get("agents", {})
                avg_response = 0
                avg_error = 0
                count = 0
                
                for agent_metrics in agents.values():
                    perf = agent_metrics.get("performance", {})
                    if perf:
                        avg_response += perf.get("avg_response_time", 0)
                        avg_error += perf.get("error_rate", 0)
                        count += 1
                
                if count > 0:
                    response_times.append(avg_response / count)
                    error_rates.append(avg_error / count)
                else:
                    response_times.append(0)
                    error_rates.append(0)
            
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, response_times, label="Avg Response Time")
            plt.title("System Performance Metrics")
            plt.xlabel("Time")
            plt.ylabel("Response Time (s)")
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, error_rates, label="Error Rate", color="red")
            plt.xlabel("Time")
            plt.ylabel("Error Rate")
            plt.legend()
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "performance_metrics.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting performance metrics: {str(e)}")
            
    def plot_agent_interactions(self):
        """Plot agent interaction graph."""
        try:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(self.interaction_graph)
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.interaction_graph, pos,
                node_color="lightblue",
                node_size=1000
            )
            
            # Draw edges
            edge_colors = []
            edge_widths = []
            for _, _, data in self.interaction_graph.edges(data=True):
                interactions = data["interactions"]
                edge_colors.append("blue")
                edge_widths.append(len(interactions))
                
            nx.draw_networkx_edges(
                self.interaction_graph, pos,
                edge_color=edge_colors,
                width=edge_widths
            )
            
            # Draw labels
            nx.draw_networkx_labels(self.interaction_graph, pos)
            
            plt.title("Agent Interaction Graph")
            plt.savefig(os.path.join(self.output_dir, "agent_interactions.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting agent interactions: {str(e)}")
            
    def plot_capability_distribution(self, agent_pools: Dict[str, List[Any]]):
        """Plot capability distribution across agent pools."""
        try:
            capabilities = {}
            for pool_name, agents in agent_pools.items():
                capabilities[pool_name] = {}
                for agent in agents:
                    for cap in agent.get_capabilities():
                        if cap.name not in capabilities[pool_name]:
                            capabilities[pool_name][cap.name] = 0
                        capabilities[pool_name][cap.name] += 1
            
            plt.figure(figsize=(12, 6))
            pools = list(capabilities.keys())
            cap_names = set()
            for pool_caps in capabilities.values():
                cap_names.update(pool_caps.keys())
            
            x = range(len(pools))
            bottom = [0] * len(pools)
            
            for cap in cap_names:
                values = [capabilities[pool].get(cap, 0) for pool in pools]
                plt.bar(x, values, bottom=bottom, label=cap)
                bottom = [b + v for b, v in zip(bottom, values)]
            
            plt.xlabel("Agent Pools")
            plt.ylabel("Number of Capabilities")
            plt.title("Capability Distribution Across Agent Pools")
            plt.xticks(x, pools, rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "capability_distribution.png"))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting capability distribution: {str(e)}")
            
    def export_metrics(self):
        """Export metrics history to JSON."""
        try:
            output_file = os.path.join(self.output_dir, "metrics_history.json")
            with open(output_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            
    def generate_report(self, system_state: Dict[str, Any], agent_pools: Dict[str, List[Any]]):
        """Generate a complete system report with all visualizations."""
        try:
            # Update metrics
            self.update_metrics(system_state)
            
            # Generate all plots
            self.plot_performance_metrics()
            self.plot_agent_interactions()
            self.plot_capability_distribution(agent_pools)
            
            # Export metrics
            self.export_metrics()
            
            # Generate HTML report
            report_file = os.path.join(self.output_dir, "system_report.html")
            with open(report_file, "w") as f:
                f.write("""
                <html>
                <head>
                    <title>System Performance Report</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        .section { margin-bottom: 30px; }
                        img { max-width: 100%; }
                    </style>
                </head>
                <body>
                    <h1>System Performance Report</h1>
                    <div class="section">
                        <h2>Performance Metrics</h2>
                        <img src="performance_metrics.png">
                    </div>
                    <div class="section">
                        <h2>Agent Interactions</h2>
                        <img src="agent_interactions.png">
                    </div>
                    <div class="section">
                        <h2>Capability Distribution</h2>
                        <img src="capability_distribution.png">
                    </div>
                </body>
                </html>
                """)
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            
    def clear_history(self):
        """Clear metrics history and reset graphs."""
        self.metrics_history = []
        self.interaction_graph.clear() 