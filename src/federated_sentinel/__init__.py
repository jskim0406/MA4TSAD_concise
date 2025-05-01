"""
Federated Sentinel - Multi-agent LLM-based Time Series Anomaly Detection library

This library implements a federated approach to time series anomaly detection using
multiple specialized LLM agents working in parallel.
"""

__version__ = "0.1.0"

from federated_sentinel.graph.workflow import create_workflow, run_workflow

__all__ = ["create_workflow", "run_workflow"]