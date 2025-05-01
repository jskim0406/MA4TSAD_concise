"""
Graph components for the Federated Sentinel library.
"""

import os
from typing import Dict, Any, List

from langsmith import traceable
from federated_sentinel.graph.workflow import create_workflow, run_workflow, TimeSeriesState

__all__ = [
    "create_workflow",
    "run_workflow",
    "TimeSeriesState",
    "get_graph"
]

@traceable
def get_graph(llm=None, tools=None):
    """
    Create and return the main Federated Sentinel graph for LangGraph integration.
    This function is used by the LangGraph CLI to construct the workflow.
    
    Args:
        llm: Optional language model to use for the agents
        tools: Optional list of tools to provide to the agents
        
    Returns:
        A compiled StateGraph for time series analysis
    """
    from langchain_google_vertexai import ChatVertexAI
    from federated_sentinel.agents import (
        create_supervisor_agent,
        create_analyst_agent,
        create_pattern_detector_agent,
        create_anomaly_detector_agent
    )
    from federated_sentinel.tools import (
        ts2img, ts2img_with_anomalies, ts2img_multi_view,
        basic_statistics, trend_analysis, seasonality_analysis,
        stationarity_test, anomaly_detection,
        get_math_calculator, rolling_window_stats
    )
    
    # Initialize LLM if not provided
    if llm is None:
        model_name = os.getenv("GOOGLE_GEN_MODEL", "gemini-1.5-flash")
        llm = ChatVertexAI(model=model_name)
    
    # Set up tools if not provided
    if tools is None:
        tools = [
            ts2img,
            ts2img_with_anomalies,
            ts2img_multi_view,
            basic_statistics,
            trend_analysis,
            seasonality_analysis,
            stationarity_test,
            anomaly_detection,
            rolling_window_stats,
            get_math_calculator(llm)
        ]
    
    # Create agents
    supervisor = create_supervisor_agent(llm)
    analyst = create_analyst_agent(llm, tools)
    pattern_detector = create_pattern_detector_agent(llm, tools)
    anomaly_detector = create_anomaly_detector_agent(llm, tools)
    
    # Create and return the workflow
    return create_workflow(
        supervisor_agent=supervisor,
        analyst_agent=analyst,
        pattern_detector_agent=pattern_detector,
        anomaly_detector_agent=anomaly_detector,
        tools=tools
    )

# Create the graph instance for export
graph = get_graph()