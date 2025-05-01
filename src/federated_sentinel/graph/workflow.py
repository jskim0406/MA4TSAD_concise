"""
Workflow definitions for the Federated Sentinel multi-agent system.
"""
import os
import operator
from typing import Annotated, List, Dict, Any, Optional, Sequence, TypedDict, Union, Literal
from dotenv import load_dotenv, find_dotenv
# Load environment variables
load_dotenv(find_dotenv())

def setup_langsmith():
    """Configure LangSmith if available in config."""
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")
setup_langsmith()

from langsmith import traceable
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Type definition for the state
class TimeSeriesState(TypedDict):
    """State for the time series analysis workflow."""
    # Original time series data
    ts_data: List[float]
    # The messages being passed between agents
    messages: List[BaseMessage]
    # Statistical analysis results (appended by agents)
    statistical_analysis: Annotated[List[Dict[str, Any]], operator.add]
    # Pattern detection results (appended by agents)
    pattern_analysis: Annotated[List[Dict[str, Any]], operator.add]
    # Anomaly detection results (appended by agents)
    anomaly_analysis: Annotated[List[Dict[str, Any]], operator.add]
    # Final aggregated analysis
    final_analysis: Optional[Dict[str, Any]]
    # Config options
    config: Optional[Dict[str, Any]]
    # 다음에 실행할 에이전트를 추적하기 위한 필드
    next_agent: Optional[str]

# --- 라우팅 함수 정의 ---
def route_next_step(state: TimeSeriesState) -> Literal["analyst", "pattern_detector", "anomaly_detector", END]:
    """Determine the next node to execute based on the current state."""
    has_stat = len(state["statistical_analysis"]) > 0
    has_pattern = len(state["pattern_analysis"]) > 0
    has_anomaly = len(state["anomaly_analysis"]) > 0

    # 모든 분석이 완료되었는지 확인
    if has_stat and has_pattern and has_anomaly:
        print("Router: All analyses complete. Ending workflow.")
        # Supervisor가 이미 final_analysis를 생성했을 것이므로 END로 라우팅
        return END
    else:
        # 아직 완료되지 않은 분석 중 하나를 선택 (순서대로)
        # 실제 병렬 실행을 원한다면 이 로직 대신 다른 방식 고려 필요
        # (예: START에서 여러 노드로 바로 연결)
        # 여기서는 단순화를 위해 순차적 라우팅을 가정합니다.
        if not has_stat:
            print("Router: Routing to analyst.")
            return "analyst"
        elif not has_pattern:
            print("Router: Routing to pattern_detector.")
            return "pattern_detector"
        elif not has_anomaly:
            print("Router: Routing to anomaly_detector.")
            return "anomaly_detector"
        else:
            # 이 경우는 이론상 발생하지 않아야 함
            print("Router: Unexpected state. Ending workflow.")
            return END

@traceable
def create_workflow(
    supervisor_agent,
    analyst_agent,
    pattern_detector_agent,
    anomaly_detector_agent,
    tools: List[Any] = None
) -> StateGraph:
    """
    Create a multi-agent workflow for time series analysis using conditional edges.

    Args:
        supervisor_agent: Agent that aggregates results
        analyst_agent: Agent that performs statistical analysis
        pattern_detector_agent: Agent that detects patterns
        anomaly_detector_agent: Agent that detects anomalies
        tools: Optional list of tools available to agents

    Returns:
        A compiled StateGraph representing the workflow
    """
    # Create the graph
    workflow = StateGraph(TimeSeriesState)

    # Add nodes to the graph
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("pattern_detector", pattern_detector_agent)
    workflow.add_node("anomaly_detector", anomaly_detector_agent)

    # Define the workflow entry point: Start -> Supervisor (초기 상태 확인 및 메시지 전달)
    workflow.add_edge(START, "supervisor")

    # Add conditional edges from supervisor based on the routing function
    workflow.add_conditional_edges(
        "supervisor",  # 라우팅 결정을 내리는 출발 노드
        route_next_step, # 상태를 평가하고 다음 노드를 결정하는 함수
        {
            # route_next_step 함수의 반환값과 다음 노드 매핑
            "analyst": "analyst",
            "pattern_detector": "pattern_detector",
            "anomaly_detector": "anomaly_detector",
            END: END  # 종료 조건 명시
        }
    )

    # All specialized agents report back to the supervisor node
    # Supervisor는 결과를 집계하고 다음 라우팅 결정을 위해 상태를 업데이트
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("pattern_detector", "supervisor")
    workflow.add_edge("anomaly_detector", "supervisor")

    # Compile the graph
    # 컴파일 전에 모든 노드와 엣지가 올바르게 정의되었는지 확인
    print("Compiling workflow...")
    compiled_workflow = workflow.compile()
    print("Workflow compiled successfully.")
    return compiled_workflow


@traceable
def run_workflow(workflow, time_series_data: Union[List[float], Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    Run the multi-agent workflow on the provided time series data.
    
    Args:
        workflow: Compiled workflow graph
        time_series_data: Time series data to analyze, or a pre-populated state dict
        **kwargs: Additional keyword arguments to pass to the workflow
        
    Returns:
        The final state of the workflow
    """
    # Check if time_series_data is already a full state dict
    if isinstance(time_series_data, dict) and "messages" in time_series_data:
        initial_state = time_series_data
    else:
        # Initialize the state with just the time series data
        initial_state = {
            "ts_data": time_series_data,
            "messages": [
                HumanMessage(content=f"Please analyze this time series data: {time_series_data[:10]}... (length: {len(time_series_data)})")
            ],
            "statistical_analysis": [],
            "pattern_analysis": [],
            "anomaly_analysis": [],
            "final_analysis": None,
            "config": kwargs.pop("config", {})
        }
    
    # Run the workflow
    final_state = workflow.invoke(initial_state, **kwargs)
    
    return final_state


def get_graph_input_schema() -> Dict[str, Any]:
    """
    Return the schema for graph inputs, for LangGraph config.
    """
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["human", "ai"]},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                }
            },
            "ts_data": {
                "type": "array",
                "items": {"type": "number"}
            }
        },
        "required": ["messages"]
    }


def get_graph_output_schema() -> Dict[str, Any]:
    """
    Return the schema for graph outputs, for LangGraph config.
    """
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["human", "ai"]},
                        "content": {"type": "string"}
                    },
                    "required": ["role", "content"]
                }
            },
            "final_analysis": {
                "type": "object"
            }
        },
        "required": ["messages"]
    }