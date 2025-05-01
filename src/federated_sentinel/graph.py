"""
Graph module for LangGraph CLI integration.

This file serves as an entry point for the LangGraph CLI and server,
replicating the workflow defined in workflow.py using conditional edges.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Union

# --- .env 로딩: 스크립트 최상단 ---
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True)) # 현재 작업 디렉토리 또는 상위에서 .env 찾기

# --- 필요한 모듈 import ---
from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# workflow.py에서 상태 정의 및 라우팅 함수 import
from federated_sentinel.graph.workflow import TimeSeriesState, route_next_step
# 에이전트 생성 함수 import
from federated_sentinel.agents import (
    create_supervisor_agent,
    create_analyst_agent,
    create_pattern_detector_agent,
    create_anomaly_detector_agent
)
# 필요시 advanced detector import
# from federated_sentinel.agents.advanced_anomaly_detector import create_advanced_anomaly_detector

# 도구 import
from federated_sentinel.tools import (
    ts2img, ts2img_with_anomalies, ts2img_multi_view,
    basic_statistics, trend_analysis, seasonality_analysis,
    stationarity_test, anomaly_detection,
    get_math_calculator, rolling_window_stats
)
# LangSmith 설정 유틸리티 (환경 변수 로딩 후 실행)
from federated_sentinel.utils.tracing import setup_tracing


# --- Helper Functions ---

def _load_model():
    """Load the language model based on environment configuration."""
    # generator_fed.py 와 동일한 로직 사용 가능
    # 여기서는 간단하게 Vertex AI만 예시로 듭니다. 필요시 확장.
    from langchain_google_vertexai import ChatVertexAI
    model_name = os.getenv("GOOGLE_GEN_MODEL", "gemini-2.5-flash-preview-04-17") # 기본값 설정
    print(f"[graph.py] Loading model: {model_name}")
    return ChatVertexAI(model=model_name)

def _load_tools(llm):
    """Load the tools for the agents."""
    print("[graph.py] Loading tools...")
    return [
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

@traceable(name="create_federated_sentinel_graph") # 추적 이름 추가
def _create_graph(llm=None, tools=None):
    """
    Create the Federated Sentinel graph using conditional edges.
    (workflow.py의 create_workflow 로직과 동일하게 구성)

    Args:
        llm: Optional language model to use
        tools: Optional list of tools to use

    Returns:
        The compiled graph
    """
    print("[graph.py] Creating graph instance...")
    # Load model and tools if not provided
    if llm is None:
        llm = _load_model()
    if tools is None:
        tools = _load_tools(llm)

    # Create agents (supervisor는 라우팅 로직이 제거된 버전 사용)
    supervisor = create_supervisor_agent(llm)
    analyst = create_analyst_agent(llm, tools)
    pattern_detector = create_pattern_detector_agent(llm, tools)
    print("[graph.py] Using standard anomaly detector.")
    anomaly_detector = create_anomaly_detector_agent(llm, tools)

    # Create the graph using StateGraph
    workflow = StateGraph(TimeSeriesState)

    # Add nodes to the graph
    print("[graph.py] Adding nodes...")
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("analyst", analyst)
    workflow.add_node("pattern_detector", pattern_detector)
    workflow.add_node("anomaly_detector", anomaly_detector)

    # Define the workflow entry point
    workflow.add_edge(START, "supervisor")

    # Add conditional edges from supervisor using the imported route_next_step function
    print("[graph.py] Adding conditional edges...")
    workflow.add_conditional_edges(
        "supervisor",
        route_next_step, # workflow.py에서 import한 라우팅 함수
        {
            "analyst": "analyst",
            "pattern_detector": "pattern_detector",
            "anomaly_detector": "anomaly_detector",
            END: END
        }
    )

    # All specialized agents report back to the supervisor
    print("[graph.py] Adding feedback edges...")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("pattern_detector", "supervisor")
    workflow.add_edge("anomaly_detector", "supervisor")

    # Compile the graph
    print("[graph.py] Compiling graph...")
    compiled_graph = workflow.compile()
    print("[graph.py] Graph compiled successfully.")
    return compiled_graph


def _convert_messages(messages):
    """Convert message dict format to LangChain messages if needed."""
    converted = []
    if not messages:
        return converted
    for msg in messages:
        if isinstance(msg, BaseMessage):
            converted.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content")
            name = msg.get("name") # 이름 속성 처리
            if role == "human":
                converted.append(HumanMessage(content=content))
            elif role == "ai":
                # AIMessage는 ToolCall 객체 등을 포함할 수 있으므로 단순 변환 주의
                # 서버 입력 형식에 따라 AIMessage 대신 dict 유지 필요할 수도 있음
                # 우선 기본 AIMessage로 변환 시도
                try:
                    # 도구 호출 등 복잡한 구조는 일단 content만 가져옴
                    converted.append(AIMessage(content=content, name=name if name else None))
                except Exception as e:
                    print(f"Warning: Could not convert AI message dict completely: {e}")
                    converted.append(AIMessage(content=str(msg))) # 실패 시 문자열로 변환
            elif role == "tool":
                 tool_call_id = msg.get("tool_call_id")
                 if tool_call_id:
                      converted.append(ToolMessage(content=content, tool_call_id=tool_call_id))
                 else:
                      print(f"Warning: Tool message dict missing tool_call_id: {msg}")
                      converted.append(ToolMessage(content=content, tool_call_id="unknown")) # ID 없으면 임의 값
            else:
                 print(f"Warning: Unknown message role '{role}' in dict: {msg}")
                 converted.append(HumanMessage(content=str(msg))) # 알 수 없는 역할은 Human으로 처리
        else:
             print(f"Warning: Unknown message type '{type(msg)}': {msg}")
             converted.append(HumanMessage(content=str(msg))) # 알 수 없는 타입은 문자열로 변환

    return converted

# LangGraph 서버 진입점 함수
@traceable(name="process_federated_sentinel_request") # 추적 이름 추가
def process_ts_data(input_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process time series data through the Federated Sentinel workflow.
    This function is used as the entry point for LangGraph server.

    Args:
        input_data: Input data containing messages and optionally time series data.
                    Expected format based on LangGraph server interaction.
        config: Configuration options passed from the server/client.

    Returns:
        Dict containing processed results in the format expected by the server.
    """
    print(f"[graph.py] Received input_data: {str(input_data)[:500]}...") # 입력 로깅
    print(f"[graph.py] Received config: {config}") # 설정 로깅

    # Setup LangSmith tracing based on environment variables (loaded at top)
    # setup_tracing()은 필요 시 호출 (예: 프로젝트 이름 동적 설정)
    # setup_tracing(os.getenv("LANGSMITH_PROJECT", "federated-sentinel-server")) # 서버용 프로젝트 이름 설정 가능

    # Create the graph instance (매번 새로 생성하거나, 전역 인스턴스 사용 가능)
    # 여기서는 요청마다 새로 생성
    graph_instance = _create_graph()

    # --- Initial State Preparation ---
    # 서버 입력 형식('input', 'config')에 맞춰 상태 준비
    # LangGraph 서버는 일반적으로 'input' 키에 상태를 전달
    state_input = input_data.get("input", {})
    if not isinstance(state_input, dict):
         print(f"Warning: Received input is not a dictionary, attempting to use as messages. Input: {state_input}")
         # 입력이 dict가 아닐 경우, 메시지로 간주 시도 (유연성 위해)
         state_input = {"messages": [HumanMessage(content=str(state_input))]}

    # 메시지 변환 (서버가 dict 로 전달하는 경우)
    messages = _convert_messages(state_input.get("messages", []))

    # 시계열 데이터 추출 (input 또는 config 에서 찾기)
    ts_data = state_input.get("ts_data")
    if ts_data is None and config:
         ts_data = config.get("ts_data") # config에서도 찾아보기

    # 기본 샘플 데이터 사용 (ts_data가 없는 경우)
    if not ts_data:
        print("[graph.py] No time series data found in input or config, using default sample.")
        # generator_fed.py의 샘플 데이터 사용
        ts_data = [753, 703, 500, 1028, 554, 1041, 603, 676, 645, 599, 502, 463, 483, 475, 526, 496, 619, 418, 895, 498, 727, 1018, 756, 763, 600, 668, 816, 490, 721, 644, 642, 347, 638, 506, 605, 578, 528, 560, 626, 649, 485, 257, 486, 649, 919, 702, 874, 614, 614, 469, 699, 430, 553, 469, 496, 934, 518, 597, 696, 602, 564, 509, 670, 775, 611, 874, 794, 613, 478, 657, 679, 644, 557, 567, 490, 685, 662, 511, 618, 606, 692, 308, 657, 583, 675, 736, 766, 811, 1042, 842, 547, 402, 1032, 598, 690, 643, 515, 621, 490, 550, 530, 500, 602, 679, 577, 573, 592, 644, 869, 811, 811, 766, 1042, 728, 527, 636, 663, 710, 297, 564, 772, 720, 687, 637, 491, 1041, 543, 518, 998, 342, 196, 702, 976, 702, 914, 891, 658, 636, 708, 1028, 743, 837, 517, 730, 607, 529, 568, 461, 598, 654, 726, 887, 356, 1042, 702, 530, 735, 691, 539, 657, 595, 509, 660, 628, 588, 631, 359, 442, 677, 619, 774, 668, 598, 623, 595, 825, 356, 725, 841, 517, 566, 516, 524, 925, 545, 665, 537, 425, 505, 559, 484, 520, 572, 663, 758, 920, 884, 818, 748, 171, 595, 464, 441, 622, 733, 543, 591, 582, 364, 562, 522, 566, 674, 633, 374, 542, 942, 876, 1006, 844, 716, 468, 555, 589, 698, 419, 525, 614, 436, 613, 691, 650, 594, 603, 596, 240, 839, 942, 702, 1023, 935, 938, 567, 790, 607, 758, 617, 577, 619, 620, 951, 752, 660, 493, 664, 545, 643, 613, 427, 999, 1024, 869, 614, 976, 869, 711, 891, 664, 783, 756, 793, 621, 833, 810, 729, 607, 655, 662, 930, 747, 674, 600, 544, 775, 695, 711, 542, 702, 944, 845, 652, 915, 710, 703, 884, 769, 701, 746, 765, 771, 751, 659, 674, 730, 702, 732, 1042, 869, 862, 1042, 942, 614, 570, 639, 685, 614, 599, 428, 635, 762, 632, 575, 810, 654, 659, 758, 538, 640, 600, 580, 914, 881, 811, 1031, 807, 614, 886, 626, 642, 668, 742, 739, 721, 502, 606, 644, 812, 582, 671, 715, 640, 653, 942, 784, 784, 631, 702, 817, 654, 760, 617, 514, 683, 667, 542, 730, 573, 681, 594, 609, 502, 599, 865, 931, 838, 675, 804, 627, 646, 757, 689, 736, 996, 761, 710, 595, 560, 657, 664, 705, 646, 671, 668, 666, 702, 708, 645, 786, 647, 781]
        if not messages: # 메시지가 비어있으면 초기 메시지 추가
             messages = [HumanMessage(content=f"Analyzing default sample time series (length: {len(ts_data)}): {str(ts_data[:10])}...")]

    # TimeSeriesState 구조에 맞춰 초기 상태 딕셔너리 생성
    initial_state = {
        "ts_data": ts_data,
        "messages": messages,
        "statistical_analysis": state_input.get("statistical_analysis", []), # 기존 상태 유지 시도
        "pattern_analysis": state_input.get("pattern_analysis", []),
        "anomaly_analysis": state_input.get("anomaly_analysis", []),
        "final_analysis": state_input.get("final_analysis"),
        "config": state_input.get("config", {}), # 입력에서 config 가져오기
        "next_agent": state_input.get("next_agent") # 기존 상태 유지 시도
    }
    # 입력받은 config를 state의 config와 병합 (입력 config 우선)
    if config:
        initial_state["config"].update(config.get("configurable", {}))

    # --- Run the graph ---
    print("[graph.py] Invoking graph instance...")
    # LangGraph 서버는 일반적으로 config를 invoke의 두 번째 인자로 전달
    # config에서 thread_id 등을 추출하여 사용
    thread_id = initial_state["config"].get("thread_id", "federated-sentinel-server-default")
    invoke_config = {"configurable": {"thread_id": thread_id}}
    print(f"[graph.py] Using config for invocation: {invoke_config}")

    result = graph_instance.invoke(initial_state, invoke_config)
    print("[graph.py] Graph invocation complete.")

    # --- Format Output ---
    # 서버가 기대하는 형식으로 결과 반환 (일반적으로 'output' 키 아래에 상태 포함)
    # 메시지를 다시 dict 형태로 변환 (선택 사항, 서버 API에 따라 다름)
    output_messages = []
    for msg in result.get("messages", []):
         if hasattr(msg, "content"):
             role = "human" if msg.type == "human" else "ai"
             msg_dict = {"role": role, "content": msg.content}
             if hasattr(msg, "name") and msg.name:
                 msg_dict["name"] = msg.name
             # ToolMessage 등의 다른 타입 처리 필요 시 추가
             output_messages.append(msg_dict)

    # 최종 반환값 구성
    output_result = {
         "output": { # LangGraph 서버는 보통 'output' 키를 기대
              "messages": output_messages,
              "final_analysis": result.get("final_analysis"),
              # 필요 시 다른 상태 값 포함
              "statistical_analysis": result.get("statistical_analysis"),
              "pattern_analysis": result.get("pattern_analysis"),
              "anomaly_analysis": result.get("anomaly_analysis"),
         }
    }
    print(f"[graph.py] Returning output: {str(output_result)[:500]}...")
    return output_result


# --- Export the graph instance for LangGraph CLI ---
# _create_graph 함수를 호출하여 최종 'graph' 객체 생성
graph = _create_graph()
print("[graph.py] Compiled graph instance exported as 'graph'.")