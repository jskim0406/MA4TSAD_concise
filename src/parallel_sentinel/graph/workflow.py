"""
병렬 시계열 분석을 위한 워크플로우 정의
"""

import os
import operator
from typing import Annotated, List, Dict, Any, Optional, TypedDict, Union, Sequence
from dotenv import load_dotenv, find_dotenv

# 환경 변수 로드
load_dotenv(find_dotenv())

def setup_langsmith():
    """LangSmith 설정 (설정된 경우)"""
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")

setup_langsmith()

from langsmith import traceable
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 시계열 상태 정의
class TimeSeriesState(TypedDict):
    """병렬 에이전트 워크플로우를 위한 시계열 분석 상태"""
    # 원본 시계열 데이터
    ts_data: List[float]
    # 에이전트 간 전달되는 메시지들
    messages: List[BaseMessage]
    # 시계열 분해 결과 (각 에이전트가 채워넣음)
    decomposition_data: Dict[str, Any]
    # 추세 분석 결과 (trend_analyzer가 추가)
    trend_analysis: Annotated[List[Dict[str, Any]], operator.add]
    # 계절성 분석 결과 (seasonality_analyzer가 추가)
    seasonality_analysis: Annotated[List[Dict[str, Any]], operator.add]
    # 잔차 분석 결과 (remainder_analyzer가 추가)
    remainder_analysis: Annotated[List[Dict[str, Any]], operator.add]
    # 최종 통합 분석
    final_analysis: Optional[Dict[str, Any]]
    # 설정 옵션
    config: Optional[Dict[str, Any]]


@traceable
def create_workflow(
    supervisor_agent,
    trend_analyzer_agent,
    seasonality_analyzer_agent,
    remainder_analyzer_agent,
    tools: List[Any] = None
) -> StateGraph:
    """
    병렬 분석을 위한 시계열 워크플로우 생성
    
    Args:
        supervisor_agent: 전체 워크플로우를 조정하는 에이전트
        trend_analyzer_agent: 추세 분석 전문 에이전트
        seasonality_analyzer_agent: 계절성 분석 전문 에이전트 
        remainder_analyzer_agent: 잔차 분석 전문 에이전트
        tools: 에이전트들이 사용 가능한 도구 목록
        
    Returns:
        StateGraph: 컴파일된 워크플로우 그래프
    """
    # 그래프 생성
    workflow = StateGraph(TimeSeriesState)
    
    # 그래프에 노드 추가
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("trend_analyzer", trend_analyzer_agent)
    workflow.add_node("seasonality_analyzer", seasonality_analyzer_agent)
    workflow.add_node("remainder_analyzer", remainder_analyzer_agent)
    
    # 워크플로우 시작점 정의: Start -> Supervisor (초기 상태 확인 및 데이터 분해)
    workflow.add_edge(START, "supervisor")
    
    # 병렬 처리를 위한 Fan-Out: Supervisor -> 각 분석 에이전트
    workflow.add_edge("supervisor", "trend_analyzer")
    workflow.add_edge("supervisor", "seasonality_analyzer")
    workflow.add_edge("supervisor", "remainder_analyzer")
    
    # 분석 결과를 취합하기 위한 Fan-In: 각 분석 에이전트 -> Supervisor
    workflow.add_edge("trend_analyzer", "supervisor")
    workflow.add_edge("seasonality_analyzer", "supervisor")
    workflow.add_edge("remainder_analyzer", "supervisor")
    
    # 워크플로우 종료 조건: Supervisor -> END
    # (모든 분석 결과가 있는 경우 종료)
    def should_end(state: TimeSeriesState) -> bool:
        """워크플로우 종료 조건 확인"""
        has_trend = len(state["trend_analysis"]) > 0
        has_seasonality = len(state["seasonality_analysis"]) > 0
        has_remainder = len(state["remainder_analysis"]) > 0
        has_final = state["final_analysis"] is not None
        
        # 모든 분석이 완료되고 최종 분석이 있는 경우 종료
        return has_trend and has_seasonality and has_remainder and has_final
    
    workflow.add_conditional_edges(
        "supervisor",
        should_end,
        {
            True: END,
            False: None  # 종료 조건이 만족되지 않으면 다른 에지로 이동
        }
    )
    
    # 그래프 컴파일 및 반환
    print("병렬 워크플로우 컴파일 중...")
    compiled_workflow = workflow.compile()
    print("워크플로우 컴파일 완료.")
    return compiled_workflow


@traceable
def run_workflow(workflow, time_series_data: Union[List[float], Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    병렬 분석 워크플로우 실행
    
    Args:
        workflow: 컴파일된 워크플로우 그래프
        time_series_data: 분석할 시계열 데이터 또는 미리 생성된 상태 딕셔너리
        **kwargs: 워크플로우에 전달할 추가 키워드 인자
        
    Returns:
        Dict[str, Any]: 워크플로우의 최종 상태
    """
    # 입력이 완전한 상태 딕셔너리인지 확인
    if isinstance(time_series_data, dict) and "messages" in time_series_data:
        initial_state = time_series_data
    else:
        # 시계열 데이터로 초기 상태 생성
        initial_state = {
            "ts_data": time_series_data,
            "messages": [
                HumanMessage(content=f"이 시계열 데이터를 분석해주세요: {time_series_data[:10]}... (길이: {len(time_series_data)})")
            ],
            "decomposition_data": {},
            "trend_analysis": [],
            "seasonality_analysis": [],
            "remainder_analysis": [],
            "final_analysis": None,
            "config": kwargs.pop("config", {})
        }
    
    # 워크플로우 실행
    print(f"병렬 워크플로우 실행 시작 - 시계열 데이터 길이: {len(initial_state['ts_data'])}")
    final_state = workflow.invoke(initial_state, **kwargs)
    print("병렬 워크플로우 실행 완료")
    
    return final_state