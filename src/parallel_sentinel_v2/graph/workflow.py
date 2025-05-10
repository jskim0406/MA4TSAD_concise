"""
병렬 시계열 분석을 위한 워크플로우 정의 - 시각적 추론 기반 버전
"""

import os
import operator
from typing import Annotated, List, Dict, Any, Optional, TypedDict, Union, Sequence, Literal
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def setup_langsmith():
    """LangSmith 설정 (설정된 경우)"""
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "parallel-sentinel-v2")
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
    # 에이전트 간 전달되는 메시지들 (Annotated와 operator.add 적용)
    messages: Annotated[List[BaseMessage], operator.add]
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
    시각적 추론 기반 병렬 분석을 위한 시계열 워크플로우 생성
    
    Args:
        supervisor_agent: 전체 워크플로우를 조정하는 에이전트
        trend_analyzer_agent: 추세 시각적 분석 전문 에이전트
        seasonality_analyzer_agent: 계절성 시각적 분석 전문 에이전트 
        remainder_analyzer_agent: 잔차 시각적 분석 전문 에이전트
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

    # 워크플로우 시작점 정의: Start -> Supervisor
    workflow.add_edge(START, "supervisor")

    # 분석 결과를 취합하기 위한 Fan-In: 각 분석 에이전트 -> Supervisor
    # 이 엣지들은 분석 에이전트가 작업을 완료한 후 supervisor를 다시 호출하도록 합니다.
    workflow.add_edge("trend_analyzer", "supervisor")
    workflow.add_edge("seasonality_analyzer", "supervisor")
    workflow.add_edge("remainder_analyzer", "supervisor")

    # Supervisor 실행 후 다음 단계를 결정하는 라우팅 함수 정의
    def route_after_supervisor(state: TimeSeriesState) -> Union[List[str], Literal[END]]:
        """Supervisor 노드 실행 후 다음 단계를 결정합니다."""
        has_trend = len(state["trend_analysis"]) > 0
        has_seasonality = len(state["seasonality_analysis"]) > 0
        has_remainder = len(state["remainder_analysis"]) > 0
        # supervisor_agent가 모든 분석 결과를 바탕으로 final_analysis를 생성했는지 확인
        has_final = state["final_analysis"] is not None

        if has_final:
            # 최종 분석이 생성되었다면 워크플로우 종료
            print("Router: 최종 분석 완료됨. 워크플로우 종료.")
            return END
        else:
            # 분석이 완료되지 않았거나 최종 분석이 아직 생성되지 않았다면,
            # 모든 분석 에이전트를 병렬로 실행합니다.
            # supervisor 에이전트 자체가 초기 분해 로직을 처리합니다.
            print("Router: 분석 미완료 또는 최종 요약 미생성. 분석 에이전트로 팬아웃.")
            # 병렬 실행을 위해 노드 이름 리스트 반환
            return ["trend_analyzer", "seasonality_analyzer", "remainder_analyzer"]

    # Supervisor에서 나가는 조건부 엣지 설정
    # supervisor 노드 실행 후 route_after_supervisor 함수 결과에 따라 분기합니다.
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor
    )

    # 그래프 컴파일 및 반환
    print("시각적 추론 기반 병렬 워크플로우 컴파일 중...")
    try:
        compiled_workflow = workflow.compile()
        print("워크플로우 컴파일 완료.")
        return compiled_workflow
    except Exception as e:
        print(f"오류: 워크플로우 생성 실패: {e}") # 컴파일 오류 시 상세 메시지 출력
        raise

@traceable
def run_workflow(workflow, time_series_data: Union[List[float], Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    시각적 추론 기반 병렬 분석 워크플로우 실행
    
    Args:
        workflow: 컴파일된 워크플로우 그래프
        time_series_data: 분석할 시계열 데이터 또는 미리 생성된 상태 딕셔너리
        **kwargs: 워크플로우에 전달할 추가 키워드 인자
        
    Returns:
        Dict[str, Any]: 워크플로우의 최종 상태
    """
    config = kwargs.pop("config", {})
    
    # 양자화 정보가 있으면 메타데이터에 추가
    if "metadata" in config and "quantization" in config["metadata"]:
        quantize_info = config["metadata"]["quantization"]
        quantize_message = ""
        if quantize_info["applied"]:
            quantize_message = f"Time series data has been quantized using {quantize_info['method']} method with range {quantize_info['range']}. "
            quantize_message += f"Original length: {quantize_info['original_length']}, Quantized length: {quantize_info['quantized_length']}."
    else:
        quantize_message = "No quantization applied."

    initial_state = {
        "ts_data": time_series_data,
        "messages": [
            HumanMessage(content=f"Please analyze this time series data. {quantize_message} Sample: {time_series_data[:10]}... (Length: {len(time_series_data)})")
        ],
        "decomposition_data": {}, # 명시적 초기화
        "trend_analysis": [],
        "seasonality_analysis": [],
        "remainder_analysis": [],
        "final_analysis": None,
        "config": config,
    }

    print(f"시각적 추론 기반 병렬 워크플로우 실행 시작 - 시계열 데이터 길이: {len(initial_state['ts_data'])}")
    final_state = workflow.invoke(initial_state, **kwargs)
    print("시각적 추론 기반 병렬 워크플로우 실행 완료")
    return final_state
