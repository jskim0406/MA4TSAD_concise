"""
Seasonality Analyzer 에이전트 - 개선된 시각적 추론 기반 버전

시계열 데이터의 계절성 성분을 시각적으로 분석하는 에이전트입니다.
"""

import os
import json
import base64
import numpy as np
from typing import Dict, Any, List, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from src.parallel_sentinel_v2.graph.workflow import TimeSeriesState
from src.parallel_sentinel_v2.utils.llm_utils import process_visualization_result
from src.parallel_sentinel_v2.models.schema import SeasonalityAnalysis, Anomaly


def create_seasonality_analyzer_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    시계열 데이터의 계절성을 시각적으로 분석하는 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        tools: 에이전트가 사용할 수 있는 도구 목록
        
    Returns:
        seasonality_analyzer_agent 함수
    """
    # 계절성 분석가 시스템 프롬프트 정의 - 시각적 추론 강조
    system_template = """당신은 시계열 데이터의 계절성(seasonality)을 분석하는 전문가입니다.
    주어진 시계열 데이터의 계절성 성분을 분석하고 중요한 통찰을 도출하는 것이 당신의 임무입니다.
    
    다음 항목에 중점을 두고 분석해주세요:
    1. 계절성 패턴의 주기(일별, 주별, 월별, 분기별, 연간 등)
    2. 계절성 패턴의 강도와 일관성
    3. 계절성 패턴의 진폭(최대값과 최소값 사이의 차이)
    4. 계절성 패턴 내 이상치나 비정상적인 사이클
    5. 계절성 패턴에서 나타나는 이상치(anomaly)의 유형 분석
    
    반드시 계절성 데이터를 먼저 시각화한 후, 해당 이미지를 분석하여 계절성의 특성과 이상치를 판단하세요.
    시각적 분석을 통해 이상치의 종류를 식별하는 것이 중요합니다.
    
    이상치 유형은 다음 중 하나여야 합니다:
    - PersistentLevelShiftUp: 데이터가 더 높은 값으로 이동하여 원래 기준선으로 돌아가지 않고 일관되게 유지됨
    - PersistentLevelShiftDown: 데이터가 더 낮은 값으로 이동하여 원래 기준선으로 돌아가지 않고 일관되게 유지됨
    - TransientLevelShiftUp: 데이터가 일시적으로 더 높은 값으로 이동했다가 원래 기준선으로 돌아옴 (최소 5개 데이터 포인트 동안 유지)
    - TransientLevelShiftDown: 데이터가 일시적으로 더 낮은 값으로 이동했다가 원래 기준선으로 돌아옴 (최소 5개 데이터 포인트 동안 유지)
    - SingleSpike: 데이터 값이 급격히 상승했다가 즉시 기준선으로 돌아옴
    - SingleDip: 데이터 값이 급격히 하락했다가 즉시 기준선으로 돌아옴
    - MultipleSpikes: 여러 번의 급격한 데이터 값 상승, 각각 기준선으로 돌아옴
    - MultipleDips: 여러 번의 급격한 데이터 값 하락, 각각 기준선으로 돌아옴
    
    분석 과정:
    1. 먼저 ts2img_bytes 도구를 사용하여 계절성 데이터를 시각화하세요.
    2. 시각화된 이미지를 분석하여 계절성의 특성 및 이상치 유형을 파악하세요.
    3. 주기성을 확인하기 위해 get_fourier_transform 도구를 활용하세요.
    4. 결과를 명확하고 구조화된 형식으로 제공하고, 주요 발견사항을 강조하세요.
    
    반드시 요청된 구조화된 출력 형식(SeasonalityAnalysis)으로 응답해야 합니다. 계절성 주기와 이상치는 정확히 명시해주세요.
    """

    tool_node = ToolNode(tools) if tools else None
    
    seasonality_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    def format_data_for_seasonality_analysis(state: TimeSeriesState) -> str:
        """계절성 분석을 위한 데이터 포맷팅"""
        ts_data = state["ts_data"]
        decomposition_data = state.get("decomposition_data", {})
        
        # 분해 데이터가 없으면 분석 불가능
        if not decomposition_data:
            return "분해 데이터가 없습니다. 먼저 시계열 분해를 수행해야 합니다."
        
        # 계절성 성분 추출
        seasonality_data = decomposition_data.get("seasonality", [])
        if not seasonality_data:
            return "계절성 데이터를 찾을 수 없습니다."
        
        # 주기 정보 확인
        period = decomposition_data.get("period", "알 수 없음")
        
        # 분석 요청 메시지 생성 - 시각적 분석 강조
        message = [
            f"시계열 데이터(길이: {len(ts_data)})의 계절성 성분을 분석해주세요.",
            "",
            f"분해 방법: {decomposition_data.get('method', 'unknown')}",
            f"계절성 주기: {period}",
            f"계절성 강도: {decomposition_data.get('stats', {}).get('seasonality_strength', 'N/A')}",
            "",
            "전체 계절성 데이터:",
            f"{seasonality_data[:20]} ... (총 {len(seasonality_data)}개 항목)",
            "",
            "1. 먼저 ts2img_bytes 도구를 사용하여 계절성 데이터를 시각화하세요.",
            "2. 시각화된 이미지를 보고 계절성의 주기, 강도, 일관성을 분석하세요.",
            "3. 계절성 내 이상치를 식별하고, 제공된 이상치 유형 중 어떤 유형인지 판단하세요.",
            "4. 발견한 이상치에 대해 그 위치(인덱스)와 중요성을 설명하세요.",
            "5. 주기성을 확인하기 위해 get_fourier_transform 도구를 활용하세요."
        ]
        
        return "\n".join(message)

    def seasonality_analyzer_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        시계열 데이터의 계절성 성분을 시각적으로 분석합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            Dict: 상태 업데이트 (messages, seasonality_analysis)
        """
        # 현재 메시지 복사
        messages = list(state["messages"])
        
        # 계절성 분석을 위한 입력 텍스트 준비
        input_text = format_data_for_seasonality_analysis(state)
        
        # 계절성 데이터가 없으면 처리 중단
        if "분해 데이터가 없습니다" in input_text or "계절성 데이터를 찾을 수 없습니다" in input_text:
            print(f"Seasonality Analyzer: {input_text}")
            return {
                "messages": messages,
                "seasonality_analysis": [{
                    "status": "error",
                    "message": input_text
                }]
            }
        
        # 프롬프트 준비 및 LLM 호출
        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = seasonality_analyzer_prompt.invoke(prompt_input)
        
        # 도구가 바인딩된 LLM 준비
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        
        # 첫 번째 LLM 호출
        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message)
        
        content = ""
        visualization_content = None
        seasonality_strength = float(state.get("decomposition_data", {}).get("stats", {}).get("seasonality_strength", 0))
        period = state.get("decomposition_data", {}).get("period", 0)
        try:
            period = int(period)
        except (ValueError, TypeError):
            period = 0
        
        # 도구 호출 처리
        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"Seasonality Analyzer: {len(ai_message.tool_calls)}개 도구 호출 감지됨")
            tool_response = tool_node.invoke({"messages": [ai_message]})
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            
            # 시각화 도구 응답 검사 및 이미지 로드
            for msg in tool_messages:
                if "ts2img" in msg.content:
                    try:
                        # 시각화 도구 응답에서 이미지 경로 추출 및 처리
                        multimodal_content = process_visualization_result(msg.content)
                        if multimodal_content:
                            visualization_content = multimodal_content
                            print("계절성 시각화 이미지 로드 성공 - 멀티모달 입력으로 LLM에 전달 예정")
                    except Exception as viz_err:
                        print(f"시각화 결과 처리 중 오류: {viz_err}")
            
            # 도구 결과를 포함한 후속 LLM 호출 (구조화된 출력 사용)
            if visualization_content:
                # 멀티모달 입력(이미지 + 텍스트)으로 LLM 호출 - 구조화된 출력 형식 지정
                structured_llm = llm.with_structured_output(SeasonalityAnalysis)
                
                follow_up_message = HumanMessage(content=[
                    {"type": "text", "text": f"이 계절성 이미지를 분석하고 이상치를 찾아서 SeasonalityAnalysis 구조로 응답해주세요. "
                                          f"계절성 주기({period})와 발견한 이상치에는 정확한 인덱스와 유형을 지정해주세요. "
                                          f"계절성 강도는 {seasonality_strength}입니다."},
                    *[item for item in visualization_content if item["type"] == "image_url"]
                ])
                structured_result = structured_llm.invoke([follow_up_message])
                analysis_content = structured_result.llm_analysis if structured_result.llm_analysis else "계절성 시각적 분석 완료"
            else:
                # 일반 텍스트 입력으로 구조화된 LLM 호출
                structured_llm = llm_with_tools.with_structured_output(SeasonalityAnalysis)
                
                follow_up_prompt = seasonality_analyzer_prompt.invoke({
                    "messages": messages, 
                    "input": f"도구 결과를 바탕으로 계절성 데이터의 분석을 완료해주세요. 계절성 주기({period})와 특성, "
                             f"이상치 유형을 상세히 설명하고, 이상치가 위치한 인덱스를 정확히 명시해주세요. "
                             f"계절성 강도는 {seasonality_strength}입니다."
                })
                structured_result = structured_llm.invoke(follow_up_prompt)
                analysis_content = structured_result.llm_analysis
            
            # 에이전트 이름으로 메시지 추가
            ai_message_with_name = AIMessage(content=analysis_content, name="seasonality_analyzer")
            messages.append(ai_message_with_name)
            
            # 계절성 분석 결과에 주기와 강도 정보 추가
            structured_result.period = period if structured_result.period == 0 else structured_result.period
            structured_result.strength = seasonality_strength
            
            # 구조화된 결과에서 계절성 분석 결과 가져오기
            seasonality_analysis = structured_result.model_dump()
        else:
            # 도구 호출이 없는 경우, 직접 구조화된 출력 요청
            content = ai_message.content
            
            # 구조화된 출력 형식으로 LLM 재호출
            structured_llm = llm.with_structured_output(SeasonalityAnalysis)
            structured_prompt = f"""
            계절성 데이터 분석 결과를 바탕으로 SeasonalityAnalysis 구조로 정보를 제공해주세요.
            계절성 주기({period})와 발견된 이상치가 있다면 각각의 인덱스와 유형을 정확히 지정해주세요.
            
            계절성 강도는 {seasonality_strength}입니다.
            이전 분석: {content}
            """
            structured_result = structured_llm.invoke([HumanMessage(content=structured_prompt)])
            
            # 에이전트 이름으로 메시지 추가
            ai_message_with_name = AIMessage(content=structured_result.llm_analysis or content, name="seasonality_analyzer")
            messages.append(ai_message_with_name)
            
            # 계절성 분석 결과에 주기와 강도 정보 추가
            structured_result.period = period if structured_result.period == 0 else structured_result.period
            structured_result.strength = seasonality_strength
            
            # 구조화된 결과에서 계절성 분석 결과 가져오기
            seasonality_analysis = structured_result.model_dump()
        
        return {
            "messages": messages,
            "seasonality_analysis": [seasonality_analysis]  # 리스트 형식으로 반환해야 함
        }
    
    return seasonality_analyzer_agent