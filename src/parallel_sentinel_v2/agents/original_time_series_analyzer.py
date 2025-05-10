"""
Original Time Series Analyzer 에이전트 - 원본 데이터 직접 분석 버전

시계열 분해 이전의 원본 시계열 데이터를 시각적으로 분석하는 에이전트입니다.
"""

import os
import json
import base64
import numpy as np
import re
from typing import Dict, Any, List, Callable
from scipy import stats

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from parallel_sentinel_v2.graph.workflow import TimeSeriesState
from parallel_sentinel_v2.utils.llm_utils import process_visualization_result


def create_original_time_series_analyzer_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    시계열 분해 이전의 원본 시계열 데이터를 시각적으로 분석하는 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        tools: 에이전트가 사용할 수 있는 도구 목록
        
    Returns:
        original_time_series_analyzer_agent 함수
    """
    # 원본 시계열 분석가 시스템 프롬프트 정의 - 시각적 추론 강조
    system_template = """당신은 원본 시계열 데이터를 직접 분석하는 전문가입니다.
    시계열 분해 이전의 원본 데이터를 시각적으로 분석하고 중요한 패턴과 이상치를 식별하는 것이 당신의 임무입니다.
    
    다음 항목에 중점을 두고 분석해주세요:
    1. 전체 시계열의 일반적인 패턴과 특성
    2. 추세와 계절성이 혼합된 상태에서의 이상치 식별
    3. 눈에 띄는 특이점이나 패턴 변화
    4. 원본 데이터에서 발견되는 이상치(anomaly)의 유형 분석
    
    반드시 원본 시계열 데이터를 먼저 시각화한 후, 해당 이미지를 분석하여 이상치의 특성과 유형을 판단하세요.
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
    1. 먼저 ts2img_bytes 도구를 사용하여 원본 시계열 데이터를 시각화하세요.
    2. 시각화된 이미지를 분석하여 시계열의 특성 및 이상치 유형을 파악하세요.
    3. 필요시 get_time_series_statistics 도구로 통계 정보를 확인하세요.
    4. 결과를 명확하고 구조화된 형식으로 제공하고, 주요 발견사항을 강조하세요.
    """

    tool_node = ToolNode(tools) if tools else None
    
    original_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    def format_data_for_original_analysis(state: TimeSeriesState) -> str:
        """원본 시계열 분석을 위한 데이터 포맷팅"""
        ts_data = state["ts_data"]
        
        # 분석 요청 메시지 생성 - 시각적 분석 지시사항 명확히 제시
        message = [
            f"원본 시계열 데이터(길이: {len(ts_data)})를 시각적 추론을 통해 분석해주세요.",
            "",
            "아래 절차를 반드시 순서대로 따라주세요:",
            "1. 먼저 ts2img_bytes 도구를 사용하여 원본 시계열 데이터를 시각화하세요 (필수).",
            "2. 시각화된 이미지를 자세히 분석하여 시계열의 패턴과 특징을 파악하세요.",
            "3. 원본 시계열에서 이상치가 있는지 찾고, 발견된 이상치의 유형을 판단하세요 (PersistentLevelShiftUp, SingleSpike 등).",
            "4. 필요하다면 추가적인 통계 도구(get_time_series_statistics 등)를 활용하세요.",
            "5. 시계열의 전반적인 특성, 이상치, 잠재적 패턴에 대한 종합적인 분석 결과를 제공하세요.",
            "",
            "전체 시계열 데이터는 다음과 같습니다:",
            f"{ts_data}"
        ]
        
        return "\n".join(message)

    def original_time_series_analyzer_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        원본 시계열 데이터를 시각적으로 분석합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            Dict: 상태 업데이트 (messages, original_ts_analysis)
        """
        # 현재 메시지 복사
        messages = list(state["messages"])
        
        # 원본 시계열 분석을 위한 입력 텍스트 준비
        input_text = format_data_for_original_analysis(state)
        
        # 프롬프트 준비 및 LLM 호출
        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = original_analyzer_prompt.invoke(prompt_input)
        
        # 도구가 바인딩된 LLM 준비
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        
        # 첫 번째 LLM 호출
        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message)
        
        content = ""
        visualization_content = None
        
        # 도구 호출 처리
        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"Original TS Analyzer: {len(ai_message.tool_calls)}개 도구 호출 감지됨")
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
                            print("원본 시계열 시각화 이미지 로드 성공 - 멀티모달 입력으로 LLM에 전달 예정")
                    except Exception as viz_err:
                        print(f"시각화 결과 처리 중 오류: {viz_err}")
            
            # 도구 결과를 포함한 후속 LLM 호출
            if visualization_content:
                # 멀티모달 입력(이미지 + 텍스트)으로 LLM 호출
                follow_up_message = HumanMessage(content=visualization_content)
                ai_follow_up = llm.invoke([follow_up_message])
            else:
                # 일반 텍스트 입력으로 LLM 호출
                follow_up_prompt = original_analyzer_prompt.invoke({
                    "messages": messages, 
                    "input": "도구 결과를 바탕으로 원본 시계열 데이터의 분석을 완료해주세요. 시계열의 특성과 이상치 유형을 상세히 설명하고, 이상치가 위치한 인덱스를 정확히 명시해주세요."
                })
                ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            
            messages.append(ai_follow_up)
            content = ai_follow_up.content
        else:
            content = ai_message.content
            
        # LLM 응답에서 이상치 정보 추출
        anomalies = []
        anomaly_types = ["PersistentLevelShiftUp", "PersistentLevelShiftDown", 
                         "TransientLevelShiftUp", "TransientLevelShiftDown", 
                         "SingleSpike", "SingleDip", "MultipleSpikes", "MultipleDips"]
        
        for anomaly_type in anomaly_types:
            if anomaly_type in content:
                # 해당 이상치 유형이 언급된 경우, 인덱스 정보 추출 시도
                # 인덱스 패턴 검색: 숫자, 인덱스, index 등의 키워드 주변 숫자 찾기
                index_patterns = [
                    rf"{anomaly_type}.*?인덱스\s*?(\d+)",
                    rf"{anomaly_type}.*?index\s*?(\d+)",
                    rf"{anomaly_type}.*?위치\s*?(\d+)",
                    rf"인덱스\s*?(\d+).*?{anomaly_type}",  # 순서가 바뀌는 경우 대응
                    rf"index\s*?(\d+).*?{anomaly_type}"    # 영어로 쓰이는 경우 대응
                ]
                
                for pattern in index_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            try:
                                index = int(match)
                                anomalies.append({
                                    "type": anomaly_type,
                                    "index": index,
                                    "confidence": 0.9  # LLM이 명시적으로 언급했으므로 높은 신뢰도
                                })
                            except ValueError:
                                pass
        
        # 에이전트 이름으로 메시지 추가 및 결과 생성
        ai_message_with_name = AIMessage(content=content, name="original_time_series_analyzer")
        
        # 원본 시계열 분석 결과 생성
        original_ts_analysis = {
            "anomalies": anomalies,
            "llm_analysis": content,
            "visual_analysis": True if visualization_content else False  # 시각적 분석 수행 여부 표시
        }
        
        return {
            "messages": messages[:-1] + [ai_message_with_name],  # 마지막 메시지만 이름 추가
            "original_ts_analysis": [original_ts_analysis]  # 리스트 형식으로 반환해야 함
        }
    
    return original_time_series_analyzer_agent