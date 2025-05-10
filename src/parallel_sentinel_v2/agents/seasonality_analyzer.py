"""
Seasonality Analyzer 에이전트 - 개선된 시각적 추론 기반 버전

시계열 데이터의 계절성 성분을 시각적으로 분석하는 에이전트입니다.
"""

import numpy as np
import json
from typing import Dict, Any, List, Callable
from scipy import stats, signal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from parallel_sentinel_v2.graph.workflow import TimeSeriesState


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
            f"{seasonality_data}",
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
        
        # 도구 호출 처리
        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"Seasonality Analyzer: {len(ai_message.tool_calls)}개 도구 호출 감지됨")
            tool_response = tool_node.invoke({"messages": [ai_message]})
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            
            # 도구 결과를 포함한 후속 LLM 호출
            follow_up_prompt = seasonality_analyzer_prompt.invoke({
                "messages": messages, 
                "input": "도구 결과를 바탕으로 계절성 데이터의 시각적 분석을 완료해주세요. 계절성의 특성과 이상치 유형을 상세히 설명하고, 이상치가 위치한 인덱스를 정확히 명시해주세요."
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
                import re
                # 인덱스 패턴 검색: 숫자, 인덱스, index 등의 키워드 주변 숫자 찾기
                index_patterns = [
                    rf"{anomaly_type}.*?인덱스\s*?(\d+)",
                    rf"{anomaly_type}.*?index\s*?(\d+)",
                    rf"{anomaly_type}.*?위치\s*?(\d+)"
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
        
        # 계절성 주기 추출
        period = state.get("decomposition_data", {}).get("period", 0)
        try:
            period = int(period)
        except (ValueError, TypeError):
            period = 0
            # LLM 응답에서 주기 추출 시도
            import re
            period_patterns = [
                r"주기[는은이가]?\s*(\d+)",
                r"period[는은이가]?\s*(\d+)",
                r"주기[는은이가]?\s*약\s*(\d+)",
                r"(\d+)\s*의\s*주기"
            ]
            for pattern in period_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    try:
                        period = int(matches[0])
                        break
                    except ValueError:
                        continue
            
        # 에이전트 이름으로 메시지 추가 및 결과 생성
        ai_message_with_name = AIMessage(content=content, name="seasonality_analyzer")
        
        # 계절성 분석 결과 생성
        seasonality_analysis = {
            "period": period,
            "anomalies": anomalies,
            "strength": float(state.get("decomposition_data", {}).get("stats", {}).get("seasonality_strength", 0)),
            "llm_analysis": content,
            "visual_analysis": True  # 시각적 분석 수행 여부 표시
        }
        
        return {
            "messages": messages[:-1] + [ai_message_with_name],  # 마지막 메시지만 이름 추가
            "seasonality_analysis": [seasonality_analysis]
        }
    
    return seasonality_analyzer_agent