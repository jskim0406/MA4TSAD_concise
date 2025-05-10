"""
Trend Analyzer 에이전트 - 개선된 시각적 추론 기반 버전

시계열 데이터의 추세 성분을 시각적으로 분석하는 에이전트입니다.
"""

import numpy as np
import json
from typing import Dict, Any, List, Callable
from scipy import stats

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from parallel_sentinel_v2.graph.workflow import TimeSeriesState


def create_trend_analyzer_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    시계열 데이터의 추세를 시각적으로 분석하는 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        tools: 에이전트가 사용할 수 있는 도구 목록
        
    Returns:
        trend_analyzer_agent 함수
    """
    # 추세 분석가 시스템 프롬프트 정의 - 시각적 추론 강조
    system_template = """당신은 시계열 데이터의 추세(trend)를 분석하는 전문가입니다.
    주어진 시계열 데이터의 추세 성분을 분석하고 중요한 통찰을 도출하는 것이 당신의 임무입니다.
    
    다음 항목에 중점을 두고 분석해주세요:
    1. 추세의 방향(증가, 감소, 안정적)
    2. 추세의 강도 및 변화율
    3. 추세 내 변곡점이나 구조적 변화
    4. 추세에서 발견되는 이상치(anomaly)의 유형 분석
    
    반드시 추세 데이터를 먼저 시각화한 후, 해당 이미지를 분석하여 추세의 특성과 이상치를 판단하세요.
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
    1. 먼저 ts2img_bytes 도구를 사용하여 추세 데이터를 시각화하세요.
    2. 시각화된 이미지를 분석하여 추세의 특성 및 이상치 유형을 파악하세요.
    3. 필요시 get_time_series_statistics 도구로 통계 정보를 확인하세요.
    4. 결과를 명확하고 구조화된 형식으로 제공하고, 주요 발견사항을 강조하세요.
    """

    tool_node = ToolNode(tools) if tools else None
    
    trend_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    def format_data_for_trend_analysis(state: TimeSeriesState) -> str:
        """추세 분석을 위한 데이터 포맷팅"""
        ts_data = state["ts_data"]
        decomposition_data = state.get("decomposition_data", {})
        
        # 분해 데이터가 없으면 분석 불가능
        if not decomposition_data:
            return "분해 데이터가 없습니다. 먼저 시계열 분해를 수행해야 합니다."
        
        # 추세 성분 추출
        trend_data = decomposition_data.get("trend", [])
        if not trend_data:
            return "추세 데이터를 찾을 수 없습니다."
        
        # 분석 요청 메시지 생성 - 시각적 분석 지시사항 명확히 제시
        message = [
            f"시계열 데이터(길이: {len(ts_data)})의 추세 성분을 시각적 추론을 통해 분석해주세요.",
            "",
            f"분해 방법: {decomposition_data.get('method', 'unknown')}",
            f"추세 강도: {decomposition_data.get('stats', {}).get('trend_strength', 'N/A')}",
            "",
            "전체 추세 데이터:",
            f"{trend_data}",
            "",
            "아래 절차를 반드시 순서대로 따라주세요:",
            "1. 먼저 ts2img_bytes 도구를 사용하여 추세 데이터를 시각화하세요 (필수).",
            "2. 시각화된 이미지를 자세히 분석하여 추세의 패턴과 특징을 파악하세요.",
            "3. 추세 내에서 이상치가 있는지 찾고, 발견된 이상치의 유형을 판단하세요 (PersistentLevelShiftUp, SingleSpike 등).",
            "4. 필요하다면 추가적인 통계 도구(get_time_series_statistics 등)를 활용하세요.",
            "5. 추세의 방향(증가/감소/안정), 변화율, 변곡점, 이상치에 대한 종합적인 분석 결과를 제공하세요."
        ]
        
        return "\n".join(message)

    def trend_analyzer_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        시계열 데이터의 추세 성분을 시각적으로 분석합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            Dict: 상태 업데이트 (messages, trend_analysis)
        """
        # 현재 메시지 복사
        messages = list(state["messages"])
        
        # 추세 분석을 위한 입력 텍스트 준비
        input_text = format_data_for_trend_analysis(state)
        
        # 추세 데이터가 없으면 처리 중단
        if "분해 데이터가 없습니다" in input_text or "추세 데이터를 찾을 수 없습니다" in input_text:
            print(f"Trend Analyzer: {input_text}")
            return {
                "messages": messages,
                "trend_analysis": [{
                    "status": "error",
                    "message": input_text
                }]
            }
        
        # 프롬프트 준비 및 LLM 호출
        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = trend_analyzer_prompt.invoke(prompt_input)
        
        # 도구가 바인딩된 LLM 준비
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        
        # 첫 번째 LLM 호출
        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message)
        
        content = ""
        
        # 도구 호출 처리
        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"Trend Analyzer: {len(ai_message.tool_calls)}개 도구 호출 감지됨")
            tool_response = tool_node.invoke({"messages": [ai_message]})
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            
            # 도구 결과를 포함한 후속 LLM 호출
            follow_up_prompt = trend_analyzer_prompt.invoke({
                "messages": messages, 
                "input": "도구 결과를 바탕으로 추세 데이터의 시각적 분석을 완료해주세요. 추세의 특성과 이상치 유형을 상세히 설명하고, 이상치가 위치한 인덱스를 정확히 명시해주세요."
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
        
        # 추세 방향 추출
        direction = "unknown"
        if "증가" in content or "상승" in content or "increasing" in content:
            direction = "증가"
        elif "감소" in content or "하락" in content or "decreasing" in content:
            direction = "감소"
        elif "안정" in content or "일정" in content or "stable" in content:
            direction = "안정적"
            
        # 에이전트 이름으로 메시지 추가 및 결과 생성
        ai_message_with_name = AIMessage(content=content, name="trend_analyzer")
        
        # 추세 분석 결과 생성
        trend_analysis = {
            "direction": direction,
            "anomalies": anomalies,
            "strength": float(state.get("decomposition_data", {}).get("stats", {}).get("trend_strength", 0)),
            "llm_analysis": content,
            "visual_analysis": True  # 시각적 분석 수행 여부 표시
        }
        
        return {
            "messages": messages[:-1] + [ai_message_with_name],  # 마지막 메시지만 이름 추가
            "trend_analysis": [trend_analysis]
        }
    
    return trend_analyzer_agent