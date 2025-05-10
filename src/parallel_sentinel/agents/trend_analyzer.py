"""
Trend Analyzer 에이전트

시계열 데이터의 추세 성분을 분석하는 에이전트입니다.
"""

import numpy as np
import json
from typing import Dict, Any, List, Callable
from scipy import stats

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from parallel_sentinel.graph.workflow import TimeSeriesState


def create_trend_analyzer_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    시계열 데이터의 추세를 분석하는 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        tools: 에이전트가 사용할 수 있는 도구 목록
        
    Returns:
        trend_analyzer_agent 함수
    """
    # 추세 분석가 시스템 프롬프트 정의
    system_template = """당신은 시계열 데이터의 추세(trend)를 분석하는 전문가입니다.
    주어진 시계열 데이터의 추세 성분을 분석하고 중요한 통찰을 도출하는 것이 당신의 임무입니다.
    
    다음 항목에 중점을 두고 분석해주세요:
    1. 추세의 방향(증가, 감소, 안정적)
    2. 추세의 강도 및 변화율
    3. 추세 내 변곡점이나 구조적 변화
    4. 추세 내 이상치나 비정상적인 패턴
    5. 추세의 미래 전망
    
    다음 도구를 사용할 수 있습니다:
    - ts2img: 시계열 데이터 시각화
    - calculate: 수학 계산 수행
    
    당신은 시계열 분해 결과 중 추세 성분만 집중적으로 분석합니다.
    명확하고 구조화된 형식으로 분석 결과를 제공하고, 주요 발견사항을 강조해주세요.
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
        
        # # 분석 요청 메시지 생성
        # message = [
        #     f"시계열 데이터(길이: {len(ts_data)})의 추세 성분을 분석해주세요.",
        #     "",
        #     f"분해 방법: {decomposition_data.get('method', 'unknown')}",
        #     f"추세 강도: {decomposition_data.get('stats', {}).get('trend_strength', 'N/A')}",
        #     "",
        #     "추세 데이터 샘플 (처음 10개 값):",
        #     f"{trend_data[:10]}",
        #     "",
        #     "추세 데이터 샘플 (마지막 10개 값):",
        #     f"{trend_data[-10:] if len(trend_data) > 10 else trend_data}",
        #     "",
        #     "추세 성분에 대한 철저한 분석을 제공해주세요. 추세 방향, 변화율, 변곡점, 그리고 추세 내 이상치를 식별해주세요."
        # ]

        # 분석 요청 메시지 생성
        message = [
            f"시계열 데이터(길이: {len(ts_data)})의 추세 성분을 분석해주세요.",
            "",
            f"분해 방법: {decomposition_data.get('method', 'unknown')}",
            f"추세 강도: {decomposition_data.get('stats', {}).get('trend_strength', 'N/A')}",
            "",
            "추세 데이터 샘플:",
            f"{trend_data}",
            "",
            "추세 성분에 대한 철저한 분석을 제공해주세요. 추세 방향, 변화율, 변곡점, 그리고 추세 내 이상치를 식별해주세요."
        ]
        
        return "\n".join(message)

    def trend_analyzer_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        시계열 데이터의 추세 성분을 분석합니다.
        
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
                "input": "도구 결과를 기반으로 추세 분석을 완료해주세요."
            })
            ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            messages.append(ai_follow_up)
            content = ai_follow_up.content
        else:
            content = ai_message.content
        
        # --- 추세 분석 수행 (NumPy 기반) ---
        decomposition_data = state.get("decomposition_data", {})
        trend_data = np.array(decomposition_data.get("trend", []))
        
        if len(trend_data) > 0:
            # 추세 방향과 속성 계산
            x = np.arange(len(trend_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, trend_data)
            
            # 변곡점 탐지 (1차 미분의 부호 변화)
            # 데이터를 부드럽게 하기 위해 이동 평균 적용
            window = max(3, len(trend_data) // 50)  # 작은 윈도우 사용
            if len(trend_data) > 2 * window:
                smoothed = np.convolve(trend_data, np.ones(window)/window, mode='valid')
                diff = np.diff(smoothed)
                inflection_points = []
                
                for i in range(1, len(diff)):
                    if (diff[i-1] > 0 and diff[i] < 0) or (diff[i-1] < 0 and diff[i] > 0):
                        inflection_points.append(i + window//2)  # 위치 조정
                
                # 최대 5개의 중요 변곡점만 사용
                if len(inflection_points) > 5:
                    # 가장 큰 변화를 보이는 변곡점 선택
                    inflection_magnitudes = [abs(diff[i] - diff[i-1]) for i in inflection_points if i < len(diff)]
                    top_indices = np.argsort(inflection_magnitudes)[-5:]
                    inflection_points = [inflection_points[i] for i in top_indices]
                
                inflection_points = sorted(inflection_points)
            else:
                inflection_points = []
            
            # 이상치 탐지 (평균에서 2 표준편차 이상 벗어난 값)
            mean = np.mean(trend_data)
            std = np.std(trend_data)
            threshold = 2.0
            
            anomaly_indices = np.where(np.abs(trend_data - mean) > threshold * std)[0].tolist()
            anomaly_values = [(int(i), float(trend_data[i])) for i in anomaly_indices]
            
            # 추세 분석 결과 생성
            trend_analysis = {
                "direction": "증가" if slope > 0.001 else "감소" if slope < -0.001 else "안정적",
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "significance": float(p_value),
                "std_error": float(std_err),
                "inflection_points": inflection_points,
                "anomalies": [{"index": idx, "value": val, "score": abs((val - mean) / std)} 
                             for idx, val in anomaly_values],
                "strength": float(decomposition_data.get("stats", {}).get("trend_strength", 0)),
                "llm_analysis": content
            }
        else:
            trend_analysis = {
                "status": "error",
                "message": "추세 데이터를 분석할 수 없습니다.",
                "llm_analysis": content
            }
        
        # 에이전트 이름으로 메시지 추가
        ai_message_with_name = AIMessage(content=content, name="trend_analyzer")
        
        return {
            "messages": messages[:-1] + [ai_message_with_name],  # 마지막 메시지만 이름 추가
            "trend_analysis": [trend_analysis]
        }
    
    return trend_analyzer_agent