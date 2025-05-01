"""
Seasonality Analyzer 에이전트

시계열 데이터의 계절성 성분을 분석하는 에이전트입니다.
"""

import numpy as np
import json
from typing import Dict, Any, List, Callable
from scipy import stats, signal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from parallel_sentinel.graph.workflow import TimeSeriesState


def create_seasonality_analyzer_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    시계열 데이터의 계절성을 분석하는 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        tools: 에이전트가 사용할 수 있는 도구 목록
        
    Returns:
        seasonality_analyzer_agent 함수
    """
    # 계절성 분석가 시스템 프롬프트 정의
    system_template = """당신은 시계열 데이터의 계절성(seasonality)을 분석하는 전문가입니다.
    주어진 시계열 데이터의 계절성 성분을 분석하고 중요한 통찰을 도출하는 것이 당신의 임무입니다.
    
    다음 항목에 중점을 두고 분석해주세요:
    1. 계절성 패턴의 주기(일별, 주별, 월별, 분기별, 연간 등)
    2. 계절성 패턴의 강도와 일관성
    3. 패턴의 진폭(최대값과 최소값 사이의 차이)
    4. 계절성 패턴 내 이상치나 비정상적인 사이클
    5. 계절성이 전체 시계열에 미치는 영향
    
    다음 도구를 사용할 수 있습니다:
    - ts2img: 시계열 데이터 시각화
    - calculate: 수학 계산 수행
    
    당신은 시계열 분해 결과 중 계절성 성분만 집중적으로 분석합니다.
    명확하고 구조화된 형식으로 분석 결과를 제공하고, 주요 발견사항을 강조해주세요.
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
        
        # 분석 요청 메시지 생성
        message = [
            f"시계열 데이터(길이: {len(ts_data)})의 계절성 성분을 분석해주세요.",
            "",
            f"분해 방법: {decomposition_data.get('method', 'unknown')}",
            f"계절성 주기: {period}",
            f"계절성 강도: {decomposition_data.get('stats', {}).get('seasonality_strength', 'N/A')}",
            "",
            "계절성 데이터 샘플 (전체 주기):"
        ]
        
        # 계절성의 전체 주기 표시
        try:
            period_int = int(period)
            if period_int > 0 and period_int <= 100:  # 합리적인 주기 범위 확인
                one_period = seasonality_data[:period_int]
                message.append(f"{one_period}")
            else:
                message.append(f"처음 50개 값: {seasonality_data[:50]}")
        except (ValueError, TypeError):
            message.append(f"처음 50개 값: {seasonality_data[:50]}")
        
        message.append("")
        message.append("계절성 성분에 대한 철저한 분석을 제공해주세요. 주기, 강도, 패턴의 일관성, 그리고 계절성 내 이상치를 식별해주세요.")
        
        return "\n".join(message)

    def seasonality_analyzer_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        시계열 데이터의 계절성 성분을 분석합니다.
        
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
                "input": "도구 결과를 기반으로 계절성 분석을 완료해주세요."
            })
            ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            messages.append(ai_follow_up)
            content = ai_follow_up.content
        else:
            content = ai_message.content
        
        # --- 계절성 분석 수행 (NumPy 기반) ---
        decomposition_data = state.get("decomposition_data", {})
        seasonality_data = np.array(decomposition_data.get("seasonality", []))
        period = decomposition_data.get("period", 0)
        
        try:
            period = int(period)
        except (ValueError, TypeError):
            period = 0
        
        if len(seasonality_data) > 0 and period > 1:
            # 계절성 속성 계산
            amplitude = np.max(seasonality_data) - np.min(seasonality_data)
            
            # 주기별 계절성 패턴 추출
            num_full_periods = len(seasonality_data) // period
            if num_full_periods > 0:
                seasons = seasonality_data[:num_full_periods * period].reshape(num_full_periods, period)
                avg_pattern = np.mean(seasons, axis=0)
                std_pattern = np.std(seasons, axis=0)
                consistency = 1 - (np.mean(std_pattern) / (np.max(avg_pattern) - np.min(avg_pattern)) if np.max(avg_pattern) != np.min(avg_pattern) else 0)
            else:
                avg_pattern = seasonality_data[:period] if period <= len(seasonality_data) else seasonality_data
                consistency = 0
            
            # 계절성 이상치 탐지
            anomalies = []
            if num_full_periods > 1:
                for i in range(num_full_periods):
                    season = seasons[i]
                    # 각 위치에서 평균 패턴과의 차이 계산
                    diffs = np.abs(season - avg_pattern)
                    thresholds = 2.0 * std_pattern  # 2 표준편차 기준
                    
                    for j in range(period):
                        if diffs[j] > thresholds[j]:
                            global_idx = i * period + j
                            anomalies.append({
                                "index": int(global_idx),
                                "value": float(seasonality_data[global_idx]),
                                "expected": float(avg_pattern[j]),
                                "score": float(diffs[j] / std_pattern[j]) if std_pattern[j] > 0 else 0
                            })
            
            # 상위 10개 이상치만 유지
            if len(anomalies) > 10:
                anomalies = sorted(anomalies, key=lambda x: x["score"], reverse=True)[:10]
            
            # 계절성 분석 결과 생성
            seasonality_analysis = {
                "period": period,
                "amplitude": float(amplitude),
                "consistency": float(consistency),
                "avg_pattern": avg_pattern.tolist()[:min(50, len(avg_pattern))],  # 최대 50개 값만 포함
                "anomalies": anomalies,
                "strength": float(decomposition_data.get("stats", {}).get("seasonality_strength", 0)),
                "llm_analysis": content
            }
        else:
            seasonality_analysis = {
                "status": "warning",
                "message": "계절성이 약하거나 주기를 식별할 수 없습니다.",
                "strength": float(decomposition_data.get("stats", {}).get("seasonality_strength", 0)),
                "llm_analysis": content
            }
        
        # 에이전트 이름으로 메시지 추가
        ai_message_with_name = AIMessage(content=content, name="seasonality_analyzer")
        
        return {
            "messages": messages[:-1] + [ai_message_with_name],  # 마지막 메시지만 이름 추가
            "seasonality_analysis": [seasonality_analysis]
        }
    
    return seasonality_analyzer_agent