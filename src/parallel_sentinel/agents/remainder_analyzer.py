"""
Remainder Analyzer 에이전트

시계열 데이터의 잔차 성분을 분석하는 에이전트입니다.
잔차 성분은 추세와 계절성으로 설명되지 않는 부분으로, 이상치 감지에 중요합니다.
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


def create_remainder_analyzer_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    시계열 데이터의 잔차를 분석하는 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        tools: 에이전트가 사용할 수 있는 도구 목록
        
    Returns:
        remainder_analyzer_agent 함수
    """
    # 잔차 분석가 시스템 프롬프트 정의
    system_template = """당신은 시계열 데이터의 잔차(remainder) 성분을 분석하는 전문가입니다.
    주어진 시계열 데이터의 잔차 성분을 분석하고 이상치를 식별하는 것이 당신의 임무입니다.
    
    잔차는 원본 시계열에서 추세와 계절성을 제거한 후 남은 부분으로, 종종 '노이즈'라고도 불립니다.
    그러나 잔차에는 중요한 이상치 정보가 포함될 수 있습니다.
    
    다음 항목에 중점을 두고 분석해주세요:
    1. 잔차의 분포와 패턴 (정규 분포를 따르는지 등)
    2. 잔차 내 명확한 이상치 식별
    3. 군집화된 이상치나 패턴
    4. 잔차의 자기상관성 여부 (잔차가 독립적인지)
    5. 잔차가 시사하는 비정상적 사건이나 이상 징후
    
    다음 도구를 사용할 수 있습니다:
    - ts2img: 시계열 데이터 시각화
    - calculate: 수학 계산 수행
    
    당신은 시계열 분해 결과 중 잔차 성분만 집중적으로 분석합니다.
    명확하고 구조화된 형식으로 분석 결과를 제공하고, 주요 발견사항을 강조해주세요.
    """

    tool_node = ToolNode(tools) if tools else None
    
    remainder_analyzer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    def format_data_for_remainder_analysis(state: TimeSeriesState) -> str:
        """잔차 분석을 위한 데이터 포맷팅"""
        ts_data = state["ts_data"]
        decomposition_data = state.get("decomposition_data", {})
        
        # 분해 데이터가 없으면 분석 불가능
        if not decomposition_data:
            return "분해 데이터가 없습니다. 먼저 시계열 분해를 수행해야 합니다."
        
        # 잔차 성분 추출
        remainder_data = decomposition_data.get("remainder", [])
        if not remainder_data:
            return "잔차 데이터를 찾을 수 없습니다."
        
        # # 분석 요청 메시지 생성
        # message = [
        #     f"시계열 데이터(길이: {len(ts_data)})의 잔차 성분을 분석해주세요.",
        #     "",
        #     f"분해 방법: {decomposition_data.get('method', 'unknown')}",
        #     f"잔차 강도: {decomposition_data.get('stats', {}).get('remainder_strength', 'N/A')}",
        #     "",
        #     "잔차 데이터 샘플 (처음 20개 값):",
        #     f"{remainder_data[:20]}",
        #     "",
        #     "잔차 데이터 샘플 (마지막 20개 값):",
        #     f"{remainder_data[-20:] if len(remainder_data) > 20 else remainder_data}",
        #     "",
        #     "잔차 성분에 대한 철저한 분석을 제공해주세요. 잔차의 분포, 패턴, 이상치를 식별하고, 시계열에서 비정상적인 부분을 찾아주세요."
        # ]

        # 분석 요청 메시지 생성
        message = [
            f"시계열 데이터(길이: {len(ts_data)})의 잔차 성분을 분석해주세요.",
            "",
            f"분해 방법: {decomposition_data.get('method', 'unknown')}",
            f"잔차 강도: {decomposition_data.get('stats', {}).get('remainder_strength', 'N/A')}",
            "",
            "잔차 데이터 샘플:",
            f"{remainder_data}",
            "",
            "잔차 성분에 대한 철저한 분석을 제공해주세요. 잔차의 분포, 패턴, 이상치를 식별하고, 시계열에서 비정상적인 부분을 찾아주세요."
        ]
        return "\n".join(message)

    def remainder_analyzer_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        시계열 데이터의 잔차 성분을 분석합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            Dict: 상태 업데이트 (messages, remainder_analysis)
        """
        # 현재 메시지 복사
        messages = list(state["messages"])
        
        # 잔차 분석을 위한 입력 텍스트 준비
        input_text = format_data_for_remainder_analysis(state)
        
        # 잔차 데이터가 없으면 처리 중단
        if "분해 데이터가 없습니다" in input_text or "잔차 데이터를 찾을 수 없습니다" in input_text:
            print(f"Remainder Analyzer: {input_text}")
            return {
                "messages": messages,
                "remainder_analysis": [{
                    "status": "error",
                    "message": input_text
                }]
            }
        
        # 프롬프트 준비 및 LLM 호출
        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = remainder_analyzer_prompt.invoke(prompt_input)
        
        # 도구가 바인딩된 LLM 준비
        llm_with_tools = llm.bind_tools(tools) if tools else llm
        
        # 첫 번째 LLM 호출
        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message)
        
        content = ""
        
        # 도구 호출 처리
        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"Remainder Analyzer: {len(ai_message.tool_calls)}개 도구 호출 감지됨")
            tool_response = tool_node.invoke({"messages": [ai_message]})
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            
            # 도구 결과를 포함한 후속 LLM 호출
            follow_up_prompt = remainder_analyzer_prompt.invoke({
                "messages": messages, 
                "input": "도구 결과를 기반으로 잔차 분석을 완료해주세요."
            })
            ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            messages.append(ai_follow_up)
            content = ai_follow_up.content
        else:
            content = ai_message.content
        
        # --- 잔차 분석 수행 (NumPy 기반) ---
        decomposition_data = state.get("decomposition_data", {})
        remainder_data = np.array(decomposition_data.get("remainder", []))
        
        if len(remainder_data) > 0:
            # 기본 통계 계산
            mean = np.mean(remainder_data)
            std = np.std(remainder_data)
            skewness = stats.skew(remainder_data)
            kurtosis = stats.kurtosis(remainder_data)
            
            # 정규성 테스트
            shapiro_test = stats.shapiro(remainder_data)
            is_normal = shapiro_test.pvalue > 0.05
            
            # 이상치 탐지
            # 방법 1: Z-score 기반 (통계적 이상치)
            z_scores = (remainder_data - mean) / std if std > 0 else np.zeros_like(remainder_data)
            z_threshold = 3.0
            z_outliers = np.where(np.abs(z_scores) > z_threshold)[0].tolist()
            
            # 방법 2: IQR 기반 (분포적 이상치)
            q1 = np.percentile(remainder_data, 25)
            q3 = np.percentile(remainder_data, 75)
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr
            iqr_outliers = np.where((remainder_data < iqr_lower) | (remainder_data > iqr_upper))[0].tolist()
            
            # 두 방법으로 감지된 이상치 결합
            all_outliers = sorted(list(set(z_outliers + iqr_outliers)))
            
            # 이상치 정보 생성
            anomalies = []
            for idx in all_outliers:
                value = remainder_data[idx]
                z_score = z_scores[idx]
                is_iqr_outlier = idx in iqr_outliers
                
                # 이상치 신뢰도 점수 계산
                score = abs(z_score)  # Z-score의 절대값을 기본 점수로 사용
                if is_iqr_outlier:
                    score += 1.0  # IQR 방법으로도 감지되면 점수 증가
                
                anomalies.append({
                    "index": int(idx),
                    "value": float(value),
                    "z_score": float(z_score),
                    "is_iqr_outlier": is_iqr_outlier,
                    "score": float(score)
                })
            
            # 상위 20개 이상치만 유지
            if len(anomalies) > 20:
                anomalies = sorted(anomalies, key=lambda x: x["score"], reverse=True)[:20]
            
            # 자기상관 검사
            acf = None
            if len(remainder_data) > 10:
                try:
                    lag = min(20, len(remainder_data) // 5)
                    acf_values = [1.0]  # 첫 번째 값은 항상 1
                    for k in range(1, lag + 1):
                        # 수동으로 자기상관 계산
                        acf_value = np.corrcoef(remainder_data[:-k], remainder_data[k:])[0, 1]
                        acf_values.append(acf_value)
                    acf = acf_values
                except:
                    acf = None
            
            # 잔차의 군집성 분석 (연속적인 잔차의 부호 변화 빈도)
            sign_changes = 0
            for i in range(1, len(remainder_data)):
                if (remainder_data[i] > 0 and remainder_data[i-1] < 0) or (remainder_data[i] < 0 and remainder_data[i-1] > 0):
                    sign_changes += 1
            
            expected_changes = (len(remainder_data) - 1) / 2  # 완전 랜덤 시 예상되는 부호 변화 수
            randomness_ratio = sign_changes / expected_changes if expected_changes > 0 else 0
            
            # 잔차 분석 결과 생성
            remainder_analysis = {
                "distribution": {
                    "mean": float(mean),
                    "std": float(std),
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "is_normal": bool(is_normal),
                    "shapiro_pvalue": float(shapiro_test.pvalue)
                },
                "noise_level": float(std / np.mean(np.abs(remainder_data)) if np.mean(np.abs(remainder_data)) > 0 else 0),
                "randomness": {
                    "sign_changes": int(sign_changes),
                    "expected_changes": float(expected_changes),
                    "randomness_ratio": float(randomness_ratio),
                    "is_random": randomness_ratio > 0.8 and randomness_ratio < 1.2  # 0.8에서 1.2 사이는 '무작위'로 간주
                },
                "autocorrelation": acf[:10] if acf and len(acf) > 10 else acf,  # 처음 10개 값만 표시
                "anomalies": anomalies,
                "strength": float(decomposition_data.get("stats", {}).get("remainder_strength", 0)),
                "llm_analysis": content
            }
        else:
            remainder_analysis = {
                "status": "error",
                "message": "잔차 데이터를 분석할 수 없습니다.",
                "llm_analysis": content
            }
        
        # 에이전트 이름으로 메시지 추가
        ai_message_with_name = AIMessage(content=content, name="remainder_analyzer")
        
        return {
            "messages": messages[:-1] + [ai_message_with_name],  # 마지막 메시지만 이름 추가
            "remainder_analysis": [remainder_analysis]
        }
    
    return remainder_analyzer_agent