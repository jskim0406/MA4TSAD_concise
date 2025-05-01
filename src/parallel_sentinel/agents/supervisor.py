"""
Supervisor 에이전트

전체 병렬 분석 워크플로우를 조정하고 최종 결과를 생성하는 에이전트입니다.
"""

import json
from typing import Dict, Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from parallel_sentinel.graph.workflow import TimeSeriesState
from parallel_sentinel.tools.decomposition import decompose_time_series


def create_supervisor_agent(llm: BaseChatModel):
    """
    전체 워크플로우를 조정하는 Supervisor 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        
    Returns:
        supervisor_agent 함수
    """
    # Supervisor 시스템 프롬프트 정의
    system_template = """당신은 시계열 데이터 분석을 위한 다중 에이전트 시스템의 감독자입니다.
    당신의 역할은 분석 워크플로우를 조정하고 다음 단계에 대한 결정을 내리는 것입니다.
    
    워크플로우에는 다음과 같은 전문 에이전트들이 있습니다:
    1. 추세 분석가(Trend Analyzer) - 시계열의 추세(trend) 성분을 분석
    2. 계절성 분석가(Seasonality Analyzer) - 시계열의 계절성(seasonality) 성분을 분석
    3. 잔차 분석가(Remainder Analyzer) - 시계열의 잔차(remainder) 성분을 분석
    
    현재 상태에 따라 다음을 결정해야 합니다:
    1. 시계열 데이터 분해를 초기화
    2. 각 전문 에이전트의 분석 결과 취합
    3. 최종 종합 분석 제공
    
    이것이 첫 단계라면, 시계열 데이터를 분해하고 세 에이전트 모두에게 전달해야 합니다.
    모든 에이전트로부터 분석 결과를 받았다면, 최종 종합 분석을 제공해야 합니다.
    
    시계열 데이터는 시간에 따른 연속적인 측정치를 나타냅니다.
    이상치는 급격한 스파이크/하락, 패턴 변화 또는 다른 비정상적인 행동일 수 있습니다.
    
    최종 분석을 제공할 때는 다음과 같은 구조로 작성하세요:
    1. 시계열 특성 요약
    2. 식별된 추세 패턴
    3. 식별된 계절성 패턴
    4. 식별된 이상치(있는 경우)와 그 잠재적 중요성
    5. 모니터링이나 추가 분석을 위한 권장사항
    """

    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])

    def format_state_for_supervisor(state: TimeSeriesState) -> str:
        """Supervisor를 위한 현재 상태 정보 포맷팅"""
        has_trend = len(state["trend_analysis"]) > 0
        has_seasonality = len(state["seasonality_analysis"]) > 0
        has_remainder = len(state["remainder_analysis"]) > 0
        has_decomposition = bool(state.get("decomposition_data"))

        message = [
            f"현재 분석 상태:",
            f"- 시계열 데이터 길이: {len(state['ts_data'])}",
            f"- 데이터 분해 완료: {'예' if has_decomposition else '아니오'}",
            f"- 추세 분석 결과 수신: {'예' if has_trend else '아니오'}",
            f"- 계절성 분석 결과 수신: {'예' if has_seasonality else '아니오'}",
            f"- 잔차 분석 결과 수신: {'예' if has_remainder else '아니오'}"
        ]

        # 이것이 첫 번째 실행이고 아직 분해가 수행되지 않았다면
        if not has_decomposition and not has_trend and not has_seasonality and not has_remainder:
            message.append("\n시계열 데이터를 분해하고 각 에이전트에게 분석을 요청해야 합니다.")
            return "\n".join(message)

        # 분석 결과 요약 추가
        if has_trend:
            message.append("\n최신 추세 분석 요약:")
            # 마지막 분석 결과만 요약 표시
            latest_trend = state["trend_analysis"][-1]
            message.append(f"- 추세 방향: {latest_trend.get('direction', '알 수 없음')}")
            if 'slope' in latest_trend:
                message.append(f"- 기울기: {latest_trend['slope']:.4f}")
            if 'strength' in latest_trend:
                message.append(f"- 추세 강도: {latest_trend['strength']:.4f}")
            if 'anomalies' in latest_trend:
                message.append(f"- 추세 내 이상치: {len(latest_trend['anomalies'])}개")

        if has_seasonality:
            message.append("\n최신 계절성 분석 요약:")
            latest_seasonality = state["seasonality_analysis"][-1]
            if 'period' in latest_seasonality:
                message.append(f"- 계절성 주기: {latest_seasonality['period']}")
            if 'strength' in latest_seasonality:
                message.append(f"- 계절성 강도: {latest_seasonality['strength']:.4f}")
            if 'anomalies' in latest_seasonality:
                message.append(f"- 계절성 내 이상치: {len(latest_seasonality['anomalies'])}개")

        if has_remainder:
            message.append("\n최신 잔차 분석 요약:")
            latest_remainder = state["remainder_analysis"][-1]
            if 'anomalies' in latest_remainder:
                message.append(f"- 발견된 이상치: {len(latest_remainder['anomalies'])}개")
                if latest_remainder['anomalies']:
                    top_anomalies = latest_remainder['anomalies'][:3]
                    message.append(f"- 상위 이상치 인덱스: {[a['index'] for a in top_anomalies]}")
            if 'noise_level' in latest_remainder:
                message.append(f"- 노이즈 수준: {latest_remainder['noise_level']:.4f}")

        # 모든 분석이 완료되었는지 확인
        if has_trend and has_seasonality and has_remainder:
            message.append("\n모든 분석 결과가 수신되었습니다. 최종 종합 분석을 제공해주세요.")
        else:
            message.append("\n추가 분석 결과를 기다리는 중...")

        return "\n".join(message)

    def supervisor_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        현재 상태를 처리하고, 필요에 따라 시계열 분해를 수행하거나 결과를 집계합니다.
        
        Args:
            state: 현재 워크플로우 상태
            
        Returns:
            Dict: 상태 업데이트 (messages, decomposition_data, final_analysis)
        """
        # Supervisor를 위한 현재 상태 포맷팅
        input_text = format_state_for_supervisor(state)
        
        # 현재 메시지 복사
        messages = list(state["messages"])
        
        # 이것이 첫 번째 실행이고 분해가 아직 수행되지 않았다면 시계열 분해 수행
        has_decomposition = bool(state.get("decomposition_data"))
        has_trend = len(state["trend_analysis"]) > 0
        has_seasonality = len(state["seasonality_analysis"]) > 0
        has_remainder = len(state["remainder_analysis"]) > 0
        
        updates: Dict[str, Any] = {}
        
        if not has_decomposition and not has_trend and not has_seasonality and not has_remainder:
            print("Supervisor: 시계열 데이터 분해 수행 중...")
            
            # 분해 도구 호출
            decomposition_result_str = decompose_time_series.invoke({"data": state["ts_data"]})
            decomposition_result = json.loads(decomposition_result_str)
            
            # 상태 업데이트에 분해 데이터 추가
            updates["decomposition_data"] = decomposition_result
            
            # 분해 결과에 대한 메시지 생성
            decomposition_message = f"""시계열 데이터 분해를 완료했습니다.
            
            방법: {decomposition_result.get('method', 'stl')}
            주기: {decomposition_result.get('period', 'N/A')}
            모델: {decomposition_result.get('model', 'additive')}
            
            추세 강도: {decomposition_result.get('stats', {}).get('trend_strength', 'N/A')}
            계절성 강도: {decomposition_result.get('stats', {}).get('seasonality_strength', 'N/A')}
            잔차 강도: {decomposition_result.get('stats', {}).get('remainder_strength', 'N/A')}
            
            각 에이전트에게 분석을 요청하겠습니다.
            """
            
            # 분해 결과 메시지를 메시지 히스토리에 추가
            messages.append(AIMessage(content=decomposition_message, name="supervisor"))
            updates["messages"] = messages
            
            print("Supervisor: 시계열 분해 완료. 각 에이전트에게 분석 요청.")
            return updates
        
        # 모든 분석이 완료되었다면 최종 분석 생성
        if has_trend and has_seasonality and has_remainder and state["final_analysis"] is None:
            print("Supervisor: 모든 분석 결과 수신. 최종 분석 생성 중...")
            
            # LLM 호출하여 최종 분석 생성
            prompt_result = supervisor_prompt.invoke({"input": input_text})
            llm_result = llm.invoke(prompt_result)
            content = llm_result.content
            
            # 최종 분석 메시지를 메시지 히스토리에 추가
            messages.append(AIMessage(content=content, name="supervisor"))
            updates["messages"] = messages
            
            # 최종 분석 구조화
            latest_trend = state["trend_analysis"][-1]
            latest_seasonality = state["seasonality_analysis"][-1]
            latest_remainder = state["remainder_analysis"][-1]
            
            # 모든 이상치 결합
            all_anomalies = []
            if 'anomalies' in latest_trend:
                all_anomalies.extend([{**a, 'source': 'trend'} for a in latest_trend['anomalies']])
            if 'anomalies' in latest_seasonality:
                all_anomalies.extend([{**a, 'source': 'seasonality'} for a in latest_seasonality['anomalies']])
            if 'anomalies' in latest_remainder:
                all_anomalies.extend([{**a, 'source': 'remainder'} for a in latest_remainder['anomalies']])
            
            # 중복 이상치 제거 (같은 인덱스의 이상치는 가장 높은 점수만 유지)
            unique_anomalies = {}
            for anomaly in all_anomalies:
                idx = anomaly['index']
                if idx not in unique_anomalies or anomaly.get('score', 0) > unique_anomalies[idx].get('score', 0):
                    unique_anomalies[idx] = anomaly
            
            final_analysis = {
                "summary": content,
                "trend_analysis": latest_trend,
                "seasonality_analysis": latest_seasonality,
                "remainder_analysis": latest_remainder,
                "combined_anomalies": list(unique_anomalies.values()),
                "decomposition_data": state["decomposition_data"]
            }
            
            updates["final_analysis"] = final_analysis
            print("Supervisor: 최종 분석 생성 완료.")
        else:
            # 진행 상황 업데이트 메시지
            prompt_result = supervisor_prompt.invoke({"input": input_text})
            llm_result = llm.invoke(prompt_result)
            content = llm_result.content
            
            messages.append(AIMessage(content=content, name="supervisor"))
            updates["messages"] = messages
            
            print("Supervisor: 진행 상황 업데이트 완료. 추가 분석 결과 대기 중...")
        
        return updates
    
    return supervisor_agent