"""
Supervisor 에이전트 - 개선된 시각적 추론 기반 버전

전체 병렬 분석 워크플로우를 조정하고 최종 결과를 생성하는 에이전트입니다.
"""

import json
from typing import Dict, Any, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from parallel_sentinel_v2.graph.workflow import TimeSeriesState
from parallel_sentinel_v2.tools.transformation import get_time_series_decomposition


def create_supervisor_agent(llm: BaseChatModel):
    """
    전체 워크플로우를 조정하는 Supervisor 에이전트 생성
    
    Args:
        llm: 에이전트가 사용할 언어 모델
        
    Returns:
        supervisor_agent 함수
    """
    # Supervisor 시스템 프롬프트 정의 - 시각적 추론 결과 통합 강조
    system_template = """당신은 시계열 데이터 분석을 위한 다중 에이전트 시스템의 감독자입니다.
    당신의 역할은 분석 워크플로우를 조정하고 각 전문 에이전트의 결과를 종합하여 최종 결론을 도출하는 것입니다.
    
    워크플로우에는 다음과 같은 전문 에이전트들이 있습니다:
    1. 추세 분석가(Trend Analyzer) - 시계열의 추세(trend) 성분을 시각적으로 분석
    2. 계절성 분석가(Seasonality Analyzer) - 시계열의 계절성(seasonality) 성분을 시각적으로 분석
    3. 잔차 분석가(Remainder Analyzer) - 시계열의 잔차(remainder) 성분을 시각적으로 분석
    
    각 전문 에이전트는 시각적 추론을 통해 시계열 데이터의 이상치를 다음 유형으로 분류했습니다:
    - PersistentLevelShiftUp: 데이터가 더 높은 값으로 이동하여 원래 기준선으로 돌아가지 않고 일관되게 유지됨
    - PersistentLevelShiftDown: 데이터가 더 낮은 값으로 이동하여 원래 기준선으로 돌아가지 않고 일관되게 유지됨
    - TransientLevelShiftUp: 데이터가 일시적으로 더 높은 값으로 이동했다가 원래 기준선으로 돌아옴 (최소 5개 데이터 포인트 동안 유지)
    - TransientLevelShiftDown: 데이터가 일시적으로 더 낮은 값으로 이동했다가 원래 기준선으로 돌아옴 (최소 5개 데이터 포인트 동안 유지)
    - SingleSpike: 데이터 값이 급격히 상승했다가 즉시 기준선으로 돌아옴
    - SingleDip: 데이터 값이 급격히 하락했다가 즉시 기준선으로 돌아옴
    - MultipleSpikes: 여러 번의 급격한 데이터 값 상승, 각각 기준선으로 돌아옴
    - MultipleDips: 여러 번의 급격한 데이터 값 하락, 각각 기준선으로 돌아옴
    
    현재 상태에 따라 다음을 결정해야 합니다:
    1. 시계열 데이터 분해를 초기화
    2. 각 전문 에이전트의 시각적 분석 결과 취합
    3. 최종 종합 분석 제공
    
    모든 에이전트로부터 분석 결과를 받았을 때, 최종 종합 분석에 다음 내용을 포함하세요:
    1. 시계열 특성 요약
    2. 각 구성 요소(추세, 계절성, 잔차)별 중요 발견사항
    3. 식별된 이상치와 그 유형
    4. 각 이상치의 시계열 내 위치(인덱스) 및 중요성
    5. 모니터링이나 추가 분석을 위한 권장사항
    
    에이전트들이 식별한 이상치 유형과 위치를 종합하여 최종 이상치 목록을 명확하게 제시하세요.
    여러 에이전트가 같은 인덱스를 이상치로 식별한 경우, 그 신뢰도가 더 높음을 강조하세요.
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
            message.append(f"- 추세 강도: {latest_trend.get('strength', 0):.4f}")
            if 'anomalies' in latest_trend and latest_trend['anomalies']:
                message.append(f"- 추세 내 이상치: {len(latest_trend['anomalies'])}개")
                for i, anomaly in enumerate(latest_trend['anomalies'][:3]):  # 최대 3개만 표시
                    message.append(f"  * {anomaly.get('type', 'Unknown')}: 인덱스 {anomaly.get('index', 'N/A')}")

        if has_seasonality:
            message.append("\n최신 계절성 분석 요약:")
            latest_seasonality = state["seasonality_analysis"][-1]
            message.append(f"- 계절성 주기: {latest_seasonality.get('period', 'N/A')}")
            message.append(f"- 계절성 강도: {latest_seasonality.get('strength', 0):.4f}")
            if 'anomalies' in latest_seasonality and latest_seasonality['anomalies']:
                message.append(f"- 계절성 내 이상치: {len(latest_seasonality['anomalies'])}개")
                for i, anomaly in enumerate(latest_seasonality['anomalies'][:3]):
                    message.append(f"  * {anomaly.get('type', 'Unknown')}: 인덱스 {anomaly.get('index', 'N/A')}")

        if has_remainder:
            message.append("\n최신 잔차 분석 요약:")
            latest_remainder = state["remainder_analysis"][-1]
            message.append(f"- 잔차 강도: {latest_remainder.get('strength', 0):.4f}")
            message.append(f"- 정규 분포 여부: {'예' if latest_remainder.get('is_normal_distribution', False) else '아니오'}")
            if 'anomalies' in latest_remainder and latest_remainder['anomalies']:
                message.append(f"- 잔차 내 이상치: {len(latest_remainder['anomalies'])}개")
                for i, anomaly in enumerate(latest_remainder['anomalies'][:3]):
                    message.append(f"  * {anomaly.get('type', 'Unknown')}: 인덱스 {anomaly.get('index', 'N/A')}")

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
            decomposition_result_str = get_time_series_decomposition({"data": state["ts_data"]})
            decomposition_result = json.loads(decomposition_result_str)
            
            # 상태 업데이트에 분해 데이터 추가
            updates["decomposition_data"] = {
                "trend": decomposition_result.get("components", {}).get("trend", []),
                "seasonality": decomposition_result.get("components", {}).get("seasonal", []),
                "remainder": decomposition_result.get("components", {}).get("residual", []),
                "method": "stl",
                "period": decomposition_result.get("period", 0),
                "model": "additive",
                "stats": decomposition_result.get("strengths", {}),
                "visualization_base64": decomposition_result.get("visualization_base64", "")
            }
            
            # 분해 결과에 대한 메시지 생성
            decomposition_message = f"""시계열 데이터 분해를 완료했습니다.
            
            방법: STL
            주기: {decomposition_result.get('period', 'N/A')}
            모델: additive
            
            추세 강도: {decomposition_result.get('strengths', {}).get('trend_strength', 'N/A')}
            계절성 강도: {decomposition_result.get('strengths', {}).get('seasonal_strength', 'N/A')}
            잔차 강도: {decomposition_result.get('strengths', {}).get('residual_strength', 'N/A')}
            
            시각화된 분해 결과를 바탕으로 각 에이전트에게 시각적 분석을 요청하겠습니다.
            """
            
            # 분해 결과 메시지를 메시지 히스토리에 추가
            messages.append(AIMessage(content=decomposition_message, name="supervisor"))
            updates["messages"] = messages
            
            print("Supervisor: 시계열 분해 완료. 각 에이전트에게 시각적 분석 요청.")
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
            
            # 모든 이상치 결합
            all_anomalies = []
            
            # 각 에이전트의 이상치 결과 취합
            latest_trend = state["trend_analysis"][-1]
            latest_seasonality = state["seasonality_analysis"][-1]
            latest_remainder = state["remainder_analysis"][-1]
            
            # 이상치 목록을 각 구성 요소별로 추가
            if 'anomalies' in latest_trend:
                all_anomalies.extend([{**a, 'source': 'trend'} for a in latest_trend['anomalies']])
            if 'anomalies' in latest_seasonality:
                all_anomalies.extend([{**a, 'source': 'seasonality'} for a in latest_seasonality['anomalies']])
            if 'anomalies' in latest_remainder:
                all_anomalies.extend([{**a, 'source': 'remainder'} for a in latest_remainder['anomalies']])
            
            # 같은 인덱스에 대해 여러 에이전트가 탐지한 경우 신뢰도 증가
            index_to_anomalies = {}
            for anomaly in all_anomalies:
                idx = anomaly['index']
                if idx not in index_to_anomalies:
                    index_to_anomalies[idx] = anomaly
                else:
                    # 이미 다른 에이전트가 식별한 이상치 - 신뢰도 증가
                    prev_confidence = index_to_anomalies[idx].get('confidence', 0)
                    new_confidence = anomaly.get('confidence', 0)
                    
                    # 더 높은 신뢰도를 가진 이상치 유형으로 업데이트
                    if new_confidence > prev_confidence:
                        index_to_anomalies[idx] = {**anomaly}
                    
                    # 신뢰도 증가 (최대 1.5까지)
                    index_to_anomalies[idx]['confidence'] = min(1.5, prev_confidence + 0.2)
                    
                    # 감지 소스 정보 업데이트
                    sources = set(index_to_anomalies[idx].get('sources', [index_to_anomalies[idx]['source']]))
                    sources.add(anomaly['source'])
                    index_to_anomalies[idx]['sources'] = list(sources)
            
            # 취합된 이상치 목록
            combined_anomalies = list(index_to_anomalies.values())
            
            # 신뢰도 기준으로 정렬
            combined_anomalies = sorted(combined_anomalies, key=lambda x: x.get('confidence', 0), reverse=True)
            
            # 최종 분석 구조화
            final_analysis = {
                "summary": content,
                "trend_analysis": latest_trend,
                "seasonality_analysis": latest_seasonality,
                "remainder_analysis": latest_remainder,
                "combined_anomalies": combined_anomalies,
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