"""
병렬 시계열 분석 에이전트 시스템 - run.py와의 통합을 위한 버전

이 모듈은 시계열 데이터의 이상치 탐지를 위한 병렬 다중 에이전트 시스템을 제공합니다.
run.py와 통합되어 사용되도록 research_agents 함수를 노출합니다.
"""

import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

# 환경 변수 로드
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# LangChain 및 관련 모듈 임포트
try:
    from langsmith import traceable, trace
    from langsmith.run_helpers import get_current_run_tree
    from langchain.globals import set_debug
    set_debug(True)
    print("langchain 디버깅 모드 활성화")
    from langchain_core.messages import AIMessage, HumanMessage
    
    # 멀티모달 LLM 지원을 위한 임포트
    from langchain_google_vertexai import ChatVertexAI
    from langchain_openai import ChatOpenAI
    
    # 병렬 워크플로우 지원을 위한 LangGraph 임포트
    from langgraph.graph import StateGraph, START, END
    
except ImportError as e:
    print(f"오류: 필요한 라이브러리를 임포트할 수 없습니다: {e}")
    print("라이브러리가 올바르게 설치되었는지 확인하세요.")
    raise

# 필요한 도구 및 에이전트 모듈 임포트
try:
    # 시각화 및 통계 분석 도구
    from src.parallel_sentinel_v2.tools import (
        visualization_tools, statistics_tools, 
        frequency_tools, transformation_tools,
        all_tools
    )
    
    # 에이전트 생성 함수
    from src.parallel_sentinel_v2.agents import (
        create_supervisor_agent, create_original_time_series_analyzer_agent,
        create_trend_analyzer_agent, create_seasonality_analyzer_agent,
        create_remainder_analyzer_agent
    )
    
    # 워크플로우 생성 및 실행 함수
    from src.parallel_sentinel_v2.graph import create_workflow, run_workflow
    from src.parallel_sentinel_v2.utils.llm_utils import init_llm
    from src.parallel_sentinel_v2.utils.ts_utils import quantize_time_series, get_quantization_info
    from src.parallel_sentinel_v2.utils.parser import extract_json_from_text, parse_final_analysis
    
except ImportError as e:
    print(f"오류: 병렬 시계열 분석 모듈을 임포트할 수 없습니다: {e}")
    print("src.parallel_sentinel_v2 패키지가 올바르게 설치되었는지 확인하세요.")
    raise

# LangSmith 추적 설정 (환경 변수에 의해 제어됨)
if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
    print("LangSmith 추적이 활성화되었습니다.")
else:
    # 추적을 사용하지 않을 경우 빈 데코레이터 제공
    if not hasattr(globals(), 'traceable'):
        def traceable(func=None, **kwargs):
            """추적을 사용하지 않을 때의 더미 데코레이터"""
            if func:
                return func
            return lambda f: f

# gRPC 및 absl 로깅 억제 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# .env 설정에서 양자화 설정 로드
DEFAULT_QUANTIZE_RANGE = int(os.getenv("QUANTIZE_RANGE", "1"))
DEFAULT_QUANTIZE_METHOD = os.getenv("QUANTIZE_METHOD", "mean")


def parse_ts_data(ts_infer_data: str) -> List[float]:
    """
    run.py에서 전달받은 시계열 데이터 문자열을 파싱하여 숫자 리스트로 변환
    
    Args:
        ts_infer_data: "인덱스 값" 형식의 줄바꿈으로 구분된 문자열
        
    Returns:
        List[float]: 시계열 값 리스트
    """
    try:
        # 줄바꿈으로 문자열 분할
        lines = ts_infer_data.strip().split('\n')
        values = []
        
        # 각 줄에서 값 추출 (형식: "인덱스 값")
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:  # 최소한 인덱스와 값이 있어야 함
                try:
                    value = float(parts[1])  # 두 번째 항목이 값
                    values.append(value)
                except (ValueError, IndexError):
                    print(f"Warning: 잘못된 데이터 형식 - {line}")
        
        return values
    except Exception as e:
        print(f"시계열 데이터 파싱 오류: {e}")
        return []


def init_custom_llm(model_name=None):
    """
    .env 파일 설정을 사용하여 LLM을 초기화합니다.
    
    Args:
        model_name: (선택사항) 사용할 모델 이름, 지정하지 않으면 환경 변수에서 가져옴
        
    Returns:
        초기화된 LLM 인스턴스
    """
    # 환경 변수에서 설정 로드
    # 먼저 모델 이름으로 어떤 제공자를 사용할지 결정
    if model_name is None:
        # 환경 변수에서 기본 모델 결정
        google_model = os.getenv("GOOGLE_MODEL_NAME", "")
        openai_model = os.getenv("OPENAI_MODEL_NAME", "")
        
        if google_model:
            model_name = google_model
            provider = "google"
        elif openai_model:
            model_name = openai_model
            provider = "openai"
        else:
            # 기본값: Gemini
            model_name = "gemini-2.5-flash-preview-04-17"
            provider = "google"
    else:
        # 모델 이름으로 제공자 결정
        if "gpt" in model_name.lower():
            provider = "openai"
        elif "claude" in model_name.lower():
            provider = "anthropic"
        else:
            provider = "google"  # 기본적으로 Gemini 모델 가정
    
    print(f"[init_llm] 모델 제공자: {provider}, 모델: {model_name}")
    
    try:
        if provider == "google":
            from langchain_google_vertexai import ChatVertexAI
            
            llm = ChatVertexAI(
                model_name=model_name,
                temperature=float(os.getenv("GOOGLE_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("GOOGLE_MAX_OUTPUT_TOKENS", "8192")),
                project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
                location=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"),
            )
            
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(
                model=model_name,
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096")),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                timeout=None,
                max_retries=3,
            )
            
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            
            llm = ChatAnthropic(
                model=model_name,
                temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.3")),
                max_tokens=int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "4096")),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            )
            
        else:
            raise ValueError(f"지원되지 않는 모델 제공자: {provider}")
        
        return llm
    
    except Exception as e:
        print(f"LLM 초기화 오류: {e}")
        raise


def format_analysis_result(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    병렬 에이전트 시스템의 최종 분석 결과를 run.py가 기대하는 형식으로 변환
    
    Args:
        final_state: 병렬 워크플로우의 최종 상태
        
    Returns:
        Dict: run.py와 호환되는 형식의 결과
    """
    # 최종 분석 결과가 없으면 빈 결과 반환
    if not final_state or not final_state.get("final_analysis"):
        return {
            "is_anomaly": False,
            "anomalies": [],
            "briefExplanation": {
                "step1_global": "분석 불가",
                "step2_local": "유효한 결과 없음",
                "step3_reassess": "이상치 미감지"
            },
            "anomaly_type": "no",
            "reason_for_anomaly_type": "no",
            "alarm_level": "no",
            "reason_for_alarm_level": "no"
        }
    
    # 최종 분석 데이터 추출
    final_analysis = final_state["final_analysis"]
    combined_anomalies = final_analysis.get("combined_anomalies", [])
    recommendations = final_analysis.get("recommendations", [])
    
    # 이상치 존재 여부 확인
    is_anomaly = len(combined_anomalies) > 0
    
    # 이상치 인덱스 목록 생성
    anomaly_indices = []
    anomaly_by_index = {}
    
    # 각 인덱스별 이상치 집계 및 신뢰도 계산
    for anomaly in combined_anomalies:
        idx = anomaly.get("index")
        if isinstance(idx, (int, float)):
            idx = int(idx)
            if idx not in anomaly_indices:
                anomaly_indices.append(idx)
            
            # 인덱스별 이상치 정보 집계
            if idx not in anomaly_by_index:
                anomaly_by_index[idx] = {
                    "occurrences": 1,
                    "types": {anomaly.get("type", "Unknown"): 1},
                    "sources": [anomaly.get("source", "Unknown")],
                    "confidence": anomaly.get("confidence", 0.5)
                }
            else:
                anomaly_by_index[idx]["occurrences"] += 1
                anomaly_type = anomaly.get("type", "Unknown")
                if anomaly_type in anomaly_by_index[idx]["types"]:
                    anomaly_by_index[idx]["types"][anomaly_type] += 1
                else:
                    anomaly_by_index[idx]["types"][anomaly_type] = 1
                
                anomaly_by_index[idx]["sources"].append(anomaly.get("source", "Unknown"))
                
                # 신뢰도 업데이트 (여러 소스에서 탐지될 경우 높은 신뢰도 적용)
                if "confidence" in anomaly and anomaly["confidence"] > anomaly_by_index[idx]["confidence"]:
                    anomaly_by_index[idx]["confidence"] = anomaly["confidence"]
                else:
                    # 소스 수에 따른 신뢰도 증가 (최대 1.5)
                    anomaly_by_index[idx]["confidence"] = min(1.5, anomaly_by_index[idx]["confidence"] + 0.1)
    
    # 가장 중요한 이상치 선택 (여러 소스에서 탐지된 인덱스 우선, 같은 경우 인덱스 번호 순)
    most_important_anomaly = None
    if anomaly_by_index:
        # 탐지 횟수와 신뢰도를 기준으로 정렬
        sorted_indices = sorted(
            anomaly_by_index.keys(),
            key=lambda x: (anomaly_by_index[x]["occurrences"], anomaly_by_index[x]["confidence"]),
            reverse=True
        )
        
        most_important_idx = sorted_indices[0]
        most_important_anomaly = anomaly_by_index[most_important_idx]
        
        # 가장 많이 탐지된 유형 결정
        dominant_type = max(
            most_important_anomaly["types"].items(),
            key=lambda x: x[1]
        )[0]
    
    # 이상치 유형 및 이유 결정
    anomaly_type = "no"
    reason_for_anomaly = "no"
    
    if most_important_anomaly:
        # 가장 많이 탐지된 유형 선택
        anomaly_type = max(most_important_anomaly["types"].items(), key=lambda x: x[1])[0]
        
        # 이유 생성
        sources_str = ", ".join(sorted(set(most_important_anomaly["sources"])))
        reason_for_anomaly = f"이 이상치는 {most_important_anomaly['occurrences']}개의 분석({sources_str})에서 감지되었으며, 주로 {anomaly_type} 유형으로 분류되었습니다."
    
    # 요약 파싱
    summary = final_analysis.get("summary", "")
    
    # 3단계 분석 설명 생성
    step1_global = ""
    step2_local = ""
    step3_reassess = ""
    
    # 요약에서 추세, 계절성, 최종 결론 관련 부분 추출
    if "추세 분석 결과" in summary:
        trend_start = summary.find("추세 분석 결과")
        trend_end = summary.find(".", trend_start) + 1
        if trend_end > trend_start:
            step1_global = summary[trend_start:trend_end].strip()
    
    if "계절성 분석 결과" in summary:
        seasonality_start = summary.find("계절성 분석 결과")
        seasonality_end = summary.find(".", seasonality_start) + 1
        if seasonality_end > seasonality_start:
            step2_local = summary[seasonality_start:seasonality_end].strip()
    
    if "종합적으로 탐지된 이상치는 다음과 같습니다" in summary:
        conclusion_start = summary.find("종합적으로 탐지된 이상치는 다음과 같습니다")
        step3_reassess = summary[conclusion_start:].strip()
        if len(step3_reassess) > 300:
            step3_reassess = step3_reassess[:297] + "..."
    
    # 설명 부분이 비어있으면 기본값 설정
    if not step1_global:
        step1_global = "전체 시계열 데이터의 추세를 분석했습니다."
    if not step2_local:
        step2_local = "지역적인 패턴과 이상치를 확인했습니다."
    if not step3_reassess:
        if is_anomaly:
            step3_reassess = f"총 {len(anomaly_indices)}개의 이상치가 감지되었습니다: {', '.join(map(str, sorted(anomaly_indices)))}"
        else:
            step3_reassess = "이상치가 감지되지 않았습니다."
    
    # 권장사항이 있으면 step3에 추가
    if recommendations and len(recommendations) > 0:
        rec_text = f" 권장사항: {recommendations[0]}"
        if len(step3_reassess) + len(rec_text) <= 300:
            step3_reassess += rec_text
    
    # 경보 수준 결정
    alarm_level = "no"
    reason_for_alarm = "no"
    
    if is_anomaly:
        # 이상치 유형과 심각도에 따른 경보 수준 설정
        severe_types = ["PersistentLevelShiftUp", "PersistentLevelShiftDown"]
        moderate_types = ["TransientLevelShiftUp", "TransientLevelShiftDown", "MultipleSpikes", "MultipleDips"]
        
        if anomaly_type in severe_types:
            alarm_level = "Urgent/Error"
            reason_for_alarm = "지속적인 수준 변화는 시스템에 중대한 영향을 미칠 수 있습니다."
        elif anomaly_type in moderate_types:
            alarm_level = "Important"
            reason_for_alarm = "일시적인 수준 변화나 다중 스파이크는 시스템 성능에 영향을 줄 수 있습니다."
        else:  # SingleSpike, SingleDip 등
            alarm_level = "Warning"
            reason_for_alarm = "단일 이상치는 일시적인 변동을 나타내지만 모니터링이 필요합니다."
        
        # 여러 소스에서 감지된 경우 경보 수준 상향 조정
        if most_important_anomaly and most_important_anomaly["occurrences"] >= 3:
            if alarm_level == "Warning":
                alarm_level = "Important"
                reason_for_alarm = "여러 분석에서 감지된 이상치로 중요도가 높습니다."
    
    # run.py와 호환되는 구조로 결과 포맷팅
    formatted_result = {
        "is_anomaly": is_anomaly,
        "anomalies": sorted(anomaly_indices),  # 인덱스 정렬
        "briefExplanation": {
            "step1_global": step1_global,
            "step2_local": step2_local,
            "step3_reassess": step3_reassess
        },
        "anomaly_type": anomaly_type,
        "reason_for_anomaly_type": reason_for_anomaly,
        "alarm_level": alarm_level,
        "reason_for_alarm_level": reason_for_alarm
    }
    
    return formatted_result


@traceable(name="research_agents", run_type="chain")
def research_agents(args, ts_infer_data: str) -> Dict[str, Any]:
    """
    병렬 다중 에이전트 시스템을 사용하여 시계열 데이터를 분석하고 이상치를 탐지
    
    Args:
        args: run.py에서 전달받은 명령줄 인자
        ts_infer_data: "인덱스 값" 형식의 줄바꿈으로 구분된 시계열 데이터 문자열
        
    Returns:
        Dict[str, Any]: 이상치 탐지 결과를 포함하는 구조화된 응답 (run.py 호환)
        {
            "is_anomaly": bool,  # 이상치 존재 여부
            "anomalies": List[int],  # 이상치 인덱스 목록
            "briefExplanation": {  # 3단계 분석 요약
                "step1_global": str,  # 전역 추세 분석
                "step2_local": str,  # 국소 패턴 분석
                "step3_reassess": str  # 최종 평가
            },
            "anomaly_type": str,  # 주요 이상치 유형
            "reason_for_anomaly_type": str,  # 이상치 유형에 대한 설명
            "alarm_level": str,  # 경보 수준 (Urgent/Error, Important, Warning, no)
            "reason_for_alarm_level": str  # 경보 수준에 대한 설명
        }
    """
    start_time = time.time()
    
    # 시계열 데이터 파싱
    time_series_data = parse_ts_data(ts_infer_data)
    if not time_series_data:
        print("오류: 시계열 데이터를 파싱할 수 없습니다.")
        return {
            "is_anomaly": False,
            "anomalies": [],
            "briefExplanation": {
                "step1_global": "데이터 파싱 오류",
                "step2_local": "유효한 시계열 데이터를 제공해주세요",
                "step3_reassess": "분석을 진행할 수 없습니다"
            },
            "anomaly_type": "no",
            "reason_for_anomaly_type": "no",
            "alarm_level": "no",
            "reason_for_alarm_level": "no"
        }
    
    print(f"[research_agents] 시계열 데이터 파싱 완료: {len(time_series_data)} 데이터 포인트")
    
    # 양자화 설정 (.env 파일에서 로드, args가 있으면 args 우선)
    quantize_range = getattr(args, 'quantize_range', DEFAULT_QUANTIZE_RANGE)
    quantize_method = getattr(args, 'quantize_method', DEFAULT_QUANTIZE_METHOD)
    
    # 데이터 양자화 (데이터가 너무 크면)
    original_length = len(time_series_data)
    if original_length > 1000 and quantize_range > 1:
        time_series_data = quantize_time_series(
            time_series_data, 
            quantize_range=quantize_range, 
            method=quantize_method
        )
        print(f"[research_agents] 시계열 데이터 양자화: {original_length} -> {len(time_series_data)} 포인트")
    
    # 양자화 정보 생성
    quantize_info = get_quantization_info(
        quantize_range, 
        quantize_method, 
        original_length
    )
    
    # LLM 초기화 - .env 파일에서 설정 로드
    try:
        # args에서 model_engine 파라미터 가져오기
        model_name = getattr(args, 'model_engine', None) 
        llm = init_custom_llm(model_name)
        print(f"[research_agents] LLM 초기화 완료: {llm}")
    except Exception as e:
        print(f"[research_agents] LLM 초기화 오류: {e}")
        return {
            "is_anomaly": False,
            "anomalies": [],
            "briefExplanation": {
                "step1_global": "LLM 초기화 오류",
                "step2_local": f"오류 메시지: {str(e)}",
                "step3_reassess": "분석을 진행할 수 없습니다"
            },
            "anomaly_type": "no",
            "reason_for_anomaly_type": "no",
            "alarm_level": "no",
            "reason_for_alarm_level": "no"
        }
    
    # 에이전트 생성
    try:
        supervisor = create_supervisor_agent(llm)
        original_ts_analyzer = create_original_time_series_analyzer_agent(llm, all_tools)
        trend_analyzer = create_trend_analyzer_agent(llm, all_tools)
        seasonality_analyzer = create_seasonality_analyzer_agent(llm, all_tools)
        remainder_analyzer = create_remainder_analyzer_agent(llm, all_tools)
        print("[research_agents] 분석 에이전트 생성 완료")
    except Exception as e:
        print(f"[research_agents] 에이전트 생성 오류: {e}")
        return {
            "is_anomaly": False,
            "anomalies": [],
            "briefExplanation": {
                "step1_global": "분석 에이전트 생성 오류",
                "step2_local": f"오류 메시지: {str(e)}",
                "step3_reassess": "분석을 진행할 수 없습니다"
            },
            "anomaly_type": "no",
            "reason_for_anomaly_type": "no",
            "alarm_level": "no",
            "reason_for_alarm_level": "no"
        }
    
    # 워크플로우 생성
    try:
        workflow = create_workflow(
            supervisor_agent=supervisor,
            original_time_series_analyzer_agent=original_ts_analyzer,
            trend_analyzer_agent=trend_analyzer,
            seasonality_analyzer_agent=seasonality_analyzer,
            remainder_analyzer_agent=remainder_analyzer,
            tools=all_tools
        )
        print("[research_agents] 분석 워크플로우 생성 완료")
    except Exception as e:
        print(f"[research_agents] 워크플로우 생성 오류: {e}")
        return {
            "is_anomaly": False,
            "anomalies": [],
            "briefExplanation": {
                "step1_global": "워크플로우 생성 오류",
                "step2_local": f"오류 메시지: {str(e)}",
                "step3_reassess": "분석을 진행할 수 없습니다"
            },
            "anomaly_type": "no",
            "reason_for_anomaly_type": "no",
            "alarm_level": "no",
            "reason_for_alarm_level": "no"
        }
    
    # 워크플로우 실행 설정
    config = {
        "metadata": {
            "quantization": quantize_info
        }
    }
    
    # LangSmith 태그 및 메타데이터 설정₩
    langsmith_tags = ["time_series", "anomaly_detection", "parallel_agents"]
    
    # 메타데이터 강화
    langsmith_metadata = {
        "model": model_name,
        "data_length": original_length,
        "quantize_range": quantize_range,
        "quantize_method": quantize_method,
        "quantized": quantize_info.get("applied", False),
    }
    
    if quantize_info.get("applied", False):
        langsmith_metadata.update({
            "compression_ratio": quantize_info.get("compression_ratio", 1.0),
            "quantized_length": quantize_info.get("quantized_length", len(time_series_data))
        })
        langsmith_tags.append(f"quantized_{quantize_method}")
        langsmith_tags.append(f"range_{quantize_range}")
    
    # 병렬 시계열 분석 워크플로우 실행
    try:
        with trace(
            name="yahoo-A1Benchmark",
            run_type="chain",
            tags=langsmith_tags,
            metadata=langsmith_metadata
        ):
            final_state = run_workflow(workflow, time_series_data, config=config)
            
        print("[research_agents] 시계열 분석 워크플로우 실행 완료")
        
        # 결과 형식 변환 및 반환
        analysis_result = format_analysis_result(final_state)
        
        # 실행 시간 기록
        elapsed_time = time.time() - start_time
        analysis_result["elapsed_time"] = elapsed_time
        
        # LangSmith에 결과 메타데이터 추가
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
            try:
                current_run = get_current_run_tree()
                if current_run:
                    current_run.metadata["execution_time_seconds"] = elapsed_time
                    current_run.metadata["is_anomaly"] = analysis_result["is_anomaly"]
                    if analysis_result["is_anomaly"]:
                        current_run.metadata["anomalies_count"] = len(analysis_result["anomalies"])
                        current_run.metadata["anomaly_type"] = analysis_result["anomaly_type"]
                        current_run.tags.append(f"anomaly_{analysis_result['anomaly_type']}")
                        current_run.tags.append(f"alert_{analysis_result['alarm_level']}")
            except Exception as e:
                # 추적 기능에 오류가 발생해도 메인 기능에 영향을 주지 않도록 처리
                print(f"[research_agents] LangSmith 메타데이터 추가 중 오류 발생(무시됨): {e}")
        return analysis_result
        
    except Exception as e:
        print(f"[research_agents] 워크플로우 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "is_anomaly": False,
            "anomalies": [],
            "briefExplanation": {
                "step1_global": "워크플로우 실행 오류",
                "step2_local": f"오류 메시지: {str(e)}",
                "step3_reassess": "분석을 진행할 수 없습니다"
            },
            "anomaly_type": "no",
            "reason_for_anomaly_type": "no",
            "alarm_level": "no",
            "reason_for_alarm_level": "no"
        }


if __name__ == "__main__":

    # 단독 실행 테스트용 코드
    class TEST:
        model_engine = None  # 환경 변수에서 모델 설정을 가져옴
        quantize_range = 1
        quantize_method = "mean"
    
    # 테스트 데이터
    test_data = "\n".join([f"{i+1} {100 + i * 5}" for i in range(100)])
    test_data += f"\n{101} 2000"  # 이상치 추가
    test_data += "\n" + "\n".join([f"{i+102} {100 + (i+101) * 5}" for i in range(100)])
    
    # 함수 실행
    args = TEST()
    result = research_agents(args, test_data)
    print("분석 결과:")
    print(json.dumps(result, indent=2))