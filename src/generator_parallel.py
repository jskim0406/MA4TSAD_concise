"""
메인 실행 스크립트 - Parallel Sentinel: 병렬 시계열 이상치 탐지 시스템
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 환경 변수 로드
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def setup_langsmith():
    """LangSmith 설정 (설정된 경우)"""
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "parallel-sentinel")

setup_langsmith()

from langsmith import traceable
from langchain.globals import set_debug
from langchain_google_vertexai import ChatVertexAI

# Parallel Sentinel 구성 요소 임포트
from parallel_sentinel.agents.supervisor import create_supervisor_agent
from parallel_sentinel.agents.trend_analyzer import create_trend_analyzer_agent
from parallel_sentinel.agents.seasonality_analyzer import create_seasonality_analyzer_agent
from parallel_sentinel.agents.remainder_analyzer import create_remainder_analyzer_agent

from parallel_sentinel.tools.visualization import (
    ts2img, ts2img_with_anomalies, ts2img_multi_view
)
from parallel_sentinel.tools.statistics import (
    basic_statistics, trend_analysis, seasonality_analysis,
    stationarity_test, anomaly_detection
)
from parallel_sentinel.tools.decomposition import decompose_time_series
from parallel_sentinel.tools.math_tools import get_math_calculator, rolling_window_stats

from parallel_sentinel.graph.workflow import create_workflow, run_workflow
from parallel_sentinel.utils.parser import parse_final_analysis


# Google Cloud 기본 환경 변수
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_NAME = os.getenv("GOOGLE_GEN_MODEL", "gemini-1.5-flash")


@traceable
def main():
    """
    Parallel Sentinel 라이브러리 데모 메인 함수
    """
    parser = argparse.ArgumentParser(description="Parallel Sentinel 시계열 이상 탐지 시스템 실행")
    parser.add_argument("--data", type=str, help="CSV 데이터 파일 경로")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                      help="사용할 LLM 모델 이름")
    parser.add_argument("--debug", action="store_true",
                      help="디버그 모드 활성화")
    parser.add_argument("--output", type=str,
                      help="결과 출력 디렉토리")
    parser.add_argument("--synthetic", action="store_true",
                      help="샘플 대신 합성 데이터 사용")
    args = parser.parse_args()
    
    # 디버그 모드 활성화 (요청된 경우)
    if args.debug:
        set_debug(True)
    
    print("=== Parallel Sentinel - 병렬 시계열 이상 탐지 시스템 시작 ===")
    
    # 사용할 모델 결정
    model_name = args.model or MODEL_NAME
    print(f"LLM 모델 사용: {model_name}")
    
    # 데이터 로드
    if args.data and os.path.exists(args.data):
        print(f"{args.data}에서 시계열 데이터 로드 중...")
        try:
            df = pd.read_csv(args.data)
            col_name = df.select_dtypes(include=[np.number]).columns[0]
            sample = df[col_name].values.tolist()
            print(f"'{col_name}' 열에서 {len(sample)}개 데이터 포인트 로드 완료")
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            print("기본 샘플 데이터 사용")
            sample = generate_sample_data(with_anomalies=True)
    elif args.synthetic:
        print("합성 데이터 생성 중...")
        sample = generate_synthetic_data()
    else:
        print("기본 샘플 데이터 사용")
        sample = generate_sample_data(with_anomalies=True)
    
    # 언어 모델 초기화
    llm = ChatVertexAI(model=model_name)
    
    # 도구 정의
    tools = [
        ts2img,
        ts2img_with_anomalies,
        ts2img_multi_view,
        basic_statistics,
        trend_analysis,
        seasonality_analysis,
        stationarity_test,
        anomaly_detection,
        decompose_time_series,
        rolling_window_stats,
        get_math_calculator(llm)
    ]
    
    # 에이전트 생성
    supervisor = create_supervisor_agent(llm)
    trend_analyzer = create_trend_analyzer_agent(llm, tools)
    seasonality_analyzer = create_seasonality_analyzer_agent(llm, tools)
    remainder_analyzer = create_remainder_analyzer_agent(llm, tools)
    
    # 워크플로우 생성
    workflow = create_workflow(
        supervisor_agent=supervisor,
        trend_analyzer_agent=trend_analyzer,
        seasonality_analyzer_agent=seasonality_analyzer,
        remainder_analyzer_agent=remainder_analyzer,
        tools=tools
    )
    
    # 출력 디렉토리 설정
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent.parent / f"results_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과는 {output_dir}에 저장됩니다")
    
    # 워크플로우 실행
    print(f"\n시계열 데이터 분석 실행 중 (길이: {len(sample)})")
    start_time = datetime.now()

    final_state = run_workflow(workflow, sample)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"분석 완료: {duration:.2f}초 소요")
    
    # 결과 처리 및 표시
    if final_state.get("final_analysis"):
        print("\n===== 최종 분석 =====")
        analysis_text = final_state["final_analysis"]["summary"]
        print(analysis_text)
        
        # 구조화된 접근을 위한 최종 분석 파싱
        structured_analysis = parse_final_analysis(analysis_text)
        
        # 이상치가 감지된 경우 표시
        if structured_analysis.anomalies:
            print("\n===== 감지된 이상치 =====")
            for i, anomaly in enumerate(structured_analysis.anomalies):
                print(f"이상치 세트 {i+1}:")
                print(f"- 인덱스: {anomaly.anomaly_indices}")
                print(f"- 값: {anomaly.anomaly_values[:5]}..." if len(anomaly.anomaly_values) > 5 else f"- 값: {anomaly.anomaly_values}")
                if anomaly.description:
                    print(f"- 설명: {anomaly.description}")
            
            # 이상치 시각화 생성
            all_indices = []
            for anomaly in structured_analysis.anomalies:
                all_indices.extend(anomaly.anomaly_indices)
            
            if all_indices:
                result_str = ts2img_with_anomalies(sample, all_indices, "감지된 이상치")
                try:
                    result = json.loads(result_str)
                    print(f"시각화 저장 경로: {result['image_path']}")
                except:
                    print("시각화 생성 오류")
        
        # 전체 분석 결과 저장
        with open(output_dir / "analysis_results.json", "w", encoding="utf-8") as f:
            json.dump({
                "time_series_length": len(sample),
                "analysis_duration_seconds": duration,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "final_analysis": final_state["final_analysis"],
                "trend_analysis": final_state["trend_analysis"][-1] if final_state["trend_analysis"] else None,
                "seasonality_analysis": final_state["seasonality_analysis"][-1] if final_state["seasonality_analysis"] else None,
                "remainder_analysis": final_state["remainder_analysis"][-1] if final_state["remainder_analysis"] else None,
                "detected_anomalies": [
                    {
                        "indices": anomaly.anomaly_indices,
                        "values": anomaly.anomaly_values,
                        "description": anomaly.description
                    }
                    for anomaly in structured_analysis.anomalies
                ] if structured_analysis.anomalies else []
            }, f, indent=2, ensure_ascii=False)
        print(f"\n결과가 {output_dir / 'analysis_results.json'}에 저장되었습니다")
    else:
        print("최종 분석이 생성되지 않았습니다. 워크플로우 실행을 확인하세요.")


def generate_sample_data(length=300, with_anomalies=True):
    """
    샘플 시계열 데이터 생성
    
    Args:
        length (int): 생성할 데이터 길이
        with_anomalies (bool): 이상치 포함 여부
        
    Returns:
        List[float]: 생성된 시계열 데이터
    """
    np.random.seed(42)  # 재현성을 위한 시드 설정
    
    # 기본 패턴 생성 (추세 + 계절성)
    t = np.arange(length)
    trend = 0.05 * t  # 선형 추세
    seasonality = 10 * np.sin(t * 0.1) + 5 * np.sin(t * 0.05)  # 계절성 (2개 주기 혼합)
    noise = 3 * np.random.randn(length)  # 잡음
    
    # 기본 시계열 = 추세 + 계절성 + 잡음
    data = trend + seasonality + noise
    
    # 이상치 추가 (요청된 경우)
    if with_anomalies:
        # 세 가지 유형의 이상치 추가
        
        # 1. 점 이상치 (큰 스파이크)
        spike_indices = [25, 80, 150, 220]
        for idx in spike_indices:
            if idx < length:
                data[idx] += (30 + 10 * np.random.random()) * (1 if np.random.random() < 0.5 else -1)
        
        # 2. 수준 이동 (level shift)
        if length > 180:
            level_shift_start = 170
            level_shift_end = 190
            data[level_shift_start:level_shift_end] += 20
        
        # 3. 패턴 변화 (계절성 패턴 파괴)
        if length > 120:
            pattern_break_start = 110
            pattern_break_end = 125
            pattern_break_length = pattern_break_end - pattern_break_start
            data[pattern_break_start:pattern_break_end] = trend[pattern_break_start:pattern_break_end] + 5 * np.random.randn(pattern_break_length)
    
    return data.tolist()


def generate_synthetic_data(length=500, anomaly_count=8):
    """
    더 복잡한 합성 시계열 데이터 생성
    
    Args:
        length (int): 생성할 데이터 길이
        anomaly_count (int): 추가할 이상치 수
        
    Returns:
        List[float]: 생성된 시계열 데이터
    """
    np.random.seed(datetime.now().microsecond)  # 현재 시간 기반 시드
    
    # 복잡한 시계열 생성
    t = np.arange(length)
    
    # 비선형 추세
    trend = 0.0001 * t**2 + 0.05 * t
    
    # 복합 계절성 (하루, 주간, 월간 패턴 흉내)
    day_cycle = 24
    week_cycle = 7 * day_cycle
    month_cycle = 30 * day_cycle
    
    seasonality = (
        12 * np.sin(2 * np.pi * t / day_cycle) +  # 일 주기
        20 * np.sin(2 * np.pi * t / week_cycle) +  # 주 주기
        30 * np.sin(2 * np.pi * t / month_cycle)   # 월 주기
    )
    
    # 임의의 변동성을 고려한 더 현실적인 잡음
    noise_scale = 5 + 0.5 * np.sin(2 * np.pi * t / (3 * day_cycle))  # 주기적으로 변화하는 잡음 규모
    noise = noise_scale * np.random.randn(length)
    
    # 기본 시계열 = 추세 + 계절성 + 잡음
    data = trend + seasonality + noise
    
    # 이상치 추가
    anomaly_types = [
        "spike",          # 급격한 스파이크
        "dip",            # 급격한 하락
        "level_shift",    # 수준 이동
        "variance_change", # 변동성 변화
        "trend_change",   # 추세 변화
        "seasonality_break" # 계절성 파괴
    ]
    
    # 사용 가능한 인덱스 범위 설정 (처음과 끝은 피함)
    available_indices = list(range(10, length - 30))
    np.random.shuffle(available_indices)
    
    # 다양한 유형의 이상치 생성
    for i in range(min(anomaly_count, len(available_indices))):
        anomaly_type = np.random.choice(anomaly_types)
        idx = available_indices[i]
        
        if anomaly_type == "spike":
            # 급격한 스파이크
            magnitude = (np.mean(data) + 3 * np.std(data)) * (0.5 + np.random.random())
            data[idx] += magnitude
            
        elif anomaly_type == "dip":
            # 급격한 하락
            magnitude = (np.mean(data) + 3 * np.std(data)) * (0.5 + np.random.random())
            data[idx] -= magnitude
            
        elif anomaly_type == "level_shift" and idx < length - 20:
            # 수준 이동
            shift_length = np.random.randint(5, 15)
            shift_mag = np.std(data) * (3 + 2 * np.random.random())
            data[idx:idx+shift_length] += shift_mag
            
        elif anomaly_type == "variance_change" and idx < length - 20:
            # 변동성 변화
            change_length = np.random.randint(10, 20)
            data[idx:idx+change_length] += 3 * noise_scale[idx:idx+change_length] * np.random.randn(change_length)
            
        elif anomaly_type == "trend_change" and idx < length - 25:
            # 추세 변화
            change_length = np.random.randint(15, 25)
            new_trend = np.linspace(0, change_length * 0.5, change_length)
            data[idx:idx+change_length] += new_trend
            
        elif anomaly_type == "seasonality_break" and idx < length - 15:
            # 계절성 파괴
            break_length = np.random.randint(day_cycle, day_cycle * 2)
            # 원래 계절성을 제거하고 다른 잡음으로 대체
            data[idx:idx+break_length] -= seasonality[idx:idx+break_length]
            data[idx:idx+break_length] += np.random.randn(break_length) * np.std(data) * 0.5
    
    return data.tolist()


if __name__ == "__main__":
    main()