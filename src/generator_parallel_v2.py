#!/usr/bin/env python3
"""
Parallel Sentinel V2 메인 실행 스크립트

시각적 추론 기반 시계열 이상치 탐지 시스템 실행을 위한 명령줄 인터페이스
"""

import os
import sys
import json
import pytz
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 환경 변수 로드
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# LangSmith 설정 함수
def setup_langsmith():
    """LangSmith 설정 (설정된 경우)"""
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "parallel-sentinel-v2")
setup_langsmith()

# LangChain 및 Parallel Sentinel V2 모듈 임포트
try:
    from langsmith import traceable
    from langchain.globals import set_debug

    # Parallel Sentinel V2 구성 요소 임포트
    from parallel_sentinel_v2.agents import (
        create_supervisor_agent, create_trend_analyzer_agent,
        create_seasonality_analyzer_agent, create_remainder_analyzer_agent
    )
    # 도구 정의 - 카테고리별로 구분하여 바인딩
    from parallel_sentinel_v2.tools import (
        visualization_tools, statistics_tools, 
        frequency_tools, transformation_tools
    )
    tools = visualization_tools + statistics_tools + frequency_tools + transformation_tools
    print(f"도구 바인딩 완료 - 총 {len(tools)}개 ({len(visualization_tools)} 시각화, "
        f"{len(statistics_tools)} 통계, {len(frequency_tools)} 주파수, "
        f"{len(transformation_tools)} 변환)")
    from parallel_sentinel_v2.graph import create_workflow, run_workflow
    from parallel_sentinel_v2.utils.llm_utils import init_llm

except ImportError as e:
    print(f"오류: 필요한 라이브러리를 임포트할 수 없습니다: {e}")
    print("라이브러리가 올바르게 설치되었는지 확인하세요 (`pip install -r requirements.txt` 또는 `pip install -e .`).")
    sys.exit(1)


def load_data(args):
    """
    명령줄 인자 또는 기본 데이터에서 시계열 데이터 로드
    
    Args:
        args: 명령줄 인자
        
    Returns:
        Tuple[List[float], str]: 시계열 데이터와 데이터 소스 설명
    """
    # 이상치가 있는 예제 데이터 (인덱스 35-36에 매우 큰 스파이크가 있음)
    DEFAULT_SAMPLE_DATA = [753, 703, 500, 1028, 554, 1041, 603, 676, 645, 599, 502, 463, 483, 475, 526, 496, 619, 418, 895, 498, 727, 1018, 756, 763, 600, 668, 816, 490, 721, 644, 642, 347, 638, 506, 605, 578, 10000, 9235, 626, 649, 485, 257, 486, 649, 919, 702, 874, 614, 614, 469, 699, 430, 553, 469, 496, 934, 518, 597, 696, 602, 564, 509, 670, 775, 611, 874, 794, 613, 478, 657, 679, 644, 557, 567, 490, 685, 662, 511, 618, 606, 692, 308, 657, 583, 675, 736, 766, 811, 1042, 842, 547, 402, 1032, 598, 690, 643, 515, 621, 490, 550, 530, 500, 602, 679, 577, 573, 592, 644, 869, 811, 811, 766, 1042, 728, 527, 636, 663, 710, 297, 564, 772, 720, 687, 637, 491, 1041, 543, 518, 998, 342, 196, 702, 976, 702, 914, 891, 658, 636, 708, 1028, 743, 837, 517, 730, 607, 529, 568, 461, 598, 654, 726, 887, 356, 1042, 702, 530, 735, 691, 539, 657, 595, 509, 660, 628, 588, 631, 359, 442, 677, 619, 774, 668, 598, 623, 595, 825, 356, 725, 841, 517, 566, 516, 524, 925, 545, 665, 537, 425, 505, 559, 484, 520, 572, 663, 758, 920, 884, 818, 748, 171, 595, 464, 441, 622, 733, 543, 591, 582, 364, 562, 522, 566, 674, 633, 374, 542, 942, 876, 1006, 844, 716, 468, 555, 589, 698, 419, 525, 614, 436, 613, 691, 650, 594, 603, 596, 240, 839, 942, 702, 1023, 935, 938, 567, 790, 607, 758, 617, 577, 619, 620, 951, 752, 660, 493, 664, 545, 643, 613, 427, 999, 1024, 869, 614, 976, 869, 711, 891, 664, 783, 756, 793, 621, 833, 810, 729, 607, 655, 662, 930, 747, 674, 600, 544, 775, 695, 711, 542, 702, 944, 845, 652, 915, 710, 703, 884, 769, 701, 746, 765, 771, 751, 659, 674, 730, 702, 732, 1042, 869, 862, 1042, 942, 614, 570, 639, 685, 614, 599, 428, 635, 762, 632, 575, 810, 654, 659, 758, 538, 640, 600, 580, 914, 881, 811, 1031, 807, 614, 886, 626, 642, 668, 742, 739, 721, 502, 606, 644, 812, 582, 671, 715, 640, 653, 942, 784, 784, 631, 702, 817, 654, 760, 617, 514, 683, 667, 542, 730, 573, 681, 594, 609, 502, 599, 865, 931, 838, 675, 804, 627, 646, 757, 689, 736, 996, 761, 710, 595, 560, 657, 664, 705, 646, 671, 668, 666, 702, 708, 645, 786, 647, 781]
    
    time_series_data = []
    data_source = ""
    
    if hasattr(args, 'data') and args.data and os.path.exists(args.data):
        data_source = f"파일 ({args.data})"
        print(f"{data_source}에서 시계열 데이터 로드 중...")
        try:
            import pandas as pd
            df = pd.read_csv(args.data)
            
            # 데이터 로드 로직 (첫 번째 숫자형 컬럼을 시계열로 가정)
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                time_series_column = numeric_cols[0]
                time_series_data = df[time_series_column].tolist()
                print(f"컬럼 '{time_series_column}'을(를) 시계열 데이터로 사용합니다.")
            else:
                raise ValueError("숫자형 컬럼을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            data_source = "기본 샘플 (로드 오류)"
            print(f"{data_source} 사용")
            time_series_data = DEFAULT_SAMPLE_DATA
    else:
        data_source = "기본 샘플"
        print(f"{data_source} 사용")
        time_series_data = DEFAULT_SAMPLE_DATA
        
    return time_series_data, data_source


@traceable
def main():
    """Parallel Sentinel V2 라이브러리 메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Parallel Sentinel V2 - 시각적 추론 기반 시계열 이상 탐지 시스템")
    parser.add_argument("--data", type=str, help="CSV 데이터 파일 경로")
    parser.add_argument("--debug", action="store_true", help="LangChain 디버그 모드 활성화")
    parser.add_argument("--output", type=str, default="output", help="결과 출력 디렉토리")
    parser.add_argument("--llm_provider", type=str, default="google", choices=["google", "anthropic", "openai"], help="멀티모달 LLM 제공자")
    args = parser.parse_args()

    # 디버그 모드 설정
    if args.debug:
        set_debug(True)
        print("--- 디버그 모드 활성화 ---")

    print("\n=== Parallel Sentinel V2 - 시각적 추론 기반 시계열 이상 탐지 시스템 시작 ===")
    
    # 데이터 로드
    time_series_data, data_source = load_data(args)
    if not time_series_data:
        print("오류: 분석할 시계열 데이터가 없습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    # 멀티모달 언어 모델 초기화
    try:
        llm = init_llm(args)
    except Exception as e:
        print(f"오류: LLM 모델 초기화 실패: {e}")
        print("모델 제공자 및 설정을 확인하세요.")
        sys.exit(1)

    # 도구 정의 - TOOL_INSTRUCTIONS에 명시된 도구들 중 핵심적인 것들 선택
    tools = [
        ts2img_bytes,  # 시각화 도구 (필수)
        get_fourier_transform,  # 주파수 분석
        get_time_series_decomposition,  # 시계열 분해
        get_time_series_statistics,  # 통계 분석
    ]
    print(f"{len(tools)}개의 도구 정의 완료")

    # 에이전트 생성
    try:
        supervisor = create_supervisor_agent(llm)
        trend_analyzer = create_trend_analyzer_agent(llm, tools)
        seasonality_analyzer = create_seasonality_analyzer_agent(llm, tools)
        remainder_analyzer = create_remainder_analyzer_agent(llm, tools)
        print("Supervisor 및 시각적 추론 기반 분석 에이전트 생성 완료")
    except Exception as e:
        print(f"오류: 에이전트 생성 실패: {e}")
        sys.exit(1)

    # 워크플로우 생성
    try:
        workflow = create_workflow(
            supervisor_agent=supervisor,
            trend_analyzer_agent=trend_analyzer,
            seasonality_analyzer_agent=seasonality_analyzer,
            remainder_analyzer_agent=remainder_analyzer,
            tools=tools
        )
        print("시각적 추론 기반 분석 워크플로우 생성 완료")
    except Exception as e:
        print(f"오류: 워크플로우 생성 실패: {e}")
        sys.exit(1)

    # 출력 디렉토리 설정
    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"결과는 다음 디렉토리에 저장됩니다: {output_dir.resolve()}")
    except OSError as e:
        print(f"오류: 출력 디렉토리 생성 실패 ({output_dir}): {e}")
        sys.exit(1)

    # 워크플로우 실행
    print(f"\n시계열 데이터 시각적 추론 분석 실행 중 (데이터 소스: {data_source}, 길이: {len(time_series_data)})")
    start_time = datetime.now()

    try:
        final_state = run_workflow(workflow, time_series_data)
    except Exception as e:
        print(f"\n오류: 워크플로우 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        final_state = None

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"분석 완료: {duration:.2f}초 소요")

    # 결과 처리 및 표시
    if final_state and final_state.get("final_analysis"):
        print("\n===== 최종 분석 요약 =====")
        final_analysis_data = final_state["final_analysis"]
        analysis_summary = final_analysis_data.get("summary", "요약 정보 없음")
        print(analysis_summary)

        # 감지된 이상치 표시 (Supervisor의 집계 결과 기준)
        detected_anomalies = final_analysis_data.get("combined_anomalies", [])
        if detected_anomalies:
            print("\n===== 감지된 이상치 (유형별) =====")
            # 유형별로 그룹화
            anomaly_types = {}
            for anomaly in detected_anomalies:
                anomaly_type = anomaly.get('type', 'Unknown')
                if anomaly_type not in anomaly_types:
                    anomaly_types[anomaly_type] = []
                anomaly_types[anomaly_type].append(anomaly)
            
            # 유형별로 출력
            for anomaly_type, anomalies in anomaly_types.items():
                print(f"\n{anomaly_type} ({len(anomalies)}개):")
                for i, anomaly in enumerate(anomalies[:5]):  # 각 유형별 최대 5개만 표시
                    sources = anomaly.get('sources', [anomaly.get('source', 'unknown')])
                    print(f"  이상치 {i+1}: 인덱스={anomaly.get('index', 'N/A')}, "
                          f"신뢰도={anomaly.get('confidence', 0):.2f}, "
                          f"감지 소스={', '.join(sources)}")
                if len(anomalies) > 5:
                    print(f"  ... 외 {len(anomalies) - 5}개")

        # 상세 결과 저장
        analysis_output = {
            "data_source": data_source,
            "time_series_length": len(time_series_data),
            "analysis_duration_seconds": duration,
            "llm_provider": args.llm_provider,
            "timestamp": datetime.now(pytz.timezone("Asia/Seoul")).isoformat(),
            "final_analysis_summary": analysis_summary,
            # 각 에이전트의 마지막 분석 결과 포함
            "trend_analysis_details": final_state.get("trend_analysis", [{}])[-1],
            "seasonality_analysis_details": final_state.get("seasonality_analysis", [{}])[-1],
            "remainder_analysis_details": final_state.get("remainder_analysis", [{}])[-1],
            "combined_anomalies_details": detected_anomalies
        }

        output_file = output_dir / "parallel_sentinel_v2_results.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # NumPy 타입을 Python 기본 타입으로 변환 후 저장
                json.dump(analysis_output, f, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            print(f"\n상세 결과가 {output_file}에 저장되었습니다.")
        except Exception as file_err:
            print(f"\n오류: 결과를 파일에 저장하는 중 오류 발생: {file_err}")

    elif final_state is None:
        print("\n오류: 워크플로우 실행 중 오류가 발생하여 최종 분석을 생성할 수 없습니다.")
    else:
        print("\n최종 분석이 생성되지 않았습니다. 워크플로우 실행 로그 및 상태를 확인하세요.")


if __name__ == "__main__":
    main()