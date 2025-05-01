# src/generator_parallel.py
"""
메인 실행 스크립트 - Parallel Sentinel: 병렬 시계열 이상치 탐지 시스템
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Literal

# 환경 변수 로드 (.env 파일 우선)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# LangSmith 설정 함수
def setup_langsmith():
    """LangSmith 설정 (설정된 경우)"""
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "parallel-sentinel") # 기본 프로젝트 이름 설정

setup_langsmith()

# 필요한 LangChain 및 Parallel Sentinel 모듈 임포트
try:
    from langsmith import traceable
    from langchain.globals import set_debug
    from langchain_google_vertexai import ChatVertexAI

    # Parallel Sentinel 구성 요소 임포트
    # (이전 단계에서 생성된 __init__.py 파일을 통해 임포트 가능)
    from parallel_sentinel.agents import (
        create_supervisor_agent, create_trend_analyzer_agent,
        create_seasonality_analyzer_agent, create_remainder_analyzer_agent
    )
    from parallel_sentinel.tools import (
        ts2img, ts2img_with_anomalies, ts2img_multi_view,
        basic_statistics, trend_analysis, seasonality_analysis,
        stationarity_test, anomaly_detection, decompose_time_series,
        get_math_calculator, rolling_window_stats
    )
    from parallel_sentinel.graph import create_workflow, run_workflow
    from parallel_sentinel.utils import parse_final_analysis

except ImportError as e:
    print(f"오류: 필요한 라이브러리를 임포트할 수 없습니다: {e}")
    print("라이브러리가 올바르게 설치되었는지 확인하세요 (`pip install -r requirements.txt` 또는 `pip install -e .`).")
    sys.exit(1)

# Default environment variables for Google Cloud
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "hd-gen-ai-proc-391223")
LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_NAME_G = os.getenv("GOOGLE_GEN_MODEL", "gemini-2.5-flash-preview-04-17")

# --- 기본 샘플 데이터 (generator_fed.py 와 동일) ---
DEFAULT_SAMPLE_DATA = [753, 703, 500, 1028, 554, 1041, 603, 676, 645, 599, 502, 463, 483, 475, 526, 496, 619, 418, 895, 498, 727, 1018, 756, 763, 600, 668, 816, 490, 721, 644, 642, 347, 638, 506, 605, 578, 10000, 9235, 626, 649, 485, 257, 486, 649, 919, 702, 874, 614, 614, 469, 699, 430, 553, 469, 496, 934, 518, 597, 696, 602, 564, 509, 670, 775, 611, 874, 794, 613, 478, 657, 679, 644, 557, 567, 490, 685, 662, 511, 618, 606, 692, 308, 657, 583, 675, 736, 766, 811, 1042, 842, 547, 402, 1032, 598, 690, 643, 515, 621, 490, 550, 530, 500, 602, 679, 577, 573, 592, 644, 869, 811, 811, 766, 1042, 728, 527, 636, 663, 710, 297, 564, 772, 720, 687, 637, 491, 1041, 543, 518, 998, 342, 196, 702, 976, 702, 914, 891, 658, 636, 708, 1028, 743, 837, 517, 730, 607, 529, 568, 461, 598, 654, 726, 887, 356, 1042, 702, 530, 735, 691, 539, 657, 595, 509, 660, 628, 588, 631, 359, 442, 677, 619, 774, 668, 598, 623, 595, 825, 356, 725, 841, 517, 566, 516, 524, 925, 545, 665, 537, 425, 505, 559, 484, 520, 572, 663, 758, 920, 884, 818, 748, 171, 595, 464, 441, 622, 733, 543, 591, 582, 364, 562, 522, 566, 674, 633, 374, 542, 942, 876, 1006, 844, 716, 468, 555, 589, 698, 419, 525, 614, 436, 613, 691, 650, 594, 603, 596, 240, 839, 942, 702, 1023, 935, 938, 567, 790, 607, 758, 617, 577, 619, 620, 951, 752, 660, 493, 664, 545, 643, 613, 427, 999, 1024, 869, 614, 976, 869, 711, 891, 664, 783, 756, 793, 621, 833, 810, 729, 607, 655, 662, 930, 747, 674, 600, 544, 775, 695, 711, 542, 702, 944, 845, 652, 915, 710, 703, 884, 769, 701, 746, 765, 771, 751, 659, 674, 730, 702, 732, 1042, 869, 862, 1042, 942, 614, 570, 639, 685, 614, 599, 428, 635, 762, 632, 575, 810, 654, 659, 758, 538, 640, 600, 580, 914, 881, 811, 1031, 807, 614, 886, 626, 642, 668, 742, 739, 721, 502, 606, 644, 812, 582, 671, 715, 640, 653, 942, 784, 784, 631, 702, 817, 654, 760, 617, 514, 683, 667, 542, 730, 573, 681, 594, 609, 502, 599, 865, 931, 838, 675, 804, 627, 646, 757, 689, 736, 996, 761, 710, 595, 560, 657, 664, 705, 646, 671, 668, 666, 702, 708, 645, 786, 647, 781]

@traceable
def main_parallel():
    """
    Parallel Sentinel 라이브러리 데모 메인 함수
    """
    parser = argparse.ArgumentParser(description="Parallel Sentinel 시계열 이상 탐지 시스템 실행")
    parser.add_argument("--data", type=str, help="CSV 데이터 파일 경로")
    parser.add_argument("--debug", action="store_true", help="LangChain 디버그 모드 활성화")
    parser.add_argument("--output", type=str, help="결과 출력 디렉토리")
    args = parser.parse_args()

    # 디버그 모드 설정
    if args.debug:
        set_debug(True)
        print("--- 디버그 모드 활성화 ---")

    print("\n=== Parallel Sentinel - 병렬 시계열 이상 탐지 시스템 시작 ===")

    # 데이터 로드
    time_series_data: List[float] = []
    data_source: str = ""
    if args.data and os.path.exists(args.data):
        data_source = f"파일 ({args.data})"
        print(f"{data_source}에서 시계열 데이터 로드 중...")
        try:
            df = pd.read_csv(args.data)
            # 첫 번째 숫자형 열을 사용
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                col_name = numeric_cols[0]
                time_series_data = df[col_name].values.tolist()
                print(f"'{col_name}' 열에서 {len(time_series_data)}개 데이터 포인트 로드 완료")
            else:
                raise ValueError("CSV 파일에 숫자형 열이 없습니다.")
        except Exception as e:
            print(f"데이터 로드 오류: {str(e)}")
            print("기본 샘플 데이터를 사용합니다.")
            time_series_data = DEFAULT_SAMPLE_DATA
            data_source = "기본 샘플 (로드 오류)"
    else:
        data_source = "기본 샘플"
        print(f"{data_source} 사용")
        time_series_data = DEFAULT_SAMPLE_DATA

    if not time_series_data:
        print("오류: 분석할 시계열 데이터가 없습니다. 스크립트를 종료합니다.")
        sys.exit(1)

    # 언어 모델 초기화
    try:

        llm = ChatVertexAI(model_name=MODEL_NAME_G)
        print(f"{MODEL_NAME_G} 모델 초기화 완료")
    except Exception as e:
        print(f"오류: LLM 모델({MODEL_NAME_G}) 초기화 실패: {e}")
        print("Google Cloud 인증 또는 모델 이름을 확인하세요.")
        sys.exit(1)

    # 도구 정의
    tools = [
        ts2img, ts2img_with_anomalies, ts2img_multi_view,
        basic_statistics, trend_analysis, seasonality_analysis,
        stationarity_test, anomaly_detection, decompose_time_series,
        rolling_window_stats, get_math_calculator(llm)
    ]
    print(f"{len(tools)}개의 도구 정의 완료")

    # 에이전트 생성
    try:
        supervisor = create_supervisor_agent(llm)
        trend_analyzer = create_trend_analyzer_agent(llm, tools)
        seasonality_analyzer = create_seasonality_analyzer_agent(llm, tools)
        remainder_analyzer = create_remainder_analyzer_agent(llm, tools)
        print("Supervisor 및 분석 에이전트 생성 완료")
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
        print("분석 워크플로우 생성 완료")
    except Exception as e:
        print(f"오류: 워크플로우 생성 실패: {e}")
        sys.exit(1)

    # 출력 디렉토리 설정
    if args.output:
        output_dir = Path(args.output)
    else:
        # 스크립트가 위치한 디렉토리의 상위 디렉토리에 results 폴더 생성
        script_dir = Path(__file__).resolve().parent
        base_dir = script_dir.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_dir / f"results_parallel_{timestamp}"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"결과는 다음 디렉토리에 저장됩니다: {output_dir.resolve()}")
    except OSError as e:
        print(f"오류: 출력 디렉토리 생성 실패 ({output_dir}): {e}")
        sys.exit(1)

    # 워크플로우 실행
    print(f"\n시계열 데이터 분석 실행 중 (데이터 소스: {data_source}, 길이: {len(time_series_data)})")
    start_time = datetime.now()

    try:
        final_state = run_workflow(workflow, time_series_data)
    except Exception as e:
        print(f"\n오류: 워크플로우 실행 중 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        final_state = None # 오류 발생 시 final_state 를 None으로 설정

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"분석 완료: {duration:.2f}초 소요")

    # 결과 처리 및 표시
    if final_state and final_state.get("final_analysis"):
        print("\n===== 최종 분석 요약 =====")
        final_analysis_data = final_state["final_analysis"]
        analysis_summary = final_analysis_data.get("summary", "요약 정보 없음")
        print(analysis_summary)

        # 구조화된 접근을 위한 최종 분석 파싱 (오류 처리 추가)
        try:
            structured_analysis = parse_final_analysis(analysis_summary)
            detected_anomalies = final_analysis_data.get("combined_anomalies", []) # Supervisor가 집계한 이상치 사용
        except Exception as parse_err:
            print(f"\n경고: 최종 분석 요약 파싱 중 오류 발생: {parse_err}")
            structured_analysis = None
            detected_anomalies = final_analysis_data.get("combined_anomalies", []) # Supervisor 결과 우선 사용

        # 감지된 이상치 표시 (Supervisor의 집계 결과 기준)
        if detected_anomalies:
            print("\n===== 감지된 이상치 (Supervisor 집계) =====")
            # 점수(score) 기준으로 정렬하여 상위 10개 표시
            detected_anomalies.sort(key=lambda x: x.get('score', 0), reverse=True)
            for i, anomaly in enumerate(detected_anomalies[:10]):
                print(f"이상치 {i+1}: 인덱스={anomaly.get('index', 'N/A')}, "
                      f"값={anomaly.get('value', 'N/A'):.2f}, "
                      f"점수={anomaly.get('score', 0):.2f}, "
                      f"출처={anomaly.get('source', 'N/A')}")
            if len(detected_anomalies) > 10:
                print(f"... (총 {len(detected_anomalies)}개 중 상위 10개 표시)")

            # 이상치 시각화 생성 (모든 집계된 이상치 인덱스 사용)
            all_indices = [a.get('index') for a in detected_anomalies if a.get('index') is not None]
            if all_indices:
                try:
                    # ts2img_with_anomalies 호출 방식 수정 (invoke 사용 및 결과 처리)
                    viz_tool_input = {"data": time_series_data, "anomaly_indices": all_indices, "title": "Detected Anomalies (Parallel)"}
                    viz_result_str = ts2img_with_anomalies.invoke(viz_tool_input)
                    viz_result = json.loads(viz_result_str) # JSON 문자열 파싱
                    if viz_result.get("status") == "success":
                        print(f"이상치 시각화 저장 경로: {viz_result.get('image_path')}")
                    else:
                        print(f"시각화 생성 실패: {viz_result.get('message')}")
                except Exception as viz_err:
                    print(f"오류: 이상치 시각화 생성 중 예외 발생: {viz_err}")
        else:
            print("\n감지된 이상치 없음 (Supervisor 집계 기준)")

        # 상세 결과 저장
        analysis_output = {
            "data_source": data_source,
            "time_series_length": len(time_series_data),
            "analysis_duration_seconds": duration,
            "model": MODEL_NAME_G,
            "timestamp": datetime.now().isoformat(),
            "final_analysis_summary": analysis_summary, # LLM 요약
             # 각 에이전트의 마지막 분석 결과 포함
            "trend_analysis_details": final_state.get("trend_analysis", [{}])[-1],
            "seasonality_analysis_details": final_state.get("seasonality_analysis", [{}])[-1],
            "remainder_analysis_details": final_state.get("remainder_analysis", [{}])[-1],
            "combined_anomalies_details": detected_anomalies # Supervisor가 집계한 상세 이상치 정보
        }

        output_file = output_dir / "parallel_analysis_results.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # NumPy 타입을 Python 기본 타입으로 변환 후 저장
                # (JSON 직렬화를 위해 필요할 수 있음)
                json.dump(analysis_output, f, indent=2, ensure_ascii=False, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            print(f"\n상세 결과가 {output_file}에 저장되었습니다.")
        except TypeError as json_err:
             print(f"\n오류: 결과를 JSON으로 저장하는 중 오류 발생: {json_err}")
             print("NumPy 배열 등 직렬화 불가능한 타입이 포함되었을 수 있습니다.")
        except Exception as file_err:
            print(f"\n오류: 결과를 파일에 저장하는 중 오류 발생: {file_err}")

    elif final_state is None:
         print("\n오류: 워크플로우 실행 중 오류가 발생하여 최종 분석을 생성할 수 없습니다.")
    else:
        print("\n최종 분석이 생성되지 않았습니다. 워크플로우 실행 로그 및 상태를 확인하세요.")
        # 실패 시 중간 상태 저장 시도
        output_file = output_dir / "parallel_analysis_failed_state.json"
        try:
            # 상태의 메시지는 직렬화가 어려울 수 있으므로 제외하거나 변환 필요
            state_to_save = {k: v for k, v in final_state.items() if k != 'messages'}
            with open(output_file, "w", encoding="utf-8") as f:
                 json.dump(state_to_save, f, indent=2, ensure_ascii=False, default=lambda x: str(x)) # 복잡한 객체는 문자열로
            print(f"실패 시점의 상태 정보가 {output_file}에 일부 저장되었습니다.")
        except Exception as save_err:
            print(f"오류: 실패 상태 저장 중 오류 발생: {save_err}")


if __name__ == "__main__":

    main_parallel()