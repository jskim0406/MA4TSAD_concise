"""
Parallel Sentinel V2 - 멀티모달 시각적 추론 기반 시계열 이상치 탐지 라이브러리

이 라이브러리는 시계열 데이터의 추세, 계절성, 잔차 성분을 병렬로 분석하고 
시각적 추론을 통해 이상치를 탐지하는 기능을 제공합니다.
"""

__version__ = "0.2.0"

from parallel_sentinel_v2.graph.workflow import create_workflow, run_workflow

__all__ = ["create_workflow", "run_workflow"]