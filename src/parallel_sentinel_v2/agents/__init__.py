"""
agents 모음
"""

# 시각화 도구
from .visualization import (
    ts2img_bytes,
    ts2img_with_anomalies,
    ts2img_multi_view
)

# 통계 분석 도구
from .statistics import (
    get_time_series_statistics,
    detect_outliers_statistics,
    get_time_series_diff,
    auto_correlation
)

# 주파수 분석 도구
from .frequency import (
    get_fourier_transform,
    wavelet_transform
)

# 변환 도구
from .transformation import (
    get_time_series_decomposition,
    symbolic_representation
)

# 도구 가져오기 편의를 위한 목록
visualization_tools = [
    ts2img_bytes,
    ts2img_with_anomalies,
    ts2img_multi_view
]

statistics_tools = [
    get_time_series_statistics,
    detect_outliers_statistics,
    get_time_series_diff,
    auto_correlation
]

frequency_tools = [
    get_fourier_transform,
    wavelet_transform
]

transformation_tools = [
    get_time_series_decomposition,
    symbolic_representation
]

# 모든 도구 목록
all_tools = visualization_tools + statistics_tools + frequency_tools + transformation_tools

__all__ = [
    # 시각화 도구
    "ts2img_bytes",
    "ts2img_with_anomalies",
    "ts2img_multi_view",
    
    # 통계 분석 도구
    "get_time_series_statistics",
    "detect_outliers_statistics",
    "get_time_series_diff",
    "auto_correlation",
    
    # 주파수 분석 도구
    "get_fourier_transform",
    "wavelet_transform",
    
    # 변환 도구
    "get_time_series_decomposition",
    "symbolic_representation",
    
    # 도구 모음
    "visualization_tools",
    "statistics_tools",
    "frequency_tools",
    "transformation_tools",
    "all_tools"
]