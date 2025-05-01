"""
시계열 분해 도구
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stl import STL
from scipy import signal

from langchain_core.tools import tool


@tool
def decompose_time_series(
    data: List[float], 
    method: str = "stl", 
    period: Optional[int] = None,
    model: str = "additive"
) -> str:
    """
    시계열 데이터를 추세(trend), 계절성(seasonality), 잔차(remainder) 성분으로 분해합니다.
    
    Args:
        data (List[float]): 분해할 시계열 데이터
        method (str): 분해 방법 ("stl" 또는 "seasonal_decompose")
        period (Optional[int]): 계절성 주기. None이면 자동 탐지 시도
        model (str): 분해 모델 ("additive" 또는 "multiplicative")
        
    Returns:
        str: 분해 결과와 시각화 경로가 포함된 JSON 문자열
    """
    try:
        data_np = np.array(data)
        
        # 주기가 지정되지 않은 경우 자동 추정
        if period is None:
            period = _estimate_seasonality_period(data_np)
        
        # 분해 수행
        if method == "stl":
            result = _decompose_stl(data_np, period)
        else:  # seasonal_decompose
            result = _decompose_seasonal(data_np, period, model)
        
        # 결과 시각화
        fig_path = _visualize_decomposition(data_np, result, method, period)
        
        # 분해 결과 정리
        decomposition_result = {
            "trend": result["trend"][:50].tolist() + result["trend"][-50:].tolist() if len(result["trend"]) > 100 else result["trend"].tolist(),
            "seasonality": result["seasonality"][:50].tolist() + result["seasonality"][-50:].tolist() if len(result["seasonality"]) > 100 else result["seasonality"].tolist(),
            "remainder": result["remainder"][:50].tolist() + result["remainder"][-50:].tolist() if len(result["remainder"]) > 100 else result["remainder"].tolist(),
            "method": method,
            "period": period,
            "model": model,
            "visualization_path": str(fig_path),
            "stats": {
                "trend_strength": _calculate_trend_strength(result["trend"], result["remainder"]),
                "seasonality_strength": _calculate_seasonality_strength(result["seasonality"], result["remainder"]),
                "remainder_strength": _calculate_remainder_strength(result["remainder"], data_np)
            },
            "status": "success"
        }
        
        return json.dumps(decomposition_result)
    
    except Exception as e:
        error_message = f"시계열 분해 오류: {e}"
        return json.dumps({"status": "error", "message": error_message})


def _estimate_seasonality_period(data: np.ndarray) -> int:
    """
    시계열 데이터의 계절성 주기를 추정합니다.
    
    Args:
        data (np.ndarray): 시계열 데이터
        
    Returns:
        int: 추정된 계절성 주기
    """
    n = len(data)
    
    # 매우 짧은 시계열의 경우 기본값 반환
    if n < 10:
        return 2
        
    # 데이터 길이에 따라 후보 주기 설정
    if n < 24:
        candidates = [2, 3, 4, 6]
    elif n < 60:
        candidates = [2, 3, 4, 6, 7, 12]
    elif n < 100:
        candidates = [2, 3, 4, 6, 7, 12, 24]
    else:
        candidates = [2, 3, 4, 6, 7, 12, 24, 30, 52, 365]
    
    # 주기 후보를 데이터 길이의 1/3로 제한
    candidates = [c for c in candidates if c < n/3]
    
    # 자기상관(autocorrelation)을 사용한 주기 추정
    if len(candidates) > 0:
        acf_values = _calculate_acf(data, max(candidates) + 1)
        
        # 자기상관이 높은 주기 찾기
        max_acf = -1
        best_period = candidates[0]  # 기본값
        
        for period in candidates:
            if period < len(acf_values) and acf_values[period] > max_acf:
                max_acf = acf_values[period]
                best_period = period
        
        # 충분히 강한 자기상관이 있는 경우에만 해당 주기 반환
        if max_acf > 0.2:
            return best_period
    
    # 적절한 주기를 찾지 못한 경우, 데이터 길이에 따른 기본값 반환
    if n < 30:
        return 7  # 주간 데이터로 가정
    elif n < 100:
        return 12  # 월간 데이터로 가정
    elif n < 500:
        return 30  # 월별 일일 데이터로 가정
    else:
        return 52  # 주간 데이터로 가정


def _calculate_acf(data: np.ndarray, max_lag: int) -> np.ndarray:
    """
    시계열 데이터의 자기상관함수(ACF)를 계산합니다.
    
    Args:
        data (np.ndarray): 시계열 데이터
        max_lag (int): 최대 지연(lag)
        
    Returns:
        np.ndarray: 자기상관 값
    """
    result = np.zeros(max_lag)
    mean = np.mean(data)
    variance = np.var(data)
    
    for lag in range(max_lag):
        if lag == 0:
            result[lag] = 1.0  # 지연이 0인 경우 자기상관은 1
        else:
            sum_of_products = np.sum((data[lag:] - mean) * (data[:-lag] - mean))
            result[lag] = sum_of_products / ((len(data) - lag) * variance)
    
    return result


def _decompose_stl(data: np.ndarray, period: int) -> Dict[str, np.ndarray]:
    """
    STL(Seasonal and Trend decomposition using Loess) 방법으로 시계열을 분해합니다.
    
    Args:
        data (np.ndarray): 시계열 데이터
        period (int): 계절성 주기
        
    Returns:
        Dict[str, np.ndarray]: 분해 결과
    """
    # 결측값 처리
    data_clean = pd.Series(data).interpolate().values
    
    # STL 분해 수행
    stl_result = STL(data_clean, period=period, robust=True).fit()
    
    return {
        "trend": stl_result.trend,
        "seasonality": stl_result.seasonal,
        "remainder": stl_result.resid
    }


def _decompose_seasonal(data: np.ndarray, period: int, model: str) -> Dict[str, np.ndarray]:
    """
    seasonal_decompose 방법으로 시계열을 분해합니다.
    
    Args:
        data (np.ndarray): 시계열 데이터
        period (int): 계절성 주기
        model (str): 분해 모델 ("additive" 또는 "multiplicative")
        
    Returns:
        Dict[str, np.ndarray]: 분해 결과
    """
    # 결측값 처리
    data_clean = pd.Series(data).interpolate().values
    
    # 계절 분해 수행
    decomposition = seasonal_decompose(data_clean, model=model, period=period)
    
    # NaN 값 처리
    trend = pd.Series(decomposition.trend).interpolate().values
    seasonal = pd.Series(decomposition.seasonal).interpolate().values
    resid = pd.Series(decomposition.resid).interpolate().values
    
    return {
        "trend": trend,
        "seasonality": seasonal,
        "remainder": resid
    }


def _visualize_decomposition(
    data: np.ndarray, 
    decomposition: Dict[str, np.ndarray], 
    method: str, 
    period: int
) -> Path:
    """
    시계열 분해 결과를 시각화합니다.
    
    Args:
        data (np.ndarray): 원본 시계열 데이터
        decomposition (Dict[str, np.ndarray]): 분해 결과
        method (str): 사용된 분해 방법
        period (int): 계절성 주기
        
    Returns:
        Path: 저장된 시각화 이미지 경로
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # 원본 데이터
    axes[0].plot(data, label='원본 데이터')
    axes[0].set_title('원본 시계열')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 추세 성분
    axes[1].plot(decomposition["trend"], label='추세', color='blue')
    axes[1].set_title('추세 성분')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 계절성 성분
    axes[2].plot(decomposition["seasonality"], label='계절성', color='green')
    axes[2].set_title(f'계절성 성분 (주기: {period})')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # 잔차 성분
    axes[3].plot(decomposition["remainder"], label='잔차', color='red')
    axes[3].set_title('잔차 성분')
    axes[3].grid(True, linestyle='--', alpha=0.6)
    
    # 전체 타이틀 설정
    plt.suptitle(f'시계열 분해 ({method.upper()} 방법)', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 이미지 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    try:
        img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
    except NameError:
        img_dir = Path("./temp_images")
    
    img_dir.mkdir(parents=True, exist_ok=True)
    save_path = img_dir / f"decomposition_{timestamp}.png"
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return save_path


def _calculate_trend_strength(trend: np.ndarray, remainder: np.ndarray) -> float:
    """
    추세 강도를 계산합니다.
    
    Args:
        trend (np.ndarray): 추세 성분
        remainder (np.ndarray): 잔차 성분
        
    Returns:
        float: 추세 강도 (0~1 사이 값)
    """
    trend_var = np.var(trend)
    remainder_var = np.var(remainder)
    
    if trend_var + remainder_var == 0:
        return 0
    
    strength = max(0, 1 - remainder_var / (trend_var + remainder_var))
    return float(strength)


def _calculate_seasonality_strength(seasonality: np.ndarray, remainder: np.ndarray) -> float:
    """
    계절성 강도를 계산합니다.
    
    Args:
        seasonality (np.ndarray): 계절성 성분
        remainder (np.ndarray): 잔차 성분
        
    Returns:
        float: 계절성 강도 (0~1 사이 값)
    """
    seasonality_var = np.var(seasonality)
    remainder_var = np.var(remainder)
    
    if seasonality_var + remainder_var == 0:
        return 0
    
    strength = max(0, 1 - remainder_var / (seasonality_var + remainder_var))
    return float(strength)


def _calculate_remainder_strength(remainder: np.ndarray, data: np.ndarray) -> float:
    """
    잔차 강도를 계산합니다.
    
    Args:
        remainder (np.ndarray): 잔차 성분
        data (np.ndarray): 원본 데이터
        
    Returns:
        float: 잔차 강도 (0~1 사이 값)
    """
    remainder_var = np.var(remainder)
    data_var = np.var(data)
    
    if data_var == 0:
        return 0
    
    strength = remainder_var / data_var
    return float(strength)