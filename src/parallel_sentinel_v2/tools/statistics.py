"""
시계열 데이터 통계 분석 도구
이 모듈은 시계열 데이터의 통계적 특성을 분석하기 위한 도구를 제공합니다.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from scipy import stats

from langchain_core.tools import tool


@tool
def get_time_series_statistics(data: List[float]) -> str:
    """
    Calculates and visualizes basic statistical measures for a time series.
    
    Args:
        data (List[float]): Time series data
        
    Returns:
        str: JSON string containing statistical measures
    """
    try:
        if not data:
            raise ValueError("Input data list is empty")
            
        data_np = np.array(data)
        
        # Calculate statistics
        stats_results = {
            "mean": float(np.mean(data_np)),
            "median": float(np.median(data_np)),
            "std": float(np.std(data_np)),
            "min": float(np.min(data_np)),
            "max": float(np.max(data_np)),
            "range": float(np.max(data_np) - np.min(data_np)),
            "q1": float(np.percentile(data_np, 25)),
            "q3": float(np.percentile(data_np, 75)),
            "iqr": float(np.percentile(data_np, 75) - np.percentile(data_np, 25)),
            "skewness": float(stats.skew(data_np)),
            "kurtosis": float(stats.kurtosis(data_np))
        }
        
        # IQR-based outlier detection
        iqr = stats_results["iqr"]
        q1 = stats_results["q1"]
        q3 = stats_results["q3"]
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_indices = np.where((data_np < lower_bound) | (data_np > upper_bound))[0].tolist()
        outlier_values = [float(data_np[i]) for i in outlier_indices]
        
        # Return results as JSON
        result = {
            "statistics": stats_results,
            "outliers": {
                "method": "IQR (1.5)",
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "indices": outlier_indices[:20] if len(outlier_indices) > 20 else outlier_indices,
                "values": outlier_values[:20] if len(outlier_values) > 20 else outlier_values,
                "count": len(outlier_indices)
            },
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error in statistical analysis: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})


@tool
def detect_outliers_statistics(data: List[float], method: str = "iqr", 
                             z_threshold: float = 3.0, iqr_multiplier: float = 1.5) -> str:
    """
    Detects outliers in time series data using multiple statistical methods.
    
    Args:
        data (List[float]): Time series data
        method (str, optional): Detection method ("iqr", "zscore", "both"). Defaults to "iqr".
        z_threshold (float, optional): Z-score threshold. Defaults to 3.0.
        iqr_multiplier (float, optional): IQR multiplier. Defaults to 1.5.
        
    Returns:
        str: JSON string containing outlier detection results
    """
    try:
        if not data:
            raise ValueError("Input data list is empty")
            
        data_np = np.array(data)
        
        # Z-score method
        z_anomalies_indices = []
        z_scores = []
        mean = np.mean(data_np)
        std = np.std(data_np)
        
        if std > 0 and (method == "zscore" or method == "both"):
            z_scores = (data_np - mean) / std
            z_anomalies_indices = np.where(np.abs(z_scores) > z_threshold)[0].tolist()
        
        # IQR method
        iqr_anomalies_indices = []
        
        if method == "iqr" or method == "both":
            q1 = np.percentile(data_np, 25)
            q3 = np.percentile(data_np, 75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            iqr_anomalies_indices = np.where((data_np < lower_bound) | (data_np > upper_bound))[0].tolist()
        
        # Combine results if using both methods
        if method == "both":
            all_indices = sorted(list(set(z_anomalies_indices + iqr_anomalies_indices)))
        elif method == "zscore":
            all_indices = z_anomalies_indices
        else:
            all_indices = iqr_anomalies_indices
            
        # Get values for the anomalies
        anomaly_values = [(int(idx), float(data_np[idx])) for idx in all_indices]
            
        # Return results
        result = {
            "detected_outliers": {
                "indices": all_indices,
                "values": [float(data_np[i]) for i in all_indices],
                "count": len(all_indices)
            },
            "methods": {
                "z_score": {
                    "threshold": float(z_threshold),
                    "mean": float(mean),
                    "std": float(std),
                    "count": len(z_anomalies_indices),
                    "indices": z_anomalies_indices[:10] if len(z_anomalies_indices) > 10 else z_anomalies_indices
                },
                "iqr": {
                    "multiplier": float(iqr_multiplier),
                    "q1": float(np.percentile(data_np, 25)),
                    "q3": float(np.percentile(data_np, 75)),
                    "iqr": float(np.percentile(data_np, 75) - np.percentile(data_np, 25)),
                    "count": len(iqr_anomalies_indices),
                    "indices": iqr_anomalies_indices[:10] if len(iqr_anomalies_indices) > 10 else iqr_anomalies_indices
                }
            },
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error in outlier detection: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})


@tool
def get_time_series_diff(data: List[float], order: int = 1) -> str:
    """
    Calculates differences between consecutive points in time series data.
    
    Args:
        data (List[float]): Time series data
        order (int, optional): Order of differencing. Defaults to 1.
        
    Returns:
        str: JSON string containing differenced time series and statistics
    """
    try:
        if not data:
            raise ValueError("Input data list is empty")
            
        data_np = np.array(data)
        
        # Calculate n-th order difference
        diff_data = data_np.copy()
        for _ in range(order):
            diff_data = np.diff(diff_data)
            
        # Calculate statistics of differenced data
        if len(diff_data) > 0:
            diff_stats = {
                "mean": float(np.mean(diff_data)),
                "std": float(np.std(diff_data)),
                "min": float(np.min(diff_data)),
                "max": float(np.max(diff_data)),
                "median": float(np.median(diff_data)),
                "range": float(np.max(diff_data) - np.min(diff_data))
            }
            
            # Detect potential change points (significant changes)
            threshold = 2.0 * diff_stats["std"]  # Points where the change is 2 std deviations above the mean
            change_points = np.where(np.abs(diff_data) > threshold)[0].tolist()
            
            result = {
                "diff_order": order,
                "original_length": len(data_np),
                "diff_length": len(diff_data),
                "diff_data": diff_data.tolist(),
                "diff_statistics": diff_stats,
                "potential_change_points": change_points[:15] if len(change_points) > 15 else change_points,
                "status": "success"
            }
        else:
            result = {
                "diff_order": order,
                "original_length": len(data_np),
                "diff_length": 0,
                "message": "Differenced data is empty, order too high for data length",
                "status": "warning"
            }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error calculating time series differences: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})


@tool
def auto_correlation(data: List[float], max_lag: Optional[int] = None) -> str:
    """
    Calculates the auto-correlation of time series data.
    
    Args:
        data (List[float]): Time series data
        max_lag (Optional[int], optional): Maximum lag to calculate (default: None, uses len(data)-1).
        
    Returns:
        str: JSON string containing auto-correlation values for different lags
    """
    try:
        if not data:
            raise ValueError("Input data list is empty")
            
        data_np = np.array(data)
        n = len(data_np)
        
        # Set max_lag if not provided
        if max_lag is None or max_lag <= 0 or max_lag >= n:
            max_lag = n - 1
        
        # Calculate mean and variance
        mean = np.mean(data_np)
        var = np.var(data_np)
        
        if var == 0:
            return json.dumps({
                "status": "warning",
                "message": "Variance is zero, cannot calculate autocorrelation",
                "autocorrelation": [1.0] + [0.0] * (max_lag)
            })
            
        # Calculate autocorrelation for each lag
        ac_values = []
        for lag in range(max_lag + 1):
            if lag == 0:
                # Lag 0 autocorrelation is always 1
                ac_values.append(1.0)
            else:
                # Calculate correlation between original series and lagged series
                ac = np.corrcoef(data_np[:-lag], data_np[lag:])[0, 1]
                ac_values.append(float(ac))
        
        # Find significant peaks in autocorrelation
        threshold = 2 / np.sqrt(n)  # Common significance threshold for autocorrelation
        significant_lags = [lag for lag, ac in enumerate(ac_values) if lag > 0 and abs(ac) > threshold]
        
        # Suggest seasonality period
        seasonality_period = None
        if significant_lags:
            # Find first significant lag with highest autocorrelation
            significant_acs = [ac_values[lag] for lag in significant_lags]
            max_idx = np.argmax(significant_acs)
            seasonality_period = significant_lags[max_idx]
        
        result = {
            "autocorrelation": ac_values,
            "max_lag": max_lag,
            "significant_lags": significant_lags[:10] if len(significant_lags) > 10 else significant_lags,
            "significance_threshold": float(threshold),
            "detected_seasonality_period": seasonality_period,
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error calculating autocorrelation: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})