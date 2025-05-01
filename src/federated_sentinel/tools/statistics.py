"""
Statistical analysis tools for time series data.
"""
import json
import numpy as np
import statsmodels

from typing import List, Dict, Any, Tuple
from scipy import stats, signal

from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg

from langchain_core.tools import tool


@tool
def basic_statistics(data: List[float]) -> str:
    """
    Calculate basic statistical properties of time series data.

    Args:
        data (List[float]): A list of numerical time series data.

    Returns:
        str: A JSON string containing basic statistics.
    """
    try:
        data_np = np.array(data)
        
        # Basic statistics
        result = {
            "mean": float(np.mean(data_np)),
            "median": float(np.median(data_np)),
            "min": float(np.min(data_np)),
            "max": float(np.max(data_np)),
            "range": float(np.max(data_np) - np.min(data_np)),
            "std": float(np.std(data_np)),
            "variance": float(np.var(data_np)),
            "skewness": float(stats.skew(data_np)),
            "kurtosis": float(stats.kurtosis(data_np)),
            "quartiles": [
                float(np.percentile(data_np, 25)),  # Q1
                float(np.percentile(data_np, 50)),  # Q2 (median)
                float(np.percentile(data_np, 75))   # Q3
            ],
            "iqr": float(np.percentile(data_np, 75) - np.percentile(data_np, 25)),
            "status": "success"
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_message = f"Error calculating statistics: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def trend_analysis(data: List[float]) -> str:
    """
    Analyze the trend in time series data.

    Args:
        data (List[float]): A list of numerical time series data.

    Returns:
        str: A JSON string containing trend analysis.
    """
    try:
        data_np = np.array(data)
        
        # Linear trend
        x = np.arange(len(data_np))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, data_np)
        
        # Non-linear trend (polynomial fit)
        poly_degree = 3
        poly_coeffs = np.polyfit(x, data_np, poly_degree)
        
        # Moving average trend
        window_size = min(20, len(data_np) // 5)
        if window_size < 2:
            window_size = 2
        moving_avg = np.convolve(data_np, np.ones(window_size)/window_size, mode='valid')
        
        # Mann-Kendall trend test
        # (simplified version for demonstration)
        n = len(data_np)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data_np[j] - data_np[i])
        
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        mk_p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        result = {
            "linear_trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value),
                "std_err": float(std_err),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            },
            "polynomial_trend": {
                "degree": poly_degree,
                "coefficients": [float(c) for c in poly_coeffs]
            },
            "moving_average": {
                "window_size": window_size,
                "first_few_values": [float(v) for v in moving_avg[:5]] if len(moving_avg) > 5 else [float(v) for v in moving_avg],
                "last_few_values": [float(v) for v in moving_avg[-5:]] if len(moving_avg) > 5 else []
            },
            "mann_kendall_test": {
                "s_statistic": float(s),
                "z_score": float(z),
                "p_value": float(mk_p_value),
                "trend": "increasing" if z > 0 and mk_p_value <= 0.05 else 
                         "decreasing" if z < 0 and mk_p_value <= 0.05 else 
                         "no significant trend"
            },
            "status": "success"
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_message = f"Error analyzing trend: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def seasonality_analysis(data: List[float], period: int = None) -> str:
    """
    Analyze seasonality in time series data.

    Args:
        data (List[float]): A list of numerical time series data.
        period (int, optional): Suspected seasonality period. If None, attempts to detect it.

    Returns:
        str: A JSON string containing seasonality analysis.
    """
    try:
        data_np = np.array(data)
        
        # Attempt to detect seasonality period if not provided
        if period is None:
            # Use FFT to find dominant frequencies
            fft_result = np.abs(np.fft.rfft(data_np - np.mean(data_np)))
            freqs = np.fft.rfftfreq(len(data_np))
            
            # Skip the DC component (0 frequency)
            indices = np.argsort(fft_result[1:])[::-1][:5] + 1
            
            # Find top 5 frequencies
            top_freqs = [(freqs[i], fft_result[i]) for i in indices]
            
            # Get the most dominant non-zero frequency
            for freq, amp in top_freqs:
                if freq > 0:
                    period = int(round(1 / freq))
                    if 2 <= period <= len(data_np) // 2:
                        break
            else:
                period = min(len(data_np) // 4, 12)  # Default fallback
        
        # Ensure period is reasonable
        if period < 2:
            period = 2
        if period > len(data_np) // 2:
            period = len(data_np) // 2
        
        # Autocorrelation function
        max_lag = min(period * 3, len(data_np) // 2)
        acf_values = acf(data_np, nlags=max_lag, fft=True)
        
        # Find peaks in ACF
        acf_peaks, _ = signal.find_peaks(acf_values, height=0.1, distance=period/2)
        
        # Check if data length is sufficient for seasonal decomposition
        decomposition = None
        if len(data_np) >= 2 * period:
            try:
                result = seasonal_decompose(data_np, model='additive', period=period)
                decomposition = {
                    "trend": [float(v) for v in result.trend if not np.isnan(v)][:5],
                    "seasonal": [float(v) for v in result.seasonal if not np.isnan(v)][:period],
                    "resid": [float(v) for v in result.resid if not np.isnan(v)][:5]
                }
            except:
                decomposition = None
        
        result = {
            "detected_period": int(period),
            "autocorrelation": {
                "acf_values": [float(v) for v in acf_values[1:11]] if len(acf_values) > 10 else [float(v) for v in acf_values[1:]],
                "acf_peaks": [int(i) for i in acf_peaks if i > 0],
                "has_significant_autocorrelation": any(v > 0.3 for v in acf_values[1:])
            },
            "seasonal_strength": float(np.std(np.array_split(data_np, len(data_np)//period)[:period], axis=0).mean() / np.std(data_np))
            if len(data_np) >= period else 0,
            "decomposition": decomposition,
            "status": "success"
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_message = f"Error analyzing seasonality: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def stationarity_test(data: List[float]) -> str:
    """
    Test for stationarity in time series data using the Augmented Dickey-Fuller test.

    Args:
        data (List[float]): A list of numerical time series data.

    Returns:
        str: A JSON string containing stationarity test results.
    """
    try:
        data_np = np.array(data)
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(data_np)
        
        # Check for enough data points to calculate rolling statistics
        rolling_window = len(data_np) // 10
        if rolling_window < 2:
            rolling_window = 2
        
        # Calculate rolling mean and std
        rolling_mean = np.array([np.mean(data_np[i:i+rolling_window]) 
                               for i in range(0, len(data_np)-rolling_window+1, rolling_window)])
        rolling_std = np.array([np.std(data_np[i:i+rolling_window]) 
                              for i in range(0, len(data_np)-rolling_window+1, rolling_window)])
        
        # Measure stationarity as the constancy of mean and variance
        mean_stationarity = np.std(rolling_mean) / np.mean(rolling_mean) if np.mean(rolling_mean) != 0 else 0
        var_stationarity = np.std(rolling_std) / np.mean(rolling_std) if np.mean(rolling_std) != 0 else 0
        
        result = {
            "adf_test": {
                "test_statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "used_lag": int(adf_result[2]),
                "nobs": int(adf_result[3]),
                "critical_values": {k: float(v) for k, v in adf_result[4].items()}
            },
            "is_stationary": bool(adf_result[1] < 0.05),
            "rolling_statistics": {
                "window_size": rolling_window,
                "mean_variation": float(mean_stationarity),
                "std_variation": float(var_stationarity),
                "rolling_mean_first_few": [float(m) for m in rolling_mean[:3]] if len(rolling_mean) > 3 else [float(m) for m in rolling_mean],
                "rolling_mean_last_few": [float(m) for m in rolling_mean[-3:]] if len(rolling_mean) > 3 else [],
                "rolling_std_first_few": [float(s) for s in rolling_std[:3]] if len(rolling_std) > 3 else [float(s) for s in rolling_std],
                "rolling_std_last_few": [float(s) for s in rolling_std[-3:]] if len(rolling_std) > 3 else []
            },
            "status": "success"
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_message = f"Error testing stationarity: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def anomaly_detection(data: List[float], threshold: float = 3.0) -> str:
    """
    Detect anomalies in time series data using multiple methods.

    Args:
        data (List[float]): A list of numerical time series data.
        threshold (float, optional): Z-score threshold for anomaly detection. Defaults to 3.0.

    Returns:
        str: A JSON string containing anomaly detection results.
    """
    try:
        data_np = np.array(data)
        
        # Z-score method
        mean = np.mean(data_np)
        std = np.std(data_np)
        z_scores = (data_np - mean) / std
        
        # Points that exceed the threshold
        z_anomalies = np.where(np.abs(z_scores) > threshold)[0].tolist()
        
        # Moving average method
        window_size = min(20, len(data_np) // 10)
        if window_size < 2:
            window_size = 2
            
        # Calculate moving average
        ma = np.convolve(data_np, np.ones(window_size)/window_size, mode='valid')
        
        # Calculate residuals (for points where moving average exists)
        residuals = np.zeros_like(data_np)
        pad = (len(data_np) - len(ma)) // 2
        residuals[pad:pad+len(ma)] = data_np[pad:pad+len(ma)] - ma
        
        # Identify points where residuals are significant
        residual_mean = np.mean(residuals[pad:pad+len(ma)])
        residual_std = np.std(residuals[pad:pad+len(ma)])
        if residual_std > 0:
            residual_z_scores = (residuals[pad:pad+len(ma)] - residual_mean) / residual_std
            ma_anomalies = (pad + np.where(np.abs(residual_z_scores) > threshold)[0]).tolist()
        else:
            ma_anomalies = []
            
        # IQR method
        q25 = np.percentile(data_np, 25)
        q75 = np.percentile(data_np, 75)
        iqr = q75 - q25
        iqr_lower = q25 - 1.5 * iqr
        iqr_upper = q75 + 1.5 * iqr
        iqr_anomalies = np.where((data_np < iqr_lower) | (data_np > iqr_upper))[0].tolist()
        
        # Combine different methods, removing duplicates
        all_anomalies = sorted(list(set(z_anomalies + ma_anomalies + iqr_anomalies)))
        
        # Get values for the anomalies
        anomaly_values = [(int(idx), float(data_np[idx])) for idx in all_anomalies]
        
        result = {
            "z_score": {
                "threshold": float(threshold),
                "anomalies": z_anomalies[:20] if len(z_anomalies) > 20 else z_anomalies,
                "count": len(z_anomalies)
            },
            "moving_average": {
                "window_size": int(window_size),
                "anomalies": ma_anomalies[:20] if len(ma_anomalies) > 20 else ma_anomalies,
                "count": len(ma_anomalies)
            },
            "iqr": {
                "q25": float(q25),
                "q75": float(q75),
                "iqr": float(iqr),
                "lower_bound": float(iqr_lower),
                "upper_bound": float(iqr_upper),
                "anomalies": iqr_anomalies[:20] if len(iqr_anomalies) > 20 else iqr_anomalies,
                "count": len(iqr_anomalies)
            },
            "combined": {
                "anomalies": all_anomalies[:20] if len(all_anomalies) > 20 else all_anomalies,
                "anomaly_values": anomaly_values[:20] if len(anomaly_values) > 20 else anomaly_values,
                "count": len(all_anomalies)
            },
            "status": "success"
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error_message = f"Error detecting anomalies: {e}"
        return json.dumps({"status": "error", "message": error_message})