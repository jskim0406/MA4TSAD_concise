"""
Statistical analysis tools for time series data.
"""
import json
import numpy as np
import statsmodels  # statsmodels 임포트 추가 (필요시)

from typing import List, Dict, Any, Optional, Tuple, Union
from scipy import stats, signal

# statsmodels 관련 기능 임포트 (adfuller 등)
try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    # from statsmodels.tsa.ar_model import AutoReg # 필요시 주석 해제
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: 'statsmodels' is not installed. Some statistical tools might not be available.")
    STATSMODELS_AVAILABLE = False
    # statsmodels 함수를 사용하는 함수들을 비활성화하거나 대체 로직 추가 필요
    adfuller = lambda data: (0, 1.0, 0, len(data), {}, 0) # 기본 반환값 (stationarity_test 용)
    acf = lambda data, nlags, fft: np.zeros(nlags + 1) # 기본 반환값 (seasonality_analysis 용)
    seasonal_decompose = None # seasonality_analysis 에서 사용 여부 확인

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
        if not data: raise ValueError("Input data list is empty")
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
            "skewness": float(stats.skew(data_np)) if len(data_np) > 1 else 0.0, # 길이가 1이면 skew 계산 불가
            "kurtosis": float(stats.kurtosis(data_np)) if len(data_np) > 1 else 0.0, # 길이가 1이면 kurtosis 계산 불가
            "quartiles": [
                float(np.percentile(data_np, 25)),  # Q1
                float(np.percentile(data_np, 50)),  # Q2 (median)
                float(np.percentile(data_np, 75))   # Q3
            ] if len(data_np) > 0 else [0.0, 0.0, 0.0],
            "iqr": float(np.percentile(data_np, 75) - np.percentile(data_np, 25)) if len(data_np) > 0 else 0.0,
            "status": "success"
        }

        return json.dumps(result)

    except Exception as e:
        error_message = f"Error calculating statistics: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def trend_analysis(data: List[float]) -> str:
    """
    Analyze the trend in time series data using linear regression and Mann-Kendall test.

    Args:
        data (List[float]): A list of numerical time series data.

    Returns:
        str: A JSON string containing trend analysis.
    """
    try:
        if not data or len(data) < 2: raise ValueError("Not enough data for trend analysis (minimum 2 points required)")
        data_np = np.array(data)

        # Linear trend
        x = np.arange(len(data_np))
        slope, intercept, r_value, p_value_lin, std_err = stats.linregress(x, data_np)

        # Mann-Kendall trend test (Simplified implementation)
        n = len(data_np)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data_np[j] - data_np[i])

        # Variance of S (assuming no ties)
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        if var_s > 0: # Avoid division by zero if n is too small
            if s > 0:
                z = (s - 1) / np.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / np.sqrt(var_s)
            else:
                z = 0
            # Two-tailed p-value
            mk_p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            z = 0
            mk_p_value = 1.0 # Not significant if variance is zero

        result = {
            "linear_trend": {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value**2),
                "p_value": float(p_value_lin),
                "std_err": float(std_err),
                "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
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
    Analyze seasonality in time series data using ACF and optionally seasonal decomposition.

    Args:
        data (List[float]): A list of numerical time series data.
        period (int, optional): Suspected seasonality period. If None, attempts to detect it using FFT.

    Returns:
        str: A JSON string containing seasonality analysis.
    """
    try:
        if not data or len(data) < 4: raise ValueError("Not enough data for seasonality analysis (minimum 4 points required)")
        data_np = np.array(data)
        n = len(data_np)
        detected_period = period

        # Attempt to detect seasonality period if not provided or invalid
        if detected_period is None or detected_period < 2 or detected_period > n // 2:
            try:
                # Use FFT to find dominant frequencies
                fft_result = np.abs(np.fft.rfft(data_np - np.mean(data_np)))
                freqs = np.fft.rfftfreq(n)

                # Skip the DC component (0 frequency) and very high frequencies
                valid_indices = np.where((freqs > 1/(n/2)) & (freqs <= 0.5))[0] # Consider periods >= 2
                if len(valid_indices) > 0:
                    # Find index with max amplitude among valid frequencies
                    dominant_idx_in_valid = np.argmax(fft_result[valid_indices])
                    dominant_freq_idx = valid_indices[dominant_idx_in_valid]
                    dominant_freq = freqs[dominant_freq_idx]

                    if dominant_freq > 0:
                         calculated_period = int(round(1 / dominant_freq))
                         # Check if calculated period is reasonable
                         if 2 <= calculated_period <= n // 2:
                             detected_period = calculated_period
                             print(f"Seasonality period auto-detected via FFT: {detected_period}")
                         else:
                              detected_period = None # Invalid period calculated
                if detected_period is None:
                     # Fallback if FFT doesn't yield a good period
                     detected_period = min(max(2, n // 4), 12) # Default fallback (e.g., quarterly/monthly-ish)
                     print(f"Could not auto-detect period via FFT, using fallback: {detected_period}")

            except Exception as fft_err:
                 print(f"Warning: FFT period detection failed: {fft_err}. Using fallback.")
                 detected_period = min(max(2, n // 4), 12)


        # Ensure period is reasonable after detection/fallback
        final_period = max(2, min(detected_period, n // 2))

        # Autocorrelation function (ACF)
        decomposition_results = None
        acf_peaks_indices = []
        significant_acf = False
        acf_values_list = []

        if STATSMODELS_AVAILABLE:
            try:
                # Calculate ACF up to a reasonable lag
                max_lag = min(final_period * 3, n - 1) # Ensure lag is less than series length
                acf_values = acf(data_np, nlags=max_lag, fft=True)
                acf_values_list = [float(v) for v in acf_values[1:max_lag+1]] # Exclude lag 0

                # Find peaks in ACF (simple peak finding)
                for i in range(1, len(acf_values_list) - 1):
                     if acf_values_list[i] > acf_values_list[i-1] and acf_values_list[i] > acf_values_list[i+1] and acf_values_list[i] > 0.1: # Basic threshold
                          acf_peaks_indices.append(i + 1) # +1 because we start from lag 1

                significant_acf = any(v > 0.3 for v in acf_values_list) # Check for significant autocorrelation

                # Seasonal Decomposition (if possible and statsmodels is available)
                if seasonal_decompose and n >= 2 * final_period:
                    decomp = seasonal_decompose(data_np, model='additive', period=final_period)
                    # Extract finite values, handle NaNs from decomposition edges
                    trend_finite = decomp.trend[np.isfinite(decomp.trend)]
                    seasonal_finite = decomp.seasonal[np.isfinite(decomp.seasonal)]
                    resid_finite = decomp.resid[np.isfinite(decomp.resid)]
                    decomposition_results = {
                        "trend_sample": trend_finite[:5].tolist() if len(trend_finite) > 0 else [],
                        "seasonal_cycle": seasonal_finite[:final_period].tolist() if len(seasonal_finite) > 0 else [],
                        "resid_sample": resid_finite[:5].tolist() if len(resid_finite) > 0 else []
                    }
            except Exception as sm_err:
                print(f"Warning: Statsmodels ACF/Decomposition failed: {sm_err}")
                # Fallback or indicate failure
                decomposition_results = {"error": f"Statsmodels analysis failed: {sm_err}"}
                if not acf_values_list: # If ACF failed too
                     acf_values_list = []
                     significant_acf = False


        # Calculate seasonality strength (simple version: std dev of period averages)
        seasonal_strength_val = 0.0
        if n >= final_period:
             try:
                 # Reshape into periods, handling potential remainder
                 num_periods = n // final_period
                 reshaped_data = data_np[:num_periods * final_period].reshape(num_periods, final_period)
                 period_means = np.mean(reshaped_data, axis=0) # Average value for each point in the cycle
                 overall_std = np.std(data_np)
                 if overall_std > 0:
                      # Strength = variation within cycle / overall variation (simplified)
                      seasonal_strength_val = float(np.std(period_means) / overall_std)
             except Exception as strength_err:
                  print(f"Warning: Seasonality strength calculation failed: {strength_err}")


        result = {
            "analyzed_period": int(final_period),
            "autocorrelation": {
                "acf_values_short": acf_values_list[:min(10, len(acf_values_list))], # First 10 lags
                "acf_peaks": acf_peaks_indices[:5], # First 5 peaks
                "has_significant_autocorrelation": bool(significant_acf)
            },
            "seasonal_strength": seasonal_strength_val,
            "decomposition": decomposition_results,
            "status": "success"
        }

        return json.dumps(result)

    except Exception as e:
        error_message = f"Error analyzing seasonality: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def stationarity_test(data: List[float]) -> str:
    """
    Test for stationarity in time series data using the Augmented Dickey-Fuller test
    and checks rolling window statistics variation.

    Args:
        data (List[float]): A list of numerical time series data.

    Returns:
        str: A JSON string containing stationarity test results.
    """
    try:
        if not data or len(data) < 10: raise ValueError("Not enough data for stationarity test (minimum 10 points recommended)")
        data_np = np.array(data)
        n = len(data_np)

        # Augmented Dickey-Fuller test (only if statsmodels is available)
        adf_results_dict = None
        is_stationary_adf = None
        if STATSMODELS_AVAILABLE:
            try:
                adf_result = adfuller(data_np)
                adf_results_dict = {
                    "test_statistic": float(adf_result[0]),
                    "p_value": float(adf_result[1]),
                    "used_lag": int(adf_result[2]),
                    "nobs": int(adf_result[3]),
                    "critical_values": {k: float(v) for k, v in adf_result[4].items()}
                }
                is_stationary_adf = bool(adf_result[1] < 0.05) # Common significance level
            except Exception as adf_err:
                print(f"Warning: ADF test failed: {adf_err}")
                adf_results_dict = {"error": f"ADF test failed: {adf_err}"}
                is_stationary_adf = None
        else:
            adf_results_dict = {"error": "Statsmodels not available for ADF test"}
            is_stationary_adf = None

        # Rolling statistics variation check
        rolling_window = max(2, n // 10) # Window size, at least 2
        rolling_stats_results = {}
        if n >= rolling_window * 2: # Need at least two full windows
            try:
                 # Calculate rolling means and std deviations
                 rolling_mean = np.array([np.mean(data_np[i:i+rolling_window]) for i in range(n - rolling_window + 1)])
                 rolling_std = np.array([np.std(data_np[i:i+rolling_window]) for i in range(n - rolling_window + 1)])

                 # Calculate variation of rolling stats (coefficient of variation)
                 mean_of_means = np.mean(rolling_mean)
                 mean_of_stds = np.mean(rolling_std)
                 mean_cv = np.std(rolling_mean) / abs(mean_of_means) if mean_of_means != 0 else 0
                 std_cv = np.std(rolling_std) / abs(mean_of_stds) if mean_of_stds != 0 else 0

                 rolling_stats_results = {
                     "window_size": rolling_window,
                     "mean_coeff_variation": float(mean_cv),
                     "std_coeff_variation": float(std_cv),
                     # Indicate stationarity based on variation (e.g., CV < 0.1 might suggest stability)
                     "mean_is_stable": bool(mean_cv < 0.1),
                     "std_is_stable": bool(std_cv < 0.1),
                 }
            except Exception as roll_err:
                 print(f"Warning: Rolling statistics calculation failed: {roll_err}")
                 rolling_stats_results = {"error": f"Rolling stats failed: {roll_err}"}
        else:
            rolling_stats_results = {"error": "Not enough data for rolling statistics"}


        # Combine results
        # Overall stationarity conclusion could combine ADF and rolling stats checks
        is_stationary_combined = is_stationary_adf if is_stationary_adf is not None else rolling_stats_results.get("mean_is_stable", False) and rolling_stats_results.get("std_is_stable", False)

        result = {
            "adf_test": adf_results_dict,
            "rolling_stats_check": rolling_stats_results,
            "is_stationary_adf": is_stationary_adf, # Result from ADF test only
            "is_stationary_combined": is_stationary_combined, # Combined heuristic
            "status": "success"
        }

        return json.dumps(result)

    except Exception as e:
        error_message = f"Error testing stationarity: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def anomaly_detection(data: List[float], z_threshold: float = 3.0, iqr_multiplier: float = 1.5) -> str:
    """
    Detect anomalies in time series data using Z-score and IQR methods.

    Args:
        data (List[float]): A list of numerical time series data.
        z_threshold (float, optional): Z-score threshold for anomaly detection. Defaults to 3.0.
        iqr_multiplier (float, optional): Multiplier for IQR bounds. Defaults to 1.5.

    Returns:
        str: A JSON string containing anomaly detection results.
    """
    try:
        if not data or len(data) < 2: raise ValueError("Not enough data for anomaly detection")
        data_np = np.array(data)

        # Z-score method
        z_anomalies_indices = []
        mean = np.mean(data_np)
        std = np.std(data_np)
        if std > 0: # Avoid division by zero
            z_scores = (data_np - mean) / std
            z_anomalies_indices = np.where(np.abs(z_scores) > z_threshold)[0].tolist()
        else:
             z_scores = np.zeros_like(data_np) # All scores are 0 if std is 0

        # IQR method
        q25 = np.percentile(data_np, 25)
        q75 = np.percentile(data_np, 75)
        iqr = q75 - q25
        lower_bound = q25 - iqr_multiplier * iqr
        upper_bound = q75 + iqr_multiplier * iqr
        iqr_anomalies_indices = np.where((data_np < lower_bound) | (data_np > upper_bound))[0].tolist()

        # Combine results
        all_anomalies_indices = sorted(list(set(z_anomalies_indices + iqr_anomalies_indices)))

        # Get values for the anomalies
        anomaly_values = [(int(idx), float(data_np[idx])) for idx in all_anomalies_indices]

        # Limit the number of anomalies reported for brevity
        max_report_count = 20
        report_indices = all_anomalies_indices[:max_report_count]
        report_values = anomaly_values[:max_report_count]

        result = {
            "z_score_method": {
                "threshold": float(z_threshold),
                "detected_indices": z_anomalies_indices[:max_report_count],
                "count": len(z_anomalies_indices)
            },
            "iqr_method": {
                "multiplier": float(iqr_multiplier),
                "q1": float(q25), "q3": float(q75), "iqr": float(iqr),
                "lower_bound": float(lower_bound), "upper_bound": float(upper_bound),
                "detected_indices": iqr_anomalies_indices[:max_report_count],
                "count": len(iqr_anomalies_indices)
            },
            "combined_results": {
                "detected_indices": report_indices,
                "anomaly_values": report_values,
                "count": len(all_anomalies_indices) # Total count
            },
            "status": "success"
        }

        return json.dumps(result)

    except Exception as e:
        error_message = f"Error detecting anomalies: {e}"
        return json.dumps({"status": "error", "message": error_message})