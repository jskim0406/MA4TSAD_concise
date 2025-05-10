"""
Time series utility functions adapted from utils.py.
"""

import os
import io
import json
import heapq
import base64
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any, Union
from matplotlib import pyplot as plt

# fastdtw는 필요시 설치해야 할 수 있습니다: pip install fastdtw
try:
    from fastdtw import fastdtw
    FASTDTW_AVAILABLE = True
except ImportError:
    print("Warning: 'fastdtw' library not found. Some similarity functions might not be available.")
    FASTDTW_AVAILABLE = False

from scipy.spatial.distance import euclidean
from scipy.stats import scoreatpercentile


def ts2img_bytes(data: List[float]) -> str:
    """
    Generates a line plot from a list of numbers and returns it as a base64 encoded PNG string.
    (참고: 이 함수는 parallel_sentinel에서 직접 사용되지 않을 수 있으나, 유틸리티로 유지)

    Args:
        data (List[float]): A list of numerical data to plot.

    Returns:
        str: A base64 encoded string representing the PNG image.
             Returns an error message string if plotting fails.
    """
    fig = None
    try:
        if not data: raise ValueError("Input data list is empty")
        data_np = np.array(data)

        fig, ax = plt.subplots()
        ax.plot(data_np)
        ax.set_title("Time Series Plot")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode bytes to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return image_base64

    except Exception as e:
        error_message = f"Error generating image: {e}"
        print(error_message)
        return error_message
    finally:
        if fig is not None:
            plt.close(fig)  # Close the figure to free memory


def affine_transform(time_series: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Apply an affine transformation to a time series.
    (참고: 이 함수는 parallel_sentinel에서 직접 사용되지 않을 수 있으나, 유틸리티로 유지)

    Args:
        time_series (np.ndarray): Input time series data
        alpha (float): Alpha parameter for scaling
        beta (float): Beta parameter for shifting

    Returns:
        np.ndarray: Transformed time series
    """
    # Calculate the 0.1% and 99% percentiles
    min_val = np.percentile(time_series, 0.1)
    max_val = np.percentile(time_series, 99)
    b = min_val - beta * (max_val - min_val)

    # Shift the series
    shifted_series = time_series - b

    # Calculate the value of 'a' using scipy's scoreatpercentile function
    a = scoreatpercentile(shifted_series, alpha)

    # If the values in the series are very small, limit 'a' to a maximum of 0.01
    if np.all(np.abs(shifted_series) < 1e-3):
        a = min(a, 0.01)

    # Handle potential division by zero or very small 'a'
    if abs(a) < 1e-9:
         print(f"Warning: Affine transform parameter 'a' is very small ({a}). Avoiding division by zero.")
         # Return shifted series or original series depending on context
         return shifted_series # or time_series depending on desired behavior

    # Apply the affine transformation
    transformed_series = shifted_series / a

    # If the minimum value in the original series is 0, keep it 0 after transformation
    transformed_series[time_series == 0] = 0

    return transformed_series


def fast_dtw_distance(series1: np.ndarray, series2: np.ndarray, dist_div_len: bool = False) -> float:
    """
    Calculate the Dynamic Time Warping (DTW) distance between two time series.
    (참고: 이 함수는 parallel_sentinel에서 직접 사용되지 않을 수 있으나, 유틸리티로 유지)

    Args:
        series1 (np.ndarray): First time series
        series2 (np.ndarray): Second time series
        dist_div_len (bool): Whether to divide distance by average length

    Returns:
        float: DTW distance between the series
    """
    if not FASTDTW_AVAILABLE:
        raise ImportError("fastdtw library is required for DTW distance calculation.")

    series1 = series1.astype(float)
    series2 = series2.astype(float)

    distance, path = fastdtw(series1, series2, dist=lambda x, y: np.linalg.norm(x - y))
    if dist_div_len:
        avg_length = (len(series1) + len(series2)) / 2
        if avg_length == 0: return 0.0 # Handle zero length case
        distance = distance / avg_length

    return distance


def find_most_similar_series_fast(
    X: np.ndarray, T_list: List[np.ndarray], top_k: int = 1, dist_div_len: bool = False
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """
    Find the most similar series to a given sequence using FastDTW.
    (참고: 이 함수는 parallel_sentinel에서 직접 사용되지 않을 수 있으나, 유틸리티로 유지)

    Args:
        X (np.ndarray): Query time series
        T_list (List[np.ndarray]): List of candidate time series
        top_k (int): Number of top matches to return
        dist_div_len (bool): Whether to divide distance by average length

    Returns:
        Tuple[List[np.ndarray], List[float], List[int]]: Top-k series, scores, and indices
    """
    if not FASTDTW_AVAILABLE:
        raise ImportError("fastdtw library is required for finding similar series.")

    # Use a heap to store the smallest k scores and the corresponding series
    heap = []

    # Iterate over all sequences
    for idx, Y in enumerate(T_list):
        if len(X) == 0 or len(Y) == 0: continue # Skip empty series
        score = fast_dtw_distance(X, Y, dist_div_len)

        # If the heap size is less than top_k, directly add
        # We store (-score) because heapq is a min-heap
        if len(heap) < top_k:
            heapq.heappush(heap, (-score, idx, Y))
        else:
            # If the current score is smaller than the largest score in the heap (smallest -score), replace it
            heapq.heappushpop(heap, (-score, idx, Y))

    # Extract the top_k series, scores, and indices
    top_k_results = sorted(heap, reverse=True) # Sort by (-score) descending, which is score ascending

    top_k_series = [series for _, _, series in top_k_results]
    top_k_scores = [-score for score, _, _ in top_k_results] # Convert scores back to positive
    top_k_indices = [idx for _, idx, _ in top_k_results]

    return top_k_series, top_k_scores, top_k_indices


def find_zero_sequences(
    data: pd.DataFrame,
    min_len: int = 100,
    max_len: int = 800,
    overlap: int = 0,
    value_col: str = 'value',
    label_col: str = 'label'
) -> List[pd.DataFrame]:
    """
    Find sequences of zeros (normal behavior) in a labeled time series.
    (참고: 이 함수는 parallel_sentinel에서 직접 사용되지 않을 수 있으나, 유틸리티로 유지)

    Args:
        data (pd.DataFrame): DataFrame with time series data and labels
        min_len (int): Minimum length of zero sequences
        max_len (int): Maximum length of zero sequences
        overlap (int): Number of overlapping points between sequences
        value_col (str): Name of column containing values
        label_col (str): Name of column containing labels (0 for normal, 1 for anomaly)

    Returns:
        List[pd.DataFrame]: List of zero sequences
    """
    zero_sequences = []
    start_index = None

    for index, row in data.iterrows():
        if row[label_col] == 0:
            if start_index is None:
                start_index = index
        else:
            if start_index is not None:
                current_len = index - start_index
                if current_len >= min_len:
                    # Append sequences up to max_len
                    for sub_start in range(start_index, index, max_len - overlap):
                        sub_end = min(sub_start + max_len, index)
                        if sub_end - sub_start >= min_len:
                            zero_sequences.append(data.loc[sub_start:sub_end-1, [value_col]])
                        if sub_end == index:
                            break
            start_index = None # Reset after finding an anomaly or end of sequence

        # Check if current sequence exceeds max_len
        if start_index is not None and (index - start_index + 1) >= max_len:
             end_index = start_index + max_len
             zero_sequences.append(data.loc[start_index : end_index-1, [value_col]])
             # Update start_index for the next potential sequence with overlap
             start_index = end_index - overlap if overlap > 0 else end_index


    # Check for the last sequence
    if start_index is not None:
        current_len = len(data) - start_index
        if current_len >= min_len:
            for sub_start in range(start_index, len(data), max_len - overlap):
                sub_end = min(sub_start + max_len, len(data))
                if sub_end - sub_start >= min_len:
                    zero_sequences.append(data.loc[sub_start:sub_end-1, [value_col]])
                if sub_end == len(data):
                    break


    return zero_sequences


def find_anomalies(
    data: pd.DataFrame,
    pad_len: int = 5,
    max_len: int = 800,
    value_col: str = 'value',
    label_col: str = 'label'
) -> Tuple[List[List[float]], List[str], List[List[int]]]:
    """
    Find and extract anomaly sequences from labeled time series data.
    (참고: 이 함수는 parallel_sentinel에서 직접 사용되지 않을 수 있으나, 유틸리티로 유지)

    Args:
        data (pd.DataFrame): DataFrame with time series data and labels
        pad_len (int): Padding length around anomalies
        max_len (int): Maximum length of anomaly sequences
        value_col (str): Name of column containing values
        label_col (str): Name of column containing labels (0 for normal, 1 for anomaly)

    Returns:
        Tuple[List[List[float]], List[str], List[List[int]]]:
            Anomaly sequences, their string representations, and labels
    """
    anomaly_sequences_values = [] # Store values
    anomaly_sequences_labels = [] # Store labels
    sequence_strings = []         # Store string representations

    start_index = -1
    for index, row in data.iterrows():
        if row[label_col] == 1 and start_index == -1: # Anomaly starts
            start_index = index
        elif row[label_col] == 0 and start_index != -1: # Anomaly ends
            end_index = index - 1
            # Extract sequence with padding
            seq_start = max(0, start_index - pad_len)
            seq_end = min(len(data), end_index + pad_len + 1)

            # Ensure sequence length doesn't exceed max_len
            if seq_end - seq_start > max_len:
                 # Prioritize keeping the anomaly core
                 core_len = end_index - start_index + 1
                 padding_needed = max_len - core_len
                 left_pad = min(pad_len, padding_needed // 2)
                 right_pad = min(pad_len, padding_needed - left_pad)
                 seq_start = max(0, start_index - left_pad)
                 seq_end = min(len(data), end_index + right_pad + 1)

            segment_df = data.iloc[seq_start:seq_end]
            values = segment_df[value_col].tolist()
            labels = segment_df[label_col].astype(int).tolist() # Ensure labels are int

            # Create string representation
            seq_str = ",".join([f"*{int(v)}*" if lbl == 1 else str(int(v)) for v, lbl in zip(values, labels)])

            anomaly_sequences_values.append(values)
            anomaly_sequences_labels.append(labels)
            sequence_strings.append(seq_str)

            start_index = -1 # Reset

    # Handle case where anomaly continues to the end
    if start_index != -1:
        end_index = len(data) - 1
        seq_start = max(0, start_index - pad_len)
        seq_end = len(data) # Go to the end

        if seq_end - seq_start > max_len:
            core_len = end_index - start_index + 1
            padding_needed = max_len - core_len
            left_pad = min(pad_len, padding_needed) # Only left padding possible
            seq_start = max(0, start_index - left_pad)
            seq_end = seq_start + max_len # Truncate

        segment_df = data.iloc[seq_start:seq_end]
        values = segment_df[value_col].tolist()
        labels = segment_df[label_col].astype(int).tolist()
        seq_str = ",".join([f"*{int(v)}*" if lbl == 1 else str(int(v)) for v, lbl in zip(values, labels)])

        anomaly_sequences_values.append(values)
        anomaly_sequences_labels.append(labels)
        sequence_strings.append(seq_str)

    return anomaly_sequences_values, sequence_strings, anomaly_sequences_labels


# --- Anomaly Detection Methods (as used by Advanced Anomaly Detector in federated_sentinel) ---

def detect_anomalies_z_score(
    time_series: List[float],
    threshold: float = 3.0,
    window_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect anomalies in a time series using Z-score method (global or rolling).

    Args:
        time_series (List[float]): Time series data
        threshold (float): Z-score threshold for anomaly detection
        window_size (Optional[int]): Size of moving window (None for global)

    Returns:
        Dict[str, Any]: Detected anomalies and analysis results
    """
    if not time_series: return {"anomaly_indices": [], "anomaly_values": [], "z_scores": [], "mean": 0.0, "std": 0.0, "threshold": threshold}
    data = np.array(time_series)
    anomalies = []
    z_scores = np.zeros_like(data, dtype=float)
    anomaly_indices = []
    mean = 0.0
    std = 0.0

    if window_size is None:
        # Global Z-score method
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:  # Handle zero standard deviation
            return {
                "anomaly_indices": [],
                "anomaly_values": [],
                "z_scores": [0] * len(data),
                "mean": float(mean),
                "std": 0.0,
                "threshold": threshold
            }

        z_scores = (data - mean) / std
        anomaly_indices = np.where(np.abs(z_scores) > threshold)[0].tolist()

    else:
        # Rolling window Z-score method
        window_size = max(2, min(window_size, len(data) // 2)) # Ensure reasonable window size
        mean = np.mean(data) # Still report global mean/std
        std = np.std(data)

        for i in range(len(data)):
            # Define window boundaries more carefully
            start = max(0, i - window_size // 2)
            end = min(len(data), i + (window_size + 1) // 2)
            if end - start < 2 : continue # Skip if window is too small

            window = data[start:end]
            window_mean = np.mean(window)
            window_std = np.std(window)

            if window_std > 1e-6: # Use small tolerance for std dev check
                z_scores[i] = (data[i] - window_mean) / window_std
            else:
                z_scores[i] = 0 if data[i] == window_mean else np.inf * np.sign(data[i] - window_mean) # Handle zero std

        anomaly_indices = np.where(np.abs(z_scores) > threshold)[0].tolist()

    anomaly_values = [(int(i), float(data[i])) for i in anomaly_indices]

    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "z_scores": z_scores.tolist(),
        "mean": float(mean), # Report global mean
        "std": float(std),   # Report global std
        "threshold": threshold
    }


def detect_anomalies_iqr(time_series: List[float], k: float = 1.5) -> Dict[str, Any]:
    """
    Detect anomalies in a time series using the IQR method.

    Args:
        time_series (List[float]): Time series data
        k (float): Multiplier for IQR range

    Returns:
        Dict[str, Any]: Detected anomalies and analysis results
    """
    if not time_series: return {"anomaly_indices": [], "anomaly_values": [], "q1": 0.0, "q3": 0.0, "iqr": 0.0, "lower_bound": 0.0, "upper_bound": 0.0, "k": k}
    data = np.array(time_series)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    anomaly_indices = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
    anomaly_values = [(int(i), float(data[i])) for i in anomaly_indices]

    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "k": k
    }


def detect_anomalies_moving_average(
    time_series: List[float],
    window_size: int = 10,
    threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Detect anomalies using moving average residuals' Z-scores.

    Args:
        time_series (List[float]): Time series data
        window_size (int): Size of moving window
        threshold (float): Threshold for residual z-scores

    Returns:
        Dict[str, Any]: Detected anomalies and analysis results
    """
    if not time_series: return {"anomaly_indices": [], "anomaly_values": [], "window_size": window_size, "moving_averages": [], "residuals": [], "residual_z_scores": [], "threshold": threshold}
    data = np.array(time_series)
    n = len(data)

    # Adjust window size if it's too large or small
    window_size = max(2, min(window_size, n // 2 if n >= 4 else n))

    if n < window_size:
         return {"anomaly_indices": [], "anomaly_values": [], "window_size": window_size, "moving_averages": [], "residuals": [], "residual_z_scores": [], "threshold": threshold, "error": "Not enough data for window size"}

    # Calculate moving average using pandas for easier handling of edges
    series = pd.Series(data)
    ma = series.rolling(window=window_size, center=True, min_periods=1).mean().values
    residuals = data - ma

    # Calculate z-scores of residuals, ignoring NaNs potentially caused by rolling window at edges
    valid_residuals = residuals[~np.isnan(residuals)]
    if len(valid_residuals) < 2: # Need at least 2 points to calculate std dev
         residual_mean = 0.0
         residual_std = 0.0
         residual_z_scores = np.zeros_like(data, dtype=float)
         anomaly_indices = []
    else:
        residual_mean = np.mean(valid_residuals)
        residual_std = np.std(valid_residuals)
        if residual_std > 1e-6:
            residual_z_scores = np.where(np.isnan(residuals), 0, (residuals - residual_mean) / residual_std)
            anomaly_indices = np.where(np.abs(residual_z_scores) > threshold)[0].tolist()
        else:
            residual_z_scores = np.zeros_like(data, dtype=float)
            anomaly_indices = [] # No anomalies if std dev is zero


    anomaly_values = [(int(i), float(data[i])) for i in anomaly_indices]

    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "window_size": window_size,
        "moving_averages": ma.tolist(), # Include NaNs for context
        "residuals": residuals.tolist(), # Include NaNs for context
        "residual_z_scores": residual_z_scores.tolist(), # Zeros where residual was NaN
        "threshold": threshold
    }


def detect_anomalies_ensemble(
    time_series: List[float],
    z_score_threshold: float = 3.0,
    iqr_k: float = 1.5,
    ma_window_size: int = 10,
    ma_threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Detect anomalies using an ensemble of methods (Z-score, IQR, Moving Average Residuals).

    Args:
        time_series (List[float]): Time series data
        z_score_threshold (float): Threshold for Z-score method
        iqr_k (float): Multiplier for IQR method
        ma_window_size (int): Window size for moving average method
        ma_threshold (float): Threshold for moving average method

    Returns:
        Dict[str, Any]: Combined anomaly detection results with confidence scores.
    """
    if not time_series: return {"anomaly_indices": [], "anomaly_values": [], "confidence_scores": {}, "z_score_anomalies": [], "iqr_anomalies": [], "ma_anomalies": [], "detection_counts": {}, "num_anomalies": 0}

    # Apply individual methods
    z_score_results = detect_anomalies_z_score(time_series, z_score_threshold)
    iqr_results = detect_anomalies_iqr(time_series, iqr_k)
    ma_results = detect_anomalies_moving_average(time_series, ma_window_size, ma_threshold)

    # Combine anomaly indices
    all_indices_set = set(z_score_results["anomaly_indices"]) | set(iqr_results["anomaly_indices"]) | set(ma_results["anomaly_indices"])
    all_indices = sorted(list(all_indices_set))

    # Get anomaly values
    data = np.array(time_series)
    anomaly_values = [(int(i), float(data[i])) for i in all_indices]

    # Count detections per method
    detection_counts = {i: 0 for i in all_indices}
    for i in z_score_results["anomaly_indices"]: detection_counts[i] += 1
    for i in iqr_results["anomaly_indices"]: detection_counts[i] += 1
    for i in ma_results["anomaly_indices"]: detection_counts[i] += 1

    # Calculate confidence scores (normalized count 0-1)
    confidence_scores = {i: count / 3.0 for i, count in detection_counts.items()}

    return {
        "anomaly_indices": all_indices,
        "anomaly_values": anomaly_values,
        "confidence_scores": confidence_scores,
        "z_score_anomalies": z_score_results["anomaly_indices"],
        "iqr_anomalies": iqr_results["anomaly_indices"],
        "ma_anomalies": ma_results["anomaly_indices"],
        "detection_counts": detection_counts,
        "num_anomalies": len(all_indices)
    }