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
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import scoreatpercentile


def ts2img_bytes(data: List[float]) -> str:
    """
    Generates a line plot from a list of numbers and returns it as a base64 encoded PNG string.

    Args:
        data (List[float]): A list of numerical data to plot.

    Returns:
        str: A base64 encoded string representing the PNG image.
             Returns an error message string if plotting fails.
    """
    try:
        data_np = np.array(data)

        fig, ax = plt.subplots()
        ax.plot(data_np)
        ax.set_title("Time Series Plot")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)  # Close the figure to free memory
        buf.seek(0)

        # Encode bytes to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return image_base64

    except Exception as e:
        error_message = f"Error generating image: {e}"
        print(error_message)
        return error_message


def affine_transform(time_series: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Apply an affine transformation to a time series.
    
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
    
    # Apply the affine transformation
    transformed_series = shifted_series / a
    
    # If the minimum value in the original series is 0, keep it 0 after transformation
    transformed_series[time_series == 0] = 0
    
    return transformed_series


def fast_dtw_distance(series1: np.ndarray, series2: np.ndarray, dist_div_len: bool = False) -> float:
    """
    Calculate the Dynamic Time Warping (DTW) distance between two time series.
    
    Args:
        series1 (np.ndarray): First time series
        series2 (np.ndarray): Second time series
        dist_div_len (bool): Whether to divide distance by average length
        
    Returns:
        float: DTW distance between the series
    """
    series1 = series1.astype(float)
    series2 = series2.astype(float)

    distance, path = fastdtw(series1, series2, dist=lambda x, y: np.linalg.norm(x - y))
    if dist_div_len:
        avg_length = (len(series1) + len(series2)) / 2
        distance = distance / avg_length

    return distance


def find_most_similar_series_fast(
    X: np.ndarray, T_list: List[np.ndarray], top_k: int = 1, dist_div_len: bool = False
) -> Tuple[List[np.ndarray], List[float], List[int]]:
    """
    Find the most similar series to a given sequence.
    
    Args:
        X (np.ndarray): Query time series
        T_list (List[np.ndarray]): List of candidate time series
        top_k (int): Number of top matches to return
        dist_div_len (bool): Whether to divide distance by average length
        
    Returns:
        Tuple[List[np.ndarray], List[float], List[int]]: Top-k series, scores, and indices
    """
    # Use a heap to store the smallest k scores and the corresponding series
    heap = []
    
    # Iterate over all sequences
    for idx, Y in enumerate(T_list):
        score = fast_dtw_distance(X, Y, dist_div_len)
        
        # If the heap size is less than top_k, directly add
        if len(heap) < top_k:
            heapq.heappush(heap, (-score, idx, Y))
        else:
            # If the current score is smaller than the largest score in the heap, replace it
            heapq.heappushpop(heap, (-score, idx, Y))
    
    # Extract the top_k series, scores, and indices
    top_k_series = []
    top_k_scores = []
    top_k_indices = []
    
    # Sort the elements in the heap in ascending order of scores
    for score, idx, series in sorted(heap, reverse=True):
        top_k_scores.append(-score)  # Convert the score back to a positive value
        top_k_indices.append(idx)
        top_k_series.append(series)
    
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
            if start_index is not None and index - start_index >= min_len:
                zero_sequences.append(data.iloc[start_index:index][[value_col]])
                start_index = None
            else:
                start_index = None

        if start_index is not None and index - start_index + 1 >= max_len:
            end_index = index
            if end_index - start_index + 1 >= min_len:
                zero_sequences.append(data.iloc[start_index:end_index + 1][[value_col]])
            start_index = index - overlap + 1 if overlap > 0 else index + 1

    if start_index is not None and len(data) - start_index >= min_len:
        zero_sequences.append(data.iloc[start_index:][[value_col]])

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
    non_zero_sequences = []  # Used to store the values of anomaly segments
    sequence_strings = []    # Used to store the string representations
    label_sequences = []     # Used to store the labels

    current_values = []      # Values of the current anomaly segment
    current_labels = []      # Labels of the current anomaly segment
    current_string = ""      # String representation of the current anomaly segment
    in_anomaly = False       # Flag indicating if an anomaly segment is being recorded

    for index, row in data.iterrows():
        # For each anomaly point
        if row[label_col] == 1:
            if not in_anomaly:
                # Start recording a new anomaly segment, add left padding
                in_anomaly = True
                start_pad_index = max(0, index - pad_len)
                try:
                    for i in range(start_pad_index, index):
                        value = data.at[i, value_col]
                        label = data.at[i, label_col]
                        current_values.append(value)
                        current_labels.append(label)
                        current_string += "{},".format("*{}*".format(int(value)) if label == 1 else value)
                except:
                    print(f"Error adding left padding at index {index}")
                    
            # Add to the current anomaly segment
            current_values.append(row[value_col])
            current_labels.append(row[label_col])
            current_string += "*{}*,".format(int(row[value_col]))

            # Check if the maximum length limit is reached
            if len(current_values) >= max_len:
                # Trim the string to remove the last comma
                current_string = current_string.rstrip(',')
                # If current_labels are not all 1
                if 0 in current_labels:    
                    non_zero_sequences.append(current_values)
                    sequence_strings.append(current_string)
                    label_sequences.append(current_labels)

                # Reset the current anomaly segment
                current_values = []
                current_labels = []
                current_string = ""
                in_anomaly = False

        else:
            # If the current anomaly segment is not empty
            if in_anomaly:
                # Add right padding
                end_pad_index = min(len(data), index + pad_len)
                for i in range(index, end_pad_index):
                    value = data.at[i, value_col]
                    label = data.at[i, label_col]
                    current_values.append(value)
                    current_labels.append(label)
                    current_string += "{},".format("*{}*".format(int(value)) if label == 1 else int(value))

                # Trim the string to remove the last comma
                current_string = current_string.rstrip(',')
                if 0 in current_labels:   
                    non_zero_sequences.append(current_values)
                    sequence_strings.append(current_string)
                    label_sequences.append(current_labels)

                # Reset the current anomaly segment
                current_values = []
                current_labels = []
                current_string = ""
                in_anomaly = False

    # Check if there are any unsaved anomaly segments
    if in_anomaly:
        # Add right padding
        end_pad_index = min(len(data), index + 1 + pad_len)
        for i in range(index + 1, end_pad_index):
            value = data.at[i, 'value']
            label = data.at[i, 'label']
            current_values.append(value)
            current_labels.append(label)
            current_string += "{},".format("*{}*".format(int(value)) if label == 1 else int(value))

        # Trim the string to remove the last comma
        current_string = current_string.rstrip(',')
        if 0 in current_labels:   
            non_zero_sequences.append(current_values)
            sequence_strings.append(current_string)
            label_sequences.append(current_labels)

    return non_zero_sequences, sequence_strings, label_sequences


def detect_anomalies_z_score(
    time_series: List[float], 
    threshold: float = 3.0, 
    window_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Detect anomalies in a time series using Z-score method.
    
    Args:
        time_series (List[float]): Time series data
        threshold (float): Z-score threshold for anomaly detection
        window_size (Optional[int]): Size of moving window (None for global)
        
    Returns:
        Dict[str, Any]: Detected anomalies and analysis results
    """
    data = np.array(time_series)
    anomalies = []
    z_scores = []
    
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
        anomaly_values = [(i, float(data[i])) for i in anomaly_indices]
        
    else:
        # Rolling window Z-score method
        window_size = min(window_size, len(data) // 2)
        window_size = max(window_size, 2)  # Ensure window size is at least 2
        
        z_scores = np.zeros_like(data, dtype=float)
        
        for i in range(len(data)):
            # Define window boundaries
            start = max(0, i - window_size)
            end = min(len(data), i + window_size + 1)
            
            # Calculate z-score within window
            window = data[start:end]
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            if window_std > 0:
                z_scores[i] = (data[i] - window_mean) / window_std
            else:
                z_scores[i] = 0
        
        anomaly_indices = np.where(np.abs(z_scores) > threshold)[0].tolist()
        anomaly_values = [(i, float(data[i])) for i in anomaly_indices]
    
    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "z_scores": z_scores.tolist(),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
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
    data = np.array(time_series)
    
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    anomaly_indices = np.where((data < lower_bound) | (data > upper_bound))[0].tolist()
    anomaly_values = [(i, float(data[i])) for i in anomaly_indices]
    
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
    Detect anomalies using moving average residuals.
    
    Args:
        time_series (List[float]): Time series data
        window_size (int): Size of moving window
        threshold (float): Threshold for residual z-scores
        
    Returns:
        Dict[str, Any]: Detected anomalies and analysis results
    """
    data = np.array(time_series)
    
    if len(data) < window_size * 2:
        window_size = len(data) // 4
        window_size = max(window_size, 2)  # Minimum window size of 2
    
    # Calculate moving average
    ma = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Calculate residuals
    residuals = np.zeros_like(data, dtype=float)
    pad = (len(data) - len(ma)) // 2
    
    # Only set residuals for points where moving average exists
    residuals[pad:pad+len(ma)] = data[pad:pad+len(ma)] - ma
    
    # Calculate z-scores of residuals
    residual_mean = np.mean(residuals[pad:pad+len(ma)])
    residual_std = np.std(residuals[pad:pad+len(ma)])
    
    if residual_std > 0:
        residual_z_scores = np.zeros_like(residuals, dtype=float)
        residual_z_scores[pad:pad+len(ma)] = (residuals[pad:pad+len(ma)] - residual_mean) / residual_std
        anomaly_indices = (pad + np.where(np.abs(residual_z_scores[pad:pad+len(ma)]) > threshold)[0]).tolist()
    else:
        residual_z_scores = np.zeros_like(residuals, dtype=float)
        anomaly_indices = []
    
    anomaly_values = [(i, float(data[i])) for i in anomaly_indices]
    
    return {
        "anomaly_indices": anomaly_indices,
        "anomaly_values": anomaly_values,
        "window_size": window_size,
        "moving_averages": ma.tolist(),
        "residuals": residuals[pad:pad+len(ma)].tolist(),
        "residual_z_scores": residual_z_scores[pad:pad+len(ma)].tolist(),
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
    Detect anomalies using an ensemble of methods.
    
    Args:
        time_series (List[float]): Time series data
        z_score_threshold (float): Threshold for Z-score method
        iqr_k (float): Multiplier for IQR method
        ma_window_size (int): Window size for moving average method
        ma_threshold (float): Threshold for moving average method
        
    Returns:
        Dict[str, Any]: Combined anomaly detection results
    """
    # Apply individual methods
    z_score_results = detect_anomalies_z_score(time_series, z_score_threshold)
    iqr_results = detect_anomalies_iqr(time_series, iqr_k)
    ma_results = detect_anomalies_moving_average(time_series, ma_window_size, ma_threshold)
    
    # Combine anomaly indices
    all_indices = sorted(list(set(
        z_score_results["anomaly_indices"] + 
        iqr_results["anomaly_indices"] + 
        ma_results["anomaly_indices"]
    )))
    
    # Get anomaly values
    data = np.array(time_series)
    anomaly_values = [(i, float(data[i])) for i in all_indices]
    
    # Count detections per method
    detection_counts = {i: 0 for i in all_indices}
    for i in z_score_results["anomaly_indices"]:
        detection_counts[i] += 1
    for i in iqr_results["anomaly_indices"]:
        detection_counts[i] += 1
    for i in ma_results["anomaly_indices"]:
        detection_counts[i] += 1
    
    # Calculate confidence scores (1-3 based on how many methods detected it)
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