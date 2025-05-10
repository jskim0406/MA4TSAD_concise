"""
시계열 데이터 시각화 도구
이 모듈은 시계열 데이터 시각화를 위한 다양한 도구를 제공합니다.
"""

import io
import os
import json
import pytz
import base64
import numpy as np
import matplotlib
matplotlib.use('agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_core.tools import tool


@tool
def ts2img_bytes(data: List[float], title: str = "Time Series Plot", highlight_indices: List[int] = None) -> str:
    """
    Generates a line plot from time series data and returns it as a base64 encoded PNG string.
    
    Args:
        data (List[float]): Time series data to visualize.
        title (str, optional): Title for the plot. Defaults to "Time Series Plot".
        highlight_indices (List[int], optional): Indices to highlight on the plot. Defaults to None.
        
    Returns:
        str: Base64 encoded string representing the PNG image.
    """
    fig = None
    try:
        if not data:
            raise ValueError("Input data list is empty")

        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot the main time series
        ax.plot(data_np, linewidth=1.5, label='Time Series')

        # Highlight specific points if requested
        if highlight_indices:
            valid_indices = [idx for idx in highlight_indices if 0 <= idx < len(data_np)]
            if valid_indices:
                ax.scatter(valid_indices, data_np[valid_indices],
                           color='red', s=50, zorder=5, label='Highlighted Points')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)

        if highlight_indices:
            ax.legend()

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


@tool
def ts2img_with_anomalies(data: List[float], anomaly_indices: List[int], title: str = "Time Series with Anomalies") -> str:
    """
    Generates a visualization of time series data highlighting anomalies and returns as base64 encoded PNG.
    
    Args:
        data (List[float]): Time series data to visualize.
        anomaly_indices (List[int]): List of indices to mark as anomalies.
        title (str, optional): Title for the plot. Defaults to "Time Series with Anomalies".
        
    Returns:
        str: Base64 encoded string representing the PNG image.
    """
    fig = None
    try:
        if not data:
            raise ValueError("Input data list is empty")

        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the main time series
        ax.plot(data_np, linewidth=1.5, label='Time Series', color='blue', alpha=0.7)

        # Highlight anomalies
        valid_indices = [idx for idx in anomaly_indices if 0 <= idx < len(data_np)]
        if valid_indices:
            ax.scatter(valid_indices, data_np[valid_indices],
                      color='red', s=50, zorder=5, label='Anomalies')

            # Add vertical lines for better visibility
            for idx in valid_indices:
                ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode bytes to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return image_base64

    except Exception as e:
        error_message = f"Error generating anomaly visualization: {e}"
        print(error_message)
        return error_message
    finally:
        if fig is not None:
            plt.close(fig)


@tool
def ts2img_multi_view(data: List[float], window_size: int = 50, num_windows: int = 4, 
                     title: str = "Multi-Window Time Series View", highlight_indices: List[int] = None) -> str:
    """
    Generates multi-window time series views for better visualization of local patterns.
    Returns as base64 encoded PNG.
    
    Args:
        data (List[float]): Time series data to visualize.
        window_size (int, optional): Size of each window. Defaults to 50.
        num_windows (int, optional): Number of windows to display. Defaults to 4.
        title (str, optional): Title for the plot. Defaults to "Multi-Window Time Series View".
        highlight_indices (List[int], optional): List of indices to highlight. Defaults to None.
        
    Returns:
        str: Base64 encoded string representing the PNG image.
    """
    fig = None
    try:
        if not data:
            raise ValueError("Input data list is empty")

        data_np = np.array(data)
        total_length = len(data_np)

        # Adjust window parameters if necessary
        window_size = min(window_size, total_length // 2) if total_length > 1 else 1
        num_windows = min(num_windows, total_length // window_size) if window_size > 0 else 1
        num_windows = max(1, num_windows)  # Ensure at least one window

        # Create figure with multiple subplots
        fig, axes = plt.subplots(num_windows, 1, figsize=(12, num_windows * 3), sharex=False)
        if num_windows == 1:
            axes = [axes]  # Ensure axes is always iterable

        fig.suptitle(title, fontsize=16)

        # Calculate window start positions
        step = (total_length - window_size) // (num_windows - 1) if num_windows > 1 else 0
        start_positions = [i * step for i in range(num_windows)] if num_windows > 1 else [0]

        for i, start_pos in enumerate(start_positions):
            end_pos = start_pos + window_size
            end_pos = min(end_pos, total_length)  # Prevent out-of-bounds

            window_data = data_np[start_pos:end_pos]
            x_indices = range(start_pos, end_pos)

            axes[i].plot(x_indices, window_data, linewidth=1.5)

            # Highlight specific points if requested
            if highlight_indices:
                window_highlights = [idx for idx in highlight_indices if start_pos <= idx < end_pos]
                if window_highlights:
                    highlight_values = data_np[window_highlights]
                    axes[i].scatter(window_highlights, highlight_values,
                                   color='red', s=50, zorder=5, label='Highlighted Points')

            axes[i].set_title(f"Window {i+1}: Indices {start_pos}-{end_pos-1}", fontsize=10)
            axes[i].set_ylabel("Value", fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.6)

            if i == num_windows - 1:  # Only add x-label to bottom subplot
                axes[i].set_xlabel("Time Index", fontsize=10)
            if highlight_indices and any(start_pos <= idx < end_pos for idx in highlight_indices):
                axes[i].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for suptitle

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Encode bytes to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return image_base64

    except Exception as e:
        error_message = f"Error generating multi-window visualization: {e}"
        print(error_message)
        return error_message
    finally:
        if fig is not None:
            plt.close(fig)