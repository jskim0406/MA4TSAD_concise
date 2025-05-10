import io
import os
import json
import pytz
import base64
import numpy as np
import matplotlib
matplotlib.use('agg') # Non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_core.tools import tool

@tool
def ts2img(data: List[float], title: str = "Time Series Plot", highlight_indices: List[int] = None) -> str:
    """
    Generates a line plot of time series data and saves it as a local PNG image.

    Args:
        data (List[float]): Time series data to visualize.
        title (str, optional): Title for the plot in English. Defaults to "Time Series Plot".
        highlight_indices (List[int], optional): Indices to highlight on the plot. Defaults to None.

    Returns:
        str: JSON string containing status, message, and image path information.
    """
    fig = None
    save_path = None
    try:
        if not data:
            raise ValueError("Input data list is empty")

        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(10, 4))

        # Plot the main time series
        ax.plot(data_np, linewidth=1.5, label='Time Series') # label 변경

        # Highlight specific points if requested
        if highlight_indices:
            valid_indices = [idx for idx in highlight_indices if 0 <= idx < len(data_np)]
            if valid_indices:
                ax.scatter(valid_indices, data_np[valid_indices],
                          color='red', s=50, zorder=5, label='Highlighted Points') # label 변경

        ax.set_title(title, fontsize=14) # title 은 인자로 받음 (기본값 영어)
        ax.set_xlabel("Time Index", fontsize=10) # xlabel 변경
        ax.set_ylabel("Value", fontsize=10) # ylabel 변경
        ax.grid(True, linestyle='--', alpha=0.6)

        if highlight_indices:
            ax.legend()

        timestamp = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")

        # Set image save path
        try:
            # Attempt relative path assuming standard project structure
            img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
            # Fallback for environments where __file__ is not defined
            img_dir = Path("./temp_images")

        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"timeseries_{timestamp}.png"

        plt.savefig(save_path, dpi=100, bbox_inches='tight')


        print(f"Image saved path: {save_path}") # 로그 메시지 변경
        return json.dumps({
            "status": "success",
            "message": f"Image generated: {save_path.name}", # 메시지 변경
            "image_path": str(save_path)
        })

    except Exception as e:
        error_message = f"Image generation error: {e}" # 메시지 변경
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})
    finally:
        # Ensure the figure is closed even if errors occur
        if fig:
             try:
                 plt.close(fig)
             except Exception:
                 pass # Ignore errors during closing

@tool
def ts2img_with_anomalies(
    data: List[float],
    anomaly_indices: List[int],
    title: str = "Time Series with Anomalies" # 기본 title 변경
) -> str:
    """
    Generates a visualization of time series data highlighting anomalies.

    Args:
        data (List[float]): Time series data to visualize.
        anomaly_indices (List[int]): List of indices to mark as anomalies.
        title (str, optional): Title for the plot. Defaults to "Time Series with Anomalies".

    Returns:
        str: JSON string containing status, message, and image path information.
    """
    fig = None # Initialize fig variable
    save_path = None
    try:
        if not data:
            raise ValueError("Input data list is empty")

        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the main time series
        ax.plot(data_np, linewidth=1.5, label='Time Series', color='blue', alpha=0.7) # label 변경

        # Highlight anomalies
        valid_indices = [idx for idx in anomaly_indices if 0 <= idx < len(data_np)]
        if valid_indices:
            ax.scatter(valid_indices, data_np[valid_indices],
                      color='red', s=50, zorder=5, label='Anomalies') # label 변경

            # Add vertical lines for better visibility
            for idx in valid_indices:
                ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)

        ax.set_title(title, fontsize=14) # title 은 인자로 받음 (기본값 영어)
        ax.set_xlabel("Time Index", fontsize=10) # xlabel 변경
        ax.set_ylabel("Value", fontsize=10) # ylabel 변경
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')

        # Create image save directory
        timestamp = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        try:
            img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
            img_dir = Path("./temp_images")

        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"anomalies_{timestamp}.png"

        plt.savefig(save_path, dpi=100, bbox_inches='tight')


        print(f"Image saved path: {save_path}") # 로그 메시지 변경
        return json.dumps({
            "status": "success",
            "message": f"Image generated: {save_path.name}", # 메시지 변경
            "image_path": str(save_path)
        })

    except Exception as e:
        error_message = f"Image generation error: {e}" # 메시지 변경
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})
    finally:
        # Ensure the figure is closed even if errors occur
        if fig:
            try:
                plt.close(fig)
            except Exception:
                pass

@tool
def ts2img_multi_view(
    data: List[float],
    window_size: int = 50, # 타입을 int로 명시
    num_windows: int = 4,  # 타입을 int로 명시
    title: str = "Multi-Window Time Series View", # 기본 title 변경
    highlight_indices: List[int] = None
) -> str:
    """
    Generates multi-window time series views for better visualization of local patterns.

    Args:
        data (List[float]): Time series data to visualize.
        window_size (int, optional): Size of each window. Defaults to 50.
        num_windows (int, optional): Number of windows to display. Defaults to 4.
        title (str, optional): Title for the plot. Defaults to "Multi-Window Time Series View".
        highlight_indices (List[int], optional): List of indices to highlight. Defaults to None.

    Returns:
        str: JSON string containing status, message, and image path information.
    """
    fig = None
    save_path = None
    try:
        if not data:
            raise ValueError("Input data list is empty")

        data_np = np.array(data)
        total_length = len(data_np)

        # Adjust window parameters if necessary
        # Ensure window_size and num_windows are integers
        window_size = int(window_size)
        num_windows = int(num_windows)

        window_size = min(window_size, total_length // 2) if total_length > 1 else 1
        num_windows = min(num_windows, total_length // window_size) if window_size > 0 else 1
        num_windows = max(1, num_windows) # Ensure at least one window

        # Create figure with multiple subplots
        fig, axes = plt.subplots(num_windows, 1, figsize=(12, num_windows * 3), sharex=False)
        if num_windows == 1:
            axes = [axes] # Ensure axes is always iterable

        fig.suptitle(title, fontsize=16) # title 은 인자로 받음 (기본값 영어)

        # Calculate window start positions
        step = (total_length - window_size) // (num_windows - 1) if num_windows > 1 else 0
        start_positions = [i * step for i in range(num_windows)] if num_windows > 1 else [0]

        for i, start_pos in enumerate(start_positions):
            end_pos = start_pos + window_size
            end_pos = min(end_pos, total_length) # Prevent out-of-bounds

            window_data = data_np[start_pos:end_pos]
            x_indices = range(start_pos, end_pos)

            axes[i].plot(x_indices, window_data, linewidth=1.5)

            # Highlight specific points if requested
            if highlight_indices:
                window_highlights = [idx for idx in highlight_indices if start_pos <= idx < end_pos]
                if window_highlights:
                    highlight_values = data_np[window_highlights] # More efficient indexing
                    axes[i].scatter(window_highlights, highlight_values,
                                   color='red', s=50, zorder=5, label='Highlighted Points') # label 추가 및 변경

            axes[i].set_title(f"Window {i+1}: Indices {start_pos}-{end_pos-1}", fontsize=10) # title 변경
            axes[i].set_ylabel("Value", fontsize=10) # ylabel 변경
            axes[i].grid(True, linestyle='--', alpha=0.6)

            if i == num_windows - 1:  # Only add x-label to bottom subplot
                axes[i].set_xlabel("Time Index", fontsize=10) # xlabel 변경
            if highlight_indices and window_highlights: # 범례 추가
                 axes[i].legend()


        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for suptitle

        # Create image save directory
        timestamp = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        try:
            img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
            img_dir = Path("./temp_images")

        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"multiview_{timestamp}.png"

        plt.savefig(save_path, dpi=100, bbox_inches='tight')


        print(f"Image saved path: {save_path}") # 로그 메시지 변경
        return json.dumps({
            "status": "success",
            "message": f"Image generated: {save_path.name}", # 메시지 변경
            "image_path": str(save_path)
        })

    except Exception as e:
        error_message = f"Image generation error: {e}" # 메시지 변경
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})
    finally:
         # Ensure the figure is closed even if errors occur
         if fig:
             try:
                 plt.close(fig)
             except Exception:
                 pass