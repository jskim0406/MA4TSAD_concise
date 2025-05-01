"""
Visualization tools for time series data.
"""

import os
import json
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool


@tool
def ts2img(data: List[float], title: str = "Time Series Plot", highlight_indices: List[int] = None) -> str:
    """
    Generates a line plot from a list of numbers and saves it as a local PNG image.

    Args:
        data (List[float]): A list of numerical time series data to plot.
        title (str, optional): The title for the plot. Defaults to "Time Series Plot".
        highlight_indices (List[int], optional): Indices to highlight on the plot. Defaults to None.

    Returns:
        str: A JSON string with status, message, and image path information.
    """
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 이미지 저장 경로 설정 (상대 경로 또는 절대 경로 확인 필요)
        try:
            # __file__을 기준으로 상위 폴더 경로 계산
             img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
             # __file__ 변수가 없는 환경 (예: 인터랙티브 세션) 대비
             img_dir = Path("./temp_images") # 현재 작업 디렉토리 기준
        
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"timeseries_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Image saved to {save_path}")
        return json.dumps({
            "status": "success", 
            "message": f"Image generated: {save_path.name}", 
            "image_path": str(save_path)
        })
    
    except Exception as e:
        # 오류 발생 시에도 fig를 닫도록 시도
        try:
            plt.close(fig)
        except NameError: # fig가 정의되기 전에 오류 발생 시
            pass
        except Exception: # 닫기 중 다른 오류 발생 시
            pass
        error_message = f"Error generating image in ts2img: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})

@tool
def ts2img_with_anomalies(
    data: List[float], 
    anomaly_indices: List[int],
    title: str = "Time Series with Anomalies"
) -> str:
    """
    Generates a line plot highlighting anomalies and saves it as a local PNG image.

    Args:
        data (List[float]): A list of numerical time series data to plot.
        anomaly_indices (List[int]): Indices of anomalies to highlight.
        title (str, optional): The title for the plot. Defaults to "Time Series with Anomalies".

    Returns:
        str: A JSON string with status, message, and image path information.
    """
    fig = None # fig 변수 초기화
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
            
            # Add vertical spans for better visibility of anomalies
            for idx in valid_indices:
                ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        
        # Create directory for images if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
             img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
             img_dir = Path("./temp_images")
        
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"anomalies_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Image saved to {save_path}")
        return json.dumps({
            "status": "success", 
            "message": f"Image generated: {save_path.name}", 
            "image_path": str(save_path)
        })
    
    except Exception as e:
        if fig: # fig 객체가 생성되었다면 닫기 시도
             try:
                 plt.close(fig)
             except Exception:
                 pass
        error_message = f"Error generating image in ts2img_with_anomalies: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})


@tool
def ts2img_multi_view(
    data: List[float],
    window_size: int = 50,
    num_windows: int = 4,
    title: str = "Multi-Window Time Series View",
    highlight_indices: List[int] = None
) -> str:
    """
    Generates multiple windowed views of a time series to better visualize local patterns.

    Args:
        data (List[float]): A list of numerical time series data to plot.
        window_size (int, optional): Size of each window. Defaults to 50.
        num_windows (int, optional): Number of windows to display. Defaults to 4.
        title (str, optional): The title for the plot. Defaults to "Multi-Window Time Series View".
        highlight_indices (List[int], optional): Indices to highlight on the plot. Defaults to None.

    Returns:
        str: A JSON string with status, message, and image path information.
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
        num_windows = max(1, num_windows) # 최소 1개의 윈도우 보장
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(num_windows, 1, figsize=(12, num_windows * 3), sharex=False)
        if num_windows == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16)
        
        # Calculate window start positions
        step = (total_length - window_size) // (num_windows - 1) if num_windows > 1 else 0
        start_positions = [i * step for i in range(num_windows)] if num_windows > 1 else [0]
        
        for i, start_pos in enumerate(start_positions):
            end_pos = start_pos + window_size
            end_pos = min(end_pos, total_length)
            
            window_data = data_np[start_pos:end_pos]
            x_indices = range(start_pos, end_pos)
            
            axes[i].plot(x_indices, window_data, linewidth=1.5)
            
            # Highlight specific points if requested
            if highlight_indices:
                window_highlights = [idx for idx in highlight_indices if start_pos <= idx < end_pos]
                if window_highlights:
                    highlight_values = [data_np[idx] for idx in window_highlights]
                    axes[i].scatter(window_highlights, highlight_values, 
                                   color='red', s=50, zorder=5)
            
            axes[i].set_title(f"Window {i+1}: Indices {start_pos}-{end_pos-1}", fontsize=10)
            axes[i].set_ylabel("Value", fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            
            if i == num_windows - 1:  # Only add x-label to bottom subplot
                axes[i].set_xlabel("Time Index", fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle
        
        # Create directory for images if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"multiview_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Image saved to {save_path}")
        return json.dumps({
            "status": "success", 
            "message": f"Image generated: {save_path.name}", 
            "image_path": str(save_path)
        })
    
    except Exception as e:
        if fig: # fig 객체가 생성되었다면 닫기 시도
             try:
                 plt.close(fig)
             except Exception:
                 pass
        error_message = f"Error generating image in ts2img_multi_view: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})