"""
시계열 데이터 시각화 도구
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
from typing import List, Dict, Any, Optional, Union

from langchain_core.tools import tool


@tool
def ts2img(data: List[float], title: str = "시계열 데이터", highlight_indices: List[int] = None) -> str:
    """
    시계열 데이터의 선 그래프를 생성하고 로컬 PNG 이미지로 저장합니다.

    Args:
        data (List[float]): 시각화할 시계열 데이터
        title (str, optional): 그래프 제목. 기본값은 "시계열 데이터"
        highlight_indices (List[int], optional): 그래프에서 강조할 인덱스. 기본값은 None

    Returns:
        str: 상태, 메시지, 이미지 경로 정보가 포함된 JSON 문자열
    """
    try:
        if not data:
            raise ValueError("입력 데이터가 비어 있습니다")
        
        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # 주요 시계열 그리기
        ax.plot(data_np, linewidth=1.5, label='시계열 데이터')
        
        # 요청된 경우 특정 지점 강조
        if highlight_indices:
            valid_indices = [idx for idx in highlight_indices if 0 <= idx < len(data_np)]
            if valid_indices:
                ax.scatter(valid_indices, data_np[valid_indices], 
                          color='red', s=50, zorder=5, label='강조 지점')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("시간 인덱스", fontsize=10)
        ax.set_ylabel("값", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if highlight_indices:
            ax.legend()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # 이미지 저장 경로 설정
        try:
            img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
            img_dir = Path("./temp_images")
        
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"timeseries_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"이미지 저장 경로: {save_path}")
        return json.dumps({
            "status": "success", 
            "message": f"이미지 생성됨: {save_path.name}", 
            "image_path": str(save_path)
        })
    
    except Exception as e:
        # 오류 발생 시에도 fig 닫기 시도
        try:
            plt.close(fig)
        except:
            pass
        
        error_message = f"이미지 생성 오류: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})


@tool
def ts2img_with_anomalies(
    data: List[float], 
    anomaly_indices: List[int],
    title: str = "이상치가 표시된 시계열 데이터"
) -> str:
    """
    이상치가 강조된 시계열 데이터 시각화를 생성합니다.

    Args:
        data (List[float]): 시각화할 시계열 데이터
        anomaly_indices (List[int]): 이상치로 표시할 인덱스 목록
        title (str, optional): 그래프 제목. 기본값은 "이상치가 표시된 시계열 데이터"

    Returns:
        str: 상태, 메시지, 이미지 경로 정보가 포함된 JSON 문자열
    """
    fig = None # fig 변수 초기화
    try:
        if not data:
            raise ValueError("입력 데이터가 비어 있습니다")
        
        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 주요 시계열 그리기
        ax.plot(data_np, linewidth=1.5, label='시계열 데이터', color='blue', alpha=0.7)
        
        # 이상치 강조
        valid_indices = [idx for idx in anomaly_indices if 0 <= idx < len(data_np)]
        if valid_indices:
            ax.scatter(valid_indices, data_np[valid_indices], 
                      color='red', s=50, zorder=5, label='이상치')
            
            # 이상치 위치에 수직선 추가 (가시성 향상)
            for idx in valid_indices:
                ax.axvline(x=idx, color='red', linestyle='--', alpha=0.3)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("시간 인덱스", fontsize=10)
        ax.set_ylabel("값", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        
        # 이미지 저장 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
            img_dir = Path("./temp_images")
        
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"anomalies_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"이미지 저장 경로: {save_path}")
        return json.dumps({
            "status": "success", 
            "message": f"이미지 생성됨: {save_path.name}", 
            "image_path": str(save_path)
        })
    
    except Exception as e:
        if fig: 
            try:
                plt.close(fig)
            except Exception:
                pass
        error_message = f"이미지 생성 오류: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})


@tool
def ts2img_multi_view(
    data: List[float],
    window_size: int = 50,
    num_windows: int = 4,
    title: str = "다중 창 시계열 뷰",
    highlight_indices: List[int] = None
) -> str:
    """
    로컬 패턴을 더 잘 시각화하기 위한 다중 창 시계열 뷰를 생성합니다.

    Args:
        data (List[float]): 시각화할 시계열 데이터
        window_size (int, optional): 각 창의 크기. 기본값은 50
        num_windows (int, optional): 표시할 창 수. 기본값은 4
        title (str, optional): 그래프 제목. 기본값은 "다중 창 시계열 뷰"
        highlight_indices (List[int], optional): 강조할 인덱스 목록. 기본값은 None

    Returns:
        str: 상태, 메시지, 이미지 경로 정보가 포함된 JSON 문자열
    """
    fig = None
    try:
        if not data:
            raise ValueError("입력 데이터가 비어 있습니다")
        
        data_np = np.array(data)
        total_length = len(data_np)
        
        # 필요한 경우 창 매개변수 조정
        window_size = min(window_size, total_length // 2) if total_length > 1 else 1
        num_windows = min(num_windows, total_length // window_size) if window_size > 0 else 1
        num_windows = max(1, num_windows) # 최소 1개 창 보장
        
        # 여러 서브플롯이 있는 그림 생성
        fig, axes = plt.subplots(num_windows, 1, figsize=(12, num_windows * 3), sharex=False)
        if num_windows == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16)
        
        # 창 시작 위치 계산
        step = (total_length - window_size) // (num_windows - 1) if num_windows > 1 else 0
        start_positions = [i * step for i in range(num_windows)] if num_windows > 1 else [0]
        
        for i, start_pos in enumerate(start_positions):
            end_pos = start_pos + window_size
            end_pos = min(end_pos, total_length)
            
            window_data = data_np[start_pos:end_pos]
            x_indices = range(start_pos, end_pos)
            
            axes[i].plot(x_indices, window_data, linewidth=1.5)
            
            # 요청된 경우 특정 지점 강조
            if highlight_indices:
                window_highlights = [idx for idx in highlight_indices if start_pos <= idx < end_pos]
                if window_highlights:
                    highlight_values = [data_np[idx] for idx in window_highlights]
                    axes[i].scatter(window_highlights, highlight_values, 
                                   color='red', s=50, zorder=5)
            
            axes[i].set_title(f"창 {i+1}: 인덱스 {start_pos}-{end_pos-1}", fontsize=10)
            axes[i].set_ylabel("값", fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.6)
            
            if i == num_windows - 1:  # 가장 아래 서브플롯에만 x축 레이블 추가
                axes[i].set_xlabel("시간 인덱스", fontsize=10)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # suptitle을 위한 공간 확보
        
        # 이미지 저장 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        try:
            img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
            img_dir = Path("./temp_images")
        
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"multiview_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"이미지 저장 경로: {save_path}")
        return json.dumps({
            "status": "success", 
            "message": f"이미지 생성됨: {save_path.name}", 
            "image_path": str(save_path)
        })
    
    except Exception as e:
        if fig:
            try:
                plt.close(fig)
            except Exception:
                pass
        error_message = f"이미지 생성 오류: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})