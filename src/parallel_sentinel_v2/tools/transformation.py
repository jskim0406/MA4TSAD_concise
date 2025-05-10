"""
시계열 데이터 변환 도구
이 모듈은, 시계열 데이터를 다양하게 변환하기 위한 도구들을 제공합니다.
"""

import io
import json
import pytz
import base64
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union

from langchain_core.tools import tool


@tool
def get_time_series_decomposition(data: List[float], period: int = None) -> str:
    """
    Decomposes time series into trend, seasonal, and residual components and saves visualization locally.
    If period is not provided, attempts to detect it automatically.
    
    Args:
        data (List[float]): Time series data
        period (int, optional): Seasonality period. Defaults to None (auto-detect).
        
    Returns:
        str: JSON string containing decomposition results and image path
    """
    try:
        try:
            from statsmodels.tsa.seasonal import STL
            STATSMODELS_AVAILABLE = True
        except ImportError:
            STATSMODELS_AVAILABLE = False
            return json.dumps({
                "status": "error", 
                "message": "statsmodels package not installed. Install with: pip install statsmodels"
            })
        
        if not STATSMODELS_AVAILABLE:
            return json.dumps({
                "status": "error", 
                "message": "statsmodels package not available"
            })
            
        if not data:
            raise ValueError("Input data list is empty")
            
        data_np = np.array(data)
        n = len(data_np)
        
        # Auto-detect period if not provided
        if period is None:
            # Use FFT to find dominant frequency
            fft_result = np.fft.rfft(data_np - np.mean(data_np))
            fft_mag = np.abs(fft_result)
            freqs = np.fft.rfftfreq(n)
            
            # Filter out DC component (index 0)
            valid_indices = np.where((freqs > 0) & (freqs <= 0.5))[0]
            if len(valid_indices) > 0:
                dominant_idx = valid_indices[np.argmax(fft_mag[valid_indices])]
                if freqs[dominant_idx] > 0:
                    period = int(round(1 / freqs[dominant_idx]))
                    period = min(max(2, period), n // 2)  # Ensure reasonable period
                else:
                    period = min(10, n // 2)  # Default fallback
            else:
                period = min(10, n // 2)  # Default fallback
        
        # Ensure minimum data length for decomposition
        if n < 2 * period:
            raise ValueError(f"Time series too short for decomposition with period {period}. Need at least {2*period} points.")
            
        # Perform decomposition with STL
        from statsmodels.tsa.seasonal import STL
        stl = STL(data_np, period=period, robust=True)
        res = stl.fit()
        
        trend = res.trend
        seasonal = res.seasonal
        residual = res.resid
        
        # Calculate component strengths
        var_components = {
            "trend": np.var(trend),
            "seasonal": np.var(seasonal),
            "residual": np.var(residual),
            "total": np.var(data_np)
        }
        
        strengths = {
            "trend_strength": max(0, 1 - var_components["residual"] / 
                               (var_components["trend"] + var_components["residual"])),
            "seasonal_strength": max(0, 1 - var_components["residual"] / 
                                  (var_components["seasonal"] + var_components["residual"])),
            "residual_strength": var_components["residual"] / var_components["total"]
        }
        
        # Visualize components - 모든 텍스트를 영어로만 사용
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Original data
        axes[0].plot(data_np, label='Original')
        axes[0].set_title("Original Time Series", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Trend
        axes[1].plot(trend, label=f'Trend (Strength: {strengths["trend_strength"]:.3f})', color='blue')
        axes[1].set_title("Trend Component", fontsize=12)
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        # Seasonal
        axes[2].plot(seasonal, label=f'Seasonal (Strength: {strengths["seasonal_strength"]:.3f}, Period: {period})', 
                     color='green')
        axes[2].set_title("Seasonal Component", fontsize=12)
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)
        
        # Residual
        axes[3].plot(residual, label=f'Residual (Strength: {strengths["residual_strength"]:.3f})', 
                     color='red')
        axes[3].set_title("Residual Component", fontsize=12)
        axes[3].legend()
        axes[3].grid(True, linestyle='--', alpha=0.6)
        axes[3].set_xlabel("Time Index", fontsize=10)
        
        plt.tight_layout()
        
        # 이미지 파일로 저장 (base64 인코딩 대신)
        timestamp = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
        try:
            img_dir = Path(__file__).resolve().parent.parent.parent.parent / "temp_images"
        except NameError:
            img_dir = Path("./temp_images")
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"decomposition_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        # Return results as JSON with image path instead of base64
        result = {
            "period": int(period),
            "strengths": {k: float(v) for k, v in strengths.items()},
            "components": {
                "trend": trend.tolist(),
                "seasonal": seasonal.tolist(),
                "residual": residual.tolist(),
            },
            "image_path": str(save_path),
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error in time series decomposition: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})


@tool
def symbolic_representation(data: List[float], alphabet_size: int = 10) -> str:
    """
    Converts time series data into symbolic representation using SAX (Symbolic Aggregate approXimation).
    
    Args:
        data (List[float]): Time series data
        alphabet_size (int, optional): Size of alphabet to use for symbolization. Defaults to 10.
        
    Returns:
        str: JSON string containing symbolic representation and analysis
    """
    fig = None
    try:
        if not data:
            raise ValueError("Input data list is empty")
            
        data_np = np.array(data)
        
        # Normalize the data (z-score normalization)
        mean = np.mean(data_np)
        std = np.std(data_np)
        
        if std == 0:
            return json.dumps({
                "status": "error",
                "message": "Cannot symbolize time series with zero standard deviation"
            })
            
        normalized_data = (data_np - mean) / std
        
        # Determine breakpoints for the specified alphabet size
        # Using Gaussian distribution breakpoints
        if alphabet_size < 2:
            alphabet_size = 2
        
        # Use percentiles to define breakpoints (Gaussian assumption)
        percentiles = [100.0 * i / alphabet_size for i in range(1, alphabet_size)]
        breakpoints = np.percentile(normalized_data, percentiles)
        
        # Create alphabet (using ASCII characters)
        alphabet = [chr(97 + i) for i in range(alphabet_size)]  # 'a', 'b', 'c', ...
        
        # Symbolize the data
        symbolic_data = []
        for val in normalized_data:
            # Find the symbol for this value
            symbol_idx = np.digitize(val, breakpoints)
            symbolic_data.append(alphabet[symbol_idx])
        
        # Create a string representation
        symbolic_string = ''.join(symbolic_data)
        
        # Find repeated patterns using a simple approach
        patterns = {}
        pattern_length = 3  # Looking for 3-symbol patterns by default
        if len(symbolic_string) >= pattern_length:
            for i in range(len(symbolic_string) - pattern_length + 1):
                pattern = symbolic_string[i:i+pattern_length]
                if pattern in patterns:
                    patterns[pattern].append(i)
                else:
                    patterns[pattern] = [i]
        
        # Filter for repeated patterns
        repeated_patterns = {pattern: indices for pattern, indices in patterns.items() if len(indices) > 1}
        
        # Visualize the symbolic representation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original series
        ax1.plot(data_np, color='blue')
        ax1.set_title("Original Time Series", fontsize=12)
        ax1.set_xlabel("Time", fontsize=10)
        ax1.set_ylabel("Value", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Symbolic representation
        # Create a colored heatmap-like representation of the symbols
        symbol_values = np.zeros(len(symbolic_data))
        for i, symbol in enumerate(symbolic_data):
            symbol_values[i] = ord(symbol) - 97  # Map 'a'->0, 'b'->1, etc.
        
        # Plot as colored blocks
        cmap = plt.get_cmap('viridis', alphabet_size)
        ax2.imshow([symbol_values], aspect='auto', cmap=cmap)
        ax2.set_title("Symbolic Representation (SAX)", fontsize=12)
        ax2.set_xlabel("Time", fontsize=10)
        ax2.set_yticks([])
        
        # Add a colorbar with symbol labels
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2, orientation='vertical', pad=0.01)
        cbar.set_ticks(np.arange(alphabet_size) + 0.5)
        cbar.set_ticklabels(alphabet)
        
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode bytes to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # Return results
        result = {
            "symbolic_string": symbolic_string,
            "alphabet_size": alphabet_size,
            "alphabet": alphabet,
            "breakpoints": breakpoints.tolist(),
            "repeated_patterns": {pattern: occurrences for pattern, occurrences in repeated_patterns.items()},
            "visualization_base64": image_base64,
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error in symbolic representation: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})
    finally:
        if fig is not None:
            plt.close(fig)