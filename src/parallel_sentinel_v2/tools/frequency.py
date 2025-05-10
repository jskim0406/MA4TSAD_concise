"""
시계열 데이터 주파수 분석 도구
이 모듈은 시계열 데이터의 주파수 특성을 분석하기 위한 도구를 제공합니다.
"""

import io
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union

from langchain_core.tools import tool


@tool
def get_fourier_transform(data: List[float]) -> str:
    """
    Performs Fourier transform on the input time series data and returns the magnitude spectrum.
    
    Args:
        data (List[float]): Time series data
        
    Returns:
        str: JSON string containing the Fourier transform results and analysis
    """
    fig = None
    try:
        if not data:
            raise ValueError("Input data list is empty")
            
        data_np = np.array(data)
        n = len(data_np)
        
        # Remove mean (DC component)
        data_centered = data_np - np.mean(data_np)
        
        # Perform FFT
        fft_result = np.fft.rfft(data_centered)
        fft_mag = np.abs(fft_result)  # Magnitude spectrum
        freqs = np.fft.rfftfreq(n)  # Normalized frequencies
        
        # Find prominent frequencies
        sorted_indices = np.argsort(fft_mag)[::-1]  # Sort by magnitude (descending)
        top_indices = sorted_indices[:5]  # Top 5 frequencies
        dominant_freqs = []
        
        for idx in top_indices:
            if idx > 0 and fft_mag[idx] > 0.01 * fft_mag.max():  # Filter out very small magnitudes
                period = 1.0 / freqs[idx] if freqs[idx] > 0 else float('inf')
                dominant_freqs.append({
                    "frequency": float(freqs[idx]),
                    "period": float(period),
                    "magnitude": float(fft_mag[idx]),
                    "normalized_magnitude": float(fft_mag[idx] / fft_mag.max())
                })
        
        # Create visualization with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original time series
        ax1.plot(data_np, linewidth=1.5)
        ax1.set_title("Original Time Series", fontsize=12)
        ax1.set_xlabel("Time Index", fontsize=10)
        ax1.set_ylabel("Value", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Magnitude spectrum
        ax2.stem(freqs, fft_mag, markerfmt=' ', basefmt='-')
        ax2.set_title("Frequency Spectrum (FFT Magnitude)", fontsize=12)
        ax2.set_xlabel("Frequency", fontsize=10)
        ax2.set_ylabel("Magnitude", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Mark dominant frequencies
        for freq_data in dominant_freqs:
            freq = freq_data["frequency"]
            mag = freq_data["magnitude"]
            period = freq_data["period"]
            ax2.annotate(f"Period: {period:.1f}", 
                         xy=(freq, mag), 
                         xytext=(freq, mag*1.1),
                         arrowprops=dict(arrowstyle="->", color="red"),
                         color="red")
        
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode bytes to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # Return results as JSON
        result = {
            "dominant_frequencies": dominant_freqs,
            "visualization_base64": image_base64,
            "suggested_seasonality": dominant_freqs[0]["period"] if dominant_freqs else None,
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error in Fourier analysis: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})
    finally:
        if fig is not None:
            plt.close(fig)


@tool
def wavelet_transform(data: List[float]) -> str:
    """
    Performs wavelet transform on time series data to analyze time-frequency components.
    
    Args:
        data (List[float]): Time series data for wavelet transform analysis
        
    Returns:
        str: JSON string containing wavelet transform results with visualization
    """
    try:
        # Check if PyWavelets is installed
        try:
            import pywt
            PYWT_AVAILABLE = True
        except ImportError:
            PYWT_AVAILABLE = False
            return json.dumps({
                "status": "error", 
                "message": "PyWavelets package not installed. Install with: pip install pywt"
            })
            
        if not PYWT_AVAILABLE:
            return json.dumps({
                "status": "error", 
                "message": "PyWavelets package not available"
            })
            
        if not data:
            raise ValueError("Input data list is empty")
            
        import pywt
        import matplotlib.cm as cm
        
        data_np = np.array(data)
        
        # Choose wavelet type
        wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet, good for identifying oscillations
        
        # Perform continuous wavelet transform
        scales = np.arange(1, min(64, len(data_np)//2))
        coef, freqs = pywt.cwt(data_np, scales, wavelet)
        
        # Calculate power (squared absolute value)
        power = (abs(coef)) ** 2
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot original time series
        ax1.plot(data_np)
        ax1.set_title('Original Time Series', fontsize=12)
        ax1.set_xlabel('Time', fontsize=10)
        ax1.set_ylabel('Amplitude', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot wavelet power spectrum
        im = ax2.imshow(power, cmap=cm.jet, aspect='auto', 
                         extent=[0, len(data_np), 1, len(scales)],
                         vmax=abs(power).max(), vmin=0)
        ax2.set_title('Wavelet Power Spectrum', fontsize=12)
        ax2.set_ylabel('Scale', fontsize=10)
        ax2.set_xlabel('Time', fontsize=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Power')
        
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode bytes to base64 string
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        # Find dominant scales/periods
        avg_power = np.mean(power, axis=1)
        dominant_scales_idx = np.argsort(avg_power)[-5:]  # Top 5 dominant scales
        dominant_scales = [float(scales[i]) for i in dominant_scales_idx]
        
        # Convert scales to approximate periods
        dominant_periods = []
        for scale in dominant_scales:
            period = scale  # Simplified relation for Morlet wavelet
            dominant_periods.append(float(period))
        
        # Identify time segments with high power
        high_power_threshold = np.percentile(power, 95)  # Top 5% power
        high_power_regions = []
        
        for scale_idx, scale in enumerate(scales):
            if scale_idx in dominant_scales_idx:
                high_power_times = np.where(power[scale_idx, :] > high_power_threshold)[0].tolist()
                if high_power_times:
                    high_power_regions.append({
                        "scale": float(scale),
                        "times": high_power_times[:10]  # Limit to first 10 times
                    })
        
        result = {
            "dominant_scales": dominant_scales,
            "dominant_periods": dominant_periods,
            "high_power_regions": high_power_regions,
            "visualization_base64": image_base64,
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error in wavelet transform analysis: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})