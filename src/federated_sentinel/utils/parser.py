"""
Output parsers for the federated sentinel system.
"""

import re
import json
from typing import Dict, Any, List, Union, Optional

from pydantic import BaseModel, Field


class AnomalyResult(BaseModel):
    """Model for anomaly detection results."""
    anomaly_indices: List[int] = Field(default_factory=list, description="Indices of detected anomalies")
    anomaly_values: List[Union[float, List[float]]] = Field(default_factory=list, description="Values of detected anomalies")
    anomaly_score: Optional[float] = Field(None, description="Confidence score for anomalies")
    description: Optional[str] = Field(None, description="Description of the anomalies")


class PatternResult(BaseModel):
    """Model for pattern detection results."""
    pattern_type: str = Field(..., description="Type of pattern (trend, seasonality, cycle)")
    description: str = Field(..., description="Description of the pattern")
    confidence: Optional[float] = Field(None, description="Confidence score for the pattern")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters describing the pattern")


class StatisticalResult(BaseModel):
    """Model for statistical analysis results."""
    mean: float = Field(..., description="Mean of the time series")
    median: float = Field(..., description="Median of the time series")
    std: float = Field(..., description="Standard deviation of the time series")
    min: float = Field(..., description="Minimum value in the time series")
    max: float = Field(..., description="Maximum value in the time series")
    stationarity: Optional[bool] = Field(None, description="Whether the time series is stationary")
    additional_stats: Optional[Dict[str, Any]] = Field(None, description="Additional statistical metrics")


class FinalAnalysisResult(BaseModel):
    """Model for the final analysis result."""
    summary: str = Field(..., description="Summary of the time series analysis")
    anomalies: List[AnomalyResult] = Field(default_factory=list, description="Detected anomalies")
    patterns: List[PatternResult] = Field(default_factory=list, description="Detected patterns")
    statistics: Optional[StatisticalResult] = Field(None, description="Statistical analysis results")
    recommendations: Optional[List[str]] = Field(None, description="Recommendations based on analysis")


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from text that might contain other content.
    
    Args:
        text (str): Text that may contain a JSON object
        
    Returns:
        Dict[str, Any]: Extracted JSON object, or empty dict if extraction fails
    """
    # Try to find JSON object between curly braces
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object between triple backticks
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    
    if code_match:
        json_str = code_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If no JSON found, return empty dict
    return {}


def parse_anomaly_results(text: str) -> List[AnomalyResult]:
    """
    Parse anomaly detection results from text.
    
    Args:
        text (str): Text containing anomaly detection results
        
    Returns:
        List[AnomalyResult]: Parsed anomaly results
    """
    results = []
    
    # Try to extract JSON first
    json_data = extract_json_from_text(text)
    if json_data and "anomalies" in json_data:
        anomalies = json_data["anomalies"]
        if isinstance(anomalies, list):
            for anomaly in anomalies:
                if isinstance(anomaly, dict):
                    results.append(AnomalyResult(**anomaly))
    
    # If JSON extraction failed, try regex patterns
    if not results:
        # Look for patterns like "Anomaly at index X with value Y"
        anomaly_matches = re.finditer(r'(?:anomaly|spike|outlier).*?(?:at|index).*?(\d+).*?(?:value|of).*?([\d.]+)', 
                                     text, re.IGNORECASE)
        
        indices = []
        values = []
        
        for match in anomaly_matches:
            try:
                index = int(match.group(1))
                value = float(match.group(2))
                indices.append(index)
                values.append(value)
            except (ValueError, IndexError):
                continue
        
        if indices:
            results.append(AnomalyResult(
                anomaly_indices=indices,
                anomaly_values=values,
                description="Anomalies extracted from text"
            ))
    
    return results


def parse_pattern_results(text: str) -> List[PatternResult]:
    """
    Parse pattern detection results from text.
    
    Args:
        text (str): Text containing pattern detection results
        
    Returns:
        List[PatternResult]: Parsed pattern results
    """
    results = []
    
    # Try to extract JSON first
    json_data = extract_json_from_text(text)
    if json_data and "patterns" in json_data:
        patterns = json_data["patterns"]
        if isinstance(patterns, list):
            for pattern in patterns:
                if isinstance(pattern, dict):
                    results.append(PatternResult(**pattern))
    
    # If JSON extraction failed, try regex patterns
    if not results:
        # Look for trend patterns
        trend_match = re.search(r'(?:trend|tendency).*?(increasing|decreasing|upward|downward|stable|flat)', 
                               text, re.IGNORECASE)
        if trend_match:
            trend_type = trend_match.group(1).lower()
            trend_dir = "increasing" if trend_type in ["increasing", "upward"] else \
                       "decreasing" if trend_type in ["decreasing", "downward"] else "stable"
            
            results.append(PatternResult(
                pattern_type="trend",
                description=f"The time series shows a {trend_dir} trend",
                parameters={"direction": trend_dir}
            ))
        
        # Look for seasonality patterns
        seasonality_match = re.search(r'(?:season|periodic|cycle).*?(?:period|length).*?(\d+)', 
                                     text, re.IGNORECASE)
        if seasonality_match:
            try:
                period = int(seasonality_match.group(1))
                results.append(PatternResult(
                    pattern_type="seasonality",
                    description=f"The time series shows seasonality with period {period}",
                    parameters={"period": period}
                ))
            except (ValueError, IndexError):
                pass
    
    return results


def parse_statistical_results(text: str) -> Optional[StatisticalResult]:
    """
    Parse statistical analysis results from text.
    
    Args:
        text (str): Text containing statistical analysis results
        
    Returns:
        Optional[StatisticalResult]: Parsed statistical results, or None if parsing fails
    """
    # Try to extract JSON first
    json_data = extract_json_from_text(text)
    if json_data and isinstance(json_data, dict):
        if all(key in json_data for key in ["mean", "median", "std", "min", "max"]):
            return StatisticalResult(**json_data)
    
    # If JSON extraction failed, try regex patterns
    mean_match = re.search(r'mean.*?([\d.]+)', text, re.IGNORECASE)
    median_match = re.search(r'median.*?([\d.]+)', text, re.IGNORECASE)
    std_match = re.search(r'(?:std|standard deviation).*?([\d.]+)', text, re.IGNORECASE)
    min_match = re.search(r'(?:min|minimum).*?([\d.]+)', text, re.IGNORECASE)
    max_match = re.search(r'(?:max|maximum).*?([\d.]+)', text, re.IGNORECASE)
    
    if mean_match and median_match and std_match and min_match and max_match:
        try:
            result = StatisticalResult(
                mean=float(mean_match.group(1)),
                median=float(median_match.group(1)),
                std=float(std_match.group(1)),
                min=float(min_match.group(1)),
                max=float(max_match.group(1))
            )
            
            # Look for stationarity
            stationary_match = re.search(r'(?:stationary|stationarity).*?(is|is not|isn\'t)', text, re.IGNORECASE)
            if stationary_match:
                is_stationary = stationary_match.group(1).lower() == "is"
                result.stationarity = is_stationary
            
            return result
        except (ValueError, IndexError):
            pass
    
    return None


def parse_final_analysis(text: str) -> FinalAnalysisResult:
    """
    Parse the final analysis result from text.
    
    Args:
        text (str): Text containing the final analysis
        
    Returns:
        FinalAnalysisResult: Parsed final analysis result
    """
    # Try to extract JSON first
    json_data = extract_json_from_text(text)
    if json_data and "summary" in json_data:
        return FinalAnalysisResult(**json_data)
    
    # If JSON extraction failed, use a more flexible approach
    summary = text.split('\n\n')[0] if '\n\n' in text else text[:200]
    
    # Extract recommendations
    recommendations = []
    rec_section = re.search(r'(?:recommendation|recommend|suggest)s?:?(.*?)(?:\n\n|$)', 
                           text, re.IGNORECASE | re.DOTALL)
    if rec_section:
        rec_text = rec_section.group(1)
        rec_items = re.findall(r'[-*]?(.*?)(?:\n|$)', rec_text)
        recommendations = [item.strip() for item in rec_items if item.strip()]
    
    # Extract anomalies, patterns, and statistics
    anomalies = parse_anomaly_results(text)
    patterns = parse_pattern_results(text)
    statistics = parse_statistical_results(text)
    
    return FinalAnalysisResult(
        summary=summary,
        anomalies=anomalies,
        patterns=patterns,
        statistics=statistics,
        recommendations=recommendations
    )