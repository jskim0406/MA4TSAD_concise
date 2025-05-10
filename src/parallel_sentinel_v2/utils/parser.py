"""
병렬 시계열 분석 시스템의 출력 파서
"""

import re
import json
from typing import Dict, Any, List, Union, Optional

from pydantic import BaseModel, Field


class AnomalyResult(BaseModel):
    """이상치 탐지 결과 모델"""
    anomaly_indices: List[int] = Field(default_factory=list, description="탐지된 이상치의 인덱스")
    anomaly_values: List[Union[float, List[float]]] = Field(default_factory=list, description="탐지된 이상치의 값")
    anomaly_score: Optional[float] = Field(None, description="이상치에 대한 신뢰도 점수")
    source: Optional[str] = Field(None, description="이상치의 출처 (추세, 계절성, 잔차)")
    description: Optional[str] = Field(None, description="이상치에 대한 설명")


class PatternResult(BaseModel):
    """패턴 탐지 결과 모델"""
    pattern_type: str = Field(..., description="패턴 유형 (추세, 계절성, 주기)")
    description: str = Field(..., description="패턴에 대한 설명")
    confidence: Optional[float] = Field(None, description="패턴에 대한 신뢰도 점수")
    parameters: Optional[Dict[str, Any]] = Field(None, description="패턴을 설명하는 추가 매개변수")


class DecompositionResult(BaseModel):
    """시계열 분해 결과 모델"""
    trend: Dict[str, Any] = Field(default_factory=dict, description="추세 성분 분석")
    seasonality: Dict[str, Any] = Field(default_factory=dict, description="계절성 성분 분석")
    remainder: Dict[str, Any] = Field(default_factory=dict, description="잔차 성분 분석")
    visualization_path: Optional[str] = Field(None, description="분해 시각화 경로")


class StatisticalResult(BaseModel):
    """통계 분석 결과 모델"""
    mean: float = Field(..., description="시계열의 평균")
    median: float = Field(..., description="시계열의 중앙값")
    std: float = Field(..., description="시계열의 표준편차")
    min: float = Field(..., description="시계열의 최소값")
    max: float = Field(..., description="시계열의 최대값")
    stationarity: Optional[bool] = Field(None, description="시계열이 정상 시계열인지 여부")
    additional_stats: Optional[Dict[str, Any]] = Field(None, description="추가 통계 지표")


class FinalAnalysisResult(BaseModel):
    """최종 분석 결과 모델"""
    summary: str = Field(..., description="시계열 분석 요약")
    anomalies: List[AnomalyResult] = Field(default_factory=list, description="탐지된 이상치")
    patterns: List[PatternResult] = Field(default_factory=list, description="탐지된 패턴")
    decomposition: Optional[DecompositionResult] = Field(None, description="시계열 분해 결과")
    statistics: Optional[StatisticalResult] = Field(None, description="통계 분석 결과")
    recommendations: Optional[List[str]] = Field(None, description="분석 기반 권장사항")


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    텍스트에서 JSON 객체를 추출합니다.
    
    Args:
        text (str): JSON 객체를 포함할 수 있는 텍스트
        
    Returns:
        Dict[str, Any]: 추출된 JSON 객체, 추출 실패 시 빈 사전
    """
    # 중괄호 사이의 JSON 객체 찾기
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 트리플 백틱 사이의 JSON 객체 찾기
    code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    
    if code_match:
        json_str = code_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # JSON을 찾지 못한 경우 빈 사전 반환
    return {}


def parse_anomaly_results(text: str) -> List[AnomalyResult]:
    """
    텍스트에서 이상치 탐지 결과를 파싱합니다.
    
    Args:
        text (str): 이상치 탐지 결과를 포함하는 텍스트
        
    Returns:
        List[AnomalyResult]: 파싱된 이상치 결과
    """
    results = []
    
    # 먼저 JSON 추출 시도
    json_data = extract_json_from_text(text)
    if json_data and "anomalies" in json_data:
        anomalies = json_data["anomalies"]
        if isinstance(anomalies, list):
            for anomaly in anomalies:
                if isinstance(anomaly, dict):
                    results.append(AnomalyResult(**anomaly))
    
    # JSON 추출에 실패한 경우 정규 표현식 패턴 시도
    if not results:
        # "인덱스 X에서 값 Y인 이상치" 같은 패턴 찾기
        anomaly_matches = re.finditer(
            r'(?:이상치|스파이크|특이치).*?(?:인덱스|위치).*?(\d+).*?(?:값|수치).*?([\d.]+)', 
            text, 
            re.IGNORECASE
        )
        
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
                description="텍스트에서 추출한 이상치"
            ))
    
    # 한국어로 된 다른 패턴도 시도 (다양한 표현 방식 고려)
    if not results:
        # "X번째 데이터 포인트에서 이상치 발견" 같은 패턴
        ko_matches = re.finditer(
            r'(\d+)(?:번째|번|위치|인덱스).*?(?:이상치|비정상|특이|특이점|스파이크)', 
            text, 
            re.IGNORECASE
        )
        
        indices = []
        
        for match in ko_matches:
            try:
                index = int(match.group(1))
                indices.append(index)
            except (ValueError, IndexError):
                continue
        
        if indices:
            # 값 정보가 없는 경우
            results.append(AnomalyResult(
                anomaly_indices=indices,
                description="텍스트에서 추출한 이상치 (값 정보 없음)"
            ))
    
    return results


def parse_pattern_results(text: str) -> List[PatternResult]:
    """
    텍스트에서 패턴 탐지 결과를 파싱합니다.
    
    Args:
        text (str): 패턴 탐지 결과를 포함하는 텍스트
        
    Returns:
        List[PatternResult]: 파싱된 패턴 결과
    """
    results = []
    
    # 먼저 JSON 추출 시도
    json_data = extract_json_from_text(text)
    if json_data and "patterns" in json_data:
        patterns = json_data["patterns"]
        if isinstance(patterns, list):
            for pattern in patterns:
                if isinstance(pattern, dict):
                    results.append(PatternResult(**pattern))
    
    # JSON 추출에 실패한 경우 정규 표현식 패턴 시도
    if not results:
        # 추세 패턴 찾기
        trend_match = re.search(
            r'(?:추세|경향|흐름).*?(증가|상승|감소|하락|안정|평탄)', 
            text, 
            re.IGNORECASE
        )
        if trend_match:
            trend_type = trend_match.group(1).lower()
            trend_dir = "증가" if trend_type in ["증가", "상승"] else \
                       "감소" if trend_type in ["감소", "하락"] else "안정"
            
            results.append(PatternResult(
                pattern_type="trend",
                description=f"시계열은 {trend_dir}하는 추세를 보입니다",
                parameters={"direction": trend_dir}
            ))
        
        # 계절성 패턴 찾기
        seasonality_match = re.search(
            r'(?:계절성|주기성|주기|순환).*?(?:주기|길이).*?(\d+)', 
            text, 
            re.IGNORECASE
        )
        if seasonality_match:
            try:
                period = int(seasonality_match.group(1))
                results.append(PatternResult(
                    pattern_type="seasonality",
                    description=f"시계열은 {period} 간격의 계절성을 보입니다",
                    parameters={"period": period}
                ))
            except (ValueError, IndexError):
                pass
    
    return results


def parse_statistical_results(text: str) -> Optional[StatisticalResult]:
    """
    텍스트에서 통계 분석 결과를 파싱합니다.
    
    Args:
        text (str): 통계 분석 결과를 포함하는 텍스트
        
    Returns:
        Optional[StatisticalResult]: 파싱된 통계 결과, 파싱 실패 시 None
    """
    # 먼저 JSON 추출 시도
    json_data = extract_json_from_text(text)
    if json_data and isinstance(json_data, dict):
        if all(key in json_data for key in ["mean", "median", "std", "min", "max"]):
            return StatisticalResult(**json_data)
    
    # JSON 추출에 실패한 경우 정규 표현식 패턴 시도
    mean_match = re.search(r'(?:평균|mean).*?([\d.]+)', text, re.IGNORECASE)
    median_match = re.search(r'(?:중앙값|중위수|median).*?([\d.]+)', text, re.IGNORECASE)
    std_match = re.search(r'(?:표준편차|std|standard deviation).*?([\d.]+)', text, re.IGNORECASE)
    min_match = re.search(r'(?:최소|min|minimum).*?([\d.]+)', text, re.IGNORECASE)
    max_match = re.search(r'(?:최대|max|maximum).*?([\d.]+)', text, re.IGNORECASE)
    
    if mean_match and median_match and std_match and min_match and max_match:
        try:
            result = StatisticalResult(
                mean=float(mean_match.group(1)),
                median=float(median_match.group(1)),
                std=float(std_match.group(1)),
                min=float(min_match.group(1)),
                max=float(max_match.group(1))
            )
            
            # 정상성 확인
            stationary_match = re.search(r'(?:정상성|stationarity).*?(이다|있다|있음|없다|없음|is|is not|isn\'t)', text, re.IGNORECASE)
            if stationary_match:
                is_stationary = stationary_match.group(1).lower() in ["이다", "있다", "있음", "is"]
                result.stationarity = is_stationary
            
            return result
        except (ValueError, IndexError):
            pass
    
    return None


def parse_final_analysis(text: str) -> FinalAnalysisResult:
    """
    텍스트에서 최종 분석 결과를 파싱합니다.
    
    Args:
        text (str): 최종 분석을 포함하는 텍스트
        
    Returns:
        FinalAnalysisResult: 파싱된 최종 분석 결과
    """
    # 먼저 JSON 추출 시도
    json_data = extract_json_from_text(text)
    if json_data and "summary" in json_data:
        return FinalAnalysisResult(**json_data)
    
    # JSON 추출에 실패한 경우 더 유연한 접근 방식 사용
    summary = text.split('\n\n')[0] if '\n\n' in text else text[:200]
    
    # 권장사항 추출
    recommendations = []
    rec_section = re.search(
        r'(?:권장사항|권고사항|제안|추천|recommend|suggest).*?[:：](.*?)(?:\n\n|$)', 
        text, 
        re.IGNORECASE | re.DOTALL
    )
    if rec_section:
        rec_text = rec_section.group(1)
        rec_items = re.findall(r'[-*•]?\s*(.*?)(?:\n|$)', rec_text)
        recommendations = [item.strip() for item in rec_items if item.strip()]
    
    # 이상치, 패턴, 통계 추출
    anomalies = parse_anomaly_results(text)
    patterns = parse_pattern_results(text)
    statistics = parse_statistical_results(text)
    
    # 분해 결과는 텍스트에서 직접 추출하기 어려우므로 None으로 설정
    decomposition = None
    
    return FinalAnalysisResult(
        summary=summary,
        anomalies=anomalies,
        patterns=patterns,
        decomposition=decomposition,
        statistics=statistics,
        recommendations=recommendations
    )