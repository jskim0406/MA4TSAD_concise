# Parallel Sentinel

병렬 LLM 기반 시계열 이상치 탐지 라이브러리

## 개요

Parallel Sentinel은 시계열 데이터의 이상치를 탐지하기 위한 멀티 에이전트 접근 방식을 구현합니다. 기존의 단일 에이전트 방식이 아닌, 시계열 데이터의 주요 구성 요소를 병렬로 분석하는 전문화된 에이전트들을 활용합니다:

1. **추세 분석가(Trend Analyzer)** - 시계열의 장기적인 방향성과 패턴 분석
2. **계절성 분석가(Seasonality Analyzer)** - 주기적인 패턴과 계절적 요소 분석
3. **잔차 분석가(Remainder Analyzer)** - 추세와 계절성으로 설명되지 않는 변동 분석
4. **감독자(Supervisor)** - 워크플로우 조정 및 결과 통합

이러한 병렬 접근 방식은 더 빠른 처리 시간과 더 깊이 있는 분석을 가능하게 합니다.

## 아키텍처

라이브러리는 LangGraph의 병렬 브랜치 실행 기능을 활용하여 다음과 같은 워크플로우를 구현합니다:

```
                     ┌───────────┐
                     │           │
                     │ Supervisor│(초기 평가)
                     │           │
                     └─────┬─────┘
                           │
                           ▼
           ┌───────────────┬───────────────┐
           │               │               │
┌──────────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐
│                 │ │              │ │              │
│ Trend Analyzer  │ │ Seasonality  │ │ Remainder    │
│                 │ │ Analyzer     │ │ Analyzer     │
│                 │ │              │ │              │
└─────────┬───────┘ └──────┬───────┘ └─────┬────────┘
          │                │               │
          └────────────────┬───────────────┘
                           │
                           ▼
                     ┌───────────┐
                     │           │
                     │ Supervisor│(최종 분석)
                     │           │
                     └───────────┘
```

## 주요 구성 요소

### 1. 시계열 분해 (Decomposition)

시계열 데이터를 세 가지 주요 구성 요소로 분해합니다:

- **추세(Trend)**: 데이터의 장기적인 증가 또는 감소 추세
- **계절성(Seasonality)**: 일정 주기로 반복되는 패턴
- **잔차(Remainder)**: 추세와 계절성으로 설명되지 않는 불규칙한 부분 (이상치 포함)

### 2. 병렬 분석

각 구성 요소는 전문화된 에이전트가 독립적으로 분석합니다:

- **추세 분석가**: 추세의 방향, 강도, 변화율, 구조적 변화를 분석
- **계절성 분석가**: 계절성 패턴의 주기, 강도, 일관성을 분석
- **잔차 분석가**: 잔차 내 이상치, 패턴, 자기상관성을 분석

### 3. 통합 분석

감독자 에이전트는 각 전문가의 결과를 통합하여 종합적인 분석을 제공합니다:

- 시계열 특성 요약
- 식별된 이상치와 그 의미
- 이상 감지된 시점의 패턴 분석
- 추가 모니터링 및 조사를 위한 권장사항

## 사용 방법

### 설치

```bash
pip install -e .
```

### 기본 사용법

```python
from langchain_google_vertexai import ChatVertexAI
from parallel_sentinel.agents import (
    create_supervisor_agent, create_trend_analyzer_agent,
    create_seasonality_analyzer_agent, create_remainder_analyzer_agent
)
from parallel_sentinel.tools import (
    ts2img, decompose_time_series, basic_statistics,
    trend_analysis, get_math_calculator
)
from parallel_sentinel.graph import create_workflow, run_workflow

# LLM 초기화
llm = ChatVertexAI(model="gemini-1.5-flash")

# 도구 정의
tools = [ts2img, decompose_time_series, basic_statistics, 
         trend_analysis, get_math_calculator(llm)]

# 에이전트 생성
supervisor = create_supervisor_agent(llm)
trend_analyzer = create_trend_analyzer_agent(llm, tools)
seasonality_analyzer = create_seasonality_analyzer_agent(llm, tools)
remainder_analyzer = create_remainder_analyzer_agent(llm, tools)

# 워크플로우 생성
workflow = create_workflow(
    supervisor_agent=supervisor,
    trend_analyzer_agent=trend_analyzer,
    seasonality_analyzer_agent=seasonality_analyzer,
    remainder_analyzer_agent=remainder_analyzer,
    tools=tools
)

# 이상치가 있는 예제 시계열 데이터
time_series_data = [100, 105, 102, 107, 109, 110, 200, 115, 118, 120]

# 분석 실행
result = run_workflow(workflow, time_series_data)

# 결과 확인
print(result["final_analysis"]["summary"])
```

### 명령줄 인터페이스

이 라이브러리는 다음과 같은 명령줄 인터페이스를 제공합니다:

```bash
# 기본 사용법
python src/parallel_sentinel/generator_parallel.py --data=your_data.csv

# 디버그 모드 활성화
python src/parallel_sentinel/generator_parallel.py --debug

# 합성 데이터 사용
python src/parallel_sentinel/generator_parallel.py --synthetic
```

## 기존 federated_sentinel과의 차이점

Parallel Sentinel은 기존 federated_sentinel의 개선된 버전으로, 다음과 같은 차이점이 있습니다:

1. **병렬 처리**: 순차적 처리 대신 병렬 분석을 통해 처리 시간 단축
2. **시계열 분해 기반**: 전체 시계열이 아닌 각 구성 요소(추세, 계절성, 잔차)에 대한 전문적 분석
3. **통합된 시각화**: 분해와 이상치 시각화 도구 개선
4. **강화된 이상치 탐지**: 각 구성 요소별 이상치 탐지로 더 정확하고 상세한 이상 감지

## 요구사항

- Python 3.9+
- LangGraph
- LangChain
- Google Vertex AI Python SDK (또는 다른 LLM 제공자)
- NumPy
- SciPy
- Matplotlib
- Pandas
- statsmodels

## 라이센스

이 프로젝트는 MIT 라이센스에 따라 배포됩니다.