### 2025.05.02 Fri.

#### `parallel_sentinel`

##### Tobe

- 현재 문제점 고민

```text
현재 로직은 형식은 Parallel + Multi-modal + Multi-agent 을 갖고 있지만, 이를 적극적으로 활용하고 있지는 않음.

특히
1. Multi-modal 활용도가 낮고
2. 이상치 탐지를 STL decomp + numpy 통계량에 크게 의존.
    - 더군다나 STL decomp의 과정이 매우 초보적. period 계산이 과학적으로 이루어지지 않고, 따라서 이후의 trend, seasonality 분석에서도 그닥 과학적이지 않게됨.
    - numpy 통계량 계산이 정확한지도 확인 필요.
```

- Tobe

```text
- STL decomp는 좋음. 다만, `STL deomp -> numpy 등 수치분석` 보다는 `STL decomp -> STL plotting -> multimodal reasoning`을 더욱 많이 하도록 유도하는 게 필요
    - 특히나 decomp 시, 필요한 필수 정보(period 등)은 더욱 정확하게 추출할 수 있도록 보완 필요
- "multi-modal로 바로 1d plot을 하고, 그 결과 알기 어렵다면 STL decomp 등 세부 분석 수행" 과 같은 로직으로 수행하도록 supervisor prompt & workflow 개선 필요
```


##### Asis
- Input: 400개 data point(`run.py`의 `infer_data`), Prompt Text
- Logic

```bash
## 1. supervisor -> STL decomp 작업 -> STL decomp 결과 바탕으로 각각 작업 지시(parallel branching)

def supervisor_task(supervisor_agent, data):

    decomposed_result = decompose(data)

    superviosr_agent(decomposed_result)
    ㄴ trend_analyzer(decomposed_result)
    ㄴ seasonality_analyzer(decomposed_result)
    ㄴ remainder_analyzer(decomposed_result)

supervisor_task(supervisor_agnet, data)

## 2. 각각 decomposed anlaysis 진행

trend_analyzer(decomposed_result)
    ㄴ bind tool
        ㄴ llm_with_tools.invoke(prompt_result)
            ㄴ if use_tool:
                ㄴ tool.invoke
            ㄴ else:
                ㄴ pass

    ㄴ 추세 분석 수행 (NumPy 기반, not llm)
        ㄴ `추세 방향과 속성 계산`
            ㄴ `변곡점 탐지 (1차 미분의 부호 변화)`
            ㄴ `데이터를 부드럽게 하기 위해 이동 평균 적용`
            ㄴ `이상치 탐지 (평균에서 2 표준편차 이상 벗어난 값)`

    ㄴ return Message(llm message + 추세 분석 수행 (NumPy 기반, not llm))

seasonality_analyzer(decomposed_result)
    ㄴ bind tool
        ㄴ llm_with_tools.invoke(prompt_result)
            ㄴ if use_tool:
                ㄴ tool.invoke
            ㄴ else:
                ㄴ pass

    ㄴ 계절성 분석 수행 (NumPy 기반, not llm)
        ㄴ # 계절성 속성 계산
            ㄴ # 주기별 계절성 패턴 추출
            ㄴ # 계절성 이상치 탐지
                ㄴ # 각 위치에서 평균 패턴과의 차이 계산(2 표준편차 기준)
            
    ㄴ return Message(llm message + 계절성 분석 수행 (NumPy 기반, not llm))

remainder_analyzer(decomposed_result)
    ㄴ bind tool
        ㄴ llm_with_tools.invoke(prompt_result)
            ㄴ if use_tool:
                ㄴ tool.invoke
            ㄴ else:
                ㄴ pass

    ㄴ 잔차 분석 수행 (NumPy 기반, not llm)
        ㄴ # 기본 통계 계산
        ㄴ # 정규성 테스트
        ㄴ # 이상치 탐지
            ㄴ # 방법 1: Z-score 기반 (통계적 이상치)
            ㄴ # 방법 2: IQR 기반 (분포적 이상치)
            ㄴ # 두 방법으로 감지된 이상치 결합
        ㄴ # 자기상관 검사
        ㄴ # 잔차의 군집성 분석 (연속적인 잔차의 부호 변화 빈도)

    ㄴ return Message(llm message + 잔차 분석 수행 (NumPy 기반, not llm))

## 3. Trend, Seasonal, Remainder 분석 결과 Agg. 최종 analysis

final_analysis = supervisor_task(supervisor_agnet, Trend_analysis, Seasonal_analysis, Remainder_analysis)

final_analysis # 최종 결과
```