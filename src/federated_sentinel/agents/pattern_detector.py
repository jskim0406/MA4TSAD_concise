"""
Pattern Detector agent for time series data.
"""

import numpy as np
from typing import Dict, Any, List, Callable
from scipy import signal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from federated_sentinel.graph.workflow import TimeSeriesState


def create_pattern_detector_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    Create a pattern detector agent that identifies patterns in time series data.
    
    Args:
        llm: Language model to use for the agent
        tools: Optional list of tools available to the agent
        
    Returns:
        A function that can be added as a node to the workflow
    """
    # Define system prompt for the pattern detector
    system_template = """
    You are a pattern detection specialist analyzing time series data.
    Your task is to identify and characterize patterns in the provided time series data.

    Focus on the following pattern types:
    1. Periodicity (daily, weekly, monthly patterns)
    2. Cycles
    3. Trends (linear, non-linear)
    4. Recurring motifs
    5. Structural breaks or regime changes

    You have access to the following tools:
    - ts2img: Generate an image of the time series
    - calculate: Perform mathematical calculations

    Provide your analysis in a clear, structured format that highlights key patterns discovered.
    """

    tool_node = ToolNode(tools) if tools else None
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) if tools else "No tools available."
    
    pattern_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template.format(tool_descriptions=tool_descriptions)),
        MessagesPlaceholder(variable_name="messages"), # 메시지 히스토리 플레이스홀더
        ("human", "{input}")
    ])

    def format_data_for_pattern_detection(state: TimeSeriesState) -> str:
        return f"Please identify patterns in the time series data provided in the state (length: {len(state['ts_data'])})."

    def pattern_detector_agent(state: TimeSeriesState) -> Dict[str, Any]:
        messages = list(state["messages"])
        input_text = format_data_for_pattern_detection(state)

        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = pattern_prompt.invoke(prompt_input)

        llm_with_tools = llm.bind_tools(tools) if tools else llm

        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message)

        content = ""

        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"PatternDetector: Detected {len(ai_message.tool_calls)} tool calls.")
            tool_response = tool_node.invoke({"messages": [ai_message]})
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            print(f"PatternDetector: Added {len(tool_messages)} tool results to history.")

            print("PatternDetector: Invoking LLM again with tool results.")
            follow_up_prompt = pattern_prompt.invoke({"messages": messages, "input": "Summarize the detected patterns based on the tool results."}) # 예시 input
            ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            messages.append(ai_follow_up)
            content = ai_follow_up.content
            print("PatternDetector: Received final response from LLM.")
        else:
            print("PatternDetector: No tool calls detected.")
            content = ai_message.content

        # --- 기본 패턴 탐지 로직 (이 부분은 그대로 유지) ---
        ts_data = np.array(state["ts_data"])
        trend = {}
        periodicity = {}
        extrema = {}
        try:
            x = np.arange(len(ts_data))
            coeffs = np.polyfit(x, ts_data, 1)
            trend = {
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1]),
                "direction": "increasing" if coeffs[0] > 0 else "decreasing" if coeffs[0] < 0 else "stable"
            }
            # FFT 주기성 분석 (간소화)
            if len(ts_data) > 1:
                 fft_result = np.abs(np.fft.rfft(ts_data - np.mean(ts_data)))
                 freqs = np.fft.rfftfreq(len(ts_data))
                 if len(fft_result) > 1:
                     dominant_idx = np.argmax(fft_result[1:]) + 1 # DC 제외
                     dominant_freq = freqs[dominant_idx]
                     period = 1 / dominant_freq if dominant_freq > 0 else float('inf')

                     # --- 수정된 부분 ---
                     # NumPy 비교 결과를 bool()을 사용하여 Python bool 타입으로 명시적 변환
                     has_periodicity_flag = bool(float(fft_result[dominant_idx]) > 0.1 * np.max(fft_result))
                     # --- 수정 끝 ---

                     periodicity = {
                         "has_periodicity": has_periodicity_flag, # 변환된 값 사용
                         "dominant_period": float(period),
                         "dominant_amplitude": float(fft_result[dominant_idx])
                     }
                 else: periodicity = {"error": "Not enough data for FFT"}
            else:
                trend = {"error": "Not enough data for trend analysis"}
                periodicity = {"error": "Not enough data for periodicity analysis"} # periodicity 초기화 추가

            # 극값 탐지 (간소화)
            if len(ts_data) > 2: # find_peaks는 최소 3개 데이터 필요
                peaks, _ = signal.find_peaks(ts_data, height=np.mean(ts_data) + np.std(ts_data))
                troughs, _ = signal.find_peaks(-ts_data, height=-(np.mean(ts_data) - np.std(ts_data)))
                extrema = {
                    "num_peaks": len(peaks),
                    "num_troughs": len(troughs),
                    "peak_indices": peaks.tolist()[:10], # 최대 10개만
                    "trough_indices": troughs.tolist()[:10] # 최대 10개만
                }
            else: extrema = {"error": "Not enough data for extrema detection"}

        except Exception as e:
            trend = {"error": f"Trend analysis failed: {str(e)}"}
            periodicity = {"error": f"Periodicity analysis failed: {str(e)}"}
            extrema = {"error": f"Extrema detection failed: {str(e)}"}
        # --- 기본 패턴 탐지 로직 끝 ---

        analysis = {
            "trend": trend,
            "periodicity": periodicity,
            "extrema": extrema,
            "llm_analysis": content
        }

        return {
            "messages": messages,
            "pattern_analysis": [analysis]
        }

    return pattern_detector_agent