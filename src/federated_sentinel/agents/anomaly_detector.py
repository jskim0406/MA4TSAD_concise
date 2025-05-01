"""
Anomaly Detector agent for time series data.
"""

import numpy as np
from typing import Dict, Any, List, Callable
from scipy import stats

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from federated_sentinel.graph.workflow import TimeSeriesState


def create_anomaly_detector_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    Create an anomaly detector agent that identifies anomalies in time series data.
    
    Args:
        llm: Language model to use for the agent
        tools: Optional list of tools available to the agent
        
    Returns:
        A function that can be added as a node to the workflow
    """
    # Define system prompt for the anomaly detector
    system_template = """You are an anomaly detection specialist analyzing time series data.
    Your task is to identify and characterize anomalies in the provided time series data.

    Focus on detecting the following anomaly types:
    1. Point anomalies (spikes, dips)
    2. Contextual anomalies (normal in general but abnormal in context)
    3. Collective anomalies (sequences of data points that are anomalous)
    4. Trend changes
    5. Level shifts

    You have access to the following tools:
    - ts2img: Generate an image of the time series
    - calculate: Perform mathematical calculations

    Provide your analysis in a clear, structured format that highlights key anomalies discovered,
    their potential significance, and possible causes.
    """

    tool_node = ToolNode(tools) if tools else None
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) if tools else "No tools available."

    anomaly_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template.format(tool_descriptions=tool_descriptions)),
        MessagesPlaceholder(variable_name="messages"), # 메시지 히스토리 플레이스홀더
        ("human", "{input}")
    ])

    def format_data_for_anomaly_detection(state: TimeSeriesState) -> str:
        return f"Please identify anomalies in the time series data provided in the state (length: {len(state['ts_data'])})."

    def anomaly_detector_agent(state: TimeSeriesState) -> Dict[str, Any]:
        messages = list(state["messages"])
        input_text = format_data_for_anomaly_detection(state)

        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = anomaly_prompt.invoke(prompt_input)

        llm_with_tools = llm.bind_tools(tools) if tools else llm

        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message)

        content = ""

        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"AnomalyDetector: Detected {len(ai_message.tool_calls)} tool calls.")
            tool_response = tool_node.invoke({"messages": [ai_message]})
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            print(f"AnomalyDetector: Added {len(tool_messages)} tool results to history.")

            print("AnomalyDetector: Invoking LLM again with tool results.")
            follow_up_prompt = anomaly_prompt.invoke({"messages": messages, "input": "Summarize the detected anomalies based on the tool results."}) # 예시 input
            ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            messages.append(ai_follow_up)
            content = ai_follow_up.content
            print("AnomalyDetector: Received final response from LLM.")
        else:
            print("AnomalyDetector: No tool calls detected.")
            content = ai_message.content

        # --- 기본 이상 탐지 로직 (이 부분은 그대로 유지) ---
        ts_data = np.array(state["ts_data"])
        all_anomalies = []
        anomaly_values = []
        z_scores_list = [] # z_scores 변수명 충돌 피하기
        residual_z_scores_list = [] # residual_z_scores 변수명 충돌 피하기
        error_msg = None

        try:
            mean = np.mean(ts_data)
            std = np.std(ts_data)
            if std > 0: # 표준편차가 0이 아닐 때만 계산
                z_scores_list = ((ts_data - mean) / std).tolist()
                threshold = 3.0
                anomaly_indices_z = np.where(np.abs(np.array(z_scores_list)) > threshold)[0].tolist()
            else:
                anomaly_indices_z = []
                z_scores_list = [0.0] * len(ts_data)

            window_size = min(20, len(ts_data) // 10) if len(ts_data) > 20 else max(2, len(ts_data) // 5) # 최소 윈도우 크기 보장
            if len(ts_data) > window_size : # 충분한 데이터가 있을 때만 이동평균 계산
                moving_avg = np.convolve(ts_data, np.ones(window_size)/window_size, mode='valid')
                residuals = np.zeros_like(ts_data, dtype=float) # float 타입 명시
                pad = (len(ts_data) - len(moving_avg)) // 2
                residuals[pad:pad+len(moving_avg)] = ts_data[pad:pad+len(moving_avg)] - moving_avg

                residual_mean = np.mean(residuals[pad:pad+len(moving_avg)])
                residual_std = np.std(residuals[pad:pad+len(moving_avg)])
                contextual_anomalies = []
                if residual_std > 0:
                    residual_z_scores_list = np.zeros_like(residuals, dtype=float) # float 타입 명시
                    residual_z_scores_list[pad:pad+len(moving_avg)] = (residuals[pad:pad+len(moving_avg)] - residual_mean) / residual_std
                    contextual_anomalies = (pad + np.where(np.abs(residual_z_scores_list[pad:pad+len(moving_avg)]) > threshold)[0]).tolist()
                else:
                     residual_z_scores_list = [0.0] * len(ts_data)
            else:
                contextual_anomalies = []
                residual_z_scores_list = [0.0] * len(ts_data)

            all_anomalies = sorted(list(set(anomaly_indices_z + contextual_anomalies)))
            anomaly_values = [(int(idx), float(ts_data[idx])) for idx in all_anomalies if idx < len(ts_data)]

        except Exception as e:
            error_msg = f"Anomaly detection failed: {str(e)}"
            print(error_msg) # 에러 로깅
            all_anomalies = []
            anomaly_values = []
        # --- 기본 이상 탐지 로직 끝 ---

        analysis = {
            "anomaly_indices": all_anomalies[:20], # 결과 제한
            "anomaly_values": anomaly_values[:20], # 결과 제한
            "num_anomalies": len(all_anomalies),
            "llm_analysis": content,
            "error": error_msg # 에러 메시지 포함
        }

        return {
            "messages": messages,
            "anomaly_analysis": [analysis]
        }

    return anomaly_detector_agent