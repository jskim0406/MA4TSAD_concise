"""
Advanced anomaly detection agent using ensemble methods.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode

from federated_sentinel.graph.workflow import TimeSeriesState
from federated_sentinel.utils.ts_utils import (
    detect_anomalies_z_score, 
    detect_anomalies_iqr,
    detect_anomalies_moving_average,
    detect_anomalies_ensemble
)


def create_advanced_anomaly_detector(llm: BaseChatModel, tools: List[Callable] = None):
    """
    Create an advanced anomaly detector agent that uses multiple methods.
    
    Args:
        llm: Language model to use for the agent
        tools: Optional list of tools available to the agent
        
    Returns:
        A function that can be added as a node to the workflow
    """
    # Define system prompt for the advanced anomaly detector
    system_template = """
    You are an advanced anomaly detection specialist analyzing time series data.
    You have access to multiple anomaly detection methods and you'll receive results from all of them.

    Your task is to:
    1. Interpret the results from different anomaly detection methods
    2. Identify which anomalies have high confidence (detected by multiple methods)
    3. Analyze the context around each anomaly to determine its significance
    4. Classify anomalies (e.g., point anomalies, contextual anomalies, collective anomalies)
    5. Assess the potential impact of each anomaly

    You have access to the following tools:
    - ts2img: Generate an image of the time series
    - ts2img_with_anomalies: Highlight anomalies in the time series
    - ts2img_multi_view: Multi-window view of time series
    - calculate: Perform mathematical calculations

    Provide your analysis in a clear, structured format that explains:
    - The number and locations of detected anomalies
    - Which anomalies have high confidence
    - The nature of each significant anomaly
    - Possible causes or explanations
    """

    tool_node = ToolNode(tools) if tools else None
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) if tools else "No tools available."

    anomaly_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template.format(tool_descriptions=tool_descriptions)),
        MessagesPlaceholder(variable_name="messages"), # 메시지 히스토리 플레이스홀더
        ("human", "{input}")
    ])

    def format_data_for_anomaly_detection(state: TimeSeriesState, anomaly_results: Dict[str, Any]) -> str:
        """Format the time series data and anomaly detection results."""
        ts_data = state["ts_data"]
        indices = anomaly_results["anomaly_indices"]
        values = [v for _, v in anomaly_results["anomaly_values"]]
        
        message_lines = [
            f"I've analyzed a time series with {len(ts_data)} data points using ensemble methods and detected {len(indices)} potential anomalies.",
            "",
            f"Time Series Summary:",
            f"- Length: {len(ts_data)}",
            f"- Mean: {np.mean(ts_data):.2f}",
            f"- Std Dev: {np.std(ts_data):.2f}",
            f"- Min: {np.min(ts_data):.2f}",
            f"- Max: {np.max(ts_data):.2f}",
            "",
            f"Ensemble Anomaly Detection Results:",
            f"- Total anomalies detected: {len(indices)}",
        ]
        
        if "confidence_scores" in anomaly_results:
            high_conf = [i for i, score in anomaly_results["confidence_scores"].items() if score > 0.66]
            med_conf = [i for i, score in anomaly_results["confidence_scores"].items() if 0.33 < score <= 0.66]
            low_conf = [i for i, score in anomaly_results["confidence_scores"].items() if score <= 0.33]
            
            message_lines.extend([
                f"- High confidence anomalies ({len(high_conf)}): {high_conf[:10]}...", # 예시: 처음 10개 인덱스
                f"- Medium confidence anomalies ({len(med_conf)}): {med_conf[:10]}...",
                f"- Low confidence anomalies ({len(low_conf)}): {low_conf[:10]}...",
            ])
        
        # Add anomaly details
        if indices:
            display_limit = min(10, len(indices))
            message_lines.append("")
            message_lines.append(f"Details for top {display_limit} anomalies:")

            anomaly_map = {idx: val for idx, val in anomaly_results.get("anomaly_values", [])}

            for i in range(display_limit):
                idx = indices[i]
                value = anomaly_map.get(idx, 'N/A')

                # Get context around the anomaly
                context_str = ""
                if idx != 'N/A':
                     start_idx = max(0, idx - 3)
                     end_idx = min(len(ts_data), idx + 4)
                     context = ts_data[start_idx:end_idx]
                     context_str = f"Context: { [f'{v:.2f}' for v in context] }"

                confidence = ""
                if "confidence_scores" in anomaly_results:
                    conf_score = anomaly_results["confidence_scores"].get(idx, 0)
                    conf_level = "high" if conf_score > 0.66 else "medium" if conf_score > 0.33 else "low"
                    confidence = f" (Confidence: {conf_level}, {conf_score:.2f})"

                message_lines.append(f"- Anomaly {i+1}: Index={idx}, Value={value:.2f if isinstance(value, float) else value}{confidence}. {context_str}")

        message_lines.append("")
        message_lines.append("Based on these ensemble results and context, please analyze these anomalies: Interpret the findings, classify the significant anomalies (point, contextual, collective), assess their potential impact, and suggest possible causes.") # 프롬프트 명확화

        return "\n".join(message_lines)

    def advanced_anomaly_detector_agent(state: TimeSeriesState) -> Dict[str, Any]:
        messages = list(state["messages"])
        ts_data = state["ts_data"]

        # Apply ensemble anomaly detection (이 부분은 LLM 호출과 별개)
        print("AdvancedAnomalyDetector: Running ensemble detection...")
        anomaly_results = detect_anomalies_ensemble(ts_data)
        print(f"AdvancedAnomalyDetector: Ensemble detection found {anomaly_results.get('num_anomalies', 0)} anomalies.")

        # Format the results for the LLM input
        input_text = format_data_for_anomaly_detection(state, anomaly_results)

        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = anomaly_prompt.invoke(prompt_input)

        llm_with_tools = llm.bind_tools(tools) if tools else llm

        # 첫 번째 LLM 호출 (분석 요청)
        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message)

        content = ""

        # 도구 호출 처리 (Advanced Anomaly Detector도 시각화 등 도구 사용 가능)
        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"AdvancedAnomalyDetector: Detected {len(ai_message.tool_calls)} tool calls.")
            tool_response = tool_node.invoke({"messages": [ai_message]})
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            print(f"AdvancedAnomalyDetector: Added {len(tool_messages)} tool results to history.")

            print("AdvancedAnomalyDetector: Invoking LLM again with tool results.")
            follow_up_prompt = anomaly_prompt.invoke({"messages": messages, "input": "Finalize the anomaly analysis based on the tool results."}) # 예시 input
            ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            messages.append(ai_follow_up)
            content = ai_follow_up.content
            print("AdvancedAnomalyDetector: Received final response from LLM.")
        else:
            print("AdvancedAnomalyDetector: No tool calls detected.")
            content = ai_message.content

        # 분석 결과 생성
        analysis = {
            "ensemble_results": anomaly_results, # 상세 결과 포함
            "anomaly_indices": anomaly_results["anomaly_indices"],
            "anomaly_values": anomaly_results["anomaly_values"],
            "confidence_scores": anomaly_results.get("confidence_scores", {}),
            "llm_analysis": content # LLM의 최종 분석 내용
        }

        return {
            "messages": messages,
            "anomaly_analysis": [analysis]
        }

    return advanced_anomaly_detector_agent