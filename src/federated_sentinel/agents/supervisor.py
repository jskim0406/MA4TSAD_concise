"""
Supervisor agent for federated sentinel.

This agent coordinates the workflow and aggregates results from specialized agents.
"""

from typing import Dict, Any, List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from federated_sentinel.graph.workflow import TimeSeriesState


def create_supervisor_agent(llm: BaseChatModel):
    """
    Create a supervisor agent that coordinates the workflow.
    
    Args:
        llm: Language model to use for the agent
        
    Returns:
        A function that can be added as a node to the workflow
    """
    # Define system prompt for the supervisor
    system_template = """You are the supervisor in a multi-agent time series analysis system.
    Your role is to coordinate the analysis workflow and make decisions about next steps.

    The workflow has the following specialized agents:
    1. Analyst - Performs statistical analysis of the time series data
    2. Pattern Detector - Identifies patterns in the time series data
    3. Anomaly Detector - Detects anomalies in the time series data

    Based on the current state of the analysis, you must decide:
    1. Which agents to invoke next, if any are needed
    2. Whether to conclude the analysis and provide a final summary

    If this is the first step, you should invoke all three specialized agents.
    If you've received analysis from all agents, you should provide a final comprehensive analysis.

    Remember:
    - The time series data represents continuous measurements over time
    - Anomalies can be sudden spikes/drops, pattern changes, or other unusual behaviors
    - Your final analysis should be clear, comprehensive, and actionable

    When providing a final analysis, structure it as follows:
    1. Summary of time series characteristics
    2. Patterns identified
    3. Anomalies detected (if any) with their potential significance
    4. Recommendations for monitoring or further analysis
    """

    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])

    def format_state_for_supervisor(state: TimeSeriesState) -> str:
        """Format the current state for the supervisor's consumption."""
        has_stat = len(state["statistical_analysis"]) > 0
        has_pattern = len(state["pattern_analysis"]) > 0
        has_anomaly = len(state["anomaly_analysis"]) > 0

        message = [
            f"Current analysis state:",
            f"- Time series data length: {len(state['ts_data'])}",
            f"- Statistical analysis received: {'Yes' if has_stat else 'No'}",
            f"- Pattern analysis received: {'Yes' if has_pattern else 'No'}",
            f"- Anomaly analysis received: {'Yes' if has_anomaly else 'No'}"
        ]

        if has_stat:
            message.append("\nLatest Statistical Analysis Summary:")
            # 마지막 분석 결과만 요약 표시 (필요 시 전체 표시 가능)
            message.append(f"- {str(state['statistical_analysis'][-1])[:200]}...")

        if has_pattern:
            message.append("\nLatest Pattern Analysis Summary:")
            message.append(f"- {str(state['pattern_analysis'][-1])[:200]}...")

        if has_anomaly:
            message.append("\nLatest Anomaly Analysis Summary:")
            message.append(f"- {str(state['anomaly_analysis'][-1])[:200]}...")

        # 모든 분석이 완료되었는지 확인
        if has_stat and has_pattern and has_anomaly:
             message.append("\nAll analyses received. Please provide a final comprehensive summary and analysis based on all gathered information.")
        else:
            message.append("\nWaiting for more analysis results...")

        return "\n".join(message)

    def supervisor_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        Process the current state, aggregate results, and potentially generate final analysis.
        (라우팅 로직 제거)

        Args:
            state: Current workflow state

        Returns:
            Dict: Dictionary containing updates to the state (messages, final_analysis)
        """
        # Format the current state for the supervisor
        input_text = format_state_for_supervisor(state)

        # Invoke the LLM
        prompt_result = supervisor_prompt.invoke({"input": input_text})
        llm_result = llm.invoke(prompt_result)
        content = llm_result.content

        # Add the supervisor's message to the state
        messages = list(state["messages"])
        messages.append(AIMessage(content=content, name="supervisor")) # 메시지에 이름 추가

        # Check if we have received analysis from all agents
        has_stat = len(state["statistical_analysis"]) > 0
        has_pattern = len(state["pattern_analysis"]) > 0
        has_anomaly = len(state["anomaly_analysis"]) > 0

        updates: Dict[str, Any] = {"messages": messages}

        # If we've received all analyses, generate and store the final analysis
        if has_stat and has_pattern and has_anomaly:
            # 여기서 content는 LLM이 생성한 최종 분석 요약
            updates["final_analysis"] = {
                "summary": content,
                "statistical_analysis": state["statistical_analysis"],
                "pattern_analysis": state["pattern_analysis"],
                "anomaly_analysis": state["anomaly_analysis"],
            }
            print("Supervisor: Final analysis generated.") # 로그 추가

        # Return only the state updates, not routing commands
        return updates

    return supervisor_agent