"""
Statistical Analyst agent for time series data.
"""

import numpy as np
from typing import Dict, Any, List, Callable

from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage

from federated_sentinel.graph.workflow import TimeSeriesState


def create_analyst_agent(llm: BaseChatModel, tools: List[Callable] = None):
    """
    Create an analyst agent that performs statistical analysis on time series data.
    
    Args:
        llm: Language model to use for the agent
        tools: Optional list of tools available to the agent
        
    Returns:
        A function that can be added as a node to the workflow
    """
    # Define system prompt for the analyst
    system_template = """
    You are a statistical analyst specializing in time series data.
    Your task is to analyze the provided time series data and extract key statistical insights.

    Perform the following analyses:
    1. Basic statistics (mean, median, min, max, standard deviation)
    2. Trend analysis (increasing, decreasing, stable)
    3. Seasonality detection
    4. Stationarity assessment
    5. Distribution characteristics

    You have access to the following tools:
    - ts2img: Generate an image of the time series
    - calculate: Perform mathematical calculations

    Provide your analysis in a clear, structured format that highlights key findings.
    """

    tool_node = ToolNode(tools) if tools else None
    # 도구 설명을 프롬프트에 추가
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) if tools else "No tools available."

    analyst_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template.format(tool_descriptions=tool_descriptions)),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])

    def format_data_for_analysis(state: TimeSeriesState) -> str:
        """Format the time series data for analysis."""
        # 데이터 자체를 직접 전달하기보다, 분석 요청 메시지를 생성
        return f"Please analyze the time series data provided in the state (length: {len(state['ts_data'])})."

    def analyst_agent(state: TimeSeriesState) -> Dict[str, Any]:
        """
        Analyze time series data and return statistical insights.

        Args:
            state: Current workflow state

        Returns:
            Updated state with statistical analysis
        """
        # 현재 메시지 히스토리 복사
        messages = list(state["messages"])
        input_text = format_data_for_analysis(state)

        # Prepare prompt input - 메시지 히스토리와 현재 요청 포함
        prompt_input = {"messages": messages, "input": input_text}
        prompt_result = analyst_prompt.invoke(prompt_input) # 프롬프트 생성

        # 모델과 도구 바인딩 (필요 시)
        llm_with_tools = llm.bind_tools(tools) if tools else llm

        # 첫 번째 LLM 호출
        ai_message = llm_with_tools.invoke(prompt_result)
        messages.append(ai_message) # AI 응답(도구 호출 포함 가능)을 히스토리에 추가

        content = "" # 최종 내용 초기화

        # 도구 호출이 있는지 확인 및 처리
        if tools and hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            print(f"Analyst: Detected {len(ai_message.tool_calls)} tool calls.")
            # ToolNode 실행 (AIMessage만 전달)
            # 중요: ToolNode는 도구 호출이 포함된 AIMessage를 입력으로 받음
            tool_response = tool_node.invoke({"messages": [ai_message]})

            # tool_response['messages']에는 AIMessage와 ToolMessage가 함께 있을 수 있음
            # 여기서 ToolMessage만 추출하여 히스토리에 추가
            tool_messages = [msg for msg in tool_response["messages"] if isinstance(msg, ToolMessage)]
            messages.extend(tool_messages)
            print(f"Analyst: Added {len(tool_messages)} tool results to history.")

            # 도구 결과가 포함된 전체 메시지 히스토리로 두 번째 LLM 호출
            print("Analyst: Invoking LLM again with tool results.")
            # 중요: 두 번째 호출 시 'input'은 필요 없을 수 있음, 히스토리 기반으로 응답 생성
            # 프롬프트 구조에 따라 invoke 인자 조정 필요
            follow_up_prompt = analyst_prompt.invoke({"messages": messages, "input": "Summarize the findings based on the tool results."})
            ai_follow_up = llm_with_tools.invoke(follow_up_prompt)
            messages.append(ai_follow_up) # 최종 AI 응답 추가
            content = ai_follow_up.content
            print("Analyst: Received final response from LLM.")

        else:
            # 도구 호출이 없었으면 첫 번째 AI 메시지의 내용 사용
            print("Analyst: No tool calls detected.")
            content = ai_message.content

        # --- Numpy 기반 분석 (이 부분은 그대로 유지) ---
        ts_data = np.array(state["ts_data"])
        basic_stats = {
            "mean": float(np.mean(ts_data)),
            "median": float(np.median(ts_data)),
            "min": float(np.min(ts_data)),
            "max": float(np.max(ts_data)),
            "std": float(np.std(ts_data)),
        }

        analysis = {
            "basic_stats": basic_stats,
            # LLM 분석 결과와 Numpy 분석 결과를 함께 저장하거나, LLM이 요약하도록 유도
            "llm_analysis": content,
        }
        # --- Numpy 기반 분석 끝 ---

        # 최종 상태 업데이트 반환
        return {
            "messages": messages, # 업데이트된 전체 메시지 히스토리 반환
            "statistical_analysis": [analysis] # 분석 결과 추가 (operator.add 사용)
        }

    return analyst_agent