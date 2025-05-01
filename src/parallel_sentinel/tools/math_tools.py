"""
Mathematical computation tools for time series data.
"""

import math
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union

import numexpr
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool, StructuredTool
from langchain_core.runnables import RunnableConfig


# Math tool description
_MATH_DESCRIPTION = (
    "math(problem: str, context: Optional[list[str]]) -> float:\n"
    " - Solves the provided math problem.\n"
    ' - `problem` can be either a simple math problem (e.g. "1 + 3") or a word problem (e.g. "how many values exceed the threshold if the threshold is 100 and there are 5 values above 100").\n'
    " - You cannot calculate multiple expressions in one call. For instance, `math('1 + 3, 2 + 4')` does not work. "
    "If you need to calculate multiple expressions, you need to call them separately.\n"
    " - Minimize the number of `math` actions as much as possible. For instance, use a single complex expression rather than multiple simple ones.\n"
    " - You can optionally provide a list of strings as `context` to help solve the problem. "
    "If there are multiple pieces of context you need to answer the question, you can provide them as a list of strings.\n"
    " - For time series analysis, provide relevant statistics or time series segments as context when needed.\n"
    " - When asking questions about `context`, specify the units. "
    'For instance, "what percentage of values exceed 100?" instead of "what values exceed 100?"\n'
)


# System prompt for the math tool
_SYSTEM_PROMPT = """
Translate a math problem into an expression that can be executed using Python's numexpr library. 
Use the output of running this code to answer the question.

Question: ${{Question with math problem.}}


${{single line mathematical expression that solves the problem}}

...numexpr.evaluate(text)...


${{Output of running the code}}

Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?
ExecuteCode({{code: "37593 * 67"}})
...numexpr.evaluate("37593 * 67")...


2518731

Answer: 2518731

Question: The mean of a time series is 752.4 and the standard deviation is 156.8. What is the z-score for a value of 1042?
ExecuteCode({{code: "(1042 - 752.4) / 156.8"}})
...numexpr.evaluate("(1042 - 752.4) / 156.8")...


1.845026148969889

Answer: 1.845026148969889

Question: If 23 out of 384 values are anomalies, what is the percentage of anomalies?
ExecuteCode({{code: "(23 / 384) * 100"}})
...numexpr.evaluate("(23 / 384) * 100")...


5.989583333333333

Answer: 5.99%
"""


# Additional context prompt
_ADDITIONAL_CONTEXT_PROMPT = """
The following additional context is provided from other functions.
Use it to substitute into any ${{#}} variables or other values in the problem.

${context}

Note that context variables are not defined in code yet.
You must extract the relevant numbers and directly put them in code.
"""


class ExecuteCode(BaseModel):
    """The input to the numexpr.evaluate() function."""

    reasoning: str = Field(
        ...,
        description="The reasoning behind the code expression, including how context is included, if applicable.",
    )

    code: str = Field(
        ...,
        description="The simple code expression to execute by numexpr.evaluate().",
    )


def _evaluate_expression(expression: str) -> str:
    """
    Evaluate a mathematical expression using numexpr.
    
    Args:
        expression (str): The mathematical expression to evaluate
        
    Returns:
        str: The result of evaluating the expression
    """
    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
    except Exception as e:
        raise ValueError(
            f'Failed to evaluate "{expression}". Raised error: {repr(e)}.'
            " Please try again with a valid numerical expression"
        )

    # Remove any leading and trailing brackets from the output
    return re.sub(r"^\[|\]$", "", output)


def get_math_calculator(llm: BaseChatModel) -> StructuredTool:
    """
    Create a mathematical calculation tool using an LLM.
    
    Args:
        llm (BaseChatModel): The language model to use for calculations
        
    Returns:
        StructuredTool: A tool for performing mathematical calculations
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_PROMPT),
            ("user", "{problem}"),
            MessagesPlaceholder(variable_name="context", optional=True),
        ]
    )
    
    extractor = prompt | llm.with_structured_output(ExecuteCode)
    
    def calculate_expression(
        problem: str,
        context: Optional[List[str]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """
        Calculate the result of a mathematical expression or problem.
        
        Args:
            problem (str): The mathematical problem to solve.
            context (Optional[List[str]]): Additional context information.
            config (Optional[RunnableConfig]): Configuration for the runnable.
            
        Returns:
            str: The calculated result as a JSON string.
        """
        try:
            chain_input = {"problem": problem}
            if context:
                context_str = "\n".join(context)
                if context_str.strip():
                    context_str = _ADDITIONAL_CONTEXT_PROMPT.format(
                        context=context_str.strip()
                    )
                    chain_input["context"] = [SystemMessage(content=context_str)]
            
            code_model = extractor.invoke(chain_input, config)
            result = _evaluate_expression(code_model.code)
            
            # Return the calculation result as JSON
            return json.dumps({
                "problem": problem,
                "result": result,
                "reasoning": code_model.reasoning,
                "status": "success"
            })
            
        except Exception as e:
            error_message = f"Error during calculation: {repr(e)}"
            return json.dumps({"status": "error", "message": error_message})
    
    return StructuredTool.from_function(
        name="math",
        func=calculate_expression,
        description=_MATH_DESCRIPTION,
    )


@tool
def rolling_window_stats(data: List[float], window_size: int = 20) -> str:
    """
    Calculate statistics using a rolling window approach.
    
    Args:
        data (List[float]): The time series data.
        window_size (int, optional): Size of the rolling window. Defaults to 20.
        
    Returns:
        str: A JSON string containing rolling window statistics.
    """
    try:
        data_np = np.array(data)
        n = len(data_np)
        
        # Adjust window size if necessary
        if window_size >= n:
            window_size = max(2, n // 2)
        
        # Calculate rolling statistics
        rolling_mean = []
        rolling_std = []
        rolling_min = []
        rolling_max = []
        
        for i in range(0, n - window_size + 1):
            window = data_np[i:i+window_size]
            rolling_mean.append(float(np.mean(window)))
            rolling_std.append(float(np.std(window)))
            rolling_min.append(float(np.min(window)))
            rolling_max.append(float(np.max(window)))
        
        # Calculate differences between consecutive means
        mean_diff = np.diff(rolling_mean)
        std_diff = np.diff(rolling_std)
        
        result = {
            "window_size": window_size,
            "num_windows": len(rolling_mean),
            "rolling_stats": {
                "mean": rolling_mean[:5] + rolling_mean[-5:] if len(rolling_mean) > 10 else rolling_mean,
                "std": rolling_std[:5] + rolling_std[-5:] if len(rolling_std) > 10 else rolling_std,
                "min": rolling_min[:5] + rolling_min[-5:] if len(rolling_min) > 10 else rolling_min,
                "max": rolling_max[:5] + rolling_max[-5:] if len(rolling_max) > 10 else rolling_max
            },
            "mean_stability": {
                "mean_change": float(np.mean(np.abs(mean_diff))),
                "max_change": float(np.max(np.abs(mean_diff))),
                "change_points": [int(i) for i in np.where(np.abs(mean_diff) > 2 * np.std(mean_diff))[0]][:10]
            },
            "std_stability": {
                "mean_change": float(np.mean(np.abs(std_diff))),
                "max_change": float(np.max(np.abs(std_diff))),
                "change_points": [int(i) for i in np.where(np.abs(std_diff) > 2 * np.std(std_diff))[0]][:10]
            },
            "status": "success"
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error calculating rolling window statistics: {e}"
        return json.dumps({"status": "error", "message": error_message})


@tool
def time_series_metrics(
    data: List[float], 
    window_size: Optional[int] = None,
    seasonality_period: Optional[int] = None
) -> str:
    """
    Calculate comprehensive time series metrics.
    
    Args:
        data (List[float]): Time series data
        window_size (Optional[int]): Window size for local metrics
        seasonality_period (Optional[int]): Expected seasonality period
        
    Returns:
        str: JSON string with time series metrics
    """
    try:
        data_np = np.array(data)
        n = len(data_np)
        
        # Basic statistics
        basic_stats = {
            "mean": float(np.mean(data_np)),
            "median": float(np.median(data_np)),
            "std": float(np.std(data_np)),
            "min": float(np.min(data_np)),
            "max": float(np.max(data_np)),
            "range": float(np.max(data_np) - np.min(data_np)),
            "q1": float(np.percentile(data_np, 25)),
            "q3": float(np.percentile(data_np, 75)),
            "iqr": float(np.percentile(data_np, 75) - np.percentile(data_np, 25))
        }
        
        # Trend analysis
        x = np.arange(n)
        slope, intercept = np.polyfit(x, data_np, 1)
        
        trend_stats = {
            "slope": float(slope),
            "intercept": float(intercept),
            "direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
        }
        
        # Variation metrics
        diffs = np.diff(data_np)
        variation_stats = {
            "mean_abs_change": float(np.mean(np.abs(diffs))),
            "max_abs_change": float(np.max(np.abs(diffs))),
            "mean_change": float(np.mean(diffs)),
            "change_points": [int(i) for i in np.where(np.abs(diffs) > 2 * np.std(diffs))[0]][:10]
        }
        
        # Window-based metrics if window_size is provided
        window_stats = {}
        if window_size and window_size < n // 2:
            # Use the provided window size
            rolling_means = [np.mean(data_np[i:i+window_size]) for i in range(0, n-window_size+1, window_size//2)]
            rolling_stds = [np.std(data_np[i:i+window_size]) for i in range(0, n-window_size+1, window_size//2)]
            
            window_stats = {
                "window_size": window_size,
                "num_windows": len(rolling_means),
                "mean_stability": float(np.std(rolling_means) / np.mean(rolling_means)) if np.mean(rolling_means) != 0 else 0,
                "std_stability": float(np.std(rolling_stds) / np.mean(rolling_stds)) if np.mean(rolling_stds) != 0 else 0
            }
        
        # Seasonality metrics if seasonality_period is provided
        seasonality_stats = {}
        if seasonality_period and seasonality_period < n // 2:
            # Calculate autocorrelation at the specified lag
            acf = np.correlate(data_np - np.mean(data_np), data_np - np.mean(data_np), mode='full')
            acf = acf[n-1:] / (np.var(data_np) * np.arange(n, 0, -1))
            
            # Get autocorrelation at the seasonality period
            if seasonality_period < len(acf):
                seasonality_acf = acf[seasonality_period]
                
                seasonality_stats = {
                    "period": seasonality_period,
                    "autocorrelation": float(seasonality_acf),
                    "strength": "strong" if seasonality_acf > 0.5 else "moderate" if seasonality_acf > 0.3 else "weak"
                }
        
        # Combine all metrics
        result = {
            "length": n,
            "basic_stats": basic_stats,
            "trend": trend_stats,
            "variation": variation_stats,
            "status": "success"
        }
        
        if window_stats:
            result["window_analysis"] = window_stats
            
        if seasonality_stats:
            result["seasonality"] = seasonality_stats
        
        return json.dumps(result)
    
    except Exception as e:
        error_message = f"Error calculating time series metrics: {e}"
        return json.dumps({"status": "error", "message": error_message})