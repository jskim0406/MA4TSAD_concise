# Federated Sentinel

A Multi-agent LLM-based Time Series Anomaly Detection library.

## Overview

Federated Sentinel implements a multi-agent approach to time series anomaly detection. Rather than relying on a single agent to perform all analysis, it divides the task among specialized agents that work in parallel:

1. **Statistical Analyst** - Analyzes statistical properties of the time series
2. **Pattern Detector** - Identifies patterns, trends, and cycles
3. **Anomaly Detector** - Detects anomalies and outliers
4. **Supervisor** - Coordinates the workflow and aggregates results

This federated approach allows for more thorough and specialized analysis of complex time series data.

## Architecture

The library is built on LangGraph's parallel branch execution strategy, where multiple agents can analyze the data simultaneously:

```
                  ┌───────────┐
                  │           │
                  │Supervisor │
                  │           │
                  └─────┬─────┘
                        │
                        ▼
        ┌───────────────┬───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼───────┐ ┌─────▼────────┐
│              │ │              │ │              │
│   Analyst    │ │   Pattern    │ │   Anomaly    │
│              │ │   Detector   │ │   Detector   │
│              │ │              │ │              │
└───────┬──────┘ └──────┬───────┘ └─────┬────────┘
        │               │               │
        └───────────────┬───────────────┘
                        │
                        ▼
                  ┌───────────┐
                  │           │
                  │Supervisor │
                  │           │
                  └───────────┘
```

## Components

### Agents

1. **Supervisor Agent**: Coordinates the workflow, assigns tasks, and aggregates results into final analysis
2. **Analyst Agent**: Performs statistical analysis of the time series data
3. **Pattern Detector Agent**: Identifies trends, seasonality, and other patterns
4. **Anomaly Detector Agent**: Finds anomalies and outliers in the data

### Tools

1. **Visualization Tools**: Generate plots and visualizations of time series data
   - `ts2img`: Basic time series plot
   - `ts2img_with_anomalies`: Highlight anomalies in the time series
   - `ts2img_multi_view`: Multi-window view for better pattern detection

2. **Statistical Tools**: Calculate statistical properties and perform analyses
   - `basic_statistics`: Calculate mean, median, std, etc.
   - `trend_analysis`: Analyze trends in the time series
   - `seasonality_analysis`: Detect and analyze seasonality
   - `stationarity_test`: Test for stationarity
   - `anomaly_detection`: Detect anomalies using various methods

3. **Math Tools**:
   - `calculate`: Perform mathematical calculations
   - `rolling_window_stats`: Calculate statistics using a rolling window

### Workflow System

The workflow is managed through a graph-based execution system:
- `create_workflow`: Creates the multi-agent workflow graph
- `run_workflow`: Executes the workflow on time series data

## Installation

1. Install the library:
```bash
pip install -e .
```

2. Set up environment variables (create a `.env` file):
```
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=false
LANGCHAIN_PROJECT=federated-sentinel
OPENAI_API_KEY=your_openai_api_key  # if using OpenAI
GOOGLE_GEN_MODEL=gemini-1.5-flash  # specify model when using Google
```

## Basic Usage

```python
from langchain_google_vertexai import ChatVertexAI
from federated_sentinel.agents import (
    create_supervisor_agent, create_analyst_agent,
    create_pattern_detector_agent, create_anomaly_detector_agent
)
from federated_sentinel.tools import (
    ts2img, ts2img_with_anomalies, basic_statistics,
    trend_analysis, seasonality_analysis, anomaly_detection,
    get_math_calculator
)
from federated_sentinel.graph import create_workflow, run_workflow

# Initialize LLM
llm = ChatVertexAI(model="gemini-1.5-flash")

# Define tools
tools = [ts2img, ts2img_with_anomalies, basic_statistics, 
         trend_analysis, seasonality_analysis, anomaly_detection,
         get_math_calculator(llm)]

# Create agents
supervisor = create_supervisor_agent(llm)
analyst = create_analyst_agent(llm, tools)
pattern_detector = create_pattern_detector_agent(llm, tools)
anomaly_detector = create_anomaly_detector_agent(llm, tools)

# Create workflow
workflow = create_workflow(
    supervisor_agent=supervisor,
    analyst_agent=analyst,
    pattern_detector_agent=pattern_detector,
    anomaly_detector_agent=anomaly_detector,
    tools=tools
)

# Run analysis
time_series_data = [1, 2, 3, 4, 5, 100, 6, 7, 8]  # Example with anomaly
result = run_workflow(workflow, time_series_data)

# Access results
print(result["final_analysis"]["summary"])
```

## LangSmith Tracing

To enable LangSmith tracing for better observability:

1. Set up your environment:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=your_langsmith_api_key
export LANGCHAIN_PROJECT=federated-sentinel
```

2. Use the traceable decorator where needed:
```python
from federated_sentinel.utils.tracing import traceable

@traceable
def my_analysis_function(data):
    # Your code here
    pass
```

3. Use the provided utilities:
```python
from federated_sentinel.utils.tracing import setup_tracing

# In your main code
setup_tracing("my-project-name")
```

## LangGraph Visualization

To visualize your workflow with LangGraph:

1. Install the LangGraph CLI:
```bash
pip install "langgraph-cli[inmem]>=0.1.58"
```

2. Run the server from the project root:
```bash
langgraph dev
```

## Command Line Interface

The library provides a command-line interface:

```bash
# Basic usage
python src/generator_fed.py --data=your_data.csv

# Enable LangSmith tracing
python src/generator_fed.py --tracing --project=my-project-name

# Use advanced anomaly detection
python src/generator_fed.py --advanced
```

## Requirements

- Python 3.9+
- LangGraph
- LangChain
- Google Vertex AI Python SDK (or another LLM provider)
- NumPy
- SciPy
- Matplotlib
- Pandas
- statsmodels

## License

This project is licensed under the MIT License - see the LICENSE file for details.