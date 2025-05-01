"""
Agents for the Federated Sentinel library.
"""

from federated_sentinel.agents.supervisor import create_supervisor_agent
from federated_sentinel.agents.analyst import create_analyst_agent
from federated_sentinel.agents.pattern_detector import create_pattern_detector_agent
from federated_sentinel.agents.anomaly_detector import create_anomaly_detector_agent

__all__ = [
    "create_supervisor_agent",
    "create_analyst_agent",
    "create_pattern_detector_agent",
    "create_anomaly_detector_agent"
]