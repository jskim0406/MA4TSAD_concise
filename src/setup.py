"""
Setup script for the federated_sentinel package.
"""

from setuptools import setup, find_packages

setup(
    name="federated_sentinel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain_google_vertexai>=0.0.1",
        "langchain_core>=0.1.0",
        "langgraph>=0.0.10",
        "langsmith>=0.0.56",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "fastdtw>=0.3.4",
        "pydantic>=2.0.0",
        "typing_extensions>=4.0.0",
        "python-dotenv>=0.19.0",
        "statsmodels>=0.13.0",
    ],
    author="Federated Sentinel Team",
    author_email="example@example.com",
    description="Multi-agent LLM-based Time Series Anomaly Detection",
    long_description=open("federated_sentinel/README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/federated_sentinel",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",
)