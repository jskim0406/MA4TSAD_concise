"""
LLM 유틸리티 모듈 - 시각적 추론 기반 에이전트를 위한 LLM 설정
"""

import os
import sys
from typing import Dict, Any, List, Optional

# 환경 변수 로드 (.env 파일 우선)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def init_llm(args):
    """
    시각적 추론 기능을 갖춘 멀티모달 LLM 초기화
    
    Args:
        args: 명령줄 인자 또는 설정 객체
        
    Returns:
        initialized_llm: 초기화된 LLM 객체
    """
    # 멀티모달 LLM 제공자 선택
    provider = getattr(args, "llm_provider", "google")

    if provider == "google":
        from langchain_google_vertexai import ChatVertexAI
        # Default environment variables for Google Cloud
        llm = ChatVertexAI(
            model_name=os.getenv("GOOGLE_MODEL_NAME", "gemini-1.5-pro"),
            temperature=float(os.getenv("GOOGLE_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("GOOGLE_MAX_OUTPUT_TOKENS", "8192")),
        )

    elif provider == "anthropic":
        # Claude
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-opus-20240229"),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192")),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )

    elif provider == "openai":
        # OpenAI GPT-4V
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL_NAME", "gpt-4-vision-preview"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "4096")),
            timeout=None,
            max_retries=3,
        )

    else:
        raise NotImplementedError("현재 지원 가능한 모델 제공자는 'google', 'anthropic', 'openai' 입니다.")
    
    print(f'{provider}의 멀티모달 언어모델이 초기화 되었습니다.')
    return llm