import os
import sys
import argparse

# 환경 변수 로드 (.env 파일 우선)
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def init_llm(args):
    if args.llm_provider=="google":

        from langchain_google_vertexai import ChatVertexAI
        # Default environment variables for Google Cloud
        llm = ChatVertexAI(
            model_name=os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash-preview-04-17"),
            temperature=os.getenv("GOOGLE_TEMPERATURE", "0.3"),
            max_tokens=os.getenv("GOOGLE_MAX_OUTPUT_TOKENS", "8192"),
        )

    elif args.llm_provider=="openai":

        # ref: https://python.langchain.com/docs/integrations/chat/openai/
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL_NAME", ""),
            temperature=os.getenv("OPENAI_TEMPERATURE", ""),
            max_tokens=os.getenv("OPENAI_MAX_OUTPUT_TOKENS", ""),
            timeout=None,
            max_retries=3,
        )

    elif args.llm_provider=="openrouter":

        # ref: https://openrouter.ai/docs/community/frameworks
        from langchain.chat_models import ChatOpenAI
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
            model_name=os.getenv("OPRNROUTER_MODEL_NAME",""),
            model_kwargs={
                "headers": {
                    "HTTP-Referer": os.getenv("HTTP_Referer", ""),
                    "X-Title": os.getenv("X_Title", ""),
                }
            },
        )

    else:
        raise NotImplementedError("현재 지원 가능한 모델 제공자는 'google', 'openai', 'opne router' 입니다. 모델 제공자(args.llm_provider)를 확인해주세요.")
    
    print(f'{args.llm_provider}의 언어모델 {llm}이 초기화 되었습니다.')
    return llm

def process_visualization_result(tool_response_str):
    """
    시각화 도구 응답에서 이미지 경로를 추출하고 이미지 파일을 로드하여 
    LLM에 전달할 수 있는 멀티모달 입력 형식으로 변환
    
    Args:
        tool_response_str: 시각화 도구의 JSON 응답 문자열
        
    Returns:
        list: LLM에 전달할 멀티모달 입력 형식의 콘텐츠 리스트 또는 None
    """
    try:
        response = json.loads(tool_response_str)
        if response.get("status") == "success" and "image_path" in response:
            image_path = response["image_path"]
            if os.path.exists(image_path):
                # 이미지 파일 읽기
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
                
                # 이미지 데이터를 LLM이 인식할 수 있는 형태로 준비
                return [
                    {"type": "text", "text": "Please analyze this time series visualization:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                ]
        return None
    except Exception as e:
        print(f"Error processing visualization result: {e}")
        return None