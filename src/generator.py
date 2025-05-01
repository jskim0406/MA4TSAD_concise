import io
import os
import re
import sys
import json
import yaml
import base64
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# LangChain imports
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Import needed for Vertex AI setup
try:
    from google.cloud import aiplatform
except ImportError:
    print("Warning: google-cloud-aiplatform is not installed. Vertex AI integration might fail.")

# --- LangSmith Tracing Import ---
try:
    from langsmith import traceable, RunTree
    LANGSMITH_AVAILABLE = True
    print("LangSmith traceable imported successfully.")
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("Warning: Could not import 'langsmith'. Install it (`pip install langsmith`) for tracing.")
    def traceable(func=None, **kwargs):
        if func: return func
        else: return lambda f: f

# --- Configuration Loading ---
def load_config():
    """Load configuration from config.yaml file."""
    try:
        script_dir = Path(__file__).resolve().parent.parent
        config_path = script_dir / "config.yaml"
        
        if not config_path.exists():
            alt_config_path = Path("config.yaml")
            if alt_config_path.exists():
                config_path = alt_config_path
            else:
                 raise FileNotFoundError("config.yaml not found in expected locations.")
        
        print(f"Loading config from: {config_path}")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: config.yaml not found. Please create it.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        sys.exit(1)

# --- LangSmith Configuration ---
def setup_langsmith(config):
    """Configure LangSmith if available in config."""
    if 'langsmith' in config and LANGSMITH_AVAILABLE:
        os.environ["LANGCHAIN_TRACING_V2"] = str(config['langsmith'].get('tracing_v2', "false")).lower()
        os.environ["LANGCHAIN_API_KEY"] = config['langsmith'].get('api_key', "")
        os.environ["LANGCHAIN_PROJECT"] = config['langsmith'].get('project', "LLMAD_Project")
        os.environ["LANGCHAIN_ENDPOINT"] = config['langsmith'].get('endpoint', "https://api.smith.langchain.com")
        
        if os.environ["LANGCHAIN_TRACING_V2"] == "true" and not os.environ["LANGCHAIN_API_KEY"]:
            print("Warning: LangSmith tracing is enabled, but LANGCHAIN_API_KEY is not set in config.yaml.")

def initialize_vertex_ai(config):
    """Initialize Vertex AI with project and location."""
    if 'google' in config and 'project_id' in config['google'] and 'region' in config['google']:
        try:
            aiplatform.init(
                project=config['google']['project_id'], 
                location=config['google']['region']
            )
            print(f"Initialized Vertex AI with project {config['google']['project_id']} in {config['google']['region']}")
            return True
        except Exception as e:
            print(f"Error initializing Vertex AI: {e}")
            return False
    else:
        print("Warning: Missing Google Cloud configuration. Need project_id and region.")
        return False

# --- Token Usage Extraction ---
def extract_token_usage(response):
    """
    Extract token usage information from Gemini response.
    Handles multimodal content by extracting modality-specific token counts.
    
    Args:
        response: The response object from the LLM
        
    Returns:
        dict: A dictionary with token usage information
    """
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "multimodal": False}
    
    # Extract token usage from response_metadata for Gemini models
    if hasattr(response, 'response_metadata') and 'usage_metadata' in response.response_metadata:
        metadata = response.response_metadata['usage_metadata']
        
        # Basic token counts
        usage["input_tokens"] = metadata.get('prompt_token_count', 0)
        usage["output_tokens"] = metadata.get('candidates_token_count', 0)
        usage["total_tokens"] = metadata.get('total_token_count', 0)
        
        # Detailed token information by modality
        modality_details = {}
        
        # Process input tokens by modality
        if 'prompt_tokens_details' in metadata:
            for detail in metadata['prompt_tokens_details']:
                modality = detail.get('modality', 1)  # Default to text (1)
                modality_name = "text" if modality == 1 else "image"
                token_count = detail.get('token_count', 0)
                
                if modality_name == "image":
                    usage["multimodal"] = True
                
                if modality_name in modality_details:
                    modality_details[modality_name]["input"] += token_count
                else:
                    modality_details[modality_name] = {"input": token_count, "output": 0}
        
        # Process output tokens by modality
        if 'candidates_tokens_details' in metadata:
            for detail in metadata['candidates_tokens_details']:
                modality = detail.get('modality', 1)  # Default to text (1)
                modality_name = "text" if modality == 1 else "image"
                token_count = detail.get('token_count', 0)
                
                if modality_name in modality_details:
                    modality_details[modality_name]["output"] += token_count
                else:
                    modality_details[modality_name] = {"input": 0, "output": token_count}
        
        # Add modality details to usage
        if modality_details:
            usage["modality_details"] = modality_details
    
    # Handle older model versions or different response formats
    elif hasattr(response, 'usage_metadata'):
        # Direct access to usage_metadata (older Gemini versions)
        usage["input_tokens"] = response.usage_metadata.get('input_tokens', 0)
        usage["output_tokens"] = response.usage_metadata.get('output_tokens', 0)
        usage["total_tokens"] = response.usage_metadata.get('total_tokens', 0)
    
    return usage

# --- Tool Definition ---
@tool
def ts2img(data: List[float], title: str = "Time Series Plot") -> str:
    """
    Generates a line plot from a list of numbers and saves it as a local PNG image.

    Args:
        data (List[float]): A list of numerical time series data to plot.
        title (str, optional): The title for the plot. Defaults to "Time Series Plot".

    Returns:
        str: A JSON string with status, message, and image path information.
    """
    try:
        if not data:
            raise ValueError("Input data list is empty")
        
        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data_np, linewidth=1.5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Create directory for images if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_dir = Path(__file__).resolve().parent.parent / "temp_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"timeseries_{timestamp}.png"
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Image saved to {save_path}")
        return json.dumps({
            "status": "success", 
            "message": f"Image generated: {save_path.name}", 
            "image_path": str(save_path)
        })
    
    except Exception as e:
        error_message = f"Error generating image: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})

# --- Extract Time Series Data from Prompt ---
def extract_time_series_from_prompt(prompt: str) -> List[float]:
    """Extracts time series data from the 'latest data points' section of the prompt."""
    data_section_match = re.search(r"latest \d+ data points for evaluation:\s*\n([\s\S]*)", prompt, re.IGNORECASE)
    if not data_section_match:
        data_section_match = re.search(r"## Data\s*.*?latest \d+ data points for evaluation:\s*\n([\s\S]*)", prompt, re.IGNORECASE | re.MULTILINE)

    if not data_section_match:
        print("Warning: Could not find 'latest data points' section in prompt.")
        return []

    data_block = data_section_match.group(1).strip()
    next_section_match = re.search(r'^\s*(##|$)', data_block, re.MULTILINE)
    if next_section_match:
        data_block = data_block[:next_section_match.start()].strip()

    time_series_values = []
    for line in data_block.split('\n'):
        line = line.strip()
        parts = line.split(maxsplit=1)
        value_str = None
        if len(parts) == 2: 
            value_str = parts[1]
        elif len(parts) == 1: 
            value_str = parts[0]

        if value_str:
            try:
                value_str_cleaned = value_str.replace('*', '')
                time_series_values.append(float(value_str_cleaned))
            except ValueError:
                continue

    if not time_series_values:
        print(f"Warning: Found data section but could not parse numeric values.")

    return time_series_values

# --- Main LLM Response Function ---
@traceable(name="get_llm_response")
def get_llm_response(args, prompt_res, structured_output=None):
    """
    Gets response from Vertex AI Gemini models with tool handling.
    
    Args:
        args: Command line arguments with model_engine and other settings
        prompt_res: The prompt text to send to the model
        structured_output: Optional schema for structured JSON output
        
    Returns:
        Tuple of (result_json, raw_result, token_usage)
    """
    model_engine = args.model_engine.lower()
    config = load_config()
    
    # Initialize Vertex AI
    initialize_vertex_ai(config)
    
    # Check if we need to import the Vertex AI model
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ImportError:
        print("Error: langchain_google_vertexai is not available. Please install it with: pip install langchain-google-vertexai")
        return None, "Error: Required package langchain-google-vertexai is not installed", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    
    # Check if model is Gemini
    if 'gemini' not in model_engine:
        print(f"Error: get_llm_response intended only for Gemini models. Got: {model_engine}.")
        return None, f"Error: Model {model_engine} is not supported", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Initialize variables
    result_json = None
    raw_result = ""
    final_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    image_path_from_tool = None
    langsmith_metadata = {"model_engine": model_engine, "prompt_mode": getattr(args, 'prompt_mode', 'N/A')}
    
    # Set start time for measuring execution time
    start_time = datetime.now()

    try:
        # --- Prepare Tools and Config ---
        ts2img_declaration = {
            "name": "ts2img",
            "description": "Generates a line plot from time series data",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array", 
                        "items": {"type": "number"},
                        "description": "List of numerical time series data points to plot."
                    },
                    "title": {
                        "type": "string", 
                        "description": "Optional title for the plot."
                    }
                },
                "required": ["data"]
            }
        }

        # --- Generation Configuration ---
        gen_config_params = {
            "temperature": config['google'].get('temperature', 0.7),
            "max_output_tokens": config['google'].get('max_output_tokens', 8192),
            "tools": [ts2img_declaration]
        }
        
        if structured_output:
             gen_config_params["response_mime_type"] = "application/json"
             if isinstance(structured_output, dict): 
                 gen_config_params["response_schema"] = structured_output
        
        # Initialize the LangChain Gemini model via Vertex AI
        llm = ChatVertexAI(
            model_name=model_engine,
            project=config['google'].get('project_id'),
            location=config['google'].get('region'),
            temperature=gen_config_params.get("temperature", 0.7),
            max_output_tokens=gen_config_params.get("max_output_tokens", 8192),
            top_p=0.95,
            top_k=40,
            verbose=True
        )
        
        # Bind tools to the model
        llm_with_tools = llm.bind_tools([ts2img])
        
        # Prepare prompt with tool usage instruction if not already included
        if "## Tool Instructions" not in prompt_res:
            prompt_res += """
            
            ## Tool Instructions
            You have access to a 'ts2img' tool that can generate time series plots. Use it to visualize the data and enhance your analysis.
            """
        
        # First model call
        start_time = datetime.now()
        response = llm_with_tools.invoke([HumanMessage(content=prompt_res)])

        # Calculate elapsed time
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"LLM response generated in {elapsed:.2f} seconds")
        
        # Get token usage (if available)
        usage = extract_token_usage(response)
        final_usage.update(usage)
        
        # For LangSmith tracing, add token usage as run metadata
        if LANGSMITH_AVAILABLE:
            import inspect
            current_frame = inspect.currentframe()
            if current_frame and hasattr(current_frame, 'f_back') and current_frame.f_back:
                run_id = getattr(current_frame.f_back, 'langsmith_run_id', None)
                if run_id:
                    from langsmith import Client
                    client = Client()
                    try:
                        # Add token usage as run metadata
                        client.update_run(
                            run_id,
                            metadata={
                                "token_usage": usage,
                                "model": model_engine,
                                "has_image": usage.get("multimodal", False)
                            }
                        )
                    except Exception as ls_err:
                        print(f"Warning: Failed to update LangSmith run metadata: {ls_err}")

        # Check if the response has tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Model requested {len(response.tool_calls)} tool calls")
            
            # For each tool call, execute the tool
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'ts2img':
                    # Extract args and handle data extraction if needed
                    args_dict = tool_call['args']
                    
                    # If data is missing, try to extract it from the prompt
                    if 'data' not in args_dict or not isinstance(args_dict['data'], list):
                        extracted_data = extract_time_series_from_prompt(prompt_res)
                        if extracted_data:
                            args_dict['data'] = extracted_data
                            print("Extracted time series data from prompt")
                        else:
                            print("Warning: Could not extract time series data from prompt")
                            continue
                    
                    # Use the invoke method instead of calling directly
                    tool_response = ts2img.invoke({
                        "data": args_dict['data'],
                        "title": args_dict.get('title', "Time Series Plot")
                    })
                    
                    # Parse the tool response
                    try:
                        tool_result = json.loads(tool_response)
                        if tool_result.get("status") == "success":
                            image_path = tool_result.get("image_path")
                            if image_path and os.path.exists(image_path):
                                image_path_from_tool = image_path
                                # Read image for the next model call
                                with open(image_path, "rb") as f:
                                    image_data = f.read()
                                
                                # Create a new message with the image
                                image_message = HumanMessage(content=[
                                    {"type": "text", "text": "Here is the visualization of the time series data:"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_data).decode('utf-8')}"}}
                                ])
                                
                                # Final model call with the image included
                                try:
                                    final_response = llm.invoke([
                                        HumanMessage(content=prompt_res),
                                        response,
                                        image_message
                                    ])

                                    # Get the final text response
                                    raw_result = final_response.content
                                    
                                    # Get token usage and update for the final call
                                    final_usage_with_image = extract_token_usage(final_response)
                                    if final_usage_with_image.get("multimodal", False):
                                        print(f"Multimodal response detected with images")
                                    
                                    # Update the token usage statistics
                                    final_usage["output_tokens"] += final_usage_with_image.get("output_tokens", 0)
                                    final_usage["total_tokens"] += final_usage_with_image.get("output_tokens", 0)
                                    
                                    # Update LangSmith with the final token usage
                                    if LANGSMITH_AVAILABLE:
                                        import inspect
                                        current_frame = inspect.currentframe()
                                        if current_frame and hasattr(current_frame, 'f_back') and current_frame.f_back:
                                            run_id = getattr(current_frame.f_back, 'langsmith_run_id', None)
                                            if run_id:
                                                from langsmith import Client
                                                client = Client()
                                                try:
                                                    client.update_run(
                                                        run_id,
                                                        metadata={
                                                            "final_token_usage": final_usage,
                                                            "with_image": True,
                                                            "image_path": image_path
                                                        }
                                                    )
                                                except Exception as ls_err:
                                                    print(f"Warning: Failed to update final LangSmith metadata: {ls_err}")
                                except Exception as img_call_error:
                                    print(f"Error in second model call with image: {img_call_error}")
                                    # Fallback to just using the first response
                                    raw_result = response.content
                            else:
                                print(f"Warning: Image file not found at {image_path}")
                                raw_result = response.content
                        else:
                            print(f"Tool execution failed: {tool_result.get('message', 'Unknown error')}")
                            raw_result = response.content
                    except json.JSONDecodeError:
                        print(f"Error: Tool response is not valid JSON: {tool_response}")
                        raw_result = response.content
                else:
                    print(f"Warning: Unknown tool call {tool_call['name']}")
                    raw_result = response.content
        else:
            # If no tool calls, use the direct response
            print("No tool calls detected in the model response")
            raw_result = response.content

        # Try to parse JSON from the result if it looks like JSON
        if raw_result and isinstance(raw_result, str) and not raw_result.lower().startswith("error:"):
            cleaned_str = re.sub(r'^```(json)?\s*|\s*```$', '', raw_result.strip(), flags=re.MULTILINE)
            if cleaned_str.startswith('{') and cleaned_str.endswith('}'):
                try:
                    result_json = json.loads(cleaned_str)
                    print("JSON parsed from response")
                except json.JSONDecodeError:
                    print(f"Warning: Response looked like JSON but failed to parse")
                    result_json = {"error": "Failed to parse LLM JSON response", "raw_response": raw_result}
        
    except Exception as e:
        print(f"Error in LLM response generation: {e}")
        import traceback
        traceback.print_exc()
        result_json = None
        raw_result = f"Error: {str(e)}"

    # Ensure raw_result is a string
    if not isinstance(raw_result, str):
        raw_result = str(raw_result)

    # Ensure the final usage numbers are set
    final_usage["total_tokens"] = final_usage["input_tokens"] + final_usage["output_tokens"]
    
    # Add a log of the final token usage
    print(f"Token usage: input={final_usage['input_tokens']}, output={final_usage['output_tokens']}, total={final_usage['total_tokens']}")
    if 'modality_details' in final_usage:
        print(f"Modal breakdown: {final_usage['modality_details']}")
    
    # Log to LangSmith as final step
    if LANGSMITH_AVAILABLE:
        import inspect
        current_frame = inspect.currentframe()
        if current_frame and hasattr(current_frame, 'f_back') and current_frame.f_back:
            run_id = getattr(current_frame.f_back, 'langsmith_run_id', None)
            if run_id:
                from langsmith import Client
                client = Client()
                try:
                    client.update_run(
                        run_id,
                        metadata={
                            "final_token_usage": final_usage,
                            "elapsed_time": (datetime.now() - start_time).total_seconds()
                        }
                    )
                except Exception as ls_err:
                    print(f"Warning: Failed to update final LangSmith metadata: {ls_err}")
    
    return result_json, raw_result, final_usage


# --- Main Execution Guard (for testing) ---
if __name__ == '__main__':
    import argparse
    try:
        from Prompt_template import PromptTemplate
    except ImportError:
        print("Error: Could not import PromptTemplate. Make sure Prompt_template.py is accessible.")
        sys.exit(1)
        
    # Add required packages check
    required_packages = ["langchain_google_vertexai", "google-cloud-aiplatform"]
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Warning: The following packages are required but not installed: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))

    parser = argparse.ArgumentParser(description="Test LLM Generator with Tool Call")
    parser.add_argument('--model_engine', type=str, default="gemini-1.5-flash-001", help="Gemini model engine name")
    parser.add_argument('--prompt_mode', type=int, default=3, help="Prompt template mode (1-4)")
    parser.add_argument('--delete_temp_image', action='store_true', help="Delete generated image after test run")
    args = parser.parse_args()

    # --- Load config and setup services ---
    config = load_config()
    setup_langsmith(config)
    
    # Initialize Vertex AI for the test
    if 'google' in config:
        initialize_vertex_ai(config)
    else:
        print("Warning: Google Cloud configuration missing. Test might fail.")

    # --- Example data and prompt construction ---
    example_data = [100, 105, 103, 108, 106, 110, 109, 150, 115, 112, 111, 109, 108, 112, 115, 110, 95, 98, 100]
    example_data_str = "\n".join([f"{i+1} {v}" for i, v in enumerate(example_data)])
    example_normal_str = "95,98,96,100,97,99,101,98"
    example_anomaly_str = "sequence 1: 100,*200*,105,102"

    try:
        prompt_template = PromptTemplate(prompt_mode=args.prompt_mode)
        prompt = prompt_template.get_template(
            normal_data=example_normal_str,
            data=example_data_str,
            data_len=len(example_data),
            anomaly_datas=[example_anomaly_str]
        )
        
        # Add tool instruction
        prompt += """

        ## Tool Instructions
        You MUST use the 'ts2img' tool to generate a plot of the time series data provided in the 'latest data points for evaluation' section. 
        Analyze the plot and the data to determine anomalies.
        """
        
        print("\n--- Generated Test Prompt ---")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print("-" * 30)
        
        # Run the test
        print(f"\n--- Running Test with Model: {args.model_engine} ---")
        json_response, raw_text, usage = get_llm_response(args, prompt)
        
        print("\n" + "=" * 30 + " Test Result " + "=" * 30)
        print(f"Model: {args.model_engine}")
        print("\nParsed JSON Response:")
        print(json.dumps(json_response, indent=2) if json_response else "None (or parsing failed)")
        print("\nRaw Response Text:")
        print(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)
        print(f"\nToken Usage: {usage}")
        print("=" * (60 + len(" Test Result ")))
        
    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()