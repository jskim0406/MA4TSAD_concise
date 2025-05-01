# src/generator.py
import io
import re
import os
import sys
import json
import time
import yaml
import base64
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

# --- Tiktoken Import ---
try:
    import tiktoken
except ImportError:
    print("Warning: 'tiktoken' library not found.")
    tiktoken = None

# --- Configuration and Client Setup ---
try:
    # Adjust path finding logic if necessary
    script_dir = Path(__file__).resolve().parent.parent
    config_path = script_dir / "config.yaml"

    if not config_path.exists():
        alt_config_path = Path("config.yaml")
        if alt_config_path.exists():
            config_path = alt_config_path
        else:
             raise FileNotFoundError("config.yaml not found in expected locations.")

    print(f"Attempting to load config from: {config_path.resolve()}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: config.yaml not found at {config_path.resolve()} or ./config.yaml. Please create it.")
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing config.yaml: {e}")
    sys.exit(1)

# LangSmith Configuration
if 'langsmith' in config:
    os.environ["LANGCHAIN_TRACING_V2"] = str(config['langsmith'].get('tracing_v2', "false")).lower()
    os.environ["LANGCHAIN_API_KEY"] = config['langsmith'].get('api_key', "")
    os.environ["LANGCHAIN_PROJECT"] = config['langsmith'].get('project', "LLMAD_Project") # Default project name
    os.environ["LANGCHAIN_ENDPOINT"] = config['langsmith'].get('endpoint', "https://api.smith.langchain.com")
    if os.environ["LANGCHAIN_TRACING_V2"] == "true" and not os.environ["LANGCHAIN_API_KEY"]:
        print("Warning: LangSmith tracing is enabled, but LANGCHAIN_API_KEY is not set in config.yaml.")

# --- LangSmith Tracing Import ---
traceable = None
if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
    try:
        from langsmith import traceable, RunTree
        print("LangSmith traceable imported successfully.")
    except ImportError:
        print("Warning: Could not import 'langsmith'. Install it (`pip install langsmith`) for tracing.")
        def traceable(func=None, **kwargs):
            if func: return func
            else: return lambda f: f
else:
    def traceable(func=None, **kwargs):
         if func: return func
         else: return lambda f: f

# --- Gemini and Tool Imports ---
try:
    from google import genai
    from google.genai import types as genai_types
    from google.generativeai import protos
    from google.protobuf.json_format import MessageToDict # Keep if using protos conversion
except ImportError as e:
    print(f"Error importing google.generativeai: {e}. Please ensure the library is installed (`pip install google-generativeai`).")
    sys.exit(1)
except AttributeError as e:
    print(f"Error: A required attribute might be missing from google.generativeai. Check SDK version? Error: {e}")
    sys.exit(1)

# --- TS2IMG Tool Definition ---
@traceable(name="ts2img_tool_execution")
def ts2img(data: List[float], title: str = "Time Series Plot", delete_image: bool = False) -> str:
    """
    Generates a line plot from a list of numbers and saves it as a local PNG image.
    Optionally deletes the image after execution. Returns a JSON string indicating success/failure.

    Args:
        data (List[float]): A list of numerical time series data to plot.
        title (str, optional): The title for the plot. Defaults to "Time Series Plot".
        delete_image (bool, optional): If True, deletes the generated image file. Defaults to False.

    Returns:
        str: A JSON string {"status": "success/error", "message": "...", "image_path": "..."}.
             image_path is included only on success.
    """
    save_path = None
    fig = None # Initialize fig to None
    try:
        if not data: raise ValueError("Input data list is empty for ts2img.")
        data_np = np.array(data)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data_np, linewidth=1.5)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Index", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        img_dir = Path(__file__).resolve().parent.parent / "temp_images"
        img_dir.mkdir(parents=True, exist_ok=True)
        save_path = img_dir / f"timeseries_{timestamp}.png"

        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"--- ts2img: Image saved to {save_path.resolve()} ---")
        return json.dumps({"status": "success", "message": f"Image generated: {save_path.name}", "image_path": str(save_path.resolve())})

    except Exception as e:
        error_message = f"Error in ts2img generating image: {e}"
        print(error_message)
        return json.dumps({"status": "error", "message": error_message})
    finally:
        if fig is not None and plt.fignum_exists(fig.number):
             plt.close(fig)
        if save_path and Path(save_path).exists() and delete_image:
            try:
                os.remove(save_path)
                print(f"--- ts2img: Deleted temporary image {save_path.name} ---")
            except OSError as remove_error:
                print(f"Warning: Could not delete temporary image {save_path}: {remove_error}")


# --- Client Initialization ---
gemini_client = None

try:
    if 'google' in config and 'project_id' in config['google'] and 'region' in config['google']:
        gemini_client = genai.Client(
            vertexai=True,
            project=config['google']['project_id'],
            location=config['google']['region']
        )
        print("Google GenAI client (Vertex AI via Client) initialized.")
    else:
        print("Warning: Google configuration missing (project_id, region). Google GenAI models not available via Vertex.")
        gemini_client = None
except Exception as e:
    print(f"Error initializing Google GenAI client: {e}")
    gemini_client = None

# --- Token Counting Helper ---
def num_tokens_from_string_openai(string: str, model_name: str) -> int:
    """Estimates token count for OpenAI models using tiktoken."""
    if not tiktoken:
        print("Tiktoken library not available, cannot count OpenAI tokens.")
        return 0
    try:
        if "gpt-4o" in model_name: encoding_name = "o200k_base"
        elif model_name.startswith("gpt-4"): encoding_name = "cl100k_base"
        elif model_name.startswith("gpt-3.5-turbo"): encoding_name = "cl100k_base"
        else:
            print(f"Warning: Unknown OpenAI model '{model_name}'. Using cl100k_base encoding as fallback.")
            encoding_name = "cl100k_base"
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(string))
    except Exception as e:
        print(f"Error counting OpenAI tokens for model {model_name}: {e}")
        return 0

# --- Extract Time Series Data ---
@traceable(name="extract_time_series_data")
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

    data_lines = data_block.split('\n')
    time_series_values = []
    for line in data_lines:
        line = line.strip()
        parts = line.split(maxsplit=1)
        value_str = None
        if len(parts) == 2: value_str = parts[1]
        elif len(parts) == 1: value_str = parts[0]

        if value_str:
            try:
                 value_str_cleaned = value_str.replace('*', '')
                 time_series_values.append(float(value_str_cleaned))
            except ValueError:
                 continue

    if not time_series_values:
        print(f"Warning: Found data section but could not parse numeric values from block:\n---\n{data_block}\n---")

    return time_series_values


# --- Internal Helper Functions for LLM Interaction ---

@traceable(name="_gemini_initial_inference")
def _gemini_initial_inference(
    client: genai.Client,
    model_engine: str,
    prompt: Optional[str],
    generation_config: genai_types.GenerateContentConfig
) -> Tuple[Optional[genai_types.GenerateContentResponse], Dict[str, int]]:
    """Performs the first Gemini inference call."""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    response = None

    if not prompt or not isinstance(prompt, str):
        print(f"Error: Invalid prompt received in _gemini_initial_inference: type={type(prompt)}")
        return None, usage

    try:
        try:
            count_response = client.models.count_tokens(model=model_engine, contents=[prompt])
            usage["input_tokens"] = count_response.total_tokens
            print(f"Gemini Initial Input Tokens Estimated: {usage['input_tokens']}")
        except Exception as count_e:
            print(f"Warning: Failed to count Gemini input tokens: {count_e}")

        print(f"--- Calling Gemini model {model_engine} for initial inference ---")
        response = client.models.generate_content(
            model=model_engine,
            contents=[prompt],
            config=generation_config,
        )
        print(f"--- Initial Gemini response received ---")

        output_text_for_count = ""
        output_has_func_call = False
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
             output_text_for_count = " ".join(p.text for p in response.candidates[0].content.parts if hasattr(p, 'text') and p.text)
             output_has_func_call = any(hasattr(p, 'function_call') and p.function_call for p in response.candidates[0].content.parts)

        if output_has_func_call:
            print("Initial response contains function call(s).")
            usage["output_tokens"] = 15 # Estimate

        if output_text_for_count:
            try:
                out_count_resp = client.models.count_tokens(model=model_engine, contents=[output_text_for_count])
                usage["output_tokens"] += out_count_resp.total_tokens # Add text tokens
                print(f"Gemini Initial Output Tokens Estimated (Text/FuncCall): {usage['output_tokens']}")
            except Exception as count_e:
                 print(f"Warning: Failed to count Gemini initial output tokens: {count_e}")
        elif not output_has_func_call:
             print("Warning: Initial Gemini response has no text or function call parts.")

        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        return response, usage

    except ValueError as ve:
        print(f"ValueError during initial Gemini inference: {ve}")
        return None, usage
    except Exception as e:
        print(f"General error during initial Gemini inference: {e}")
        import traceback
        traceback.print_exc()
        return None, usage

@traceable(name="_execute_tool_call")
def _execute_tool_call(
    function_call: Union[protos.FunctionCall, genai_types.FunctionCall],
    delete_temp_image: bool
) -> Tuple[Optional[Dict], Optional[str]]:
    """Executes the requested tool call (currently only ts2img)."""
    tool_response_dict = None
    image_path = None

    try:
        if isinstance(function_call, protos.FunctionCall):
            function_name = function_call.name
            function_args_dict = MessageToDict(function_call.args) if getattr(function_call, 'args', None) else {}
        elif isinstance(function_call, genai_types.FunctionCall):
             function_name = function_call.name
             function_args_dict = dict(function_call.args) if getattr(function_call, 'args', None) else {}
        else:
            raise TypeError(f"Unexpected function_call type: {type(function_call)}")

        print(f"--- Gemini requested tool call: {function_name} with args: {function_args_dict} ---")
        function_to_call = globals().get(function_name)

        if function_to_call and function_name == 'ts2img':
            try:
                if 'data' not in function_args_dict or not isinstance(function_args_dict['data'], list):
                     extracted_data = []
                     prompt_arg = function_args_dict.get('prompt')
                     if prompt_arg and isinstance(prompt_arg, str):
                         extracted_data = extract_time_series_from_prompt(prompt_arg)
                     if not extracted_data:
                         raise ValueError(f"Tool '{function_name}' called without valid 'data' list argument and could not extract from other args. Args received: {function_args_dict}")
                     else:
                          print("Warning: Extracted time series data from a different argument for ts2img.")
                          function_args_dict['data'] = extracted_data

                tool_title = function_args_dict.get('title', "Time Series Plot")

                print(f"--- Executing tool: {function_name} (delete_image={delete_temp_image}) ---")
                tool_response_str = function_to_call(
                    data=function_args_dict['data'],
                    title=tool_title,
                    delete_image=delete_temp_image
                )
                try:
                    tool_response_dict = json.loads(tool_response_str)
                    if tool_response_dict.get("status") == "success":
                        image_path = tool_response_dict.get("image_path")
                except json.JSONDecodeError:
                    print(f"Warning: Tool '{function_name}' did not return valid JSON: {tool_response_str}")
                    tool_response_dict = {"status": "error", "message": "Tool response not valid JSON", "raw_tool_response": tool_response_str}

            except ValueError as tool_val_err:
                print(f"ValueError during tool '{function_name}' execution/validation: {tool_val_err}")
                tool_response_dict = {"status": "error", "message": str(tool_val_err)}
            except Exception as tool_exec_e:
                print(f"General error executing/processing tool '{function_name}': {tool_exec_e}")
                tool_response_dict = {"status": "error", "message": str(tool_exec_e)}
        elif function_to_call:
            print(f"Warning: Gemini requested tool '{function_name}' which is defined but not handled in this flow. Ignoring.")
            tool_response_dict = {"status": "error", "message": f"Tool '{function_name}' defined but not configured for execution here."}
        else:
            print(f"Warning: Function '{function_name}' requested by Gemini but not defined locally.")
            tool_response_dict = {"status": "error", "message": f"Function '{function_name}' not defined."}

    except Exception as e:
        print(f"Error processing function call object: {e}")
        tool_response_dict = {"status": "error", "message": f"Internal error processing function call: {e}"}

    return tool_response_dict, image_path


@traceable(name="_gemini_inference_after_tool")
def _gemini_inference_after_tool(
    client: genai.Client,
    model_engine: str,
    history: List[Union[str, protos.Content, genai_types.Content]], # Accept SDK types
    tool_response: Dict, # Tool response is dict
    tool_name: str,
    image_path: Optional[str], # Add image_path parameter
    generation_config: genai_types.GenerateContentConfig
) -> Tuple[Optional[genai_types.GenerateContentResponse], Dict[str, int]]:
    """
    Performs the second Gemini inference call after a tool was executed.
    Handles potential image input generated by the tool using genai_types.
    Uses genai_types for constructing history parts. Corrected Part.from_text usage.
    """
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    response = None

    # --- Prepare History (Contents) for Second Call using genai_types ---
    print("--- Preparing history (contents) for Gemini call after tool execution ---")
    final_contents: List[genai_types.Content] = [] # Use SDK Content type

    # 1. Process existing history
    for item in history:
        if isinstance(item, str):
            # *** CORRECTED: Use keyword argument 'text=' ***
            final_contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=item)]))
        elif isinstance(item, genai_types.Content):
             final_contents.append(item)
        elif isinstance(item, protos.Content):
             # Convert proto Content to SDK Content
             try:
                  sdk_parts = []
                  for part_proto in item.parts:
                      if part_proto.text:
                           # *** CORRECTED: Use keyword argument 'text=' ***
                           sdk_parts.append(genai_types.Part.from_text(text=part_proto.text))
                      elif hasattr(part_proto, 'inline_data') and part_proto.inline_data:
                           sdk_parts.append(genai_types.Part.from_bytes(mime_type=part_proto.inline_data.mime_type, data=part_proto.inline_data.data))
                      elif hasattr(part_proto, 'function_call') and part_proto.function_call:
                           print("Warning: Skipping proto FunctionCall conversion in history.")
                           continue # Skip complex proto conversions for now
                  if sdk_parts:
                       final_contents.append(genai_types.Content(role=item.role or "user", parts=sdk_parts))
                       print("Converted proto.Content to genai_types.Content for history.")
                  else:
                       print("Warning: Skipping proto.Content with no convertible parts.")
             except Exception as proto_conv_err:
                  print(f"Warning: Skipping proto.Content conversion in history: {proto_conv_err}")
        else:
             print(f"Warning: Skipping unknown item type in history: {type(item)}")

    # 2. Add the tool response part
    try:
        tool_response_part = genai_types.Part.from_function_response(
            name=tool_name,
            response=tool_response
        )
        final_contents.append(genai_types.Content(role="tool", parts=[tool_response_part]))
        print(f"--- Added tool response part for '{tool_name}' ---")
    except Exception as part_err:
        print(f"Error creating Part from tool response dict: {part_err}. Tool response was: {tool_response}")
        return None, usage

    # 3. Add the image part if an image was generated
    image_part = None
    if image_path and os.path.exists(image_path):
        try:
            print(f"--- Reading image from path: {image_path} ---")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            ext = Path(image_path).suffix.lower()
            mime_type = "image/png"
            if ext in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
            elif ext == ".webp": mime_type = "image/webp"

            image_part = genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
            print(f"--- Created image part ({mime_type}) from {image_path} ---")
        except FileNotFoundError:
            print(f"Error: Image file not found at path: {image_path}")
            image_part = None
        except Exception as img_err:
            print(f"Error reading or creating image part: {img_err}")
            image_part = None

    # 4. Add the image part to the contents list
    if image_part:
        final_contents.append(genai_types.Content(role="user", parts=[
            # genai_types.Part.from_text("Here is the image generated by the tool:"), # Optional context
            image_part
        ]))
        print("--- Added image part to contents for the next call ---")

    # --- Make the Second API Call ---
    try:
        try:
             count_response = client.models.count_tokens(model=model_engine, contents=final_contents)
             usage["input_tokens"] = count_response.total_tokens
             print(f"Gemini Second Call Input Tokens Estimated (structure included): {usage['input_tokens']}")
        except Exception as count_e:
             print(f"Warning: Failed to count Gemini second call input tokens: {count_e}")

        print(f"--- Calling Gemini model {model_engine} after tool execution (contents length: {len(final_contents)}) ---")
        response = client.models.generate_content(
            model=model_engine,
            contents=final_contents,
            config=generation_config,
        )
        print(f"--- Final Gemini response received ---")

        final_text_output = ""
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            final_text_output = getattr(response.candidates[0].content.parts[0], 'text', "")

        if final_text_output:
            try:
                out_count_resp = client.models.count_tokens(model=model_engine, contents=[final_text_output])
                usage["output_tokens"] = out_count_resp.total_tokens
                print(f"Gemini Final Output Tokens Estimated: {usage['output_tokens']}")
            except Exception as count_e:
                 print(f"Warning: Failed to count Gemini final output tokens: {count_e}")
        else:
             usage["output_tokens"] = 0

        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        return response, usage

    except Exception as e:
        print(f"Error during Gemini inference after tool call: {e}")
        import traceback
        traceback.print_exc()
        return None, usage


# --- Main LLM Response Function (Orchestrator) ---
@traceable(name="get_llm_response")
def get_llm_response(args, prompt_res, structured_output=None):
    """
    Gets response synchronously ONLY from Vertex AI Gemini models.
    Handles manual tool declaration/invocation for Gemini's ts2img tool.
    Orchestrates the inference flow (initial -> tool -> final).
    Calculates total token usage and returns it along with JSON and raw results.
    """
    result_json = None
    raw_result = ""
    final_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    image_path_from_tool = None # Track generated image path from the tool call

    model_engine = args.model_engine.lower()
    langsmith_metadata = {"model_engine": model_engine, "prompt_mode": getattr(args, 'prompt_mode', 'N/A')}

    # --- Check Client and Model ---
    if 'gemini' not in model_engine:
        print(f"Error: get_llm_response intended only for Gemini models. Got: {model_engine}.")
        raw_result = f"Error: Model engine '{model_engine}' is not a Gemini model."
        return None, raw_result, final_usage
    if not gemini_client:
        print("Error: Gemini client (Vertex AI) not initialized.")
        raw_result = "Gemini client not initialized"
        return None, raw_result, final_usage

    print(f"--- Gemini Call Flow Starting: {model_engine} ---")

    try:
        # --- Prepare Tools and Config ---
        ts2img_declaration = genai_types.FunctionDeclaration(
            name='ts2img',
            description='Generates a line plot from a list of numbers (time series data). Input is a list of floats. Returns a JSON string indicating success or failure and optionally the image path.',
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'data': genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.NUMBER, format="float"), description='List of numerical time series data points to plot.'),
                    'title': genai_types.Schema(type=genai_types.Type.STRING, description='Optional title for the plot.')
                },
                required=['data']
             )
        )
        manual_tool = genai_types.Tool(function_declarations=[ts2img_declaration])

        # --- Generation Configuration ---
        gen_config_params = {
            "temperature": config['google'].get('temperature', 0.7),
            "max_output_tokens": config['google'].get('max_output_tokens', 8192),
            "tools": [manual_tool]
        }
        if structured_output:
             gen_config_params["response_mime_type"] = "application/json"
             if isinstance(structured_output, dict): gen_config_params["response_schema"] = structured_output
             else:
                 print(f"Warning: Unsupported structured_output type: {type(structured_output)}. Ignoring JSON mode.")
                 if "response_mime_type" in gen_config_params: del gen_config_params["response_mime_type"]
                 if "response_schema" in gen_config_params: del gen_config_params["response_schema"]
        gen_config_params = {k: v for k, v in gen_config_params.items() if v is not None}

        try:
             generation_config_object = genai_types.GenerateContentConfig(**gen_config_params)
        except TypeError as config_err:
             print(f"Error creating GenerateContentConfig: {config_err}. Using default with only tool.")
             generation_config_object = genai_types.GenerateContentConfig(tools=[manual_tool])

        # === Step 1: Initial Gemini Inference ===
        initial_response, initial_usage = _gemini_initial_inference(
            gemini_client, model_engine, prompt_res, generation_config_object
        )
        final_usage["input_tokens"] += initial_usage.get("input_tokens", 0)
        final_usage["output_tokens"] += initial_usage.get("output_tokens", 0)

        if initial_response is None:
            raw_result = "Error: Initial Gemini inference failed."
            print(raw_result)
            final_usage["total_tokens"] = final_usage["input_tokens"]
            return None, raw_result, final_usage

        if not initial_response.candidates:
            finish_reason_val = getattr(initial_response, 'prompt_feedback', None)
            block_reason = getattr(finish_reason_val, 'block_reason', 'UNKNOWN') if finish_reason_val else 'UNKNOWN'
            if block_reason != 'UNKNOWN' and block_reason != 'BLOCK_REASON_UNSPECIFIED':
                 raw_result = f"Error: Initial Gemini inference blocked. Reason: {block_reason}."
            else:
                 raw_result = "Error: Initial Gemini inference returned no candidates (unknown reason)."
            print(raw_result)
            final_usage["total_tokens"] = final_usage["input_tokens"] + final_usage["output_tokens"]
            return None, raw_result, final_usage

        candidate = initial_response.candidates[0]
        content = getattr(candidate, 'content', None)
        parts = getattr(content, 'parts', []) if content else []

        # === Step 2: Check for Tool Call and Execute ===
        function_call_part = None
        if parts:
             for part in parts:
                 if hasattr(part, 'function_call') and part.function_call:
                     function_call_part = part.function_call
                     break

        if function_call_part:
             should_delete_image = getattr(args, 'delete_temp_image', False)
             tool_response_dict, image_path_from_tool = _execute_tool_call(function_call_part, should_delete_image) # Capture image_path

             if tool_response_dict and tool_response_dict.get("status") == "success":
                # === Step 3: Gemini Inference After Tool ===
                history_before_tool_resp = [prompt_res]
                if content: history_before_tool_resp.append(content)
                else: print("Warning: No Content object found in initial response containing function call.")

                final_response, tool_followup_usage = _gemini_inference_after_tool(
                    gemini_client,
                    model_engine,
                    history_before_tool_resp,
                    tool_response_dict,
                    function_call_part.name,
                    image_path_from_tool, # Pass the path here
                    generation_config_object
                )

                final_usage["input_tokens"] += tool_followup_usage.get("input_tokens", 0)
                final_usage["output_tokens"] += tool_followup_usage.get("output_tokens", 0)

                if final_response and final_response.candidates:
                    final_candidate = final_response.candidates[0]
                    final_content = getattr(final_candidate, 'content', None)
                    final_parts = getattr(final_content, 'parts', []) if final_content else []
                    if final_parts:
                        raw_result = getattr(final_parts[0], 'text', "")
                        if not raw_result:
                             final_finish_reason = getattr(final_candidate, 'finish_reason', protos.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED)
                             if final_finish_reason == protos.Candidate.FinishReason.SAFETY: raw_result = f"Error: Final response blocked by safety settings. Details: {getattr(final_candidate, 'safety_ratings', [])}"
                             elif final_finish_reason == protos.Candidate.FinishReason.RECITATION: raw_result = "Error: Final response blocked due to recitation."
                             elif final_finish_reason != protos.Candidate.FinishReason.STOP: raw_result = f"Error: Final candidate has no text (Reason: {final_finish_reason})."
                             else: raw_result = "Error: No text in final response after tool call (unknown reason)."
                             print(f"Warning: {raw_result}")
                    else:
                        raw_result = "Error: No parts in final response content after tool call."
                        print(f"Warning: {raw_result} (Final Response: {final_response})")
                else:
                    final_block_reason = getattr(final_response.prompt_feedback, 'block_reason', 'UNKNOWN') if getattr(final_response, 'prompt_feedback', None) else 'UNKNOWN'
                    if final_block_reason != 'UNKNOWN' and final_block_reason != 'BLOCK_REASON_UNSPECIFIED': raw_result = f"Error: Final Gemini response blocked. Reason: {final_block_reason}."
                    else: raw_result = "Error: Could not extract final response after tool call (no candidates)."
                    print(f"Warning: {raw_result}")

             else: # Tool execution failed
                  raw_result = f"Error executing tool '{function_call_part.name}': {tool_response_dict.get('message', 'Unknown tool error')}" if tool_response_dict else f"Error: Tool '{function_call_part.name}' execution failed unexpectedly."
                  print(raw_result)

        else: # No tool call requested
             if parts:
                 raw_result = getattr(parts[0], 'text', "")
                 if not raw_result:
                     finish_reason = getattr(candidate, 'finish_reason', protos.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED)
                     if finish_reason == protos.Candidate.FinishReason.SAFETY: raw_result = f"Error: Response blocked by safety settings. Details: {getattr(candidate, 'safety_ratings', [])}"
                     elif finish_reason == protos.Candidate.FinishReason.RECITATION: raw_result = "Error: Response blocked due to recitation."
                     elif finish_reason != protos.Candidate.FinishReason.STOP: raw_result = f"Error: Candidate has no text (Reason: {finish_reason})."
                     else: raw_result = "Error: Could not extract text from initial response (empty parts?)."
                     print(f"Warning: {raw_result}")
             else:
                 finish_reason = getattr(candidate, 'finish_reason', protos.Candidate.FinishReason.FINISH_REASON_UNSPECIFIED)
                 if finish_reason == protos.Candidate.FinishReason.SAFETY: raw_result = f"Error: Response blocked by safety settings. Details: {getattr(candidate, 'safety_ratings', [])}"
                 elif finish_reason == protos.Candidate.FinishReason.RECITATION: raw_result = "Error: Response blocked due to recitation."
                 else: raw_result = f"Error: Initial response candidate has no content/parts (Reason: {finish_reason})."
                 print(f"Warning: {raw_result}")

    # --- Overall Exception Handling ---
    except Exception as e:
        print(f"Major error in get_llm_response orchestration for {model_engine}: {e}")
        import traceback
        traceback.print_exc()
        if 'rate limit' in str(e).lower():
             print("Rate limit suspected. Consider adding retry logic or sleeping.")
        result_json = None
        raw_result = f"Error in get_llm_response orchestration: {str(e)}"
        final_usage["output_tokens"] = 0

    # --- Final JSON Parsing ---
    if raw_result and isinstance(raw_result, str) and not raw_result.lower().startswith("error:"):
         cleaned_str = re.sub(r'^```(json)?\s*|\s*```$', '', raw_result.strip(), flags=re.MULTILINE)
         if cleaned_str.startswith('{') and cleaned_str.endswith('}'):
              try:
                  result_json = json.loads(cleaned_str)
                  print("--- JSON parsed from final raw result ---")
              except json.JSONDecodeError as json_err:
                  print(f"Warning: Final raw result looked like JSON but failed to parse. Error: {json_err}. Raw: '{raw_result[:100]}...'")
                  result_json = {"error": "Failed to parse LLM JSON response", "raw_response": raw_result}
         elif structured_output:
             print("Warning: Expected JSON output but parsing failed or result was not JSON.")
             result_json = {"error": "Failed to get expected JSON output", "raw_response": raw_result}
    elif raw_result and isinstance(raw_result, str) and raw_result.lower().startswith("error:"):
         result_json = {"error": raw_result}

    # --- Final Touches ---
    if not isinstance(raw_result, str): raw_result = str(raw_result)
    final_usage["total_tokens"] = final_usage["input_tokens"] + final_usage["output_tokens"]

    if image_path_from_tool:
        print(f"Generated image path (retention depends on args.delete_temp_image): {image_path_from_tool}")

    return result_json, raw_result


# --- Main Execution Guard (for testing) ---
if __name__ == '__main__':
    import argparse
    try:
        from Prompt_template import PromptTemplate
    except ImportError:
        print("Error: Could not import PromptTemplate. Make sure Prompt_template.py is in the same directory or accessible via PYTHONPATH.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Test LLM Generator with Tool Call")
    parser.add_argument('--model_engine', type=str, default="gemini-1.5-flash-001", help="Gemini model engine name (e.g., gemini-1.5-flash-001).")
    parser.add_argument('--prompt_mode', type=int, default=3, help="Prompt template mode (e.g., 1 for WSD, 2 for KPI, 3/4 for Yahoo).")
    parser.add_argument('--delete_temp_image', action='store_true', default=False, help="Delete generated image after test run.")
    args = parser.parse_args()

    # --- Example Data and Prompt Construction ---
    example_data_vals = [100, 105, 103, 108, 106, 110, 109, 150, 115, 112, 111, 109, 108, 112, 115, 110, 95, 98, 100]
    example_data_str = "\n".join([f"{i+1} {v}" for i, v in enumerate(example_data_vals)])
    example_normal_str = "95,98,96,100,97,99,101,98"
    example_anomaly_str = "sequence 1: 100,*200*,105,102"

    try:
        prompt_template_instance = PromptTemplate(prompt_mode=args.prompt_mode)
        prompt_for_llm = prompt_template_instance.get_template(
            normal_data=example_normal_str,
            data=example_data_str,
            data_len=len(example_data_vals),
            anomaly_datas=[example_anomaly_str]
        )
        prompt_for_llm += """

        ## Tool Instructions
        You MUST use the 'ts2img' tool to generate a plot of the time series data provided in the 'latest data points for evaluation' section. Analyze the plot and the data to determine anomalies.
        """
        print("--- Generated Prompt for Test ---")
        print(prompt_for_llm)
        print("-" * 30)
    except ValueError as prompt_err:
        print(f"Error creating prompt template: {prompt_err}")
        sys.exit(1)

    # --- Function to Run the Test ---
    def run_sync_test():
        print(f"\n--- Running Sync Test with Model: {args.model_engine} ---")
        if not (args.model_engine.startswith('gemini') and gemini_client):
            print(f"Error: Client for Gemini model {args.model_engine} not initialized or model is not Gemini. Check config.yaml.")
            return

        print("Calling get_llm_response with tool test prompt...")
        start_time = time.time()
        json_response, raw_text, usage = get_llm_response(args, prompt_for_llm, structured_output=None)
        end_time = time.time()
        elapsed = end_time - start_time

        print("\n" + "=" * 30 + " Test Result " + "=" * 30)
        print(f"Model: {args.model_engine}")
        print(f"Total Time: {elapsed:.2f} seconds")
        print("\nParsed JSON Response:")
        print(json.dumps(json_response, indent=2) if json_response else "None (or parsing failed)")
        print("\nRaw Response Text:")
        print(raw_text)
        print(f"\nToken Usage: {usage}")
        print("=" * (60 + len(" Test Result ")))

    # --- Execute Test ---
    try:
        run_sync_test()
    except Exception as main_e:
        print(f"Error during test execution: {main_e}")
        import traceback
        traceback.print_exc()