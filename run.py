# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import os
import sys
import json
import time
sys.path.append('..')
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.generator import get_llm_response
from src.generator_parallel_agents import research_agents

from Prompt_template import PromptTemplate
from utils import find_most_similar_series_fast, affine_transform, find_anomalies, find_zero_sequences


try:
    import tiktoken
except ImportError:
    print("Warning: 'tiktoken' library not found. Manual OpenAI token counting in run.py won't work (using SDK usage instead).")
    tiktoken = None

try:
    # run.py is likely executed from the LLMAD directory
    config_path = Path("config.yaml")
    if not config_path.exists():
         # Fallback if run from a different CWD
         config_path = Path(__file__).parent / "config.yaml" # Assumes run.py is in LLMAD/

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Error: config.yaml not found at expected location: {config_path}. Please create it.")
    sys.exit(1)
except NameError: # If Path wasn't imported yet
    from pathlib import Path
    config_path = Path("config.yaml")
    if not config_path.exists():
         config_path = Path(__file__).parent / "config.yaml"
    try:
        with open(config_path, "r") as file:
             config = yaml.safe_load(file)
    except FileNotFoundError:
         print(f"Error: config.yaml not found at expected location: {config_path}. Please create it.")
         sys.exit(1)
    except yaml.YAMLError as e:
         print(f"Error parsing config.yaml: {e}")
         sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing config.yaml: {e}")
    sys.exit(1)


# Check if OpenAI config is available for get_gpt_response
if 'openai' in config and 'api_key' in config['openai']:
    os.environ["OPENAI_API_KEY"] = config['openai']['api_key']
    if 'base_url' in config['openai']:
        os.environ["OPENAI_API_BASE"] = config['openai']['base_url']
    print("OpenAI API key configured for get_gpt_response.")
else:
    print("Warning: OpenAI config missing in config.yaml. get_gpt_response will fail if called.")


# --- LangSmith Configuration ---
if 'langsmith' in config:
    os.environ["LANGCHAIN_TRACING_V2"] = config['langsmith'].get('tracing_v2', "false")
    os.environ["LANGCHAIN_API_KEY"] = config['langsmith'].get('api_key', "")
    os.environ["LANGCHAIN_PROJECT"] = config['langsmith'].get('project', "Default Project") # Adjust project name if needed
    os.environ["LANGCHAIN_ENDPOINT"] = config['langsmith'].get('endpoint', "https://api.smith.langchain.com")
    if os.environ["LANGCHAIN_TRACING_V2"] == "true" and not os.environ["LANGCHAIN_API_KEY"]:
        print("Warning: LangSmith tracing is enabled, but LANGCHAIN_API_KEY is not set in config.yaml.")

# --- LangSmith Tracing Import ---
traceable = None
if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
    try:
        from langsmith import traceable
        print("LangSmith traceable imported successfully.")
    except ImportError:
        print("Warning: Could not import 'langsmith'. LangChain tracing might rely solely on environment variables.")
        # Define a dummy traceable decorator if Langsmith is not used or fails to import
        def traceable(func=None, **kwargs):
            """Dummy decorator that does nothing."""
            if func: return func
            else: return lambda f: f
else:
    # Define a dummy traceable decorator if Langsmith is not used
    def traceable(func=None, **kwargs):
         """Dummy decorator that does nothing."""
         if func: return func
         else: return lambda f: f


@traceable(name="get_gpt_response")
def get_gpt_response(args, prompt_res):
    """
    Calls the OpenAI API using LangChain and returns parsed response, raw response, and usage metadata.
    """
    result = None
    raw_result = ""
    usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        # Initialize the LangChain OpenAI model
        model_name = args.model_engine if hasattr(args, 'model_engine') and 'gpt' in args.model_engine.lower() else "gpt-4o-mini"
        
        # Use config values if available
        temperature = config['openai'].get('temperature', 0.7) if 'openai' in config else 0.7
        max_tokens = config['openai'].get('max_output_tokens', 2000) if 'openai' in config else 2000
        
        # Create the LangChain ChatOpenAI model
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={"response_format": {"type": "text"}},
        )
        
        # Prepare the message and invoke the model
        message = HumanMessage(content=prompt_res)
        response = llm.invoke([message])
        
        # Extract the response text
        raw_result = response.content
        
        # Get token usage (if available)
        if hasattr(response, 'response_metadata') and 'token_usage' in response.response_metadata:
            usage = response.response_metadata['token_usage']
            usage_metadata["input_tokens"] = usage.get('prompt_tokens', 0)
            usage_metadata["output_tokens"] = usage.get('completion_tokens', 0)
            usage_metadata["total_tokens"] = usage.get('total_tokens', 0)
        
        # --- Parse JSON ---
        if raw_result and isinstance(raw_result, str):
            # Clean potential markdown code blocks
            cleaned_str = re.sub(r'^```(json)?\s*|\s*```$', '', raw_result.strip(), flags=re.MULTILINE)
            if cleaned_str.startswith('{') and cleaned_str.endswith('}'):
                try:
                    result = json.loads(cleaned_str)
                    print("--- OpenAI JSON parsed ---")
                except json.JSONDecodeError as json_err:
                    print(f"Warning: OpenAI raw result looked like JSON but failed to parse. Error: {json_err}.")
                    result = {"error": "Failed to parse OpenAI JSON response", "raw": raw_result}
        
        # Add delay based on model type if needed
        if model_name and 'gpt-4o' in model_name.lower():
            time.sleep(1)

    except Exception as e:
        print(f"Error during OpenAI API call or processing: {e}")
        import traceback
        traceback.print_exc()
        
        if 'rate limit' in str(e).lower():
            print("Rate limit suspected. Sleeping...")
            time.sleep(20)  # Sleep longer for rate limits
        
        # Ensure structure is returned even on error
        result = None
        raw_result = f"Error in get_gpt_response: {str(e)}"
        # Reset usage on error
        usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # Ensure raw_result is a string
    if not isinstance(raw_result, str):
        raw_result = str(raw_result)

    return result, raw_result, usage_metadata

# --- Argument Parsing ---
def parse_args():
    arg_parser = argparse.ArgumentParser()
    # Keep existing arguments...
    arg_parser.add_argument('--infer_data_path', type=str, default="data/AIOPS_hard",required=False, help="Path to the inference data.")
    arg_parser.add_argument('--retreive_data_path', type=str, default="data/AIOPS",required=False, help="Path to the data to be retrieved.")
    arg_parser.add_argument('--sub_company', type=str, default="real",required=False, help="Subsidiary company.")
    # Defaulting to a llm model as per original script logic, user overrides via command line
    arg_parser.add_argument('--model_engine', type=str, default="gemini-1.5-flash-001", required=False, help="The model engine to be used (e.g., gpt-4o-mini, gemini-1.5-flash-001).")
    arg_parser.add_argument('--window_size', type=int, default=400, required=False, help="Window size for the time series.")
    arg_parser.add_argument('--test_ratio', type=float, default=0.05, required=False, help="Ratio of the data to be used for testing.")
    arg_parser.add_argument('--trans_alpha', type=int, default=95, required=False, help="Alpha parameter for transformation.")
    arg_parser.add_argument('--trans_beta', type=float, default=0, required=False, help="Beta parameter for transformation.")
    arg_parser.add_argument('--retreive_ratio', type=float, default=0.50, required=False, help="Ratio of data to be retrieved.")
    arg_parser.add_argument('--prompt_mode', type=int, default=13, required=False, help="Mode for the prompt.")
    arg_parser.add_argument('--result_save_dir', type=str, default='LLM_AD/GPT_AD_exp', required=False, help="Directory to save the results.")
    # Default run name might need adjustment based on model
    arg_parser.add_argument('--delete_temp_image', action='store_true', default=False, required=False, help="Delete the temporary image generated by ts2img tool after use.")
    arg_parser.add_argument('--run_name', type=str, default='LLMAD_Run_DefaultModel', required=False, help="Name of the run.")
    arg_parser.add_argument('--retrieve_positive_num', type=int, default=1, required=False, help="Number of positive instances to retrieve.")
    arg_parser.add_argument('--use_positive', action='store_false', default=True, required=False, help="Whether to use positive instances.")
    arg_parser.add_argument('--retrieve_negative_num', type=int, default=1, required=False, help="Number of negative instances to retrieve.")
    arg_parser.add_argument('--use_negative', action='store_false', default=True, required=False, help="Whether to use negative instances.")
    arg_parser.add_argument('--retrieve_negative_len', type=int, default=800, required=False, help="Length of negative instances to retrieve.")
    arg_parser.add_argument('--retrieve_negative_overlap', type=int, default=0, required=False, help="Overlap of negative instances to retrieve.")
    arg_parser.add_argument('--retrieve_database_ratio', type=float, default=0.1, required=False, help="Ratio of database to retrieve.")
    arg_parser.add_argument('--run_only_anomaly', type=bool, default=False, required=False, help="Whether to run only on anomalies.")
    arg_parser.add_argument('--overlap', type=bool, default=False, required=False, help="Whether to use overlap.")
    arg_parser.add_argument('--cross_retrieve', type=bool, default=True, required=False, help="Whether to use cross instance retrieval.")
    arg_parser.add_argument('--with_value', type=bool, default=False, required=False, help="Whether to predict value while testing.")
    arg_parser.add_argument('--dist_div_len', type=bool, default=False, required=False, help="Whether to divide the distance by length.")
    arg_parser.add_argument('--delete_zero', type=bool, default=False, required=False, help="Whether to delete zeros values.")
    arg_parser.add_argument('--interpolate_zero', type=bool, default=False, required=False, help="Whether to interpolate zeros values.")
    arg_parser.add_argument('--cost_analysis', type=bool, default=False, required=False, help="Whether to perform cost analysis.")
    return arg_parser.parse_args()

# --- Main Function ---
def main():
    args = parse_args()
    print(args)
    files = os.listdir(args.infer_data_path)
    run_dir = os.path.join(args.result_save_dir, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    if args.cross_retrieve:
        ad_list = []
        ad_str_list = []
        ad_label_list = []
        T_list = []
        retrieve_path = args.retreive_data_path
        retrieve_files = os.listdir(retrieve_path)
        retrieve_database_ratio = args.retrieve_database_ratio

        for idx, file in tqdm(enumerate(retrieve_files)):
            if not file.startswith(args.sub_company) and args.sub_company != 'all':
                continue
            path = os.path.join(retrieve_path, file)
            all_datas = pd.read_csv(path)
            # Fill missing values with 0
            all_datas.fillna(0, inplace=True)

            # Delete rows with value 0 and reindex
            if args.delete_zero:
                all_datas = all_datas[all_datas['value'] != 0]
                all_datas = all_datas.reset_index(drop=True)
                
            if args.interpolate_zero:
                # Replace 0 values with NaN for interpolation
                all_datas['value'] = all_datas['value'].replace(0, np.nan)
                # Perform linear interpolation
                all_datas['value'] = all_datas['value'].interpolate(method='linear')
                
            # Rename 'value' column to 'raw_value'
            all_datas.rename(columns={'value': 'raw_value'}, inplace=True)
            # Value transformation
            alpha = args.trans_alpha  # Percentile, can be adjusted as needed
            beta = args.trans_beta  # Beta hyperparameter, can be adjusted as needed
            transform_series = affine_transform(all_datas['raw_value'].values, alpha, beta)
    
            all_datas['value'] = (transform_series * 1000).astype(int)
            train_data = all_datas[: int(args.retreive_ratio * len(all_datas))]

            # jskim0406
            # Yahoo raw data에는 기존 코드에서의 column naming convention인 'label' 컬럼이 없음
            ### WSD raw data에는 'label'컬럼 모두 존재
            ### KPI raw data에는 'label' 혹은 'anomaly' 여부를 보여주는 일관된 컬럼명 찾기 어려움
            # 따라서 'is_anomaly'(A1, A2, A3), 'anomaly'(A4)를 'label'로 컬럼 rename 수행하는 전처리 과정 추가
            if 'label' not in train_data.columns:
                if 'is_anomaly' in train_data.columns:
                    train_data = train_data.rename(columns={'is_anomaly': 'label'})
                elif 'anomaly' in train_data.columns:
                    train_data = train_data.rename(columns={'anomaly': 'label'})
                else:
                    print(f"Warning: No anomaly label column found in data. Now, available columns: {train_data.columns.tolist()}")

            ad_list_, ad_str_list_, ad_label_list_ = find_anomalies(train_data, pad_len=20, max_len=200)
            
            ad_list_ = [np.squeeze(np.asarray(series)) for series in ad_list_]
            ad_list.extend(ad_list_)
            ad_str_list.extend(ad_str_list_)
            ad_label_list.extend(ad_label_list_)
        
        # Shuffle and take 5%
        ad_list = np.asarray(ad_list, dtype=object)
        ad_str_list = np.asarray(ad_str_list, dtype=object)
        ad_label_list = np.asarray(ad_label_list, dtype=object)
        
        shuffle_index = np.random.permutation(len(ad_list))
        ad_list = ad_list[shuffle_index]
        ad_str_list = ad_str_list[shuffle_index]
        ad_label_list = ad_label_list[shuffle_index]
        ad_list = ad_list[:int(len(ad_list) * retrieve_database_ratio)]
        ad_str_list = ad_str_list[:int(len(ad_str_list) * retrieve_database_ratio)]
        ad_label_list = ad_label_list[:int(len(ad_label_list) * retrieve_database_ratio)]
        
        print('ad_list len: ', len(ad_list))
        
    if args.cost_analysis:
        cost_cnt = 0
    for idx, file in enumerate(files):
        file_name = file.split('.')[0]
        print('file_name: ', file_name)
        if not file.startswith(args.sub_company) and args.sub_company != 'all':
            continue
        base_dir = os.path.join(run_dir, file_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        path = os.path.join(args.infer_data_path, file)
        all_datas = pd.read_csv(path)
        # Fill missing values with 0
        all_datas.fillna(0, inplace=True)
        
        # Delete rows with value 0
        if args.delete_zero:
            all_datas = all_datas[all_datas['value'] != 0]
        if args.interpolate_zero:
            # Replace 0 values with NaN for interpolation
            all_datas['value'] = all_datas['value'].replace(0, np.nan)
            # Perform linear interpolation
            all_datas['value'] = all_datas['value'].interpolate(method='linear')
            
        # Rename 'value' column to 'raw_value'
        all_datas.rename(columns={'value': 'raw_value'}, inplace=True)
        # Value transformation
        alpha = args.trans_alpha  # Percentile, can be adjusted as needed
        beta = args.trans_beta  # Beta hyperparameter, can be adjusted as needed
        transform_series = affine_transform(all_datas['raw_value'].values, alpha, beta)
        all_datas['value'] = (transform_series * 1000).astype(int)

        # jskim0406
        # Yahoo raw data에는 기존 코드에서의 column naming convention인 'label' 컬럼이 없음.
        ### WSD raw data에는 'label'컬럼 모두 존재
        ### KPI raw data에는 'label' 혹은 'anomaly' 여부를 보여주는 일관된 컬럼명 찾기 어려움
        # 따라서 'is_anomaly'(A1, A2), 'anomaly'(A3, A4)를 'label'로 컬럼 rename 수행하는 전처리 과정 추가
        if 'label' not in all_datas.columns:
            if 'is_anomaly' in all_datas.columns:
                all_datas = all_datas.rename(columns={'is_anomaly': 'label'})
            elif 'anomaly' in all_datas.columns:
                all_datas = all_datas.rename(columns={'anomaly': 'label'})
            else:
                print(f"Warning: No anomaly label column found in data. Now, available columns: {all_datas.columns.tolist()}")

        # jskim0406
        # Yahoo raw data > 'A3', 'A4'는 'timestamps' 컬럼, 'A1', 'A2'는 'timestamp' 컬럼.
        ## 소문자 's'가 A1, A2 <> A3, A4 간 통일 X.
        ## 기존 코드의 infer_data slicing column name convention에 맞춰, 'timestamp'로 rename force
        if 'timestamps' in all_datas.columns:
            all_datas = all_datas.rename(columns={'timestamps': 'timestamp'})
        
        infer_data = all_datas[int((1 - args.test_ratio) * len(all_datas)):]
        # infer_data = all_datas[int(0.35 * len(all_datas)):]
        infer_data = infer_data.reset_index(drop=True)
        train_data = all_datas[: int(args.retreive_ratio * len(all_datas))]
        if args.overlap:
            result_path = os.path.join(base_dir, 'part1.csv')
            result_overlap_path = os.path.join(base_dir, 'part2.csv')
        else:
            result_path = os.path.join(base_dir, 'predict.csv')
        log_path = os.path.join(base_dir, 'log.csv')

        infer_data['idx'] = range(len(infer_data))
        
        infer_data = infer_data[['idx', 'timestamp', 'raw_value', 'label', 'value']]
        # Check if the file exists
        if not os.path.exists(result_path):
            # Write the result header
            infer_result = pd.DataFrame(columns=['idx', 'timestamp', 'raw_value', 'label', 'value', 'predict', 'ad_len', 'alarm_level', 'anomaly_type'])
            infer_result.to_csv(result_path, mode='w', header=True, index=False)
            log_result = pd.DataFrame(columns=['ground_truth', 'predict', 'data', 'normal_data', 'anomaly_data', 'anomaly_label', 'scores','ad_scores', 'briefExplanation', 'anomaly_type', 'reason', 'alarm_level', 'prompt_res', 'raw_response', 'elapsed_time', 'api_elapsed_time'])
            log_result.to_csv(log_path, mode='w', header=True, index=False)
        else:
            exits_data = pd.read_csv(result_path)
            # Remove already predicted data from infer_data
            infer_data = infer_data[~infer_data['idx'].isin(exits_data['idx'])]    
            exits_data.sort_values(by=['idx'], inplace=True)    
            
        # Print the remaining data count
        print('infer_data len: ', len(infer_data))

        positive_count = 0
        negative_count = 0

        # Find continuous data with label 0 from train_data
        if not args.cross_retrieve:

            ad_list, ad_str_list, ad_label_list = find_anomalies(train_data, pad_len=20, max_len=0)
            ad_list = [np.squeeze(np.asarray(series)) for series in ad_list]

        T_list = find_zero_sequences(train_data, min_len=args.retrieve_negative_len, max_len=args.retrieve_negative_len, overlap=0)
        temp_len = args.retrieve_negative_len
        while len(T_list) == 0:
            temp_len = temp_len // 2
            T_list = find_zero_sequences(train_data, min_len=temp_len, max_len=temp_len, overlap=0)

        T_list = [np.squeeze(np.asarray(series)) for series in T_list]
        print('T_list len: ', len(T_list))
        print('ad_list len: ', len(ad_list))

        if len(T_list) == 0:
            continue

        retrieve_positive_num = min(args.retrieve_positive_num, len(ad_list))
        retrieve_negative_num = min(args.retrieve_negative_num, len(T_list))

        prompt_template = PromptTemplate(prompt_mode=args.prompt_mode)

        if args.overlap:
            increment = args.window_size // 2
        else:
            increment = args.window_size
        begin = False
        end = False
        for i in range(0, len(infer_data), increment):
            if args.cost_analysis:
                cost_cnt += 1
                if cost_cnt > 100:
                    break
            start_time = time.time()
            
            if i + args.window_size > len(infer_data):
                data = infer_data[i:]
                end = True
            else:
                data = infer_data[i: i + args.window_size]
            
            if i == 0:
                begin = True
            
            refer_data = data.idx.tolist()[0]
            X = data.value.tolist()
            X = np.squeeze(np.asarray(X))
            # Find the sequence most similar to X
            most_similar_series, scores, _ = find_most_similar_series_fast(X, T_list, top_k=retrieve_negative_num, dist_div_len=args.dist_div_len)

            # most_similar_series = most_similar_series[0]
            ad_demo_series, ad_scores, ad_demo_indexes = find_most_similar_series_fast(X, ad_list, top_k=retrieve_positive_num, dist_div_len=args.dist_div_len)

            ad_demo_series = [ad_demo_series[i].tolist() for i in range(len(ad_demo_series))]
            if len(most_similar_series) == 1:
                most_similar_series = most_similar_series[0]
                # Convert to int
                
                normal_data = ",".join(most_similar_series.astype(int).astype(str))
                most_similar_series = [most_similar_series.tolist()]
            else:
                normal_data = ""
                for idx, series in enumerate(most_similar_series):
                    
                    normal_data += "series_{}:".format(idx+1) + ",".join(series.astype(str)) + "\n"
                most_similar_series = [series.tolist() for series in most_similar_series]
            # Add * to the anomaly data
            anomaly_datas = []
            anomaly_labels = []
            for index in ad_demo_indexes:
                anomaly_datas.append(ad_str_list[index])
                anomaly_label = ad_label_list[index]
                anomaly_label = [int(i) for i in anomaly_label]    
                anomaly_labels.append(anomaly_label)

            # Data consists of index and value, separated by space, and each pair is separated by \n, forming a string
            cur_data = "\n".join([" ".join([str(index+1), str(value)]) for index, value in enumerate(data.value)]) 
            
            # Extract the relative index of the data with label 1
            label_index = data[data['label'] == 1].index - refer_data + 1
            
            if len(label_index) == 0 and args.run_only_anomaly:
                continue
            print('ground truth: ', label_index.tolist())
            
            if not args.use_positive:
                anomaly_datas = ['']
            if not args.use_negative:
                normal_data = ""
            prompt_res = prompt_template.get_template(normal_data=normal_data, data=cur_data, data_len=len(data), anomaly_datas=anomaly_datas)
            print(prompt_res)

            infer_start_time = time.time()



            ### WIP ###

            breakpoint()
            response = research_agents(args, ts_infer_data=cur_data)
            breakpoint()

            ### WIP ###



            end_time = time.time()
            elapsed_time = end_time - start_time
            api_elapsed_time = end_time - infer_start_time
            print('response: ', response)
            print('elapsed_time: ', elapsed_time)
            print('-------------------') 
            # Save the result, save it as a sequence of 0, 1
            
            if response is not None:
                is_anomaly = response['is_anomaly']
                anomalies = response['anomalies']
                if args.with_value:
                    anomalies = [anomaly[0] for anomaly in anomalies]
                briefExplanation = ""
                if 'briefExplanation' in response.keys():
                    briefExplanation = response['briefExplanation']
                anomaly_type = ""
                if 'anomaly_type' in response.keys():
                    anomaly_type = response['anomaly_type']
                reason = ""
                if 'reason' in response.keys():
                    reason = response['reason']
                alarm_level = ""
                if 'alarm_level' in response.keys():
                    alarm_level = response['alarm_level']
                if is_anomaly:
                    anomaly_index = [index + refer_data for index in anomalies]
                    # import pdb; pdb.set_trace()
                    predict = [1 if index + refer_data + 1 in anomaly_index else 0 for index in range(len(data))]
                    # append write save result to csv file, infer_result is a copy of data
                    infer_result = data.copy()
                    infer_result['predict'] = predict
                    infer_result['ad_len'] = [len(anomalies) for index in range(len(data))]
                    infer_result['alarm_level'] = [alarm_level for index in range(len(data))]
                    infer_result['anomaly_type'] = [anomaly_type for index in range(len(data))]
                else:
                    infer_result = data.copy()
                    infer_result['predict'] = [0 for index in range(len(data))]
                    infer_result['ad_len'] = [0 for index in range(len(data))]
                    infer_result['alarm_level'] = [alarm_level for index in range(len(data))]
                    infer_result['anomaly_type'] = [anomaly_type for index in range(len(data))]
                if args.overlap:
                    half_len = len(infer_result) // 2
                    if begin:# The first half is deposited in part2.csv
                        infer_result[:half_len].to_csv(result_overlap_path, mode='a', header=False, index=False)
                        begin = False
                    # First half deposited in part1.csv
                    infer_result[:half_len].to_csv(result_path, mode='a', header=False, index=False)
                    if end:
                        infer_result[half_len:].to_csv(result_path, mode='a', header=False, index=False)
                        end = False  
                    # Second half deposited in part2.csv      
                    infer_result[half_len:].to_csv(result_overlap_path, mode='a', header=False, index=False)
                else:
                    infer_result.to_csv(result_path, mode='a', header=False, index=False)
                log_result = pd.DataFrame(columns=['ground_truth', 'predict', 'data', 'normal_data', 'anomaly_data', 'anomaly_label', 'scores', 'ad_scores', 'briefExplanation', 'anomaly_type', 'reason', 'alarm_level', 'prompt_res', 'raw_response', 'elapsed_time', 'api_elapsed_time'])
                # Add a new data
                log_result.loc[0] = [json.dumps(label_index.tolist()), json.dumps(anomalies), json.dumps(data.value.tolist()), json.dumps(most_similar_series), 
                                    json.dumps(ad_demo_series), json.dumps(anomaly_labels), json.dumps(scores), json.dumps(ad_scores), briefExplanation, anomaly_type, reason, alarm_level, prompt_res, raw_response, elapsed_time, api_elapsed_time]
                
                log_result.to_csv(log_path, mode='a', header=False, index=False)
        # Read result_path for sorting
        infer_result = pd.read_csv(result_path)
        infer_result.sort_values(by=['idx'], inplace=True)
        infer_result.to_csv(result_path, mode='w', header=True, index=False)    
        print('finish ')
        
if __name__ == '__main__':
    main()