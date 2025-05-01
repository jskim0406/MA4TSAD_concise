"""
Test script for the Federated Sentinel library.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse

from langchain_google_vertexai import ChatVertexAI
from dotenv import load_dotenv, find_dotenv

# Import Federated Sentinel components
from federated_sentinel.agents import (
    create_supervisor_agent,
    create_analyst_agent,
    create_pattern_detector_agent,
    create_anomaly_detector_agent
)
from federated_sentinel.agents.advanced_anomaly_detector import create_advanced_anomaly_detector
from federated_sentinel.tools import (
    ts2img, ts2img_with_anomalies, ts2img_multi_view,
    basic_statistics, trend_analysis, seasonality_analysis,
    stationarity_test, anomaly_detection,
    get_math_calculator, rolling_window_stats
)
from federated_sentinel.graph import create_workflow, run_workflow
from federated_sentinel.utils.parser import parse_final_analysis
from federated_sentinel.utils.ts_utils import detect_anomalies_ensemble


def load_sample_data():
    """
    Load sample time series data with anomalies.
    """
    # Normal pattern with random noise
    np.random.seed(42)
    x = np.arange(0, 300)
    normal = 100 + 30 * np.sin(x * 0.05) + 10 * np.random.randn(len(x))
    
    # Add some anomalies
    data = normal.copy()
    # Point anomalies
    data[50] = normal[50] + 100  # Spike
    data[150] = normal[150] - 80  # Dip
    
    # Collective anomaly (level shift)
    data[200:230] = normal[200:230] + 50
    
    # Contextual anomaly (pattern break)
    data[80:100] = normal[80:100] * 0.5 + 20 * np.random.randn(20)
    
    return data.tolist()


def load_real_data(file_path=None):
    """
    Load real time series data from file if provided, else use the sample from utils.py.
    """
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'value' in df.columns:
                return df['value'].values.tolist()
            else:
                # Use the first numeric column
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        return df[col].values.tolist()
            
            raise ValueError("No suitable numeric column found in the CSV file")
        except Exception as e:
            print(f"Error loading data from {file_path}: {str(e)}")
            print("Using sample data instead")
    
    # Default sample from the original generator_fed.py
    sample = [753, 703, 500, 1028, 554, 1041, 603, 676, 645, 599, 502, 463, 483, 475, 526, 496, 619, 418, 895, 498, 727, 1018, 756, 763, 600, 668, 816, 490, 721, 644, 642, 347, 638, 506, 605, 578, 528, 560, 626, 649, 485, 257, 486, 649, 919, 702, 874, 614, 614, 469, 699, 430, 553, 469, 496, 934, 518, 597, 696, 602, 564, 509, 670, 775, 611, 874, 794, 613, 478, 657, 679, 644, 557, 567, 490, 685, 662, 511, 618, 606, 692, 308, 657, 583, 675, 736, 766, 811, 1042, 842, 547, 402, 1032, 598, 690, 643, 515, 621, 490, 550, 530, 500, 602, 679, 577, 573, 592, 644, 869, 811, 811, 766, 1042, 728, 527, 636, 663, 710, 297, 564, 772, 720, 687, 637, 491, 1041, 543, 518, 998, 342, 196, 702, 976, 702, 914, 891, 658, 636, 708, 1028, 743, 837, 517, 730, 607, 529, 568, 461, 598, 654, 726, 887, 356, 1042, 702, 530, 735, 691, 539, 657, 595, 509, 660, 628, 588, 631, 359, 442, 677, 619, 774, 668, 598, 623, 595, 825, 356, 725, 841, 517, 566, 516, 524, 925, 545, 665, 537, 425, 505, 559, 484, 520, 572, 663, 758, 920, 884, 818, 748, 171, 595, 464, 441, 622, 733, 543, 591, 582, 364, 562, 522, 566, 674, 633, 374, 542, 942, 876, 1006, 844, 716, 468, 555, 589, 698, 419, 525, 614, 436, 613, 691, 650, 594, 603, 596, 240, 839, 942, 702, 1023, 935, 938, 567, 790, 607, 758, 617, 577, 619, 620, 951, 752, 660, 493, 664, 545, 643, 613, 427, 999, 1024, 869, 614, 976, 869, 711, 891, 664, 783, 756, 793, 621, 833, 810, 729, 607, 655, 662, 930, 747, 674, 600, 544, 775, 695, 711, 542, 702, 944, 845, 652, 915, 710, 703, 884, 769, 701, 746, 765, 771, 751, 659, 674, 730, 702, 732, 1042, 869, 862, 1042, 942, 614, 570, 639, 685, 614, 599, 428, 635, 762, 632, 575, 810, 654, 659, 758, 538, 640, 600, 580, 914, 881, 811, 1031, 807, 614, 886, 626, 642, 668, 742, 739, 721, 502, 606, 644, 812, 582, 671, 715, 640, 653, 942, 784, 784, 631, 702, 817, 654, 760, 617, 514, 683, 667, 542, 730, 573, 681, 594, 609, 502, 599, 865, 931, 838, 675, 804, 627, 646, 757, 689, 736, 996, 761, 710, 595, 560, 657, 664, 705, 646, 671, 668, 666, 702, 708, 645, 786, 647, 781]
    return sample


def generate_synthetic_data(length=500, num_anomalies=5):
    """
    Generate synthetic time series data with anomalies.
    """
    # Base signal with trend and seasonality
    x = np.arange(length)
    trend = 0.05 * x
    seasonality = 10 * np.sin(x * 0.1) + 5 * np.sin(x * 0.05)
    noise = 3 * np.random.randn(length)
    
    data = trend + seasonality + noise
    
    # Add anomalies
    anomaly_indices = np.random.choice(range(10, length-10), num_anomalies, replace=False)
    for idx in anomaly_indices:
        # Point anomaly
        if np.random.random() < 0.7:
            data[idx] += (20 + 10 * np.random.random()) * (1 if np.random.random() < 0.5 else -1)
        # Short sequence anomaly
        else:
            length = np.random.randint(3, 8)
            shift = (10 + 5 * np.random.random()) * (1 if np.random.random() < 0.5 else -1)
            data[idx:idx+length] += shift
    
    return data.tolist(), anomaly_indices.tolist()


def run_test(data, model_name="gemini-2.5-flash", use_advanced=True, output_dir=None):
    """
    Run the Federated Sentinel workflow on the provided data.
    
    Args:
        data (List[float]): Time series data
        model_name (str): LLM model name to use
        use_advanced (bool): Whether to use the advanced anomaly detector
        output_dir (str): Directory to save results
        
    Returns:
        Dict: Results of the analysis
    """
    # Ensure output directory exists
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"./results_{timestamp}")
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the LLM
    llm = ChatVertexAI(model=model_name)
    
    # Define tools
    tools = [
        ts2img,
        ts2img_with_anomalies,
        ts2img_multi_view,
        basic_statistics,
        trend_analysis,
        seasonality_analysis,
        stationarity_test,
        anomaly_detection,
        rolling_window_stats,
        get_math_calculator(llm)
    ]
    
    # Create agents
    supervisor = create_supervisor_agent(llm)
    analyst = create_analyst_agent(llm, tools)
    pattern_detector = create_pattern_detector_agent(llm, tools)
    
    # Choose between regular and advanced anomaly detector
    if use_advanced:
        anomaly_detector = create_advanced_anomaly_detector(llm, tools)
    else:
        anomaly_detector = create_anomaly_detector_agent(llm, tools)
    
    # Create the workflow
    workflow = create_workflow(
        supervisor_agent=supervisor,
        analyst_agent=analyst,
        pattern_detector_agent=pattern_detector,
        anomaly_detector_agent=anomaly_detector,
        tools=tools
    )
    
    # Get initial anomaly detection results (without LLM)
    print("Performing initial anomaly detection...")
    initial_anomalies = detect_anomalies_ensemble(data)
    
    # Save the data and initial analysis
    with open(output_path / "data.json", "w") as f:
        json.dump({
            "time_series": data,
            "length": len(data),
            "initial_anomalies": initial_anomalies
        }, f, indent=2)
    
    # Run the workflow
    print(f"Running Federated Sentinel with {model_name}...")
    print(f"Time series length: {len(data)}")
    print(f"Using {'advanced' if use_advanced else 'standard'} anomaly detector")
    
    start_time = datetime.now()
    final_state = run_workflow(workflow, data)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    print(f"Analysis completed in {duration:.2f} seconds")
    
    # Save results
    if final_state.get("final_analysis"):
        print("\n===== Final Analysis =====")
        analysis_text = final_state["final_analysis"]["summary"]
        print(analysis_text)
        
        # Parse for structured access
        structured_analysis = parse_final_analysis(analysis_text)
        
        # Display anomalies if detected
        if structured_analysis.anomalies:
            print("\n===== Detected Anomalies =====")
            for i, anomaly in enumerate(structured_analysis.anomalies):
                print(f"Anomaly Set {i+1}:")
                print(f"- Indices: {anomaly.anomaly_indices}")
                if anomaly.description:
                    print(f"- Description: {anomaly.description}")
            
            # Create a visualization of the anomalies
            all_indices = []
            for anomaly in structured_analysis.anomalies:
                all_indices.extend(anomaly.anomaly_indices)
            
            if all_indices:
                viz_result = json.loads(ts2img_with_anomalies(data, all_indices, "Detected Anomalies"))
                print(f"Visualization saved to {viz_result.get('image_path')}")
        
        # Save the final analysis
        with open(output_path / "analysis_results.json", "w") as f:
            json.dump({
                "final_analysis": final_state["final_analysis"],
                "duration_seconds": duration,
                "model": model_name,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        
        return final_state
    else:
        print("No final analysis was produced. Check the workflow execution.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test the Federated Sentinel library")
    parser.add_argument("--data", type=str, help="Path to CSV file with time series data")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-preview-04-17", 
                      help="LLM model to use")
    parser.add_argument("--synthetic", action="store_true", 
                      help="Use synthetic data instead of sample data")
    parser.add_argument("--standard", action="store_true", 
                      help="Use standard anomaly detector instead of advanced")
    parser.add_argument("--output", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # Set up Google Cloud environment variables if not already set
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "hd-gen-ai-proc-391223"
    if not os.getenv("GOOGLE_CLOUD_REGION"):
        os.environ["GOOGLE_CLOUD_REGION"] = "us-central1"
    
    # Determine which data to use
    if args.synthetic:
        print("Generating synthetic data with anomalies...")
        data, true_anomalies = generate_synthetic_data()
        print(f"Generated synthetic data with {len(true_anomalies)} anomalies")
    elif args.data:
        print(f"Loading data from {args.data}...")
        data = load_real_data(args.data)
    else:
        print("Using sample data...")
        data = load_real_data()
    
    # Run the test
    run_test(
        data, 
        model_name=args.model, 
        use_advanced=not args.standard,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()