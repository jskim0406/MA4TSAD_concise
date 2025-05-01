# at root(`LLMAD`)
# python src/generator_fed.py --debug --output src/federated_sentinel/output_test
"""
Main script for Federated Sentinel - Multi-agent Time Series Anomaly Detection
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv, find_dotenv
# Load environment variables
load_dotenv(find_dotenv())

def setup_langsmith():
    """Configure LangSmith if available in config."""
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "")
setup_langsmith()

from langsmith import traceable
from langchain.globals import set_debug
from langchain_google_vertexai import ChatVertexAI

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


# Default environment variables for Google Cloud
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "hd-gen-ai-proc-391223")
LOCATION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
MODEL_NAME_G = os.getenv("GOOGLE_GEN_MODEL", "gemini-2.5-flash-preview-04-17")

@traceable
def main_fed():
    """
    Main function to demonstrate the Federated Sentinel library.
    """
    parser = argparse.ArgumentParser(description="Run Federated Sentinel Time Series Analysis")
    parser.add_argument("--data", type=str, help="Path to CSV data file")
    parser.add_argument("--model", type=str, default=MODEL_NAME_G, 
                       help="LLM model name to use")
    parser.add_argument("--advanced", action="store_true", 
                       help="Use advanced anomaly detection")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    parser.add_argument("--output", type=str, 
                       help="Output directory for results")
    args = parser.parse_args()
    
    # Enable debug mode if requested
    if args.debug:
        set_debug(True)
    
    print("Starting Federated Sentinel - Multi-agent Time Series Anomaly Detection")
    
    # Determine model to use
    model_name = args.model or MODEL_NAME_G
    print(f"Using LLM model: {model_name}")
    
    # Load data
    if args.data and os.path.exists(args.data):
        print(f"Loading time series data from {args.data}")
        try:
            df = pd.read_csv(args.data)
            col_name = df.select_dtypes(include=[np.number]).columns[0]
            sample = df[col_name].values.tolist()
            print(f"Loaded {len(sample)} data points from column '{col_name}'")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Using default sample data instead")
            # Default sample data
            sample = [753, 703, 500, 1028, 554, 1041, 603, 676, 645, 599, 502, 463, 483, 475, 526, 496, 619, 418, 895, 498, 727, 1018, 756, 763, 600, 668, 816, 490, 721, 644, 642, 347, 638, 506, 605, 578, 528, 560, 626, 649, 485, 257, 486, 649, 919, 702, 874, 614, 614, 469, 699, 430, 553, 469, 496, 934, 518, 597, 696, 602, 564, 509, 670, 775, 611, 874, 794, 613, 478, 657, 679, 644, 557, 567, 490, 685, 662, 511, 618, 606, 692, 308, 657, 583, 675, 736, 766, 811, 1042, 842, 547, 402, 1032, 598, 690, 643, 515, 621, 490, 550, 530, 500, 602, 679, 577, 573, 592, 644, 869, 811, 811, 766, 1042, 728, 527, 636, 663, 710, 297, 564, 772, 720, 687, 637, 491, 1041, 543, 518, 998, 342, 196, 702, 976, 702, 914, 891, 658, 636, 708, 1028, 743, 837, 517, 730, 607, 529, 568, 461, 598, 654, 726, 887, 356, 1042, 702, 530, 735, 691, 539, 657, 595, 509, 660, 628, 588, 631, 359, 442, 677, 619, 774, 668, 598, 623, 595, 825, 356, 725, 841, 517, 566, 516, 524, 925, 545, 665, 537, 425, 505, 559, 484, 520, 572, 663, 758, 920, 884, 818, 748, 171, 595, 464, 441, 622, 733, 543, 591, 582, 364, 562, 522, 566, 674, 633, 374, 542, 942, 876, 1006, 844, 716, 468, 555, 589, 698, 419, 525, 614, 436, 613, 691, 650, 594, 603, 596, 240, 839, 942, 702, 1023, 935, 938, 567, 790, 607, 758, 617, 577, 619, 620, 951, 752, 660, 493, 664, 545, 643, 613, 427, 999, 1024, 869, 614, 976, 869, 711, 891, 664, 783, 756, 793, 621, 833, 810, 729, 607, 655, 662, 930, 747, 674, 600, 544, 775, 695, 711, 542, 702, 944, 845, 652, 915, 710, 703, 884, 769, 701, 746, 765, 771, 751, 659, 674, 730, 702, 732, 1042, 869, 862, 1042, 942, 614, 570, 639, 685, 614, 599, 428, 635, 762, 632, 575, 810, 654, 659, 758, 538, 640, 600, 580, 914, 881, 811, 1031, 807, 614, 886, 626, 642, 668, 742, 739, 721, 502, 606, 644, 812, 582, 671, 715, 640, 653, 942, 784, 784, 631, 702, 817, 654, 760, 617, 514, 683, 667, 542, 730, 573, 681, 594, 609, 502, 599, 865, 931, 838, 675, 804, 627, 646, 757, 689, 736, 996, 761, 710, 595, 560, 657, 664, 705, 646, 671, 668, 666, 702, 708, 645, 786, 647, 781]
    else:
        print("Using default sample data")
        # sample = [753, 703, 500, 1028, 554, 1041, 603, 676, 645, 599, 502, 463, 483, 475, 526, 496, 619, 418, 895, 498, 727, 1018, 756, 763, 600, 668, 816, 490, 721, 644, 642, 347, 638, 506, 605, 578, 528, 560, 626, 649, 485, 257, 486, 649, 919, 702, 874, 614, 614, 469, 699, 430, 553, 469, 496, 934, 518, 597, 696, 602, 564, 509, 670, 775, 611, 874, 794, 613, 478, 657, 679, 644, 557, 567, 490, 685, 662, 511, 618, 606, 692, 308, 657, 583, 675, 736, 766, 811, 1042, 842, 547, 402, 1032, 598, 690, 643, 515, 621, 490, 550, 530, 500, 602, 679, 577, 573, 592, 644, 869, 811, 811, 766, 1042, 728, 527, 636, 663, 710, 297, 564, 772, 720, 687, 637, 491, 1041, 543, 518, 998, 342, 196, 702, 976, 702, 914, 891, 658, 636, 708, 1028, 743, 837, 517, 730, 607, 529, 568, 461, 598, 654, 726, 887, 356, 1042, 702, 530, 735, 691, 539, 657, 595, 509, 660, 628, 588, 631, 359, 442, 677, 619, 774, 668, 598, 623, 595, 825, 356, 725, 841, 517, 566, 516, 524, 925, 545, 665, 537, 425, 505, 559, 484, 520, 572, 663, 758, 920, 884, 818, 748, 171, 595, 464, 441, 622, 733, 543, 591, 582, 364, 562, 522, 566, 674, 633, 374, 542, 942, 876, 1006, 844, 716, 468, 555, 589, 698, 419, 525, 614, 436, 613, 691, 650, 594, 603, 596, 240, 839, 942, 702, 1023, 935, 938, 567, 790, 607, 758, 617, 577, 619, 620, 951, 752, 660, 493, 664, 545, 643, 613, 427, 999, 1024, 869, 614, 976, 869, 711, 891, 664, 783, 756, 793, 621, 833, 810, 729, 607, 655, 662, 930, 747, 674, 600, 544, 775, 695, 711, 542, 702, 944, 845, 652, 915, 710, 703, 884, 769, 701, 746, 765, 771, 751, 659, 674, 730, 702, 732, 1042, 869, 862, 1042, 942, 614, 570, 639, 685, 614, 599, 428, 635, 762, 632, 575, 810, 654, 659, 758, 538, 640, 600, 580, 914, 881, 811, 1031, 807, 614, 886, 626, 642, 668, 742, 739, 721, 502, 606, 644, 812, 582, 671, 715, 640, 653, 942, 784, 784, 631, 702, 817, 654, 760, 617, 514, 683, 667, 542, 730, 573, 681, 594, 609, 502, 599, 865, 931, 838, 675, 804, 627, 646, 757, 689, 736, 996, 761, 710, 595, 560, 657, 664, 705, 646, 671, 668, 666, 702, 708, 645, 786, 647, 781]
        sample_anomaly = [753, 703, 500, 1028, 554, 1041, 603, 676, 645, 599, 502, 463, 483, 475, 526, 496, 619, 418, 895, 498, 727, 1018, 756, 763, 600, 668, 816, 490, 721, 644, 642, 347, 638, 506, 605, 578, 10000, 9235, 626, 649, 485, 257, 486, 649, 919, 702, 874, 614, 614, 469, 699, 430, 553, 469, 496, 934, 518, 597, 696, 602, 564, 509, 670, 775, 611, 874, 794, 613, 478, 657, 679, 644, 557, 567, 490, 685, 662, 511, 618, 606, 692, 308, 657, 583, 675, 736, 766, 811, 1042, 842, 547, 402, 1032, 598, 690, 643, 515, 621, 490, 550, 530, 500, 602, 679, 577, 573, 592, 644, 869, 811, 811, 766, 1042, 728, 527, 636, 663, 710, 297, 564, 772, 720, 687, 637, 491, 1041, 543, 518, 998, 342, 196, 702, 976, 702, 914, 891, 658, 636, 708, 1028, 743, 837, 517, 730, 607, 529, 568, 461, 598, 654, 726, 887, 356, 1042, 702, 530, 735, 691, 539, 657, 595, 509, 660, 628, 588, 631, 359, 442, 677, 619, 774, 668, 598, 623, 595, 825, 356, 725, 841, 517, 566, 516, 524, 925, 545, 665, 537, 425, 505, 559, 484, 520, 572, 663, 758, 920, 884, 818, 748, 171, 595, 464, 441, 622, 733, 543, 591, 582, 364, 562, 522, 566, 674, 633, 374, 542, 942, 876, 1006, 844, 716, 468, 555, 589, 698, 419, 525, 614, 436, 613, 691, 650, 594, 603, 596, 240, 839, 942, 702, 1023, 935, 938, 567, 790, 607, 758, 617, 577, 619, 620, 951, 752, 660, 493, 664, 545, 643, 613, 427, 999, 1024, 869, 614, 976, 869, 711, 891, 664, 783, 756, 793, 621, 833, 810, 729, 607, 655, 662, 930, 747, 674, 600, 544, 775, 695, 711, 542, 702, 944, 845, 652, 915, 710, 703, 884, 769, 701, 746, 765, 771, 751, 659, 674, 730, 702, 732, 1042, 869, 862, 1042, 942, 614, 570, 639, 685, 614, 599, 428, 635, 762, 632, 575, 810, 654, 659, 758, 538, 640, 600, 580, 914, 881, 811, 1031, 807, 614, 886, 626, 642, 668, 742, 739, 721, 502, 606, 644, 812, 582, 671, 715, 640, 653, 942, 784, 784, 631, 702, 817, 654, 760, 617, 514, 683, 667, 542, 730, 573, 681, 594, 609, 502, 599, 865, 931, 838, 675, 804, 627, 646, 757, 689, 736, 996, 761, 710, 595, 560, 657, 664, 705, 646, 671, 668, 666, 702, 708, 645, 786, 647, 781]
        sample = sample_anomaly
    
    # Initialize the language model
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
    if args.advanced:
        print("Using advanced anomaly detector")
        anomaly_detector = create_advanced_anomaly_detector(llm, tools)
    else:
        print("Using standard anomaly detector")
        anomaly_detector = create_anomaly_detector_agent(llm, tools)
    
    # Create the workflow
    workflow = create_workflow(
        supervisor_agent=supervisor,
        analyst_agent=analyst,
        pattern_detector_agent=pattern_detector,
        anomaly_detector_agent=anomaly_detector,
        tools=tools
    )
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).resolve().parent.parent / f"results_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to {output_dir}")
    
    # Run the workflow
    print(f"\nRunning analysis on time series (length: {len(sample)})")
    start_time = datetime.now()

    final_state = run_workflow(workflow, sample)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Analysis completed in {duration:.2f} seconds")
    
    # Process and display results
    if final_state.get("final_analysis"):
        print("\n===== Final Analysis =====")
        analysis_text = final_state["final_analysis"]["summary"]
        print(analysis_text)
        
        # Parse the final analysis for structured access
        structured_analysis = parse_final_analysis(analysis_text)
        
        # Display anomalies if detected
        if structured_analysis.anomalies:
            print("\n===== Detected Anomalies =====")
            for i, anomaly in enumerate(structured_analysis.anomalies):
                print(f"Anomaly Set {i+1}:")
                print(f"- Indices: {anomaly.anomaly_indices}")
                print(f"- Values: {anomaly.anomaly_values[:5]}..." if len(anomaly.anomaly_values) > 5 else f"- Values: {anomaly.anomaly_values}")
                if anomaly.description:
                    print(f"- Description: {anomaly.description}")
            
            # Create a visualization of the anomalies
            all_indices = []
            for anomaly in structured_analysis.anomalies:
                all_indices.extend(anomaly.anomaly_indices)
            
            if all_indices:
                input_dict = {
                        "data": sample,
                        "anomaly_indices": all_indices,
                        "title": "Detected Anomalies"
                    }
                result_str = ts2img_with_anomalies.invoke(input_dict)
                try:
                    result = json.loads(result_str)
                    print(f"Visualization saved to: {result['image_path']}")
                except:
                    print("Error creating visualization")
        
        # Save the final analysis to file
        with open(output_dir / "analysis_results.json", "w") as f:
            json.dump({
                "time_series_length": len(sample),
                "analysis_duration_seconds": duration,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "final_analysis": final_state["final_analysis"],
                "detected_anomalies": [
                    {
                        "indices": anomaly.anomaly_indices,
                        "values": anomaly.anomaly_values,
                        "description": anomaly.description
                    }
                    for anomaly in structured_analysis.anomalies
                ] if structured_analysis.anomalies else []
            }, f, indent=2)
        print(f"\nResults saved to {output_dir / 'analysis_results.json'}")
    else:
        print("No final analysis was produced. Check the workflow execution.")


if __name__ == "__main__":
    main_fed()