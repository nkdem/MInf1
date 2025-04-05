#!/usr/bin/env python3
"""
Model benchmarking script using thop (pytorch-OpCounter) for measuring
model efficiency metrics such as FLOPs and parameters.

References:
- Green AI (https://arxiv.org/abs/1907.10597)
- pytorch-OpCounter (https://github.com/Lyken17/pytorch-OpCounter)
"""

import os
import argparse
import json
import time
import torch
import numpy as np
from datetime import datetime
from thop import profile, clever_format

from models import CRNN

def benchmark_model_with_thop(model, input_shape, model_name, device='cpu', 
                             num_iterations=100, save_results=True, output_dir='output/benchmarks'):
    """
    Benchmark a model using thop and measure inference time.
    
    Args:
        model: PyTorch model
        input_shape: Input shape for the model (excluding batch dimension)
        model_name: Name of the model
        device: Device to run benchmarks on ('cpu', 'cuda', 'mps')
        num_iterations: Number of iterations for timing measurement
        save_results: Whether to save results to a JSON file
        output_dir: Directory to save results
    
    Returns:
        Dictionary containing benchmark results
    """
    print(f"\nBenchmarking {model_name} on {device}...")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect results
    results = {}
    
    # Use thop for FLOPs and parameter counting
    try:
        print("\n--- thop results ---")
        input_tensor = torch.randn(1, *input_shape).to(device).squeeze(0)
        flops, params = profile(model, inputs=(input_tensor,))
        flops_readable, params_readable = clever_format([flops, params], "%.3f")
        
        print(f"FLOPs: {flops_readable}, Parameters: {params_readable}")
        
        results["flops"] = float(flops)
        results["params"] = float(params)
        results["flops_readable"] = flops_readable
        results["params_readable"] = params_readable
    except Exception as e:
        print(f"Error with thop: {e}")
    
    # Measure inference time
    try:
        print("\n--- Inference time measurement ---")
        input_tensor = torch.randn(1, *input_shape).to(device).squeeze(0)
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
                input_tensor = torch.randn(1, *input_shape).to(device).squeeze(0)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(input_tensor)
                input_tensor = torch.randn(1, *input_shape).to(device).squeeze(0)
                # Synchronize if using GPU
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"Average inference time: {avg_time:.2f} ms (±{std_time:.2f} ms)")
        
        # Calculate operations per second (GOPS)
        ops_per_second = flops / (avg_time / 1000)  # Convert ms to seconds
        ops_per_second_readable = f"{ops_per_second/1e9:.2f} GOPS"
        
        print(f"Operations per second: {ops_per_second_readable}")
        
        results["avg_time_ms"] = float(avg_time)
        results["std_time_ms"] = float(std_time)
        results["ops_per_second"] = float(ops_per_second)
        results["ops_per_second_readable"] = ops_per_second_readable
        
        # Calculate operations per parameter
        if params > 0:
            ops_per_param = ops_per_second / params
            results["ops_per_parameter"] = float(ops_per_param)
            print(f"Operations per parameter: {ops_per_param:.2f}")
    except Exception as e:
        print(f"Error measuring inference time: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL METRICS")
    print("="*50)
    
    print(f"\nPARAMETER COUNTS:")
    print(f"  Total Parameters:      {params:,}")
    
    print(f"\nINFERENCE PERFORMANCE:")
    print(f"  Avg Inference Time:    {avg_time:.2f} ms")
    print(f"  Std Inference Time:    {std_time:.2f} ms")
    print(f"  FLOPs per Inference:   {flops:,}")
    print(f"  Operations per Second: {ops_per_second/1e9:.2f} GOPS")
    if params > 0:
        print(f"  OPS per Parameter:     {ops_per_param:.2f}")
    
    print("\n" + "="*50)
    
    # Save results
    if save_results and results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"{model_name}_benchmark_{device}_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {filename}")
    
    return results


# benchmark the CRNN model
model = CRNN()
benchmark_model_with_thop(model, (1, 205, 161), "CRNN", device='cuda', save_results=True, output_dir='output/benchmarks')
benchmark_model_with_thop(model, (1, 205, 161), "CRNN", device='cpu', save_results=True, output_dir='output/benchmarks')