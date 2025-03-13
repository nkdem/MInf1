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

from models import AudioCNN, CNNSpeechEnhancer
from constants import MODELS

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
        input_tensor = torch.randn(1, *input_shape).to(device)
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
        input_tensor = torch.randn(1, *input_shape).to(device)
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(input_tensor)
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

def benchmark_audio_cnn_models(device='cpu', save_results=True, output_dir='output/benchmarks'):
    """
    Benchmark all AudioCNN models defined in constants.MODELS
    
    Args:
        device: Device to run benchmarks on ('cpu', 'cuda', 'mps')
        save_results: Whether to save results to a JSON file
        output_dir: Directory to save results
    """
    print(f"Benchmarking AudioCNN models on {device}...")
    
    # Create output directory if it doesn't exist
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Results dictionary
    results = {}
    
    # Input shape for AudioCNN: (1, 40, 500) - (channels, mel_bands, time_frames)
    input_shape = (1, 40, 500)
    
    # Benchmark each model
    for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in MODELS.items():
        print(f"\nBenchmarking {model_name}...")
        
        # Create model
        model = AudioCNN(num_classes=15, 
                         cnn1_channels=cnn1_channels, 
                         cnn2_channels=cnn2_channels, 
                         fc_neurons=fc_neurons)
        
        # Benchmark model
        model_results = benchmark_model_with_thop(
            model=model,
            input_shape=input_shape,
            model_name=model_name,
            device=device,
            save_results=False  # We'll save all results together
        )
        
        # Store results
        results[model_name] = model_results
    
    # Save combined results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"audiocnn_benchmark_{device}_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nCombined results saved to {filename}")
    
    return results

def benchmark_speech_enhancer(device='cpu', save_results=True, output_dir='output/benchmarks'):
    """
    Benchmark the CNNSpeechEnhancer model
    
    Args:
        device: Device to run benchmarks on ('cpu', 'cuda', 'mps')
        save_results: Whether to save results to a JSON file
        output_dir: Directory to save results
    """
    print(f"Benchmarking CNNSpeechEnhancer on {device}...")
    
    # Create model
    model = CNNSpeechEnhancer()
    
    # Input shape for CNNSpeechEnhancer: (40, 1, 500) - (channels, height, width)
    input_shape = (40, 1, 500)
    
    # Benchmark model
    results = benchmark_model_with_thop(
        model=model,
        input_shape=input_shape,
        model_name="CNNSpeechEnhancer",
        device=device,
        save_results=save_results,
        output_dir=output_dir
    )
    
    return results

def benchmark_model_from_path(model_path, model_type, input_shape=None, num_classes=14, 
                             device='cpu', save_results=True, output_dir='output/benchmarks'):
    """
    Benchmark a model from a saved checkpoint
    
    Args:
        model_path: Path to the model checkpoint
        model_type: Type of model ('audiocnn' or 'speech_enhancer')
        input_shape: Input shape for the model (excluding batch dimension)
        num_classes: Number of output classes for classification models
        device: Device to run benchmarks on ('cpu', 'cuda', 'mps')
        save_results: Whether to save results to a JSON file
        output_dir: Directory to save results
    """
    print(f"Benchmarking model from {model_path} on {device}...")
    
    # Set default input shape based on model type if not provided
    if input_shape is None:
        if model_type.lower() == 'audiocnn':
            input_shape = (1, 40, 500)
        elif model_type.lower() == 'speech_enhancer':
            input_shape = (40, 1, 500)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model
    if model_type.lower() == 'audiocnn':
        # Extract model configuration from path if possible
        model_name = os.path.basename(os.path.dirname(model_path))
        if model_name in MODELS:
            cnn1_channels, cnn2_channels, fc_neurons = MODELS[model_name]
            model = AudioCNN(num_classes=num_classes, 
                            cnn1_channels=cnn1_channels, 
                            cnn2_channels=cnn2_channels, 
                            fc_neurons=fc_neurons)
        else:
            # Default configuration
            model = AudioCNN(num_classes=num_classes, 
                            cnn1_channels=64, 
                            cnn2_channels=128, 
                            fc_neurons=256)
    elif model_type.lower() == 'speech_enhancer':
        model = CNNSpeechEnhancer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Benchmark model
    results = benchmark_model_with_thop(
        model=model,
        input_shape=input_shape,
        model_name=os.path.basename(os.path.dirname(model_path)),
        device=device,
        save_results=save_results,
        output_dir=output_dir
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Benchmark models using thop')
    parser.add_argument('--model_type', type=str, choices=['audiocnn', 'speech_enhancer', 'all'],
                        default='all', help='Type of model to benchmark')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run benchmarks on (cpu, cuda, mps)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a specific model checkpoint to benchmark')
    parser.add_argument('--input_shape', type=str, default=None,
                        help='Input shape for the model (comma-separated integers, e.g., "1,40,500")')
    parser.add_argument('--num_classes', type=int, default=14,
                        help='Number of output classes for classification models')
    parser.add_argument('--output_dir', type=str, default='output/benchmarks',
                        help='Directory to save benchmark results')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save benchmark results')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'
    
    # Benchmark specific model from path
    if args.model_path:
        input_shape = None
        if args.input_shape:
            input_shape = tuple(map(int, args.input_shape.split(',')))
        
        benchmark_model_from_path(
            model_path=args.model_path,
            model_type=args.model_type,
            input_shape=input_shape,
            num_classes=args.num_classes,
            device=device,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        return
    
    # Benchmark models
    if args.model_type in ['audiocnn', 'all']:
        benchmark_audio_cnn_models(
            device=device,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
    
    if args.model_type in ['speech_enhancer', 'all']:
        benchmark_speech_enhancer(
            device=device,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )

if __name__ == '__main__':
    main() 