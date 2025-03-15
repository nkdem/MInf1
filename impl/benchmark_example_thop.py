#!/usr/bin/env python3
"""
Example script demonstrating how to use the model benchmarking functionality with
thop (pytorch-OpCounter) for measuring model efficiency metrics.

This script benchmarks all models defined in constants.MODELS and the CNNSpeechEnhancer model.

References:
- Green AI (https://arxiv.org/abs/1907.10597)
- pytorch-OpCounter (https://github.com/Lyken17/pytorch-OpCounter)
"""

import os
import argparse
import torch
from model_benchmark_thop import (
    benchmark_audio_cnn_models,
    benchmark_speech_enhancer,
    benchmark_model_from_path
)

def main():
    parser = argparse.ArgumentParser(description='Example script for model benchmarking with thop')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to run benchmarks on (cpu, cuda, mps)')
    parser.add_argument('--output_dir', type=str, default='output/benchmarks',
                        help='Directory to save benchmark results')
    parser.add_argument('--specific_model', type=str, default=None,
                        help='Path to a specific model checkpoint to benchmark')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['audiocnn', 'speech_enhancer'],
                        help='Type of the specific model to benchmark')
    parser.add_argument('--num_classes', type=int, default=14,
                        help='Number of output classes for classification models')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = 'cpu'
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Benchmark a specific model if provided
    if args.specific_model:
        if not args.model_type:
            print("Error: --model_type must be specified when using --specific_model")
            return
        
        if args.model_type == 'audiocnn':
            input_shape = (1, 40, 500)
        else:  # speech_enhancer
            input_shape = (40, 1, 500)
        
        print(f"Benchmarking specific model: {args.specific_model}")
        benchmark_model_from_path(
            model_path=args.specific_model,
            model_type=args.model_type,
            input_shape=input_shape,
            num_classes=args.num_classes,
            device=device,
            output_dir=args.output_dir
        )
        return
    
    # Benchmark all AudioCNN models
    print("Benchmarking AudioCNN models...")
    benchmark_audio_cnn_models(device=device, output_dir=args.output_dir)
    
    # Benchmark CNNSpeechEnhancer
    print("\nBenchmarking CNNSpeechEnhancer...")
    benchmark_speech_enhancer(device=device, output_dir=args.output_dir)
    
    print(f"\nAll benchmarks completed. Results saved to {args.output_dir}")
    print("You can include these metrics in your report to compare model efficiency.")
    print("\nReferences:")
    print("- Green AI (https://arxiv.org/abs/1907.10597)")
    print("- pytorch-OpCounter (https://github.com/Lyken17/pytorch-OpCounter)")

if __name__ == '__main__':
    main() 