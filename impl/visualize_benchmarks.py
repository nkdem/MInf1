#!/usr/bin/env python3
"""
Script to visualize the benchmark results from the thop-based benchmarking.

This script reads the JSON files in the output/benchmarks directory and creates
visualizations of the model metrics.

References:
- Green AI (https://arxiv.org/abs/1907.10597)
- pytorch-OpCounter (https://github.com/Lyken17/pytorch-OpCounter)
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_model_metrics(benchmark_dir='output/benchmarks', output_dir='output/visualizations/model_metrics'):
    """
    Plot model metrics from the benchmark results.
    
    Args:
        benchmark_dir: Directory containing benchmark results
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the benchmark directory
    json_files = glob.glob(os.path.join(benchmark_dir, '*.json'))
    
    if not json_files:
        print(f"No benchmark results found in {benchmark_dir}")
        return
    
    # Collect metrics for all models
    model_metrics = {}
    
    for json_file in json_files:
        model_name = os.path.basename(json_file).split('_benchmark_')[0]
        
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
                
                # Handle both single model and multiple model formats
                if isinstance(data, dict) and 'flops' in data:
                    # Single model format
                    if model_name not in model_metrics:
                        model_metrics[model_name] = []
                    model_metrics[model_name].append(data)
                else:
                    # Multiple model format
                    for model, metrics in data.items():
                        if model not in model_metrics:
                            model_metrics[model] = []
                        model_metrics[model].append(metrics)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {json_file}")
    
    if not model_metrics:
        print("No valid benchmark metrics found")
        return
    
    # Plot metrics for each model
    for model_name, metrics_list in model_metrics.items():
        if not metrics_list or not all(isinstance(m, dict) for m in metrics_list):
            continue
        
        # Extract metrics
        flops = [m.get('flops', 0) for m in metrics_list if 'flops' in m]
        params = [m.get('params', 0) for m in metrics_list if 'params' in m]
        ops_per_second = [m.get('ops_per_second', 0) for m in metrics_list if 'ops_per_second' in m]
        avg_time_ms = [m.get('avg_time_ms', 0) for m in metrics_list if 'avg_time_ms' in m]
        
        if not flops or not params:
            continue
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot FLOPs
        axes[0, 0].bar(range(len(flops)), [f / 1e6 for f in flops])
        axes[0, 0].set_title(f'FLOPs - {model_name}')
        axes[0, 0].set_xlabel('Run')
        axes[0, 0].set_ylabel('FLOPs (millions)')
        
        # Plot Parameters
        axes[0, 1].bar(range(len(params)), [p / 1e3 for p in params])
        axes[0, 1].set_title(f'Parameters - {model_name}')
        axes[0, 1].set_xlabel('Run')
        axes[0, 1].set_ylabel('Parameters (thousands)')
        
        # Plot Operations per Second
        if ops_per_second:
            axes[1, 0].bar(range(len(ops_per_second)), [o / 1e9 for o in ops_per_second])
            axes[1, 0].set_title(f'Operations per Second - {model_name}')
            axes[1, 0].set_xlabel('Run')
            axes[1, 0].set_ylabel('GOPS')
        
        # Plot Inference Time
        if avg_time_ms:
            axes[1, 1].bar(range(len(avg_time_ms)), avg_time_ms)
            axes[1, 1].set_title(f'Inference Time - {model_name}')
            axes[1, 1].set_xlabel('Run')
            axes[1, 1].set_ylabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_metrics.png'))
        plt.close()
        
        # Create a table with average metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.axis('tight')
        
        # Calculate average metrics
        avg_metrics = {
            'FLOPs (millions)': np.mean([f / 1e6 for f in flops]),
            'Parameters (thousands)': np.mean([p / 1e3 for p in params]),
        }
        
        if ops_per_second:
            avg_metrics['Operations per Second (GOPS)'] = np.mean([o / 1e9 for o in ops_per_second])
        
        if avg_time_ms:
            avg_metrics['Inference Time (ms)'] = np.mean(avg_time_ms)
        
        if ops_per_second and params:
            avg_metrics['OPS per Parameter'] = np.mean([o / p if p > 0 else 0 for o, p in zip(ops_per_second, params)])
        
        # Create table
        table_data = [[k, f"{v:,.2f}"] for k, v in avg_metrics.items()]
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Average Value'], 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.title(f'Average Model Metrics - {model_name}', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_metrics_table.png'))
        plt.close()
    
    # Compare models if there are multiple
    if len(model_metrics) > 1:
        # Calculate average metrics for each model
        avg_metrics_by_model = {}
        for model_name, metrics_list in model_metrics.items():
            if not metrics_list or not all(isinstance(m, dict) for m in metrics_list):
                continue
            
            flops = [m.get('flops', 0) for m in metrics_list if 'flops' in m]
            params = [m.get('params', 0) for m in metrics_list if 'params' in m]
            ops_per_second = [m.get('ops_per_second', 0) for m in metrics_list if 'ops_per_second' in m]
            avg_time_ms = [m.get('avg_time_ms', 0) for m in metrics_list if 'avg_time_ms' in m]
            
            if not flops or not params:
                continue
            
            avg_metrics_by_model[model_name] = {
                'FLOPs (millions)': np.mean([f / 1e6 for f in flops]),
                'Parameters (thousands)': np.mean([p / 1e3 for p in params]),
            }
            
            if ops_per_second:
                avg_metrics_by_model[model_name]['Operations per Second (GOPS)'] = np.mean([o / 1e9 for o in ops_per_second])
            
            if avg_time_ms:
                avg_metrics_by_model[model_name]['Inference Time (ms)'] = np.mean(avg_time_ms)
            
            if ops_per_second and params:
                avg_metrics_by_model[model_name]['OPS per Parameter'] = np.mean([o / p if p > 0 else 0 for o, p in zip(ops_per_second, params)])
        
        if not avg_metrics_by_model:
            return
        
        # Create comparison plots
        metrics_to_plot = ['FLOPs (millions)', 'Parameters (thousands)', 
                          'Operations per Second (GOPS)', 'Inference Time (ms)', 
                          'OPS per Parameter']
        
        # Filter metrics that are available for all models
        available_metrics = []
        for metric in metrics_to_plot:
            if all(metric in model_metrics for model_metrics in avg_metrics_by_model.values()):
                available_metrics.append(metric)
        
        if not available_metrics:
            return
        
        # Create a figure with subplots for each metric
        fig, axes = plt.subplots(len(available_metrics), 1, figsize=(12, 5 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            model_names = list(avg_metrics_by_model.keys())
            values = [avg_metrics_by_model[model].get(metric, 0) for model in model_names]
            
            axes[i].bar(model_names, values)
            axes[i].set_title(f'Comparison of {metric}')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric)
            
            # Rotate x-axis labels if there are many models
            if len(model_names) > 3:
                axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_metrics_comparison.png'))
        plt.close()
        
        # Create comparison table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')
        ax.axis('tight')
        
        # Prepare table data
        table_data = []
        for model_name in avg_metrics_by_model:
            row = [model_name]
            for metric in available_metrics:
                row.append(f"{avg_metrics_by_model[model_name].get(metric, 0):,.2f}")
            table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, 
                         colLabels=['Model'] + available_metrics, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        plt.title('Model Metrics Comparison', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_metrics_comparison_table.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--benchmark_dir', type=str, default='output/benchmarks',
                        help='Directory containing benchmark results')
    parser.add_argument('--output_dir', type=str, default='output/visualizations/model_metrics',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    plot_model_metrics(benchmark_dir=args.benchmark_dir, output_dir=args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")
    print("\nReferences:")
    print("- Green AI (https://arxiv.org/abs/1907.10597)")
    print("- pytorch-OpCounter (https://github.com/Lyken17/pytorch-OpCounter)")

if __name__ == '__main__':
    main() 