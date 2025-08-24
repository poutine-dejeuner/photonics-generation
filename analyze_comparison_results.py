"""
Results analysis and comparison script for generative models.
Loads results from different model experiments and creates comparison plots.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any
import argparse


def load_experiment_results(results_dir: str) -> Dict[str, Dict]:
    """Load results from all model experiments."""
    results = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return results
    
    # Look for model subdirectories
    for model_dir in results_path.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            summary_file = model_dir / "results_summary.json"
            
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        results[model_name] = json.load(f)
                    print(f"Loaded results for {model_name}")
                except Exception as e:
                    print(f"Error loading results for {model_name}: {e}")
            else:
                print(f"No results summary found for {model_name}")
    
    return results


def plot_fom_comparison(results: Dict[str, Dict], save_path: str = None):
    """Create FOM comparison plots."""
    if not results:
        print("No results to plot")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Comparison: Figure of Merit Analysis', fontsize=16)
    
    # Extract data
    models = list(results.keys())
    fom_means = [results[model]['fom_mean'] for model in models]
    fom_stds = [results[model]['fom_std'] for model in models]
    fom_maxs = [results[model]['fom_max'] for model in models]
    fom_mins = [results[model]['fom_min'] for model in models]
    
    # 1. Bar plot of mean FOM
    axes[0, 0].bar(models, fom_means, yerr=fom_stds, capsize=5, alpha=0.7)
    axes[0, 0].set_title('Mean FOM Comparison')
    axes[0, 0].set_ylabel('FOM')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(fom_means, fom_stds)):
        axes[0, 0].text(i, mean + std + 0.01, f'{mean:.3f}', 
                       ha='center', va='bottom')
    
    # 2. Box plot style comparison
    fom_data = []
    model_labels = []
    for model in models:
        # Create synthetic data points around mean/std for visualization
        n_points = 100
        synthetic_fom = np.random.normal(results[model]['fom_mean'], 
                                       results[model]['fom_std'], n_points)
        fom_data.extend(synthetic_fom)
        model_labels.extend([model] * n_points)
    
    df = pd.DataFrame({'Model': model_labels, 'FOM': fom_data})
    sns.boxplot(data=df, x='Model', y='FOM', ax=axes[0, 1])
    axes[0, 1].set_title('FOM Distribution Comparison')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Range comparison (min to max)
    x_pos = np.arange(len(models))
    axes[1, 0].errorbar(x_pos, fom_means, 
                       yerr=[np.array(fom_means) - np.array(fom_mins),
                             np.array(fom_maxs) - np.array(fom_means)],
                       fmt='o', capsize=5, capthick=2)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].set_title('FOM Range (Min-Max)')
    axes[1, 0].set_ylabel('FOM')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary statistics table
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    table_data = []
    for model in models:
        table_data.append([
            model,
            f"{results[model]['fom_mean']:.4f}",
            f"{results[model]['fom_std']:.4f}",
            f"{results[model]['fom_max']:.4f}",
            f"{results[model]['fom_min']:.4f}",
            f"{results[model]['n_samples']}"
        ])
    
    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Model', 'Mean', 'Std', 'Max', 'Min', 'N Samples'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def create_performance_report(results: Dict[str, Dict], save_path: str = None):
    """Create a detailed performance report."""
    if not results:
        print("No results to analyze")
        return
    
    report = []
    report.append("=" * 80)
    report.append("GENERATIVE MODEL COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Sort models by mean FOM (descending)
    sorted_models = sorted(results.keys(), 
                          key=lambda x: results[x]['fom_mean'], 
                          reverse=True)

    report.append("RANKING BY MEAN FOM:")
    report.append("-" * 30)
    for i, model in enumerate(sorted_models, 1):
        fom_mean = results[model]['fom_mean']
        fom_std = results[model]['fom_std']
        report.append(f"{i}. {model}: {fom_mean:.4f} ± {fom_std:.4f}")
    
    report.append("")
    report.append("DETAILED RESULTS:")
    report.append("-" * 30)
    
    for model in sorted_models:
        res = results[model]
        report.append(f"\nModel: {model.upper()}")
        report.append(f"  Mean FOM: {res['fom_mean']:.4f}")
        report.append(f"  Std FOM:  {res['fom_std']:.4f}")
        report.append(f"  Max FOM:  {res['fom_max']:.4f}")
        report.append(f"  Min FOM:  {res['fom_min']:.4f}")
        report.append(f"  Samples:  {res['n_samples']}")
        
        # Calculate coefficient of variation
        cv = res['fom_std'] / res['fom_mean'] if res['fom_mean'] != 0 else float('inf')
        report.append(f"  CV:       {cv:.4f}")
    
    # Analysis
    report.append("")
    report.append("ANALYSIS:")
    report.append("-" * 30)
    
    best_model = sorted_models[0]
    best_fom = results[best_model]['fom_mean']
    
    report.append(f"Best performing model: {best_model} (FOM: {best_fom:.4f})")
    
    # Calculate improvements over other models
    for model in sorted_models[1:]:
        improvement = ((best_fom - results[model]['fom_mean']) / 
                      results[model]['fom_mean']) * 100
        report.append(f"  {improvement:.1f}% better than {model}")
    
    # Most consistent model (lowest CV)
    cv_dict = {model: results[model]['fom_std'] / results[model]['fom_mean'] 
               for model in results.keys() if results[model]['fom_mean'] != 0}
    most_consistent = min(cv_dict.keys(), key=lambda x: cv_dict[x])
    report.append(f"\nMost consistent model: {most_consistent} (CV: {cv_dict[most_consistent]:.4f})")
    
    report.append("")
    report.append("=" * 80)
    
    # Print and save report
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze and compare generative model results')
    parser.add_argument('--results_dir', type=str, 
                       default='~/scratch/nanophoto/comparison/experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./comparison_analysis',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Expand paths
    results_dir = os.path.expanduser(args.results_dir)
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("Loading experiment results...")
    results = load_experiment_results(results_dir)
    
    if not results:
        print("No results found. Make sure experiments have been run.")
        return
    
    print(f"Found results for {len(results)} models: {list(results.keys())}")
    
    # Create comparison plot
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plot_fom_comparison(results, plot_path)
    
    # Create performance report
    report_path = os.path.join(output_dir, "performance_report.txt")
    create_performance_report(results, report_path)
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
