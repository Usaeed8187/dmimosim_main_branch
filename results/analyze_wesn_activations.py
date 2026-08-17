#!/usr/bin/env python3
"""Analyze WESN activation patterns across different IBOs.

This script investigates whether WESN activations operate in the linear or
non-linear region of the tanh activation function at different IBO levels.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
import json

# Add the repo root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dmimo.channel.configured_wesn_pred import ConfiguredWESNPredictor


class ActivationAnalyzer:
    """Analyzes WESN activation patterns during prediction."""
    
    def __init__(self, model: ConfiguredWESNPredictor):
        """Initialize analyzer with a WESN model.
        
        Args:
            model: ConfiguredWESNPredictor instance to analyze
        """
        self.model = model
        self.activations = []
        self.pre_activations = []
        self.ibo_db = None
        
    def capture_activations(self, u_seq: np.ndarray) -> np.ndarray:
        """Run prediction while capturing pre-activation values.
        
        Args:
            u_seq: Input sequence [T, B, input_dim]
            
        Returns:
            States array from the ESN
        """
        T, B, _ = u_seq.shape
        state = np.zeros((B, self.model.state_dim), dtype=self.model.dtype)
        states = np.zeros((T, B, self.model.state_dim), dtype=self.model.dtype)
        pre_acts = []
        
        for t in range(T):
            # Compute pre-activation values
            projected_input = (
                u_seq[t]
                if getattr(self.model, "enable_residue_low_rank", False)
                else self.model.input_scale * u_seq[t]
            )
            if self.model.W_res.ndim == 1:
                recurrent = state * self.model.W_res[None, :]
            else:
                recurrent = state @ self.model.W_res.T
            
            pre_act = (projected_input @ self.model.W_in.T) + recurrent
            pre_acts.append(pre_act)
            
            # Apply activation
            state = self.model._activation(pre_act).astype(self.model.dtype, copy=False)
            states[t] = state
        
        self.pre_activations = np.array(pre_acts)  # [T, B, state_dim]
        return states
    
    def analyze_activation_distribution(self) -> Dict:
        """Analyze the distribution of pre-activation values.
        
        Returns:
            Dictionary with activation statistics
        """
        if len(self.pre_activations) == 0:
            raise ValueError("No activations captured. Call capture_activations first.")
        
        # Flatten to get all pre-activation values
        pre_acts_flat = self.pre_activations.flatten()
        pre_acts_real = np.real(pre_acts_flat)
        pre_acts_imag = np.imag(pre_acts_flat)
        
        # Define linear region as approximately [-1, 1] for tanh
        # (tanh is approximately linear in this range)
        linear_threshold = 1.0
        
        real_linear = np.abs(pre_acts_real) <= linear_threshold
        imag_linear = np.abs(pre_acts_imag) <= linear_threshold
        both_linear = real_linear & imag_linear
        
        stats = {
            'mean_real': np.mean(pre_acts_real),
            'mean_imag': np.mean(pre_acts_imag),
            'std_real': np.std(pre_acts_real),
            'std_imag': np.std(pre_acts_imag),
            'min_real': np.min(pre_acts_real),
            'max_real': np.max(pre_acts_real),
            'min_imag': np.min(pre_acts_imag),
            'max_imag': np.max(pre_acts_imag),
            'pct_real_linear': 100.0 * np.mean(real_linear),
            'pct_imag_linear': 100.0 * np.mean(imag_linear),
            'pct_both_linear': 100.0 * np.mean(both_linear),
            'pct_real_saturated': 100.0 * np.mean(np.abs(pre_acts_real) > 3.0),
            'pct_imag_saturated': 100.0 * np.mean(np.abs(pre_acts_imag) > 3.0),
            'num_samples': len(pre_acts_real),
        }
        
        return stats


def find_prediction_results(ibo_db: float, base_dir: str = "results/channels_multiple_mu_mimo") -> List[str]:
    """Find prediction result files for a given IBO.
    
    Args:
        ibo_db: IBO value in dB
        base_dir: Base results directory
        
    Returns:
        List of matching file paths
    """
    # Convert IBO to string format used in filenames
    ibo_str = str(ibo_db).replace(".", "p")
    pattern = f"*ibo_db_{ibo_str}*.npz"
    
    matching_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if "ibo_db" in file:
                file_path = os.path.join(root, file)
                if f"ibo_db_{ibo_str}" in file:
                    matching_files.append(file_path)
    
    return matching_files


def load_wesn_from_npz(npz_file: str) -> Tuple[ConfiguredWESNPredictor, Dict]:
    """Load WESN predictor from NPZ file if it contains predictions.
    
    Args:
        npz_file: Path to NPZ file
        
    Returns:
        Tuple of (predictor, metadata) or (None, None) if file doesn't have predictor
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Check if this is a WESN prediction result
        if 'predictor' not in data:
            return None, None
        
        predictor = data['predictor'].item()
        metadata = {}
        
        # Extract metadata
        for key in data.files:
            if key != 'predictor':
                val = data[key]
                if isinstance(val, np.ndarray):
                    if val.size == 1:
                        metadata[key] = float(val)
                    else:
                        metadata[key] = val
                else:
                    metadata[key] = val
        
        return predictor, metadata
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None, None


def analyze_ibo_sweep(ibo_values: List[float] = None, base_dir: str = "results/channels_multiple_mu_mimo"):
    """Analyze activations across different IBO values.
    
    Args:
        ibo_values: List of IBO values to analyze (in dB)
        base_dir: Base results directory
    """
    if ibo_values is None:
        ibo_values = [0.0, 3.0, 5.0, 6.5, 7.0, 9.0]
    
    results_by_ibo = {}
    
    for ibo_db in ibo_values:
        print(f"\n{'='*60}")
        print(f"Analyzing IBO = {ibo_db} dB")
        print(f"{'='*60}")
        
        # Find result files for this IBO
        result_files = find_prediction_results(ibo_db, base_dir)
        
        if not result_files:
            print(f"No results found for IBO = {ibo_db} dB")
            continue
        
        # Filter for WESN results (balanced_lite variant)
        wesn_files = [f for f in result_files if 'wesn_balanced_lite' in f or 'configured_wesn_balanced_lite' in f]
        
        if not wesn_files:
            print(f"No WESN predictions found for IBO = {ibo_db} dB")
            continue
        
        print(f"Found {len(wesn_files)} WESN result files")
        
        # Analyze each file
        ibo_stats = []
        for result_file in wesn_files[:3]:  # Limit to first 3 files per IBO
            print(f"\nAnalyzing: {Path(result_file).name[:80]}...")
            predictor, metadata = load_wesn_from_npz(result_file)
            
            if predictor is None:
                print("  Skipped (no predictor found)")
                continue
            
            # Try to extract channel observations
            try:
                data = np.load(result_file, allow_pickle=True)
                
                # Look for channel or signal observations
                if 'h_true' in data:
                    obs = data['h_true']
                elif 'h_pred' in data:
                    obs = data['h_pred']
                else:
                    # Use a default input based on the predictor
                    obs = None
                
                if obs is not None:
                    # Create synthetic input for analysis
                    # Reshape to [T, B, input_dim]
                    if obs.ndim == 3:
                        obs_seq = obs[:, np.newaxis, :]  # [T, 1, D]
                    else:
                        obs_seq = obs[:, np.newaxis, :]
                    
                    analyzer = ActivationAnalyzer(predictor)
                    analyzer.ibo_db = ibo_db
                    states = analyzer.capture_activations(obs_seq)
                    stats = analyzer.analyze_activation_distribution()
                    ibo_stats.append(stats)
                    
                    print(f"  Mean pre-activation (real): {stats['mean_real']:.4f}")
                    print(f"  Mean pre-activation (imag): {stats['mean_imag']:.4f}")
                    print(f"  Pct in linear region: {stats['pct_both_linear']:.1f}%")
                    print(f"  Pct saturated: {stats['pct_real_saturated']:.1f}% (real), {stats['pct_imag_saturated']:.1f}% (imag)")
            except Exception as e:
                print(f"  Error processing file: {e}")
                import traceback
                traceback.print_exc()
        
        if ibo_stats:
            # Average stats across files
            avg_stats = {}
            for key in ibo_stats[0].keys():
                if isinstance(ibo_stats[0][key], (int, float)):
                    avg_stats[key] = np.mean([s[key] for s in ibo_stats])
            
            results_by_ibo[ibo_db] = avg_stats
            print(f"\nAveraged statistics for IBO = {ibo_db} dB:")
            print(f"  Pct in linear region: {avg_stats['pct_both_linear']:.1f}%")
            print(f"  Mean magnitude (real): {abs(avg_stats['mean_real']):.4f}")
            print(f"  Mean magnitude (imag): {abs(avg_stats['mean_imag']):.4f}")
    
    return results_by_ibo


def plot_activation_analysis(results_by_ibo: Dict[float, Dict]):
    """Create visualization of activation patterns across IBOs.
    
    Args:
        results_by_ibo: Dictionary with results for each IBO
    """
    if not results_by_ibo:
        print("No results to plot")
        return
    
    ibos = sorted(results_by_ibo.keys())
    pct_linear = [results_by_ibo[ibo]['pct_both_linear'] for ibo in ibos]
    mean_real = [abs(results_by_ibo[ibo]['mean_real']) for ibo in ibos]
    mean_imag = [abs(results_by_ibo[ibo]['mean_imag']) for ibo in ibos]
    pct_saturated = [results_by_ibo[ibo]['pct_real_saturated'] for ibo in ibos]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Percentage in linear region
    axes[0, 0].plot(ibos, pct_linear, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    axes[0, 0].axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Fully linear')
    axes[0, 0].set_xlabel('IBO (dB)', fontsize=11)
    axes[0, 0].set_ylabel('% of Activations', fontsize=11)
    axes[0, 0].set_title('Percentage of Activations in Linear Region (|x| ≤ 1)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Mean magnitude of pre-activations
    axes[0, 1].plot(ibos, mean_real, 'o-', label='Real part', linewidth=2, markersize=8)
    axes[0, 1].plot(ibos, mean_imag, 's-', label='Imag part', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Linear threshold')
    axes[0, 1].set_xlabel('IBO (dB)', fontsize=11)
    axes[0, 1].set_ylabel('Mean |Pre-activation|', fontsize=11)
    axes[0, 1].set_title('Mean Magnitude of Pre-Activations', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: Percentage saturated
    axes[1, 0].plot(ibos, pct_saturated, 'o-', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=0, color='g', linestyle='--', alpha=0.5, label='No saturation')
    axes[1, 0].set_xlabel('IBO (dB)', fontsize=11)
    axes[1, 0].set_ylabel('% Saturated (|x| > 3)', fontsize=11)
    axes[1, 0].set_title('Percentage of Saturated Activations', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Tanh linearity comparison
    x_range = np.linspace(-5, 5, 1000)
    y_tanh = np.tanh(x_range)
    y_linear = x_range
    
    axes[1, 1].plot(x_range, y_tanh, 'b-', linewidth=2, label='tanh(x)')
    axes[1, 1].plot(x_range, y_linear, 'r--', linewidth=1.5, alpha=0.7, label='y = x (linear)')
    axes[1, 1].fill_between([-1, 1], -1.2, 1.2, alpha=0.2, color='green', label='Linear region')
    axes[1, 1].set_xlabel('x', fontsize=11)
    axes[1, 1].set_ylabel('tanh(x)', fontsize=11)
    axes[1, 1].set_title('tanh Activation Function', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_xlim(-5, 5)
    axes[1, 1].set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig('results/wesn_activation_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: results/wesn_activation_analysis.png")
    plt.show()


def create_summary_report(results_by_ibo: Dict[float, Dict]):
    """Create a text summary report of the analysis.
    
    Args:
        results_by_ibo: Dictionary with results for each IBO
    """
    report = []
    report.append("=" * 70)
    report.append("WESN ACTIVATION LINEARITY ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")
    
    if not results_by_ibo:
        report.append("No results available for analysis.")
    else:
        report.append("Summary of Activation Patterns Across IBOs:")
        report.append("")
        report.append(f"{'IBO (dB)':<12} {'% Linear':<12} {'% Saturated':<14} {'Mean |Re|':<12} {'Mean |Im|':<12}")
        report.append("-" * 70)
        
        for ibo in sorted(results_by_ibo.keys()):
            stats = results_by_ibo[ibo]
            report.append(f"{ibo:<12.1f} {stats['pct_both_linear']:<12.1f} "
                         f"{stats['pct_real_saturated']:<14.1f} "
                         f"{abs(stats['mean_real']):<12.4f} "
                         f"{abs(stats['mean_imag']):<12.4f}")
        
        report.append("")
        report.append("INTERPRETATION:")
        report.append("-" * 70)
        
        # Find min and max linearity
        pct_linear_vals = [results_by_ibo[ibo]['pct_both_linear'] for ibo in results_by_ibo]
        min_linear = min(pct_linear_vals)
        max_linear = max(pct_linear_vals)
        
        if min_linear > 90:
            report.append("• HEAVILY LINEAR: >90% of activations operate in the linear region")
            report.append("  This explains why WESN performance is similar to linear KF methods.")
            report.append("  The non-linear capabilities of WESN are NOT being utilized.")
        elif min_linear > 70:
            report.append("• MOSTLY LINEAR: 70-90% of activations in linear region")
            report.append("  WESN is operating mostly in the linear regime but with some")
            report.append("  non-linearity at higher IBOs or more challenging conditions.")
        else:
            report.append("• MIXED OPERATION: Significant non-linear activation")
            report.append("  WESN is utilizing non-linear capabilities effectively.")
        
        report.append("")
        report.append("RECOMMENDATION:")
        report.append("-" * 70)
        
        if min_linear > 90:
            report.append("To improve WESN performance relative to KF:")
            report.append("1. Increase input scaling to drive activations away from linear region")
            report.append("2. Use ReLU or other activations that have wider non-linear regions")
            report.append("3. Analyze if the channel characteristics are too smooth/linear")
            report.append("4. Check if PA non-linearity is being properly modeled in the inputs")
        else:
            report.append("WESN appears to be using non-linear capabilities appropriately.")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    with open('results/wesn_activation_analysis_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to: results/wesn_activation_analysis_report.txt")
    
    return report_text


if __name__ == "__main__":
    print("WESN Activation Analysis")
    print("=" * 60)
    
    # Analyze activations across IBO sweep
    results = analyze_ibo_sweep([0.0, 3.0, 5.0, 6.5, 7.0, 9.0])
    
    # Create visualization
    if results:
        plot_activation_analysis(results)
        
        # Create report
        create_summary_report(results)
    else:
        print("No results found. Please ensure the IBO sweep has been run and results are available.")
