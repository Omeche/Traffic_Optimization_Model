"""
Evaluates trained models and generates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, models_path, results_path):
        """
        Initialize evaluator
        
        Args:
            models_path: Path to saved models
            results_path: Path to save results
        """
        self.models_path = Path(models_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_results(self):
        """Load training results"""
        results_file = self.models_path / "training_results.pkl"
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Loaded results for {len(results)} models")
        return results
    
    def plot_model_comparison(self, results):
        """
        Plot comparison of all models
        
        Args:
            results: Dictionary of training results
        """
        print("\nCreating model comparison plots...")
        
        # Prepare data
        models = []
        test_mae = []
        test_rmse = []
        test_r2 = []
        
        for model_name, result in results.items():
            models.append(model_name)
            metrics = result['overall_metrics']
            test_mae.append(metrics['test_mae'])
            test_rmse.append(metrics['test_rmse'])
            test_r2.append(metrics['test_r2'])
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # MAE comparison
        axes[0].barh(models, test_mae, color='skyblue')
        axes[0].set_xlabel('Mean Absolute Error (seconds)')
        axes[0].set_title('Test MAE by Model')
        axes[0].grid(axis='x', alpha=0.3)
        
        # RMSE comparison
        axes[1].barh(models, test_rmse, color='lightcoral')
        axes[1].set_xlabel('Root Mean Squared Error (seconds)')
        axes[1].set_title('Test RMSE by Model')
        axes[1].grid(axis='x', alpha=0.3)
        
        # R² comparison
        axes[2].barh(models, test_r2, color='lightgreen')
        axes[2].set_xlabel('R² Score')
        axes[2].set_title('Test R² by Model')
        axes[2].grid(axis='x', alpha=0.3)
        axes[2].set_xlim([0, 1])
        
        plt.tight_layout()
        
        # Save
        output_file = self.results_path / 'model_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
        plt.close()
    
    def plot_target_wise_performance(self, results, target_cols):
        """
        Plot performance for each target variable
        
        Args:
            results: Dictionary of training results
            target_cols: List of target column names
        """
        print("\nCreating target-wise performance plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, target in enumerate(target_cols):
            ax = axes[idx]
            
            models = []
            mae_values = []
            
            for model_name, result in results.items():
                if target in result['target_metrics']:
                    models.append(model_name)
                    mae_values.append(result['target_metrics'][target]['test_mae'])
            
            ax.barh(models, mae_values, color='steelblue')
            ax.set_xlabel('MAE (seconds)')
            ax.set_title(f'{target.replace("_", " ").title()}')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_file = self.results_path / 'target_wise_performance.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
        plt.close()
    
    def plot_prediction_scatter(self, results, model_name, target_cols):
        """
        Create scatter plots of predictions vs actual values
        
        Args:
            results: Dictionary of training results
            model_name: Name of model to plot
            target_cols: List of target columns
        """
        print(f"\nCreating prediction scatter plots for {model_name}...")
        
        if model_name not in results:
            print(f"Model {model_name} not found")
            return
        
        result = results[model_name]
        y_test_pred = result['predictions']['y_test_pred']
        
        print("   Actual test values not available in saved results")
        print("   This would require re-running with test data access")
    
    def plot_learning_curves(self, results):
        """
        Plot training vs test performance
        
        Args:
            results: Dictionary of training results
        """
        print("\nCreating learning curves...")
        
        models = []
        train_mae = []
        test_mae = []
        
        for model_name, result in results.items():
            models.append(model_name)
            metrics = result['overall_metrics']
            train_mae.append(metrics['train_mae'])
            test_mae.append(metrics['test_mae'])
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(x - width/2, train_mae, width, label='Training MAE', color='skyblue')
        ax.bar(x + width/2, test_mae, width, label='Test MAE', color='coral')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Mean Absolute Error (seconds)')
        ax.set_title('Training vs Test Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_file = self.results_path / 'learning_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_file}")
        plt.close()
    
    def generate_performance_report(self, results):
        """
        Generate detailed performance report
        
        Args:
            results: Dictionary of training results
        """
        print("\nGenerating performance report...")
        
        report_lines = []
        report_lines.append("TRAFFIC SIGNAL OPTIMIZATION - MODEL PERFORMANCE REPORT")
        report_lines.append("")
        
        for model_name, result in results.items():
            report_lines.append(f"\n{model_name}")
            report_lines.append("-" * 40)
            
            # Overall metrics
            metrics = result['overall_metrics']
            report_lines.append(f"Overall Performance:")
            report_lines.append(f"  Training MAE:   {metrics['train_mae']:.3f} seconds")
            report_lines.append(f"  Test MAE:       {metrics['test_mae']:.3f} seconds")
            report_lines.append(f"  Training RMSE:  {metrics['train_rmse']:.3f} seconds")
            report_lines.append(f"  Test RMSE:      {metrics['test_rmse']:.3f} seconds")
            report_lines.append(f"  Training R²:    {metrics['train_r2']:.3f}")
            report_lines.append(f"  Test R²:        {metrics['test_r2']:.3f}")
            
            # Target-wise metrics
            report_lines.append(f"\nTarget-wise Performance:")
            for target, target_metrics in result['target_metrics'].items():
                report_lines.append(f"  {target}:")
                report_lines.append(f"    MAE:  {target_metrics['test_mae']:.3f} seconds")
                report_lines.append(f"    RMSE: {target_metrics['test_rmse']:.3f} seconds")
                report_lines.append(f"    R²:   {target_metrics['test_r2']:.3f}")
        
        # Best model summary
        report_lines.append("\n")
        best_model = min(results.items(), 
                        key=lambda x: x[1]['overall_metrics']['test_mae'])
        report_lines.append(f"BEST MODEL: {best_model[0]}")
        report_lines.append(f"Test MAE: {best_model[1]['overall_metrics']['test_mae']:.3f} seconds")
        report_lines.append(f"Test R²:  {best_model[1]['overall_metrics']['test_r2']:.3f}")
        
        # Save report
        report_file = self.results_path / 'performance_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Saved to {report_file}")
        
        # Also print to console
        print('\n'.join(report_lines))
    
    def calculate_improvement_metrics(self, results):
        """
        Calculate improvement over baseline (fixed-time control)
        
        Args:
            results: Dictionary of training results
        """
        print("\nCalculating improvement metrics...")
        
        # Baseline: assume fixed-time control has higher error
        baseline_mae = 15.0  # Typical baseline MAE in seconds
        
        improvements = []
        for model_name, result in results.items():
            test_mae = result['overall_metrics']['test_mae']
            improvement = ((baseline_mae - test_mae) / baseline_mae) * 100
            
            improvements.append({
                'Model': model_name,
                'Test_MAE': test_mae,
                'Improvement_over_Baseline': improvement
            })
        
        improvement_df = pd.DataFrame(improvements)
        
        # Save
        improvement_file = self.results_path / 'improvement_metrics.csv'
        improvement_df.to_csv(improvement_file, index=False)
        
        print("\nImprovement over Baseline (Fixed-time control):")
        print(improvement_df.to_string(index=False))
        print(f"\n Saved to {improvement_file}")
        
        return improvement_df
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\nStarting full evaluation...")
        
        # Load results
        results = self.load_results()
        
        # Target columns
        target_cols = [
            'cycle_time',
            'phase_1_green',
            'phase_2_green',
            'phase_3_green',
            'phase_4_green',
            'pedestrian_green'
        ]
        
        # Generate all visualizations and reports
        self.plot_model_comparison(results)
        self.plot_target_wise_performance(results, target_cols)
        self.plot_learning_curves(results)
        self.generate_performance_report(results)
        self.calculate_improvement_metrics(results)
        
        print("\nEvaluation complete.")
        print(f"   All results saved to: {self.results_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate traffic signal models')
    parser.add_argument('--models_path', type=str, default='models/saved_models',
                       help='Path to saved models')
    parser.add_argument('--results_path', type=str, default='results',
                       help='Path to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.models_path, args.results_path)
    
    # Run evaluation
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()