"""
Trains multiple ML models for traffic signal optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class TrafficSignalModelTrainer:
    """Train ML models for traffic signal optimization"""
    
    def __init__(self, random_state=42):
        """
        Initialize trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Define target columns (signal timings to predict)
        self.target_cols = [
            'cycle_time',
            'phase_1_green',
            'phase_2_green',
            'phase_3_green',
            'phase_4_green',
            'pedestrian_green'
        ]
        
        # Define feature columns to exclude
        self.exclude_cols = [
            'time_bin', 'timestamp', 'recording_id',
            'cycle_time', 'phase_1_green', 'phase_2_green',
            'phase_3_green', 'phase_4_green', 'pedestrian_green',
            'yellow_time', 'all_red_time'
        ]
    
    def prepare_data(self, df, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            df: Labeled DataFrame
            test_size: Proportion of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        
        # Remove rows with missing values in targets
        df_clean = df.dropna(subset=self.target_cols)
        
        # Select feature columns
        feature_cols = [col for col in df_clean.columns 
                       if col not in self.exclude_cols]
        
        X = df_clean[feature_cols].values
        y = df_clean[self.target_cols].values
        
        # Handle any remaining NaN values in features
        X = np.nan_to_num(X, nan=0.0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Targets: {len(self.target_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols
    
    def initialize_models(self):
        """Initialize all ML models"""
        
        self.models = {
            'Decision Tree': MultiOutputRegressor(
                DecisionTreeRegressor(
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=self.random_state
                )
            ),
            
            'Random Forest': MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            ),
            
            'Gradient Boosting': MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state
                )
            ),
            
            'SVR': MultiOutputRegressor(
                SVR(
                    kernel='rbf',
                    C=10,
                    epsilon=0.1
                )
            ),
            
            'Neural Network': MultiOutputRegressor(
                MLPRegressor(
                    hidden_layer_sizes=(100, 50, 25),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    random_state=self.random_state,
                    early_stopping=True
                )
            )
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """
        Train a single model and evaluate
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train, y_train: Training data
            X_test, y_test: Testing data
            
        Returns:
            Dictionary with results
        """
        print(f"\nTraining {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics for each target
        target_metrics = {}
        for i, target in enumerate(self.target_cols):
            target_metrics[target] = {
                'train_mae': mean_absolute_error(y_train[:, i], y_train_pred[:, i]),
                'test_mae': mean_absolute_error(y_test[:, i], y_test_pred[:, i]),
                'train_rmse': np.sqrt(mean_squared_error(y_train[:, i], y_train_pred[:, i])),
                'test_rmse': np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i])),
                'train_r2': r2_score(y_train[:, i], y_train_pred[:, i]),
                'test_r2': r2_score(y_test[:, i], y_test_pred[:, i])
            }
        
        # Overall metrics
        overall_metrics = {
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        print(f"{model_name} - Test MAE: {overall_metrics['test_mae']:.2f}, Test R²: {overall_metrics['test_r2']:.3f}")
        
        return {
            'model': model,
            'predictions': {
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            },
            'target_metrics': target_metrics,
            'overall_metrics': overall_metrics
        }
    
    def train_all_models(self, X_train, y_train, X_test, y_test, models_to_train=None):
        """
        Train all models
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            models_to_train: Optional list of model names to train
        """
        if models_to_train is None:
            models_to_train = list(self.models.keys())
        
        print(f"\nTraining {len(models_to_train)} models...")
        
        for model_name in models_to_train:
            if model_name in self.models:
                self.results[model_name] = self.train_model(
                    model_name,
                    self.models[model_name],
                    X_train, y_train,
                    X_test, y_test
                )
            else:
                print(f"Model {model_name} not found")
    
    def compare_models(self):
        """Compare all trained models"""
        print("\nModel Comparison:")
        print(f"{'Model':<20} {'Train MAE':<12} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10}")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            metrics = result['overall_metrics']
            print(f"{model_name:<20} {metrics['train_mae']:<12.2f} {metrics['test_mae']:<12.2f} "
                  f"{metrics['test_rmse']:<12.2f} {metrics['test_r2']:<10.3f}")
            
            comparison_data.append({
                'Model': model_name,
                'Train_MAE': metrics['train_mae'],
                'Test_MAE': metrics['test_mae'],
                'Train_RMSE': metrics['train_rmse'],
                'Test_RMSE': metrics['test_rmse'],
                'Train_R2': metrics['train_r2'],
                'Test_R2': metrics['test_r2']
            })
        
        
        # Find best model
        best_model = min(comparison_data, key=lambda x: x['Test_MAE'])
        print(f"\nBest Model: {best_model['Model']} (Test MAE: {best_model['Test_MAE']:.2f})")
        
        return pd.DataFrame(comparison_data)
    
    def get_feature_importance(self, model_name, feature_names):
        """
        Get feature importance for tree-based models
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importances
        """
        if model_name not in self.results:
            return None
        
        model = self.results[model_name]['model']
        
        # Extract feature importances
        if hasattr(model, 'estimators_'):
            # For MultiOutputRegressor
            importances_list = []
            for estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    importances_list.append(estimator.feature_importances_)
            
            if importances_list:
                # Average across all output estimators
                avg_importances = np.mean(importances_list, axis=0)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': avg_importances
                }).sort_values('importance', ascending=False)
                
                return importance_df
        
        return None
    
    def save_models(self, output_path):
        """
        Save trained models and results
        
        Args:
            output_path: Path to save models
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving models to {output_path}")
        
        # Save each model
        for model_name, result in self.results.items():
            model_file = output_path / f"{model_name.replace(' ', '_').lower()}_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': result['model'],
                    'scaler': self.scaler,
                    'target_cols': self.target_cols
                }, f)
            print(f"Saved {model_name}")
        
        # Save comparison results
        comparison_df = self.compare_models()
        comparison_file = output_path / "model_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        # Save detailed results
        results_file = output_path / "training_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Saved comparison and results")


def main():
    parser = argparse.ArgumentParser(description='Train traffic signal optimization models')
    parser.add_argument('--data_path', type=str, default='data/labels/labeled_data_adaptive.csv',
                       help='Path to labeled data')
    parser.add_argument('--output_path', type=str, default='models/saved_models',
                       help='Path to save trained models')
    parser.add_argument('--models', type=str, default='all',
                       help='Models to train (comma-separated or "all")')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set proportion')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading labeled data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples")
    
    # Initialize trainer
    trainer = TrafficSignalModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(df, args.test_size)
    
    # Initialize models
    trainer.initialize_models()
    
    # Determine which models to train
    if args.models == 'all':
        models_to_train = list(trainer.models.keys())
    else:
        models_to_train = [m.strip() for m in args.models.split(',')]
    
    # Train models
    trainer.train_all_models(X_train, y_train, X_test, y_test, models_to_train)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Show feature importance for Random Forest
    print("\nTop 10 Important Features (Random Forest):")
    importance_df = trainer.get_feature_importance('Random Forest', feature_names)
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))
    
    # Save models
    trainer.save_models(args.output_path)
    
    print("\nTraining complete.")


if __name__ == "__main__":
    main()