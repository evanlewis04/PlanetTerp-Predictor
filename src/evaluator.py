"""
Model evaluation and cross-validation functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from config.config import CV_FOLDS, RANDOM_STATE, OUTPUT_DIR
from utils.helpers import create_output_directory
import os


class ModelEvaluator:
    """Handles model evaluation and cross-validation"""
    
    def __init__(self):
        """Initialize model evaluator"""
        create_output_directory(OUTPUT_DIR)
    
    def evaluate_models_with_cross_validation(self, X: pd.DataFrame, y: pd.Series, 
                                            cv: int = CV_FOLDS) -> pd.DataFrame:
        """
        Perform cross-validation to evaluate multiple regression models
        
        Args:
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds
            
        Returns:
            DataFrame with cross-validation results
        """
        # Standardize features for Ridge regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        # Define the three regression models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=RANDOM_STATE),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        }

        # Define evaluation metrics
        r2_scorer = make_scorer(r2_score)

        # Store results from cross-validation
        results = {
            'Model': [],
            'R² Mean': [],
            'R² Std': [],
            'MSE Mean': [],
            'MSE Std': []
        }

        # Define cross-validation strategy with shuffling
        kf = KFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

        # Evaluate each model with cross-validation
        for name, model in models.items():
            # For Ridge use scaled features; otherwise use original features
            X_to_use = X_scaled_df if name in ['Ridge Regression'] else X

            # Calculate R² scores across CV folds
            r2_scores = cross_val_score(model, X_to_use, y, cv=kf, scoring=r2_scorer)

            # Calculate MSE scores (negative of MSE for cross_val_score)
            mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
            mse_scores = -cross_val_score(model, X_to_use, y, cv=kf, scoring=mse_scorer)

            # Store results for this model
            results['Model'].append(name)
            results['R² Mean'].append(r2_scores.mean())
            results['R² Std'].append(r2_scores.std())
            results['MSE Mean'].append(mse_scores.mean())
            results['MSE Std'].append(mse_scores.std())

        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame(results)

        # Create visualization of cross-validation results
        self._plot_cv_results(results_df)

        return results_df
    
    def _plot_cv_results(self, results_df: pd.DataFrame) -> None:
        """
        Plot cross-validation results
        
        Args:
            results_df: DataFrame with CV results
        """
        plt.figure(figsize=(12, 6))

        # Plot R² results
        plt.subplot(1, 2, 1)
        plt.bar(results_df['Model'], results_df['R² Mean'])
        plt.errorbar(results_df['Model'], results_df['R² Mean'],
                     yerr=results_df['R² Std'], fmt='o', color='black')
        plt.title('R² by Model (Cross-Validation)')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot MSE results
        plt.subplot(1, 2, 2)
        plt.bar(results_df['Model'], results_df['MSE Mean'])
        plt.errorbar(results_df['Model'], results_df['MSE Mean'],
                     yerr=results_df['MSE Std'], fmt='o', color='black')
        plt.title('MSE by Model (Cross-Validation)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'cross_validation_results.png'))
        plt.close()
        
    def create_performance_comparison_plot(self, results_table: pd.DataFrame) -> None:
        """
        Create model performance comparison plots
        
        Args:
            results_table: DataFrame with model performance metrics
        """
        models = results_table['Model']
        mse_scores = results_table['MSE']
        r2_scores = results_table['R²']

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        bars1 = plt.bar(models, mse_scores)
        plt.title('MSE by Model (lower is better)')
        plt.ylim(0, max(mse_scores) * 1.2)
        # Add values on top of bars
        for bar in bars1:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')

        plt.subplot(1, 2, 2)
        bars2 = plt.bar(models, r2_scores)
        plt.title('R² by Model (higher is better)')
        plt.ylim(0, 1)
        # Add values on top of bars
        for bar in bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
        plt.close()
    
    def create_feature_importance_plots(self, importances_dict: dict) -> None:
        """
        Create feature importance plots for all models
        
        Args:
            importances_dict: Dictionary mapping model names to importance DataFrames
        """
        num_models = len(importances_dict)
        plt.figure(figsize=(15, 4 * num_models))

        for i, (model_name, importance_df) in enumerate(importances_dict.items(), 1):
            plt.subplot(num_models, 1, i)
            top_features = importance_df.head(10)
            plt.barh(top_features['Feature'], top_features['Importance'])
            plt.xlabel('Importance')
            plt.title(f'Top 10 Important Features ({model_name})')
            plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_all_models.png'))
        plt.close()
    
    def create_prediction_scatter_plot(self, y_test: pd.Series, y_pred: np.ndarray, 
                                     model_name: str) -> None:
        """
        Create scatter plot of predicted vs actual ratings
        
        Args:
            y_test: Actual test values
            y_pred: Predicted values
            model_name: Name of the model for the title
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual Rating')
        plt.ylabel('Predicted Rating')
        plt.title(f'Actual vs Predicted Professor Ratings ({model_name})')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_scatter.png'))
        plt.close()
    
    def create_residual_plot(self, y_pred: np.ndarray, y_test: pd.Series, 
                           model_name: str) -> None:
        """
        Create residual plot for model evaluation
        
        Args:
            y_pred: Predicted values
            y_test: Actual test values
            model_name: Name of the model
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, y_pred - y_test, alpha=0.5)
        plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), color='red', linestyle='--')
        plt.xlabel('Predicted Rating')
        plt.ylabel('Residuals')
        plt.title(f'{model_name} Residual Plot')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name.lower().replace(" ", "_")}_residuals.png'))
        plt.close()