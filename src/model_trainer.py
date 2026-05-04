"""
Model training and evaluation functionality
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from config.config import RANDOM_STATE, TEST_SIZE, RIDGE_ALPHAS, OUTPUT_DIR
from src.evaluator import ModelEvaluator
import os


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self):
        """Initialize model trainer"""
        self.evaluator = ModelEvaluator()
    
    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Train and evaluate all three models on the dataset
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (best_model, feature_importance_df)
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Scale features for Ridge regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train all models and collect results
        results = []
        feature_importances = {}
        
        # 1. Linear Regression Model
        lr_model, lr_results, lr_importance = self._train_linear_regression(
            X_train, X_test, y_train, y_test
        )
        results.append(lr_results)
        feature_importances['Linear Regression'] = lr_importance
        
        # 2. Random Forest Model
        rf_model, rf_results, rf_importance = self._train_random_forest(
            X_train, X_test, y_train, y_test
        )
        results.append(rf_results)
        feature_importances['Random Forest'] = rf_importance
        
        # 3. Ridge Regression Model
        ridge_model, ridge_results, ridge_importance = self._train_ridge_regression(
            X_train_scaled, X_test_scaled, y_train, y_test, X.columns
        )
        results.append(ridge_results)
        feature_importances['Ridge Regression'] = ridge_importance

        # Create results table and visualizations
        results_table = pd.DataFrame(results)
        self.evaluator.create_performance_comparison_plot(results_table)
        self.evaluator.create_feature_importance_plots(feature_importances)

        # Find and return best model
        best_model, best_importance = self._find_best_model(
            results_table, 
            {'Linear Regression': lr_model, 'Random Forest': rf_model, 'Ridge Regression': ridge_model},
            feature_importances,
            y_test
        )

        print("\nModel Performance Comparison:")
        print(results_table.sort_values('R2', ascending=False).to_string(index=False))

        return best_model, best_importance
    
    def _train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model"""
        print("\nTraining Linear Regression model")
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Linear Regression - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(model.coef_)
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 important features for Linear Regression:")
        print(importance.head(5).to_string(index=False))
        
        # Create residual plot
        self.evaluator.create_residual_plot(y_pred, y_test, 'Linear Regression')
        
        return model, {'Model': 'Linear Regression', 'MSE': mse, 'R2': r2}, importance
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        print("\nTraining Random Forest model")
        
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Random Forest - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 important features for Random Forest:")
        print(importance.head(5).to_string(index=False))
        
        # Create residual plot
        self.evaluator.create_residual_plot(y_pred, y_test, 'Random Forest')
        
        return model, {'Model': 'Random Forest', 'MSE': mse, 'R2': r2}, importance
    
    def _train_ridge_regression(self, X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
        """Train Ridge Regression model with alpha tuning"""
        print("\nTraining Ridge Regression model")
        
        # Test different alpha values
        ridge_r2_scores = []
        for alpha in RIDGE_ALPHAS:
            ridge = Ridge(alpha=alpha, random_state=RANDOM_STATE)
            ridge.fit(X_train_scaled, y_train)
            pred = ridge.predict(X_test_scaled)
            ridge_r2_scores.append(r2_score(y_test, pred))

        # Find best alpha
        best_alpha_idx = ridge_r2_scores.index(max(ridge_r2_scores))
        best_alpha = RIDGE_ALPHAS[best_alpha_idx]
        print(f"Best alpha for Ridge Regression: {best_alpha}")

        # Train final model
        model = Ridge(alpha=best_alpha, random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Ridge Regression - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(model.coef_)
        }).sort_values('Importance', ascending=False)
        
        print("Top 5 important features for Ridge Regression:")
        print(importance.head(5).to_string(index=False))
        
        # Create residual plot
        self.evaluator.create_residual_plot(y_pred, y_test, 'Ridge Regression')
        
        # Plot alpha tuning results
        self._plot_ridge_alpha_tuning(RIDGE_ALPHAS, ridge_r2_scores, best_alpha)
        
        return model, {'Model': 'Ridge Regression', 'MSE': mse, 'R2': r2}, importance
    
    def _plot_ridge_alpha_tuning(self, alphas, r2_scores, best_alpha):
        """Plot Ridge regression alpha tuning results"""
        plt.figure(figsize=(10, 6))
        plt.plot(alphas, r2_scores, 'o-')
        plt.axvline(x=best_alpha, color='red', linestyle='--', label=f'Best alpha: {best_alpha}')
        plt.xscale('log')
        plt.xlabel('Alpha (Regularization Strength)')
        plt.ylabel('R2 Score')
        plt.title('Ridge Regression: Alpha vs R2 Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(OUTPUT_DIR, 'ridge_alpha_tuning.png'))
        plt.close()
    
    def _find_best_model(self, results_table, models_dict, feature_importances, y_test):
        """Find and return the best performing model"""
        best_row = results_table.loc[results_table['R2'].idxmax()]
        best_model_name = best_row['Model']
        best_model = models_dict[best_model_name]
        best_importance = feature_importances[best_model_name]
        
        print(f"\nBest model: {best_model_name}")
        
        # Print most important feature for each model
        print("\nMost important feature for each model:")
        for model_name, importance_df in feature_importances.items():
            top_feature = importance_df.iloc[0]
            print(f"{model_name}: {top_feature['Feature']} (Importance: {top_feature['Importance']:.4f})")
        
        return best_model, best_importance
