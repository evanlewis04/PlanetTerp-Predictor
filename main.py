"""
Main script to run PlanetTerp professor rating prediction analysis
"""

from sklearn.impute import SimpleImputer
import pandas as pd
from typing import Any
from planetterp_predictor.experiment_tracking import ExperimentTracker
from src.data_processor import PlanetTerpDataProcessor
from src.feature_extractor import FeatureExtractor
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from utils.helpers import filter_valid_reviews, get_feature_summary
from config.config import MAX_PROFESSORS, MIN_REVIEWS


def run_planetterp_analysis(num_professors: int = MAX_PROFESSORS,
                           min_reviews: int = MIN_REVIEWS,
                           professors: list[dict[str, Any]] | None = None,
                           snapshot_metadata: dict[str, Any] | None = None,
                           experiment_name: str | None = None,
                           save_experiment: bool = True) -> tuple:
    """
    Complete workflow for PlanetTerp professor rating prediction
    
    Args:
        num_professors: Maximum number of professors to fetch
        min_reviews: Minimum number of reviews required per professor
        
    Returns:
        Tuple of (best_model, feature_importance, X, y)
    """
    print(f"Analysis will fetch and process up to {num_professors} professors with at least {min_reviews} reviews")
    print("="*70)

    # Step 1: Fetch or load data
    if professors is None:
        print("Step 1: Fetching professor data from PlanetTerp API")
        data_processor = PlanetTerpDataProcessor()
        professors = data_processor.fetch_professor_data(max_professors=num_professors)
        print(f"Successfully fetched data for {len(professors)} professors")
    else:
        print("Step 1: Using provided professor dataset")
        print(f"Loaded data for {len(professors)} professors")

    # Step 2: Filter professors with sufficient reviews
    print(f"\nStep 2: Filtering professors with at least {min_reviews} reviews")
    valid_professors = filter_valid_reviews(professors, min_reviews)
    
    if len(valid_professors) < 10:
        print("Error: Too few professors found with sufficient reviews!")
        return None, None, None, None

    # Step 3: Extract features
    print("\nStep 3: Processing data and extracting features")
    feature_extractor = FeatureExtractor()
    X, y = feature_extractor.prepare_data_for_modeling(valid_professors)
    
    if X is None or y is None:
        print("Error: Failed to prepare data for modeling")
        return None, None, None, None

    # Print dataset summary
    summary = get_feature_summary(pd.DataFrame(X).assign(avg_rating=y))
    print(f"\nDataset Summary:")
    print(f"- Number of professors: {summary['num_samples']}")
    print(f"- Number of features: {summary['num_features']}")
    print(f"- Target range: {summary['target_range'][0]:.2f} - {summary['target_range'][1]:.2f}")
    print(f"- Features: {', '.join(summary['feature_names'])}")

    # Step 4: Handle missing values
    imputation_strategy = None
    if X.isnull().sum().sum() > 0:
        print(f"\nStep 4: Handling {X.isnull().sum().sum()} missing values")
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        imputation_strategy = "mean"
    else:
        print("\nStep 4: No missing values found")

    # Step 5: Cross-validation evaluation
    print("\nStep 5: Evaluating models with cross-validation")
    evaluator = ModelEvaluator()
    cv_results = evaluator.evaluate_models_with_cross_validation(X, y)
    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    # Step 6: Train and evaluate final models
    print("\nStep 6: Training and evaluating final models")
    trainer = ModelTrainer()
    best_model, feature_importance, holdout_results, best_model_name = trainer.train_and_evaluate_models(X, y)

    experiment_run_path = None
    if save_experiment and best_model is not None:
        target_summary = {
            "count": len(y),
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
        }
        run_id, run_dir = ExperimentTracker().save_run(
            experiment_name=experiment_name,
            snapshot_metadata=snapshot_metadata,
            feature_columns=list(X.columns),
            target_summary=target_summary,
            imputation_strategy=imputation_strategy,
            cv_results=cv_results,
            holdout_results=holdout_results,
            best_model_name=best_model_name,
            best_model=best_model,
            best_feature_importance=feature_importance,
        )
        experiment_run_path = run_dir
        print(f"\nSaved experiment run: {run_id}")
        print(f"Experiment artifacts: {run_dir}")

    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    best_cv_r2 = cv_results["R2 Mean"].max()
    print(f"- Processed {len(valid_professors)} professors")
    print(f"- Extracted {len(X.columns)} features")
    print(f"- Best cross-validation R2 score: {best_cv_r2:.4f}")
    print("- Generated plots saved to the configured output directory")
    if experiment_run_path:
        print(f"- Experiment artifacts saved to {experiment_run_path}")
    
    if feature_importance is not None:
        print(f"\nTop 3 most predictive features:")
        for i, (_, row) in enumerate(feature_importance.head(3).iterrows()):
            print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")

    return best_model, feature_importance, X, y


if __name__ == "__main__":
    # Run the complete analysis
    try:
        best_model, feature_importance, X, y = run_planetterp_analysis()
        
        if best_model is not None:
            print(f"\nAnalysis completed successfully!")
            print(f"Check the 'outputs/' folder for detailed visualizations")
        else:
            print("Analysis failed - check error messages above")
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
