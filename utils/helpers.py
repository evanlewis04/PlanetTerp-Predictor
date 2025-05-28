"""
Utility functions and helpers for PlanetTerp analysis
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

def create_output_directory(output_dir: str) -> None:
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, return default if denominator is 0"""
    if denominator == 0:
        return default
    return numerator / denominator

def safe_variance(values: List[float], default: float = 0.0) -> float:
    """Safely calculate variance, return default if not enough values"""
    if len(values) <= 1:
        return default
    return np.var(values)

def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Safely calculate mean, return default if empty list"""
    if not values:
        return default
    return sum(values) / len(values)

def filter_valid_grades(reviews: List[Dict], grade_map: Dict[str, Optional[float]]) -> List[float]:
    """Extract and filter valid numerical grades from reviews"""
    grades = [grade_map.get(review.get('expected_grade')) for review in reviews
              if review.get('expected_grade') in grade_map]
    return [grade for grade in grades if grade is not None]

def filter_valid_reviews(professors: List[Dict], min_reviews: int) -> List[Dict]:
    """Filter professors with at least min_reviews reviews"""
    valid_professors = []
    skipped_count = 0
    
    for professor in professors:
        if isinstance(professor, dict) and professor.get('reviews'):
            if len(professor.get('reviews')) >= min_reviews:
                valid_professors.append(professor)
            else:
                skipped_count += 1
    
    print(f"Found {len(valid_professors)} professors with at least {min_reviews} reviews")
    print(f"Skipped {skipped_count} professors with fewer than {min_reviews} reviews")
    
    return valid_professors

def log_progress(current: int, total: int, interval: int, message: str = "Processing") -> None:
    """Log progress at specified intervals"""
    if current % interval == 0 or current == total:
        percentage = (current / total) * 100
        print(f"{message}: {current}/{total} ({percentage:.1f}%)")

def extract_course_level(course_code: str) -> int:
    """Extract course level from course code (e.g., 'CMSC131' -> 1)"""
    import re
    match = re.search(r'\d+', course_code)
    if match:
        return int(match.group()) // 100
    return 0

def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for the feature DataFrame"""
    return {
        'num_samples': len(df),
        'num_features': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'feature_names': list(df.columns),
        'target_range': (df['avg_rating'].min(), df['avg_rating'].max()) if 'avg_rating' in df.columns else None
    }