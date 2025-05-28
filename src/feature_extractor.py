"""
Feature extraction methods for professor data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from utils.helpers import (safe_divide, safe_variance, safe_mean, 
                          filter_valid_grades, extract_course_level)
from config.config import GRADE_MAP

# Download NLTK data if needed
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')


class FeatureExtractor:
    """Extracts features from professor review data"""
    
    def __init__(self):
        """Initialize feature extractor"""
        pass
    
    def extract_course_features(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Extract features related to courses taught by the professor
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Dictionary of course-related features
        """
        if not reviews:
            return {
                'unique_courses': 0,
                'avg_course_level': 0,
                'upper_level_ratio': 0
            }

        course_codes = [review.get('course', '') for review in reviews if review.get('course')]
        unique_courses = len(set(course_codes))

        course_levels = [extract_course_level(code) for code in course_codes]
        course_levels = [level for level in course_levels if level > 0]

        if course_levels:
            avg_course_level = safe_mean(course_levels)
            upper_level_ratio = safe_divide(
                sum(1 for level in course_levels if level >= 3), 
                len(course_levels)
            )
        else:
            avg_course_level = 0
            upper_level_ratio = 0

        return {
            'unique_courses': unique_courses,
            'avg_course_level': avg_course_level,
            'upper_level_ratio': upper_level_ratio
        }

    def extract_grade_features(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Extract features related to expected grades from reviews
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Dictionary of grade-related features
        """
        if not reviews:
            return {
                'avg_expected_grade': 0,
                'grade_variance': 0,
                'grade_a_ratio': 0
            }

        grades = filter_valid_grades(reviews, GRADE_MAP)

        if grades:
            avg_expected_grade = safe_mean(grades)
            grade_variance = safe_variance(grades)
            grade_a_ratio = safe_divide(
                sum(1 for grade in grades if grade >= 3.7), 
                len(grades)
            )
        else:
            avg_expected_grade = 0
            grade_variance = 0
            grade_a_ratio = 0

        return {
            'avg_expected_grade': avg_expected_grade,
            'grade_variance': grade_variance,
            'grade_a_ratio': grade_a_ratio
        }

    def extract_review_stats(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Extract basic statistical features from reviews
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Dictionary of review statistics
        """
        if not reviews:
            return {
                'num_reviews': 0,
                'avg_review_length': 0,
                'has_reviews': 0
            }

        num_reviews = len(reviews)
        review_texts = [review.get('review', '') for review in reviews if review.get('review')]
        review_lengths = [len(text) for text in review_texts]

        avg_review_length = safe_mean(review_lengths)
        has_reviews = 1 if num_reviews > 0 else 0

        return {
            'num_reviews': num_reviews,
            'avg_review_length': avg_review_length,
            'has_reviews': has_reviews
        }

    def extract_sentiment_features(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Extract sentiment features from review text using VADER
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Dictionary of sentiment features
        """
        try:
            sia = SentimentIntensityAnalyzer()

            if not reviews:
                return {
                    'avg_sentiment_pos': 0,
                    'avg_sentiment_neg': 0,
                    'avg_sentiment_compound': 0,
                    'sentiment_variance': 0
                }

            review_texts = [review.get('review', '') for review in reviews if review.get('review')]

            if not review_texts:
                return {
                    'avg_sentiment_pos': 0,
                    'avg_sentiment_neg': 0,
                    'avg_sentiment_compound': 0,
                    'sentiment_variance': 0
                }

            sentiment_scores = [sia.polarity_scores(text) for text in review_texts]

            compound_scores = [score['compound'] for score in sentiment_scores]
            positive_scores = [score['pos'] for score in sentiment_scores]
            negative_scores = [score['neg'] for score in sentiment_scores]

            return {
                'avg_sentiment_pos': safe_mean(positive_scores),
                'avg_sentiment_neg': safe_mean(negative_scores),
                'avg_sentiment_compound': safe_mean(compound_scores),
                'sentiment_variance': safe_variance(compound_scores)
            }
            
        except ImportError:
            print("NLTK not available. Skipping sentiment analysis.")
            return {
                'avg_sentiment_pos': 0,
                'avg_sentiment_neg': 0,
                'avg_sentiment_compound': 0,
                'sentiment_variance': 0
            }

    def process_professor_data(self, professors: List[Dict]) -> pd.DataFrame:
        """
        Process professor data and extract all features
        
        Args:
            professors: List of professor dictionaries
            
        Returns:
            DataFrame with extracted features
        """
        professor_features = []

        for professor in professors:
            if not isinstance(professor, dict) or not professor.get('reviews'):
                continue

            prof_info = {
                'professor_id': professor.get('name'),
                'department': professor.get('department', 'Unknown')
            }

            # Extract all feature categories
            review_stats = self.extract_review_stats(professor.get('reviews', []))
            course_features = self.extract_course_features(professor.get('reviews', []))
            grade_features = self.extract_grade_features(professor.get('reviews', []))
            sentiment_features = self.extract_sentiment_features(professor.get('reviews', []))

            # Combine all features
            features = {
                **prof_info,
                **review_stats,
                **course_features,
                **grade_features,
                **sentiment_features
            }

            # Calculate target variable (average rating)
            ratings = [review.get('rating') for review in professor.get('reviews', [])
                      if isinstance(review.get('rating'), (int, float))]
            if ratings:
                features['avg_rating'] = safe_mean(ratings)
            else:
                features['avg_rating'] = None

            professor_features.append(features)

        return pd.DataFrame(professor_features)

    def prepare_data_for_modeling(self, professors: List[Dict]) -> tuple:
        """
        Prepare complete dataset for modeling
        
        Args:
            professors: List of professor dictionaries
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        features_df = self.process_professor_data(professors)

        if features_df.empty:
            return None, None

        target = features_df['avg_rating']
        features = features_df.drop(['avg_rating', 'professor_id'], axis=1)

        # One-hot encode categorical variables
        features = pd.get_dummies(features, drop_first=True)

        return features, target