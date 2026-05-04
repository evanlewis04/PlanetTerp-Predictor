"""
Feature extraction methods for professor data
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict
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


KEYWORD_CATEGORIES = {
    'difficulty': [
        'difficult', 'hard', 'tough', 'challenging', 'brutal', 'confusing',
        'stressful', 'demanding'
    ],
    'clarity': [
        'clear', 'unclear', 'organized', 'disorganized', 'explains',
        'explanation', 'understand', 'confusing'
    ],
    'helpfulness': [
        'helpful', 'office hours', 'responsive', 'available', 'supportive',
        'accommodating', 'approachable'
    ],
    'workload': [
        'homework', 'assignment', 'project', 'workload', 'busywork', 'paper',
        'reading', 'quiz', 'quizzes'
    ],
    'exams': [
        'exam', 'exams', 'midterm', 'final', 'test', 'tests'
    ],
    'grading': [
        'grade', 'grades', 'grading', 'curve', 'rubric', 'extra credit',
        'lenient', 'harsh'
    ],
    'lectures': [
        'lecture', 'lectures', 'slides', 'notes', 'recording', 'presentation'
    ],
    'attendance': [
        'attendance', 'mandatory', 'participation', 'absent', 'lecture hall'
    ],
}


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
                'course_level_variance': 0,
                'lower_level_ratio': 0,
                'mid_level_ratio': 0,
                'upper_level_ratio': 0,
                'graduate_level_ratio': 0,
                'lower_level_course_count': 0,
                'mid_level_course_count': 0,
                'upper_level_course_count': 0,
                'graduate_level_course_count': 0
            }

        course_codes = [review.get('course', '') for review in reviews if review.get('course')]
        unique_courses = len(set(course_codes))

        course_levels = [extract_course_level(code) for code in course_codes]
        course_levels = [level for level in course_levels if level > 0]

        if course_levels:
            avg_course_level = safe_mean(course_levels)
            course_level_variance = safe_variance(course_levels)
            lower_level_count = sum(1 for level in course_levels if level in [1, 2])
            mid_level_count = sum(1 for level in course_levels if level == 3)
            upper_level_count = sum(1 for level in course_levels if level == 4)
            graduate_level_count = sum(1 for level in course_levels if level >= 5)
            lower_level_ratio = safe_divide(lower_level_count, len(course_levels))
            mid_level_ratio = safe_divide(mid_level_count, len(course_levels))
            upper_level_ratio = safe_divide(
                sum(1 for level in course_levels if level >= 3), 
                len(course_levels)
            )
            graduate_level_ratio = safe_divide(graduate_level_count, len(course_levels))
        else:
            avg_course_level = 0
            course_level_variance = 0
            lower_level_count = 0
            mid_level_count = 0
            upper_level_count = 0
            graduate_level_count = 0
            lower_level_ratio = 0
            mid_level_ratio = 0
            upper_level_ratio = 0
            graduate_level_ratio = 0

        return {
            'unique_courses': unique_courses,
            'avg_course_level': avg_course_level,
            'course_level_variance': course_level_variance,
            'lower_level_ratio': lower_level_ratio,
            'mid_level_ratio': mid_level_ratio,
            'upper_level_ratio': upper_level_ratio,
            'graduate_level_ratio': graduate_level_ratio,
            'lower_level_course_count': lower_level_count,
            'mid_level_course_count': mid_level_count,
            'upper_level_course_count': upper_level_count,
            'graduate_level_course_count': graduate_level_count
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
                'grade_a_ratio': 0,
                'grade_b_or_above_ratio': 0,
                'grade_c_or_below_ratio': 0,
                'valid_expected_grade_count': 0,
                'missing_expected_grade_ratio': 0
            }

        grades = filter_valid_grades(reviews, GRADE_MAP)
        missing_grade_count = sum(1 for review in reviews if not review.get('expected_grade'))

        if grades:
            avg_expected_grade = safe_mean(grades)
            grade_variance = safe_variance(grades)
            grade_a_ratio = safe_divide(
                sum(1 for grade in grades if grade >= 3.7), 
                len(grades)
            )
            grade_b_or_above_ratio = safe_divide(
                sum(1 for grade in grades if grade >= 3.0),
                len(grades)
            )
            grade_c_or_below_ratio = safe_divide(
                sum(1 for grade in grades if grade <= 2.0),
                len(grades)
            )
        else:
            avg_expected_grade = 0
            grade_variance = 0
            grade_a_ratio = 0
            grade_b_or_above_ratio = 0
            grade_c_or_below_ratio = 0

        return {
            'avg_expected_grade': avg_expected_grade,
            'grade_variance': grade_variance,
            'grade_a_ratio': grade_a_ratio,
            'grade_b_or_above_ratio': grade_b_or_above_ratio,
            'grade_c_or_below_ratio': grade_c_or_below_ratio,
            'valid_expected_grade_count': len(grades),
            'missing_expected_grade_ratio': safe_divide(missing_grade_count, len(reviews))
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
                'median_review_length': 0,
                'review_length_variance': 0,
                'avg_review_word_count': 0,
                'median_review_word_count': 0,
                'empty_review_ratio': 0,
                'has_reviews': 0
            }

        num_reviews = len(reviews)
        review_texts = [review.get('review', '') for review in reviews if review.get('review')]
        review_lengths = [len(text) for text in review_texts]
        review_word_counts = [len(re.findall(r'\b\w+\b', text)) for text in review_texts]
        empty_review_count = num_reviews - len(review_texts)

        avg_review_length = safe_mean(review_lengths)
        median_review_length = float(np.median(review_lengths)) if review_lengths else 0
        review_length_variance = safe_variance(review_lengths)
        avg_review_word_count = safe_mean(review_word_counts)
        median_review_word_count = float(np.median(review_word_counts)) if review_word_counts else 0
        has_reviews = 1 if num_reviews > 0 else 0

        return {
            'num_reviews': num_reviews,
            'avg_review_length': avg_review_length,
            'median_review_length': median_review_length,
            'review_length_variance': review_length_variance,
            'avg_review_word_count': avg_review_word_count,
            'median_review_word_count': median_review_word_count,
            'empty_review_ratio': safe_divide(empty_review_count, num_reviews),
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
                    'sentiment_variance': 0,
                    'positive_review_ratio': 0,
                    'negative_review_ratio': 0,
                    'neutral_review_ratio': 0
                }

            review_texts = [review.get('review', '') for review in reviews if review.get('review')]

            if not review_texts:
                return {
                    'avg_sentiment_pos': 0,
                    'avg_sentiment_neg': 0,
                    'avg_sentiment_compound': 0,
                    'sentiment_variance': 0,
                    'positive_review_ratio': 0,
                    'negative_review_ratio': 0,
                    'neutral_review_ratio': 0
                }

            sentiment_scores = [sia.polarity_scores(text) for text in review_texts]

            compound_scores = [score['compound'] for score in sentiment_scores]
            positive_scores = [score['pos'] for score in sentiment_scores]
            negative_scores = [score['neg'] for score in sentiment_scores]

            return {
                'avg_sentiment_pos': safe_mean(positive_scores),
                'avg_sentiment_neg': safe_mean(negative_scores),
                'avg_sentiment_compound': safe_mean(compound_scores),
                'sentiment_variance': safe_variance(compound_scores),
                'positive_review_ratio': safe_divide(
                    sum(1 for score in compound_scores if score >= 0.05),
                    len(compound_scores)
                ),
                'negative_review_ratio': safe_divide(
                    sum(1 for score in compound_scores if score <= -0.05),
                    len(compound_scores)
                ),
                'neutral_review_ratio': safe_divide(
                    sum(1 for score in compound_scores if -0.05 < score < 0.05),
                    len(compound_scores)
                )
            }
            
        except ImportError:
            print("NLTK not available. Skipping sentiment analysis.")
            return {
                'avg_sentiment_pos': 0,
                'avg_sentiment_neg': 0,
                'avg_sentiment_compound': 0,
                'sentiment_variance': 0,
                'positive_review_ratio': 0,
                'negative_review_ratio': 0,
                'neutral_review_ratio': 0
            }

    def extract_keyword_features(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Extract interpretable keyword category features from review text.
        """
        review_texts = [review.get('review', '') for review in reviews if review.get('review')]
        combined_text = ' '.join(review_texts).lower()
        total_words = len(re.findall(r'\b\w+\b', combined_text))
        features = {}

        for category, keywords in KEYWORD_CATEGORIES.items():
            keyword_hits = 0
            reviews_with_keyword = 0
            for text in review_texts:
                normalized_text = text.lower()
                review_has_keyword = False
                for keyword in keywords:
                    hits = normalized_text.count(keyword)
                    keyword_hits += hits
                    if hits > 0:
                        review_has_keyword = True
                if review_has_keyword:
                    reviews_with_keyword += 1

            features[f'{category}_keyword_count'] = keyword_hits
            features[f'{category}_keyword_density'] = safe_divide(keyword_hits, total_words)
            features[f'{category}_review_ratio'] = safe_divide(
                reviews_with_keyword,
                len(review_texts)
            )

        return features

    def extract_readability_features(self, reviews: List[Dict]) -> Dict[str, float]:
        """
        Extract lightweight readability and punctuation features from review text.
        """
        review_texts = [review.get('review', '') for review in reviews if review.get('review')]
        if not review_texts:
            return {
                'avg_sentence_count': 0,
                'avg_words_per_sentence': 0,
                'question_mark_ratio': 0,
                'exclamation_mark_ratio': 0
            }

        sentence_counts = []
        words_per_sentence = []
        question_mark_count = 0
        exclamation_mark_count = 0

        for text in review_texts:
            words = re.findall(r'\b\w+\b', text)
            sentences = [part for part in re.split(r'[.!?]+', text) if part.strip()]
            sentence_count = max(len(sentences), 1)
            sentence_counts.append(sentence_count)
            words_per_sentence.append(safe_divide(len(words), sentence_count))
            question_mark_count += text.count('?')
            exclamation_mark_count += text.count('!')

        return {
            'avg_sentence_count': safe_mean(sentence_counts),
            'avg_words_per_sentence': safe_mean(words_per_sentence),
            'question_mark_ratio': safe_divide(question_mark_count, len(review_texts)),
            'exclamation_mark_ratio': safe_divide(exclamation_mark_count, len(review_texts))
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
            keyword_features = self.extract_keyword_features(professor.get('reviews', []))
            readability_features = self.extract_readability_features(professor.get('reviews', []))

            # Combine all features
            features = {
                **prof_info,
                **review_stats,
                **course_features,
                **grade_features,
                **sentiment_features,
                **keyword_features,
                **readability_features
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
