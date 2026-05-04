"""Compatibility constants for the original analysis modules."""

from planetterp_predictor.settings import settings

# Data fetching parameters
MAX_PROFESSORS = settings.max_professors
MIN_REVIEWS = settings.min_reviews
PROFESSORS_PER_BATCH = settings.professors_per_batch

# Model parameters
RANDOM_STATE = settings.random_state
TEST_SIZE = settings.test_size
CV_FOLDS = settings.cv_folds

# Ridge regression alpha values to test
RIDGE_ALPHAS = list(settings.ridge_alphas)

# Grade mapping for GPA calculation
GRADE_MAP = {
    'A+': 4.0, 'A': 4.0, 'A-': 3.7,
    'B+': 3.3, 'B': 3.0, 'B-': 2.7,
    'C+': 2.3, 'C': 2.0, 'C-': 1.7,
    'D+': 1.3, 'D': 1.0, 'D-': 0.7,
    'F': 0.0, 'W': None, 'Other': None
}

# Output settings
OUTPUT_DIR = settings.output_dir
FIGURE_DPI = settings.figure_dpi
FIGURE_FORMAT = settings.figure_format

# Logging settings
LOG_LEVEL = settings.log_level
PROGRESS_INTERVAL = settings.progress_interval  # Log progress every N professors
