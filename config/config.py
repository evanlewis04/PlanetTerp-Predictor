"""
Configuration settings for PlanetTerp analysis
"""

# Data fetching parameters
MAX_PROFESSORS = 1000
MIN_REVIEWS = 10
PROFESSORS_PER_BATCH = 100

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

# Ridge regression alpha values to test
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Grade mapping for GPA calculation
GRADE_MAP = {
    'A+': 4.0, 'A': 4.0, 'A-': 3.7,
    'B+': 3.3, 'B': 3.0, 'B-': 2.7,
    'C+': 2.3, 'C': 2.0, 'C-': 1.7,
    'D+': 1.3, 'D': 1.0, 'D-': 0.7,
    'F': 0.0, 'W': None, 'Other': None
}

# Output settings
OUTPUT_DIR = 'outputs'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Logging settings
LOG_LEVEL = 'INFO'
PROGRESS_INTERVAL = 100  # Log progress every N professors