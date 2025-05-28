# 🎓 PlanetTerp Professor Rating Predictor

This is a machine learning project that predicts professor ratings using sentiment analysis, grade distributions, and course characteristics from PlanetTerp review data.

## 📋 Features

- **Data Fetching**: Automated retrieval of professor and review data from PlanetTerp API
- **Feature Engineering**: 
  - Sentiment analysis of student reviews using VADER
  - Grade distribution analysis and statistics
  - Course level and teaching breadth metrics
  - Review quantity and quality indicators
- **Machine Learning Models**:
  - Linear Regression
  - Ridge Regression with hyperparameter tuning
  - Random Forest Regression
- **Model Evaluation**: 10-fold cross-validation with comprehensive metrics
- **Visualizations**: Automated generation of performance plots and feature importance charts

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis**:
   ```bash
   python main.py
   ```

## 📁 Project Structure

```
planetterp_predictor/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── main.py                  # Main script to run analysis
├── config/
│   └── config.py            # Configuration settings
├── src/
│   ├── __init__.py
│   ├── data_processor.py    # Data fetching from PlanetTerp API
│   ├── feature_extractor.py # Feature engineering and extraction
│   ├── model_trainer.py     # Model training and evaluation
│   └── evaluator.py         # Cross-validation and plotting
├── utils/
│   ├── __init__.py
│   └── helpers.py           # Utility functions
└── outputs/                 # Generated plots and results
```

## ⚙️ Configuration

Edit `config/config.py` to customize:

- **MAX_PROFESSORS**: Number of professors to fetch (default: 1000)
- **MIN_REVIEWS**: Minimum reviews required per professor (default: 10)
- **CV_FOLDS**: Cross-validation folds (default: 10)
- **RIDGE_ALPHAS**: Alpha values for Ridge regression tuning

## 📊 Output Files

The analysis generates several visualization files in the `outputs/` directory:

- `cross_validation_results.png` - Cross-validation performance comparison
- `model_comparison.png` - Final model performance metrics
- `feature_importance_all_models.png` - Feature importance for each model
- `*_residuals.png` - Residual plots for each model
- `ridge_alpha_tuning.png` - Ridge regression hyperparameter tuning
- `prediction_scatter.png` - Actual vs predicted ratings scatter plot

## 🔬 Model Features

The predictor uses the following feature categories:

### 📝 Review Statistics
- Number of reviews
- Average review length
- Review presence indicator

### 📚 Course Features
- Unique courses taught
- Average course level (100, 200, 300, 400 level)
- Ratio of upper-level courses

### 📈 Grade Features
- Average expected grade (GPA scale)
- Grade variance
- Ratio of A-level grades

### 🎭 Sentiment Features
- Average positive sentiment
- Average negative sentiment
- Overall compound sentiment
- Sentiment variance across reviews

## 📈 Performance

Typical performance metrics:
- **R² Score**: 0.77-0.87 
- **Cross-validation**: 10-fold with standard deviation reporting
- **Best Model**: Ridge Regression

