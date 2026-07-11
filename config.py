# ============================================================
#  config.py — Central configuration for Disease Predictor
# ============================================================

# Model parameters
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
CV_FOLDS       = 5

RF_PARAMS = {
    'n_estimators'   : 200,
    'max_depth'      : None,
    'min_samples_split': 2,
    'min_samples_leaf' : 1,
    'max_features'   : 'sqrt',
    'random_state'   : RANDOM_STATE,
    'n_jobs'         : -1
}

XGB_PARAMS = {
    'n_estimators'      : 200,
    'max_depth'         : 6,
    'learning_rate'     : 0.1,
    'subsample'         : 0.8,
    'colsample_bytree'  : 0.8,
    'use_label_encoder' : False,
    'eval_metric'       : 'mlogloss',
    'random_state'      : RANDOM_STATE,
    'n_jobs'            : -1
}

# File paths
TRAIN_CSV       = 'data/Training.csv'
TEST_CSV        = 'data/Testing.csv'
MODEL_PATH      = 'models/disease_model.pkl'
XGB_MODEL_PATH  = 'models/disease_model_xgb.pkl'
ENCODER_PATH    = 'models/label_encoder.pkl'
SYMPTOM_PATH    = 'models/symptom_list.pkl'
ASSETS_DIR      = 'assets/'

# Diseases requiring urgent attention
URGENT_DISEASES = [
    'Heart attack', 'Typhoid', 'Malaria', 'Dengue',
    'Tuberculosis', 'Hepatitis B', 'Hepatitis C',
    'Hepatitis D', 'Hepatitis E', 'AIDS'
]

LOW_CONFIDENCE_THRESHOLD = 0.70   # below this → flag as urgent

# App info
APP_TITLE       = 'Disease Prediction System'
APP_ICON        = '🩺'
APP_VERSION     = '1.0.0'
DATASET_SOURCE  = 'https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning'