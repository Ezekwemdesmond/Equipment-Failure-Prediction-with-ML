import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('predictive_maintenance.csv')

# Filter out 'Random Failures'
df = df[df['Failure Type'] != 'Random Failures']

# Drop unnecessary columns
df.drop(['Product ID', 'UDI', 'Failure Type'], axis=1, inplace=True)

# Separate features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Define categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),  # Scale numerical columns
        ('cat', OrdinalEncoder(), categorical_columns)  # Encode categorical columns
    ]
)

# Create a pipeline for preprocessing
preprocess_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline on the data
preprocess_pipeline.fit(X)

# Save the preprocessing pipeline
joblib.dump(preprocess_pipeline, 'models/preprocessing_pipeline.pkl')

# Transform the data
X_transformed = preprocess_pipeline.transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define LightGBM model
lgb_model = LGBMClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='binary',  # Change if it's a multi-class problem
    metric='auc'  # Metric to optimize
)

# Set up stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(
    lgb_model,
    X_train_resampled,
    y_train_resampled,
    cv=skf,
    scoring='roc_auc'  # Change scoring metric as needed
)

print(f"Cross-validation ROC-AUC scores: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean()}")

# Train the model on the full training set
lgb_model.fit(X_train_resampled, y_train_resampled)

# Save the trained model
joblib.dump(lgb_model, 'models/failure_prediction_model.pkl')

# Evaluate on the test set
y_pred = lgb_model.predict(X_test)
y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
test_roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"Test ROC-AUC: {test_roc_auc}")