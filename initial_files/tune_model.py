import pandas as pd
from sklearn import pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix#, make_scorer, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

INPUT_FILE = "training_data.csv"

def tune_model():
    print("Loading data for tuning...")
    df = pd.read_csv(INPUT_FILE)
    df.dropna(inplace=True) # Drop NaNs
    
    features = [col for col in df.columns if col not in 
                ['timestamp', 'threshold', 'is_degraded', 'target_5m_degraded']]
    
    X = df[features]
    y = df['target_5m_degraded']

    # Split Holdout Set (We will NOT touch this until the very end)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Starting Grid Search on {len(X_train)} rows...")

    # 1. Define the Pipeline
    # SMOTE happens INSIDE the cross-validation loop. 
    # This prevents Data Leakage.
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    # 2. Define the Hyperparameter Grid (The "Contestants")
    # We test combinations of these settings.
    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 4, 5],
        'xgb__learning_rate': [0.05, 0.1],
        'xgb__scale_pos_weight': [1, 3, 5]  # Reduced from [1, 10, 25]
    }

    # 3. Run Grid Search
    # We optimize for 'f1' because it balances precision and recall (ideal for crash detection).
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',    
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # 4. Results
    print("\n" + "="*40)
    print("BEST PARAMETERS FOUND")
    print("="*40)
    print(grid_search.best_params_)
    print(f"Best CV Recall Score: {grid_search.best_score_:.2f}")

    # 5. Final Test on the Holdout Set
    print("\n Testing Best Model on Holdout Data...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")

    # Save the winner
    # We need to extract the XGBoost model from the pipeline to save it properly
    final_xgb = best_model.named_steps['xgb']
    final_xgb.save_model("tournament_model_2.json")
    print("\n Best model saved to tournament_model_2.json")

if __name__ == "__main__":
    tune_model()