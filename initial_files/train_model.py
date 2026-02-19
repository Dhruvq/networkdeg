import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt



INPUT_FILE = "training_data.csv"
MODEL_FILE = "model.json"

def train_model():
    # 1. Load Data
    print("Loading training data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Feature Selection
    features = [col for col in df.columns if col not in 
                ['timestamp', 'threshold', 'is_degraded', 'target_5m_degraded']]
    
    X = df[features]
    y = df['target_5m_degraded']
    
    # 2. Chronological Split (Train on Past, Test on Future)
    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # 3. Aggressive Training
    # We increase the weight significantly to force the model to care about failures
    count_class_0, count_class_1 = y_train.value_counts()
    scale_weight = count_class_0 / count_class_1 
    
    print(f"Calculated Scale Weight: {scale_weight:.2f} (This tells model to treat 1 failure as equal to {scale_weight:.0f} successes)")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=300,       # More trees
        learning_rate=0.05,     # Slower learning (better generalization)
        max_depth=4,            # Shallower trees (prevents overfitting)
        scale_pos_weight=scale_weight * 1.5, # Boost the weight even more
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("Training the brain...")
    model.fit(X_train, y_train)
    
    # 4. THRESHOLD TUNING (The "Secret Sauce")
    # Instead of just asking for 0 or 1, we ask for the PROBABILITY (0.0 to 1.0)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # We lower the bar: If risk is > 30%, call it a crash.
    custom_threshold = 0.30
    y_pred_aggressive = (y_proba >= custom_threshold).astype(int)
    
    # 5. Evaluate
    print("\n" + "="*40)
    print(f"🤖 MODEL PERFORMANCE (Threshold > {custom_threshold})")
    print("="*40)
    
    print(classification_report(y_test, y_pred_aggressive))
    
    # Confusion Matrix (Visualizing the "Misses")
    cm = confusion_matrix(y_test, y_pred_aggressive)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Correct Silence): {cm[0][0]}")
    print(f"False Positives (False Alarm):    {cm[0][1]}")
    print(f"False Negatives (Missed Crash):   {cm[1][0]}")
    print(f"True Positives (Caught Crash):    {cm[1][1]}")
    
    # 6. Feature Importance
    print("\n ROOT CAUSE ANALYSIS:")
    importance = model.feature_importances_
    
    model.save_model(MODEL_FILE)

if __name__ == "__main__":
    train_model()