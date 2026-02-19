import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = "training_data.csv"
MODEL_FILE = "netprophet_model.json"

def train_with_smote():
    print("Loading data...")
    df = pd.read_csv(INPUT_FILE)
    
    # 1. CLEANING: Drop rows with NaN (Crucial for SMOTE)
    original_len = len(df)
    df.dropna(inplace=True)
    print(f"🧹 Dropped {original_len - len(df)} rows containing NaNs.")
    
    # Features
    features = [col for col in df.columns if col not in 
                ['timestamp', 'threshold', 'is_degraded', 'target_5m_degraded']]
    
    X = df[features]
    y = df['target_5m_degraded']
    
    # 2. Random Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📉 Original Training Count: {len(X_train)} rows")
    print(f"   - Normal: {sum(y_train == 0)}")
    print(f"   - Crashes: {sum(y_train == 1)}")

    # 3. APPLY SMOTE
    # k_neighbors must be smaller than the number of crashes.
    # Since you have 39 crashes, k=3 is safe.
    smote = SMOTE(random_state=42, k_neighbors=3) 
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"📈 SMOTE Training Count: {len(X_train_resampled)} rows")
    print(f"   - Normal: {sum(y_train_resampled == 0)}")
    print(f"   - Crashes: {sum(y_train_resampled == 1)}")
    
    # 4. Train XGBoost
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    print("🧠 Training the brain...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    
    print("\n" + "="*40)
    print("🤖 MODEL PERFORMANCE (With SMOTE)")
    print("="*40)
    
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives:  {cm[1][1]}")

    # 6. Feature Importance
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    importance = model.feature_importances_
    feature_names = X.columns
    sorted_idx = importance.argsort()[::-1]
    for index in sorted_idx:
        print(f"   • {feature_names[index]}: {importance[index]:.4f}")
    
    # Save model
    model.save_model(MODEL_FILE)

if __name__ == "__main__":
    train_with_smote()