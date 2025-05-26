import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import optuna
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, average_precision_score

# 1. Load data
print("Loading original features...")
original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
print("Loading odd one out features...")
odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")
print("Loading participant responses...")
participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=int(1e6))

# Filter participant_responses to only include 'mdt' test
participant_responses = participant_responses[participant_responses['test'] == 'mdt']

# Drop duration features from original and odd-one-out features
duration_cols_orig = [col for col in original_features.columns if 'duration_features.' in col]
duration_cols_ooo = [col for col in odd_one_out_features.columns if 'duration_features.' in col]
print("Dropping duration features from original:", duration_cols_orig)
print("Dropping duration features from odd-one-out:", duration_cols_ooo)
original_features = original_features.drop(columns=duration_cols_orig)
odd_one_out_features = odd_one_out_features.drop(columns=duration_cols_ooo)

# Merge features on melody_id
if 'melody_id' not in original_features.columns:
    raise ValueError('melody_id column missing from original_features')
if 'melody_id' not in odd_one_out_features.columns:
    raise ValueError('melody_id column missing from odd_one_out_features')

features_merged = original_features.merge(odd_one_out_features, on='melody_id', suffixes=('_orig', '_ooo'))

# Only use numeric columns for difference calculation
orig_num = features_merged.filter(regex='_orig$').select_dtypes(include=[np.number]).copy()
ooo_num = features_merged.filter(regex='_ooo$').select_dtypes(include=[np.number]).copy()
feature_diffs = ooo_num.values - orig_num.values
feature_diffs = pd.DataFrame(feature_diffs, columns=[col.replace('_orig', '') + '_diff' for col in orig_num.columns])

# Drop any columns with zero variance
non_zero_var_cols = feature_diffs.columns[feature_diffs.var() != 0]
feature_diffs = feature_diffs[non_zero_var_cols]

# Concatenate all features
features_final = pd.concat([features_merged, feature_diffs], axis=1)

# Drop zero variance columns
zero_var_cols = [col for col in features_final.columns if features_final[col].nunique() == 1]
print("Dropping zero variance columns:", zero_var_cols)
features_final = features_final.drop(columns=zero_var_cols)

# Drop duration features which are constant across the dataset
duration_cols = [col for col in features_final.columns if 'duration' in col.lower()]
print("Dropping duration columns:", duration_cols)
features_final = features_final.drop(columns=duration_cols)

# Drop any rows with missing values
features_final = features_final.dropna()

# Get raw scores from participant_responses instead of mean scores
raw_scores = participant_responses[['item_id', 'score']]

# Merge features with raw scores
data_raw = features_final.merge(raw_scores, left_on='melody_id', right_on='item_id')

# Prepare X and y using raw scores
exclude_cols = {'melody_id', 'item_id', 'score'}
feature_cols = [col for col in data_raw.columns if col not in exclude_cols]
numeric_feature_cols = data_raw[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X_raw = data_raw[numeric_feature_cols].values
y_raw = data_raw['score'].values

# Train/test split by melody_id to avoid data leakage
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
groups = data_raw['melody_id'].values

for train_idx, test_idx in gss.split(X_raw, y_raw, groups=groups):
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train_raw, y_test_raw = y_raw[train_idx], y_raw[test_idx]

def optimize_xgb_classifier(X, y, groups):
    # Calculate scale_pos_weight from the full training set
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    scale_pos_weight_default = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Auto-calculated scale_pos_weight: {scale_pos_weight_default:.3f}")
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', max(0.5, scale_pos_weight_default * 0.5), scale_pos_weight_default * 2),
            'random_state': 123,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
        for train_idx, test_idx in gss.split(X, y, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        xgb_clf = XGBClassifier(**params)
        xgb_clf.fit(X_train, y_train)
        proba_test = xgb_clf.predict_proba(X_test)[:, 1]
        pred_test = (proba_test > 0.5).astype(int)
        f1 = f1_score(y_test, pred_test)
        return f1
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40, n_jobs=-1)
    print('Best parameters:', study.best_params)
    return study.best_params

def xgboost_raw_classifier():
    # Make sure y_raw is binary (0 or 1)
    assert set(np.unique(y_raw)).issubset({0, 1}), "y_raw must be binary for classification!"

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Calculate scale_pos_weight for the training set
    n_pos = np.sum(y_train_raw == 1)
    n_neg = np.sum(y_train_raw == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Auto-calculated scale_pos_weight for fit: {scale_pos_weight:.3f}")
    # --- Run Optuna optimization ---
    best_params = optimize_xgb_classifier(X_train_scaled, y_train_raw, groups[train_idx])
    # Remove params not accepted by fit
    best_params_fit = best_params.copy()
    best_params_fit.pop('eval_metric', None)
    # --- Fit classifier with best params, using auto scale_pos_weight if not present ---
    if 'scale_pos_weight' not in best_params_fit:
        best_params_fit['scale_pos_weight'] = scale_pos_weight
    xgb_clf = XGBClassifier(**best_params_fit, eval_metric='logloss')

    # Combine X and y for easy resampling
    Xy = np.hstack([X_train_scaled, y_train_raw.reshape(-1, 1)])
    majority = Xy[y_train_raw == 1]
    minority = Xy[y_train_raw == 0]

    # Downsample majority
    majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=123)
    Xy_balanced = np.vstack([majority_downsampled, minority])
    np.random.shuffle(Xy_balanced)

    X_train_bal = Xy_balanced[:, :-1]
    y_train_bal = Xy_balanced[:, -1]

    # Fit model on balanced data
    xgb_clf.fit(X_train_bal, y_train_bal)

    # Get predictions
    pred_train_raw = xgb_clf.predict(X_train_scaled)
    pred_test_raw = xgb_clf.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test_raw, pred_test_raw))
    print("F1:", f1_score(y_test_raw, pred_test_raw))
    print("ROC AUC:", roc_auc_score(y_test_raw, xgb_clf.predict_proba(X_test_scaled)[:, 1]))

    # Create confusion matrices
    cm_train = confusion_matrix(y_train_raw, pred_train_raw)
    cm_test = confusion_matrix(y_test_raw, pred_test_raw)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Train set confusion matrix
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix: Train Set')

    # Test set confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix: Test Set')

    plt.tight_layout()
    plt.show()
    # Get feature importances
    importances = xgb_clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 Feature Importances:")
    for f in range(min(20, len(numeric_feature_cols))):
        print("%d. %s (%f)" % (f + 1, numeric_feature_cols[indices[f]], importances[indices[f]]))
        
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Most Important Features for Raw Score Prediction", pad=20, fontsize=14)
    
    bars = plt.barh(range(19, -1, -1), importances[indices[:20]], color='skyblue', alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.yticks(range(19, -1, -1), [numeric_feature_cols[i] for i in indices[:20]],
               ha='right', fontsize=10)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def logistic_regression_baseline(X_train_raw, y_train_raw, X_test_raw, y_test_raw):
    print("\n--- Logistic Regression Baseline ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=123)
    lr.fit(X_train_scaled, y_train_raw)
    pred_train = lr.predict(X_train_scaled)
    pred_test = lr.predict(X_test_scaled)
    print("Train set:")
    print("  Accuracy:", accuracy_score(y_train_raw, pred_train))
    print("  Recall:", recall_score(y_train_raw, pred_train))
    print("  Precision:", precision_score(y_train_raw, pred_train))
    print("  F1:", f1_score(y_train_raw, pred_train))
    print("  Confusion matrix:\n", confusion_matrix(y_train_raw, pred_train))
    print("Test set:")
    print("  Accuracy:", accuracy_score(y_test_raw, pred_test))
    print("  Recall:", recall_score(y_test_raw, pred_test))
    print("  Precision:", precision_score(y_test_raw, pred_test))
    print("  F1:", f1_score(y_test_raw, pred_test))
    print("  Confusion matrix:\n", confusion_matrix(y_test_raw, pred_test))
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train set confusion matrix
    cm_train = confusion_matrix(y_train_raw, pred_train)
    sns.heatmap(cm_train, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual') 
    ax1.set_title('Confusion Matrix: Train Set')

    # Test set confusion matrix
    cm_test = confusion_matrix(y_test_raw, pred_test)
    sns.heatmap(cm_test, annot=True, fmt='d', ax=ax2, cmap='Blues')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix: Test Set')

    plt.tight_layout()
    plt.show()

def xgboost_compare_strategies():
    from sklearn.utils import resample
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    from xgboost import XGBClassifier
    import numpy as np

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Calculate scale_pos_weight for the training set
    n_pos = np.sum(y_train_raw == 1)
    n_neg = np.sum(y_train_raw == 0)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # --- (a) No resampling, use only scale_pos_weight ---
    print("\n--- XGBoost: No Resampling, Only scale_pos_weight ---")
    xgb_a = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=123, n_jobs=-1, eval_metric='logloss')
    xgb_a.fit(X_train_scaled, y_train_raw)
    pred_test_a = xgb_a.predict(X_test_scaled)
    proba_test_a = xgb_a.predict_proba(X_test_scaled)[:, 1]
    print("Test set:")
    print("  Accuracy:", accuracy_score(y_test_raw, pred_test_a))
    print("  Recall:", recall_score(y_test_raw, pred_test_a))
    print("  Precision:", precision_score(y_test_raw, pred_test_a))
    print("  F1:", f1_score(y_test_raw, pred_test_a))
    print("  Confusion matrix:\n", confusion_matrix(y_test_raw, pred_test_a))

    # --- (d) Precision-Recall Curve for (a) ---
    precision, recall, thresholds = precision_recall_curve(y_test_raw, proba_test_a)
    avg_prec = average_precision_score(y_test_raw, proba_test_a)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP={avg_prec:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (XGBoost, scale_pos_weight only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# xgboost_raw_classifier()
# logistic_regression_baseline(X_train_raw, y_train_raw, X_test_raw, y_test_raw)
xgboost_compare_strategies()
