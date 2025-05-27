import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_recall_curve, PrecisionRecallDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def irtxgboostclassifier():
    """XGBoost classifier incorporating IRT features from R analysis"""
    print("\nRunning XGBoost Classifier with IRT features...")

    # Load data
    print("Loading original features...")
    original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/testing.csv")
    print("Loading odd one out features...")
    odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/miq_mels2.csv")
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

    # Drop zero variance columns before merging with mean scores
    zero_var_cols = [col for col in features_final.columns if features_final[col].nunique() == 1]
    print("Dropping zero variance columns:", zero_var_cols)
    features_final = features_final.drop(columns=zero_var_cols)
    
    # Drop duration features which are constant across the dataset
    duration_cols = [col for col in features_final.columns if 'duration' in col.lower()]
    print("Dropping duration columns:", duration_cols)
    features_final = features_final.drop(columns=duration_cols)

    # Get IRT features
    irt_cols = ['item_id', 'ability_WL', 'difficulty', 'score']
    irt_features = participant_responses[irt_cols].copy()
    irt_features = irt_features.groupby('item_id').agg({
        'ability_WL': 'mean',
        'difficulty': 'mean',
        'score': 'first'
    }).reset_index()

    # Merge with existing features
    data_with_irt = features_final.merge(irt_features, left_on='melody_id', right_on='item_id', how='left')

    # Prepare feature columns
    exclude_cols = {'melody_id', 'item_id', 'score'}
    feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]
    numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # Prepare X and y
    X = data_with_irt[numeric_feature_cols].values
    y = data_with_irt['score'].values
    groups = data_with_irt['melody_id'].values

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    # Initialize XGBoost classifier
    xgb = XGBClassifier(random_state=8, n_jobs=-1)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    # Create validation set for early stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, val_idx in gss.split(X_train, y_train, groups=X_train_groups):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        groups_tr = X_train_groups[train_idx]

    # Set up GroupKFold cross validation
    gkf = GroupKFold(n_splits=5)
    
    # Perform grid search
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=gkf.split(X_tr, y_tr, groups=groups_tr),
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_tr, y_tr)
    
    print("\nXGBoost Classifier with IRT features - optimized parameters:")
    print("Best parameters:", grid_search.best_params_)
    print("Best CV ROC AUC:", round(grid_search.best_score_, 4))

    # Get best model
    best_xgb = grid_search.best_estimator_

    # Evaluate model
    pred_train = best_xgb.predict(X_train)
    pred_test = best_xgb.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)
    train_auc = roc_auc_score(y_train, best_xgb.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])

    # After predictions
    train_f1 = f1_score(y_train, pred_train)
    test_f1 = f1_score(y_test, pred_test)

    print("\nTrain metrics:")
    print("Accuracy:", round(train_accuracy, 4))
    print("F1 Score:", round(train_f1, 4))
    print("ROC AUC:", round(train_auc, 4))
    print("\nTest metrics:")
    print("Accuracy:", round(test_accuracy, 4))
    print("F1 Score:", round(test_f1, 4))
    print("ROC AUC:", round(test_auc, 4))

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train set confusion matrix
    cm_train = confusion_matrix(y_train, pred_train)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=['Incorrect', 'Correct'])
    disp_train.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix (Train Set)')
    
    # Test set confusion matrix
    cm_test = confusion_matrix(y_test, pred_test)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['Incorrect', 'Correct'])
    disp_test.plot(ax=ax2, cmap='Blues')
    ax2.set_title('Confusion Matrix (Test Set)')
    
    plt.tight_layout()
    plt.show()

    # Precision-Recall curve for the test set
    y_scores = best_xgb.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title("Precision-Recall Curve (Test Set)")
    plt.show()

    # Feature importance analysis
    feature_names = numeric_feature_cols
    importances = best_xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 Feature Importances:")
    for f in range(min(20, len(feature_names))):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Most Important Features", pad=20, fontsize=14)
    
    bars = plt.barh(range(19, -1, -1), importances[indices[:20]], color='skyblue', alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.yticks(range(19, -1, -1), [feature_names[i] for i in indices[:20]],
               ha='right', fontsize=10)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    irtxgboostclassifier()
