import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from adjustText import adjust_text
import seaborn as sns
import os

def load_data():
    """
    Load and prepare all necessary data for the IRT-XGBoost classifier.
    Includes original features, odd one out features, and their differences.
    
    Returns:
    --------
    tuple
        (features_final, participant_responses, item_bank)
    """
    cache_path_full = "modelling_paper/data_cache.pkl"
    cache_path_merged = "modelling_paper/data_cache_merged.pkl"

    # Try loading fully processed cache first
    if os.path.exists(cache_path_full):
        print("Loading fully processed data from cache...")
        features_final, participant_responses, item_bank = pd.read_pickle(cache_path_full)
        return features_final, participant_responses, item_bank

    # Try loading merged cache
    if os.path.exists(cache_path_merged):
        print("Loading merged data from cache...")
        data_with_irt, participant_responses, item_bank = pd.read_pickle(cache_path_merged)
    else:
        print("Loading item bank...")
        item_bank = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/item-bank.csv")
        item_bank = item_bank.rename(columns={'id': 'item_id'})
        print(f"Loaded item bank with {len(item_bank)} items")

        print("Loading participant responses...")
        participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=1e7)
        participant_responses = participant_responses[participant_responses['test'] == 'mdt']
        print(f"Loaded {len(participant_responses)} participant responses (test == 'mdt')")

        print("Loading original features...")
        original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/testing.csv")
        melody_ids = original_features['melody_id'].copy()
        print(f"Loaded original features: {original_features.shape}")

        print("Loading odd one out features...")
        odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/miq_mels2.csv")
        print(f"Loaded odd one out features: {odd_one_out_features.shape}")

        # Remove non-numeric columns and melody_id
        original_features = original_features.select_dtypes(include=[np.number])
        odd_one_out_features = odd_one_out_features.select_dtypes(include=[np.number])
        print(f"Numeric original features: {original_features.shape}")
        print(f"Numeric odd one out features: {odd_one_out_features.shape}")

        # Ensure both feature sets have the same columns
        common_cols = list(set(original_features.columns) & set(odd_one_out_features.columns))
        original_features = original_features[common_cols]
        odd_one_out_features = odd_one_out_features[common_cols]
        print(f"Aligned features columns: {len(common_cols)}")

        # Remove zero variance columns
        zero_var_cols_orig = original_features.columns[original_features.var() == 0]
        zero_var_cols_odd = odd_one_out_features.columns[odd_one_out_features.var() == 0]
        zero_var_cols = list(set(zero_var_cols_orig) | set(zero_var_cols_odd))
        original_features = original_features.drop(columns=zero_var_cols)
        odd_one_out_features = odd_one_out_features.drop(columns=zero_var_cols)
        print(f"Removed zero variance columns: {len(zero_var_cols)}")

        # Remove tempo features
        tempo_cols_orig = [col for col in original_features.columns if 'duration_features.tempo' in col]
        tempo_cols_odd = [col for col in odd_one_out_features.columns if 'duration_features.tempo' in col]
        tempo_cols = list(set(tempo_cols_orig) | set(tempo_cols_odd))
        original_features = original_features.drop(columns=tempo_cols)
        odd_one_out_features = odd_one_out_features.drop(columns=tempo_cols)
        print(f"Removed tempo columns: {len(tempo_cols)}")

        # Calculate feature differences
        feature_diffs = original_features - odd_one_out_features
        print(f"Feature differences shape: {feature_diffs.shape}")

        # Scale features
        imputer = SimpleImputer(strategy='mean')
        original_features_scaled = pd.DataFrame(
            imputer.fit_transform(original_features),
            columns=[f'orig_{col}' for col in original_features.columns]
        )
        odd_one_out_features_scaled = pd.DataFrame(
            imputer.fit_transform(odd_one_out_features),
            columns=[f'odd_{col}' for col in odd_one_out_features.columns]
        )
        feature_diffs_scaled = pd.DataFrame(
            imputer.fit_transform(feature_diffs),
            columns=[f'diff_{col}' for col in feature_diffs.columns]
        )
        print("Scaled features.")

        # Combine all features
        features_final = pd.concat([
            original_features_scaled,
            odd_one_out_features_scaled,
            feature_diffs_scaled
        ], axis=1)
        features_final['melody_id'] = melody_ids.values
        print(f"Combined features_final shape: {features_final.shape}")

        # Get IRT features for each trial
        irt_cols = ['item_id', 'ability_WL', 'score'] 
        irt_features = participant_responses[irt_cols].copy()
        print(f"IRT features shape: {irt_features.shape}")

        # Direct merge on melody_id and item_id
        data_with_irt = features_final.merge(irt_features, left_on='melody_id', right_on='item_id', how='right')
        print(f"Shape of data_with_irt after merge: {data_with_irt.shape}")

        # Now merge with item_bank
        data_with_irt = data_with_irt.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')
        print(f"Shape after merging with item_bank: {data_with_irt.shape}")

        # Save intermediate cache
        pd.to_pickle((data_with_irt, participant_responses, item_bank), cache_path_merged)
        print("Saved merged data to cache.")

    # Continue processing from merged data
    print("Continuing processing from merged data...")
    # Convert oddity to categorical and create dummy variables
    data_with_irt['oddity'] = data_with_irt['oddity'].astype('category')
    oddity_dummies = pd.get_dummies(data_with_irt['oddity'], prefix='oddity')
    data_with_irt = pd.concat([data_with_irt, oddity_dummies], axis=1)
    print(f"Shape after adding oddity dummies: {data_with_irt.shape}")

    # Prepare feature columns
    exclude_cols = {'melody_id', 'item_id', 'score', 'oddity'}
    feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]
    numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    irt_feature_cols = ['ability_WL'] + [col for col in numeric_feature_cols if col.startswith('oddity_')]
    print(f"Number of numeric feature columns: {len(numeric_feature_cols)}")

    # Prepare X and y
    X = data_with_irt[numeric_feature_cols].values
    y = data_with_irt['score'].values
    groups = data_with_irt['item_id'].values
    print(f"Shape of X: {X.shape}, y: {y.shape}, groups: {groups.shape}")

    # Handle NaNs
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    print("Imputed missing values in X.")

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]
    print(f"Split data: X_train {X_train.shape}, X_test {X_test.shape}")

    # Impute train/test separately
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print("Imputed train/test separately.")

    # Sample X_train if too large for 32-bit LAPACK
    max_elements = 2_000_000_000  # safety margin below 2^31-1
    n_elements = X_train.shape[0] * X_train.shape[1]
    if n_elements > max_elements:
        max_rows = max_elements // X_train.shape[1]
        print(f"Sampling X_train from {X_train.shape[0]} to {max_rows} rows to fit 32-bit LAPACK limit.")
        sample_idx = np.random.choice(X_train.shape[0], max_rows, replace=False)
        X_train = X_train[sample_idx]
        y_train = y_train[sample_idx]
        X_train_groups = X_train_groups[sample_idx]
        print(f"Sampled X_train shape: {X_train.shape}")

    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=0.95, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"Number of PCA components: {X_train_pca.shape[1]}")

    # Add ability_WL and oddity dummies back after PCA
    ability_WL_train = data_with_irt.iloc[train_idx]['ability_WL'].values.reshape(-1, 1)
    ability_WL_test = data_with_irt.iloc[test_idx]['ability_WL'].values.reshape(-1, 1)
    oddity_cols = [col for col in data_with_irt.columns if col.startswith('oddity_')]
    oddity_train = data_with_irt.iloc[train_idx][oddity_cols].values
    oddity_test = data_with_irt.iloc[test_idx][oddity_cols].values
    X_train_final = np.hstack((X_train_pca, ability_WL_train, oddity_train))
    X_test_final = np.hstack((X_test_pca, ability_WL_test, oddity_test))
    print(f"Final train/test shapes: {X_train_final.shape}, {X_test_final.shape}")

    # Save fully processed cache
    features_final = features_final  # for compatibility with rest of code
    pd.to_pickle((features_final, participant_responses, item_bank), cache_path_full)
    print("Saved fully processed data to cache.")
    return features_final, participant_responses, item_bank

def irtxgboost_classifier(features_final, participant_responses, item_bank):
    """
    XGBoost classifier incorporating IRT features and PCA transformation for raw score prediction.
    Predicts scores for each individual trial using melody features.
    
    Parameters:
    -----------
    features_final : pd.DataFrame
        DataFrame containing the final features
    participant_responses : pd.DataFrame
        DataFrame containing participant responses and IRT features
    item_bank : pd.DataFrame
        DataFrame containing item bank information
    
    Returns:
    --------
    dict
        Dictionary containing model performance metrics and predictions
    """
    print("\nRunning XGBoost Classifier with IRT features and PCA...")

    # Get IRT features for each trial
    irt_cols = ['item_id', 'ability_WL', 'score'] 
    irt_features = participant_responses[irt_cols].copy()

    # Direct merge on melody_id and item_id (like updated_analysis.py)
    data_with_irt = features_final.reset_index().merge(irt_features, left_on='melody_id', right_on='item_id', how='right')

    # Now merge with item_bank
    data_with_irt = data_with_irt.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')

    # Print shape for debugging
    print(f"Shape of data_with_irt after merge: {data_with_irt.shape}")

    # Convert oddity to categorical and create dummy variables
    data_with_irt['oddity'] = data_with_irt['oddity'].astype('category')
    oddity_dummies = pd.get_dummies(data_with_irt['oddity'], prefix='oddity')
    data_with_irt = pd.concat([data_with_irt, oddity_dummies], axis=1)

    # Prepare feature columns
    exclude_cols = {'melody_id', 'item_id', 'score', 'oddity'}
    feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]
    numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    irt_feature_cols = ['ability_WL'] + [col for col in numeric_feature_cols if col.startswith('oddity_')]

    # Prepare X and y
    X = data_with_irt[numeric_feature_cols].values
    y = data_with_irt['score'].values
    groups = data_with_irt['item_id'].values  # Use item_id for grouping to ensure same items stay together

    # Handle NaNs
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    # Impute train/test separately
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Sample X_train if too large for 32-bit LAPACK
    max_elements = 2_000_000_000  # safety margin below 2^31-1
    n_elements = X_train.shape[0] * X_train.shape[1]
    if n_elements > max_elements:
        max_rows = max_elements // X_train.shape[1]
        print(f"Sampling X_train from {X_train.shape[0]} to {max_rows} rows to fit 32-bit LAPACK limit.")
        sample_idx = np.random.choice(X_train.shape[0], max_rows, replace=False)
        X_train = X_train[sample_idx]
        y_train = y_train[sample_idx]
        X_train_groups = X_train_groups[sample_idx]
        print(f"Sampled X_train shape: {X_train.shape}")

    # Apply PCA
    print("Applying PCA...")
    pca = PCA(n_components=0.95, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"Number of PCA components: {X_train_pca.shape[1]}")

    # Add ability_WL and oddity dummies back after PCA
    ability_WL_train = data_with_irt.iloc[train_idx]['ability_WL'].values.reshape(-1, 1)
    ability_WL_test = data_with_irt.iloc[test_idx]['ability_WL'].values.reshape(-1, 1)
    
    oddity_cols = [col for col in data_with_irt.columns if col.startswith('oddity_')]
    oddity_train = data_with_irt.iloc[train_idx][oddity_cols].values
    oddity_test = data_with_irt.iloc[test_idx][oddity_cols].values
    
    X_train_final = np.hstack((X_train_pca, ability_WL_train, oddity_train))
    X_test_final = np.hstack((X_test_pca, ability_WL_test, oddity_test))

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

    # Create validation set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, val_idx in gss.split(X_train_final, y_train, groups=X_train_groups):
        X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        groups_tr = X_train_groups[train_idx]

    # Set up GroupKFold cross validation
    gkf = GroupKFold(n_splits=5)

    # Check for cached parameters
    cache_file = 'modelling_paper/xgb_irt_pca_best_params.npy'
    try:
        best_params = np.load(cache_file, allow_pickle=True).item()
        print("\nLoading cached best parameters:", best_params)
        best_xgb = XGBClassifier(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(X_tr, y_tr)
        grid_search = None
    except:
        print("\nNo cached parameters found. Running grid search...")
        grid_search = GridSearchCV(
            xgb,
            param_grid, 
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_tr, y_tr)
        np.save(cache_file, grid_search.best_params_)
        best_xgb = grid_search.best_estimator_

    if grid_search is not None:
        print("\nXGBoost Classifier with IRT features and PCA - optimized parameters:")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV accuracy:", round(grid_search.best_score_, 4))
    else:
        print("\nXGBoost Classifier with IRT features and PCA - loaded cached parameters.")

    # Evaluate model
    pred_train = best_xgb.predict(X_train_final)
    pred_test = best_xgb.predict(X_test_final)
    
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)

    # Probabilities for ROC curve
    if hasattr(best_xgb, "predict_proba"):
        y_test_proba = best_xgb.predict_proba(X_test_final)[:, 1]
    else:
        y_test_proba = best_xgb.decision_function(X_test_final)

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # Precision, recall, F1
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)

    print("\nModel Performance:")
    print("\nTrain metrics:")
    print("Accuracy:", round(train_accuracy, 4))
    print("\nTest metrics:")
    print("Accuracy:", round(test_accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    # Plot confusion matrices side by side
    plt.figure(figsize=(15, 6))
    def plot_confusion_matrix(cm, title):
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)'
        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap='Blues',
            cbar=cm.max() > 10,  # Only show colorbar if counts are large
            vmin=0,
            vmax=cm.max() if cm.max() > 0 else 1,
            xticklabels=['Incorrect', 'Correct'],
            yticklabels=['Incorrect', 'Correct'],
            annot_kws={"size": 18, "weight": 'bold'},
            linewidths=1,
            linecolor='gray',
            square=True
        )
        plt.title(title, fontsize=18)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    plt.subplot(1, 2, 1)
    cm_train = confusion_matrix(y_train, pred_train)
    plot_confusion_matrix(cm_train, 'Train Confusion Matrix')
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, pred_test)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Feature importance analysis
    feature_names = [f'PC{i+1}' for i in range(X_train_pca.shape[1])] + ['ability_WL'] + oddity_cols
    importances = best_xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop Feature Importances (including ability_WL and oddity categories):")
    for f in range(len(feature_names)):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances (PCA Components + IRT Features)", pad=20, fontsize=14)
    
    bars = plt.barh(range(len(importances)-1, -1, -1), importances[indices], color='skyblue', alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.yticks(range(len(importances)-1, -1, -1), [feature_names[i] for i in indices],
               ha='right', fontsize=10)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        'model': best_xgb,
        'pca': pca,
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        },
        'predictions': {
            'train': pred_train,
            'test': pred_test
        },
        'feature_importances': dict(zip(feature_names, importances))
    }

def irtxgboost_classifier_no_pca(features_final, participant_responses, item_bank):
    print("\nRunning XGBoost Classifier with IRT features (No PCA)...")

    irt_cols = ['item_id', 'ability_WL', 'score']
    irt_features = participant_responses[irt_cols].copy()
    data_with_irt = features_final.reset_index().merge(irt_features, left_on='melody_id', right_on='item_id', how='right')
    data_with_irt = data_with_irt.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')
    print(f"Shape of data_with_irt after merge: {data_with_irt.shape}")

    data_with_irt['oddity'] = data_with_irt['oddity'].astype('category')
    oddity_dummies = pd.get_dummies(data_with_irt['oddity'], prefix='oddity')
    data_with_irt = pd.concat([data_with_irt, oddity_dummies], axis=1)

    exclude_cols = {'melody_id', 'item_id', 'score', 'oddity'}
    feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]
    numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    irt_feature_cols = ['ability_WL'] + [col for col in numeric_feature_cols if col.startswith('oddity_')]

    X = data_with_irt[numeric_feature_cols].values
    y = data_with_irt['score'].values
    groups = data_with_irt['item_id'].values

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    max_elements = 2_000_000_000
    n_elements = X_train.shape[0] * X_train.shape[1]
    if n_elements > max_elements:
        max_rows = max_elements // X_train.shape[1]
        print(f"Sampling X_train from {X_train.shape[0]} to {max_rows} rows to fit 32-bit LAPACK limit.")
        sample_idx = np.random.choice(X_train.shape[0], max_rows, replace=False)
        X_train = X_train[sample_idx]
        y_train = y_train[sample_idx]
        X_train_groups = X_train_groups[sample_idx]
        print(f"Sampled X_train shape: {X_train.shape}")

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

    # Create validation set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, val_idx in gss.split(X_train, y_train, groups=X_train_groups):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        groups_tr = X_train_groups[train_idx]

    # Set up GroupKFold cross validation
    gkf = GroupKFold(n_splits=5)

    # Check for cached parameters
    cache_file = 'modelling_paper/xgb_irt_no_pca_best_params.npy'
    try:
        best_params = np.load(cache_file, allow_pickle=True).item()
        print("\nLoading cached best parameters:", best_params)
        best_xgb = XGBClassifier(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(X_tr, y_tr)
        grid_search = None
    except:
        print("\nNo cached parameters found. Running grid search...")
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_tr, y_tr)
        np.save(cache_file, grid_search.best_params_)
        best_xgb = grid_search.best_estimator_

    if grid_search is not None:
        print("\nXGBoost Classifier with IRT features (No PCA) - optimized parameters:")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV accuracy:", round(grid_search.best_score_, 4))
    else:
        print("\nXGBoost Classifier with IRT features (No PCA) - loaded cached parameters.")

    # Evaluate model
    pred_train = best_xgb.predict(X_train)
    pred_test = best_xgb.predict(X_test)
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)

    # Probabilities for ROC curve
    if hasattr(best_xgb, "predict_proba"):
        y_test_proba = best_xgb.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = best_xgb.decision_function(X_test)

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # Precision, recall, F1
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)

    print("\nModel Performance:")
    print("\nTrain metrics:")
    print("Accuracy:", round(train_accuracy, 4))
    print("\nTest metrics:")
    print("Accuracy:", round(test_accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    # Plot confusion matrices side by side
    plt.figure(figsize=(15, 6))
    def plot_confusion_matrix(cm, title):
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)'
        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap='Blues',
            cbar=cm.max() > 10,
            vmin=0,
            vmax=cm.max() if cm.max() > 0 else 1,
            xticklabels=['Incorrect', 'Correct'],
            yticklabels=['Incorrect', 'Correct'],
            annot_kws={"size": 18, "weight": 'bold'},
            linewidths=1,
            linecolor='gray',
            square=True
        )
        plt.title(title, fontsize=18)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    plt.subplot(1, 2, 1)
    cm_train = confusion_matrix(y_train, pred_train)
    plot_confusion_matrix(cm_train, 'Train Confusion Matrix')
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, pred_test)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        'model': best_xgb,
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        },
        'predictions': {
            'train': pred_train,
            'test': pred_test
        }
    }

def irtxgboost_classifier_irt_only(participant_responses, item_bank):
    print("\nRunning XGBoost Classifier with IRT features only (ability_WL and oddity)...")

    irt_cols = ['item_id', 'ability_WL', 'score']
    irt_features = participant_responses[irt_cols].copy()
    data_with_irt = irt_features.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')
    print(f"Shape of data_with_irt after merge: {data_with_irt.shape}")

    data_with_irt['oddity'] = data_with_irt['oddity'].astype('category')
    oddity_dummies = pd.get_dummies(data_with_irt['oddity'], prefix='oddity')
    data_with_irt = pd.concat([data_with_irt, oddity_dummies], axis=1)

    exclude_cols = {'item_id', 'score', 'oddity'}
    feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]
    numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    irt_feature_cols = ['ability_WL'] + [col for col in numeric_feature_cols if col.startswith('oddity_')]

    X = data_with_irt[numeric_feature_cols].values
    y = data_with_irt['score'].values
    groups = data_with_irt['item_id'].values

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    max_elements = 2_000_000_000
    n_elements = X_train.shape[0] * X_train.shape[1]
    if n_elements > max_elements:
        max_rows = max_elements // X_train.shape[1]
        print(f"Sampling X_train from {X_train.shape[0]} to {max_rows} rows to fit 32-bit LAPACK limit.")
        sample_idx = np.random.choice(X_train.shape[0], max_rows, replace=False)
        X_train = X_train[sample_idx]
        y_train = y_train[sample_idx]
        X_train_groups = X_train_groups[sample_idx]
        print(f"Sampled X_train shape: {X_train.shape}")

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

    # Create validation set
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, val_idx in gss.split(X_train, y_train, groups=X_train_groups):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        groups_tr = X_train_groups[train_idx]

    # Set up GroupKFold cross validation
    gkf = GroupKFold(n_splits=5)

    # Check for cached parameters
    cache_file = 'modelling_paper/xgb_irt_only_best_params.npy'
    try:
        best_params = np.load(cache_file, allow_pickle=True).item()
        print("\nLoading cached best parameters:", best_params)
        best_xgb = XGBClassifier(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(X_tr, y_tr)
        grid_search = None
    except:
        print("\nNo cached parameters found. Running grid search...")
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_tr, y_tr)
        np.save(cache_file, grid_search.best_params_)
        best_xgb = grid_search.best_estimator_

    if grid_search is not None:
        print("\nXGBoost Classifier with IRT features only - optimized parameters:")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV accuracy:", round(grid_search.best_score_, 4))
    else:
        print("\nXGBoost Classifier with IRT features only - loaded cached parameters.")

    # Evaluate model
    pred_train = best_xgb.predict(X_train)
    pred_test = best_xgb.predict(X_test)
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)

    # Probabilities for ROC curve
    if hasattr(best_xgb, "predict_proba"):
        y_test_proba = best_xgb.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = best_xgb.decision_function(X_test)

    # ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)

    # Precision, recall, F1
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)

    print("\nModel Performance:")
    print("\nTrain metrics:")
    print("Accuracy:", round(train_accuracy, 4))
    print("\nTest metrics:")
    print("Accuracy:", round(test_accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    # Plot confusion matrices side by side
    plt.figure(figsize=(15, 6))
    def plot_confusion_matrix(cm, title):
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_percent = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]} ({cm_percent[i, j]:.1f}%)'
        sns.heatmap(
            cm,
            annot=annot,
            fmt='',
            cmap='Blues',
            cbar=cm.max() > 10,
            vmin=0,
            vmax=cm.max() if cm.max() > 0 else 1,
            xticklabels=['Incorrect', 'Correct'],
            yticklabels=['Incorrect', 'Correct'],
            annot_kws={"size": 18, "weight": 'bold'},
            linewidths=1,
            linecolor='gray',
            square=True
        )
        plt.title(title, fontsize=18)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

    plt.subplot(1, 2, 1)
    cm_train = confusion_matrix(y_train, pred_train)
    plot_confusion_matrix(cm_train, 'Train Confusion Matrix')
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, pred_test)
    plot_confusion_matrix(cm_test, 'Test Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        'model': best_xgb,
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        },
        'predictions': {
            'train': pred_train,
            'test': pred_test
        }
    }

if __name__ == "__main__":
    # Load data
    features_final, participant_responses, item_bank = load_data()
    
    # Run classifier
    results = irtxgboost_classifier_irt_only(participant_responses, item_bank)
    
    # Print final results
    print("\nFinal Results:")
    print("Train Accuracy:", round(results['metrics']['train_accuracy'], 4))
    print("Test Accuracy:", round(results['metrics']['test_accuracy'], 4))

