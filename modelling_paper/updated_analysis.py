import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from tqdm import tqdm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import optuna
import json
import os
from datetime import datetime
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer

def get_final_ability_WL(participant_responses):
    """Get the final ability_WL value for each participant and update all earlier occurrences."""
    # Sort by participant and timestamp to ensure chronological order
    sorted_responses = participant_responses.sort_values(['user_id', 'timestamp'])
    
    # Get the final ability_WL value for each participant
    final_abilities = sorted_responses.groupby('user_id')['ability_WL'].last()
    
    # Create a mapping of user_id to their final ability_WL
    ability_map = dict(zip(final_abilities.index, final_abilities.values))
    
    # Create a copy of the responses to avoid modifying the original
    updated_responses = participant_responses.copy()
    
    # Update all ability_WL values to the final value for each participant
    updated_responses['ability_WL'] = updated_responses['user_id'].map(ability_map)
    
    return updated_responses

# 1. Load data
print("Loading original features...")
original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/testing.csv")
print("Loading odd one out features...")
odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/miq_mels2.csv")
print("Loading participant responses...")
participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv")
print("Loading item bank...")
item_bank = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/item-bank.csv")
item_bank = item_bank.rename(columns={'id': 'item_id'})

# Filter participant_responses to only include 'mdt' test
participant_responses = participant_responses[participant_responses['test'] == 'mdt']

# Update ability_WL values to use final values for each participant
participant_responses = get_final_ability_WL(participant_responses)

# Compute mean score per melody
mean_scores = participant_responses.groupby('item_id')['score'].mean().reset_index()
mean_scores = mean_scores.rename(columns={'score': 'mean_score'})

# Calculate basic statistics for mean scores
mean_score = mean_scores['mean_score'].mean()
std_score = mean_scores['mean_score'].std()
min_score = mean_scores['mean_score'].min()
max_score = mean_scores['mean_score'].max()

print("\nBasic Statistics for Mean Scores per Melody:")
print(f"Mean: {mean_score:.3f}")
print(f"Standard Deviation: {std_score:.3f}")
print(f"Range: [{min_score:.3f}, {max_score:.3f}]")

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

# Add oddity score from item bank
data_with_irt = data_with_irt.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')

# Prepare X and y
exclude_cols = {'melody_id', 'item_id', 'score'}
feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]

# Only keep numeric columns for modeling
numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [col for col in feature_cols if col not in numeric_feature_cols]
if non_numeric_cols:
    print("Dropping non-numeric columns:", non_numeric_cols)

X = data_with_irt[numeric_feature_cols].values
y = data_with_irt['score'].values  # Changed from 'mean_score' to 'score'

# Train/test split by melody_id
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
groups = data_with_irt['melody_id'].values
for train_idx, test_idx in gss.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train_indices, X_test_indices = train_idx, test_idx

X_train_groups = groups[X_train_indices]

# Print shapes
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 3. PCA
# Handle NaNs before PCA (impute with mean)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

pca = PCA(n_components=0.95, svd_solver='full')  # keep 95% variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"Number of PCA components: {X_train_pca.shape[1]}")
# 4. Linear Regression (no random effect)

def linear_regression_pca():
    linreg = LinearRegression()
    linreg.fit(X_train_pca, y_train)

    # Predict and evaluate on train set
    pred_train = linreg.predict(X_train_pca)
    r2_train = r2_score(y_train, pred_train)

    # Predict and evaluate on test set
    pred_test = linreg.predict(X_test_pca)
    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)

    print("\nLinear Regression:")
    print("Train R-squared:", round(r2_train, 4))
    print("\nTest metrics:")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))
    # Calculate adjusted R-squared
    n = X_test.shape[0]  # number of observations
    p = X_test_pca.shape[1]  # number of predictors
    adj_r2_test = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
    print("Adjusted R-squared:", round(adj_r2_test, 4))


def random_forest_pca():
    # 5. Random Forest with PCA components
    # Define parameter grid for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [5, 10, 20],
        'min_samples_split': [5, 10, 20],
        'max_features': ['sqrt', 0.5]
    }

    # Use the same user_ids for grouping as in your split
    gkf = GroupKFold(n_splits=3)

    rf = RandomForestRegressor(random_state=8, n_jobs=-1)
    rf_grid_search = GridSearchCV(
        estimator=rf,
        param_grid=rf_param_grid,
        cv=gkf.split(X_train_pca, y_train, groups=groups),
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )

    print("Starting Random Forest grid search...")
    rf_grid_search.fit(X_train_pca, y_train)
    print("Best parameters:", rf_grid_search.best_params_)
    print("Best CV score (MSE):", -rf_grid_search.best_score_)

    # Use the best estimator to predict on the test set
    best_rf = rf_grid_search.best_estimator_
    rf_pred_test = best_rf.predict(X_test_pca)

    rf_mse_test = mean_squared_error(y_test, rf_pred_test)
    rf_rmse_test = np.sqrt(rf_mse_test)
    rf_r2_test = r2_score(y_test, rf_pred_test)

    print("\nRandom Forest (Test) with best params:")
    print("MSE:", round(rf_mse_test, 4))
    print("RMSE:", round(rf_rmse_test, 4))
    print("R-squared:", round(rf_r2_test, 4))

    # Plot actual vs predicted for Random Forest test set
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, rf_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Random Forest (Tuned): Actual vs Predicted Mean Scores (Test Set)")
    plt.show()

def xgboost_pca():
    # Define a more regularized parameter grid for XGBoost
    xgb_param_grid = {
        'n_estimators': [25, 50, 100],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0],
        'min_child_weight': [5, 10, 20],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10]
    }
    # Use the same user_ids for grouping as in your split
    gkf = GroupKFold(n_splits=3)

    # Split off a validation set from the training data for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(X_train_pca, y_train, test_size=0.2, random_state=8)

    xgb = XGBRegressor(random_state=8, n_jobs=-1)
    xgb_grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=xgb_param_grid,
        cv=gkf.split(X_tr, y_tr, groups=groups),
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )

    print("Starting XGBoost grid search with regularization and early stopping...")
    xgb_grid_search.fit(X_tr, y_tr)
    print("Best parameters:", xgb_grid_search.best_params_)
    print("Best CV score (MSE):", -xgb_grid_search.best_score_)

    # Use the best estimator to predict on train and test sets
    best_xgb = xgb_grid_search.best_estimator_
    xgb_pred_train = best_xgb.predict(X_train_pca)
    xgb_pred_test = best_xgb.predict(X_test_pca)

    # Calculate metrics for train set
    xgb_mse_train = mean_squared_error(y_train, xgb_pred_train)
    xgb_rmse_train = np.sqrt(xgb_mse_train)
    xgb_r2_train = r2_score(y_train, xgb_pred_train)

    print("\nXGBoost (Train) with best params:")
    print("MSE:", round(xgb_mse_train, 4))
    print("RMSE:", round(xgb_rmse_train, 4))
    print("R-squared:", round(xgb_r2_train, 4))

    # Calculate metrics for test set
    xgb_mse_test = mean_squared_error(y_test, xgb_pred_test)
    xgb_rmse_test = np.sqrt(xgb_mse_test)
    xgb_r2_test = r2_score(y_test, xgb_pred_test)

    print("\nXGBoost (Test) with best params:")
    print("MSE:", round(xgb_mse_test, 4))
    print("RMSE:", round(xgb_rmse_test, 4))
    print("R-squared:", round(xgb_r2_test, 4))

    # Plot actual vs predicted for XGBoost test set
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, xgb_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("XGBoost (Tuned & Regularized): Actual vs Predicted Mean Scores (Test Set)")
    plt.show()

def ridge_pca():
    from sklearn.linear_model import RidgeCV

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    ridge.fit(X_train_pca, y_train)
    pred_test = ridge.predict(X_test_pca)

    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)
    print("\nRidge Regression (Test):")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))

def lasso_pca():
    from sklearn.linear_model import LassoCV

    lasso = LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000, random_state=8)
    lasso.fit(X_train_pca, y_train)
    pred_test = lasso.predict(X_test_pca)

    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)
    print("\nLasso Regression (Test):")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))
    
def elasticnet_pca():
    from sklearn.linear_model import ElasticNetCV

    elasticnet = ElasticNetCV(l1_ratio=np.linspace(0.1, 1.0, 10), alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000, random_state=8)
    elasticnet.fit(X_train_pca, y_train)
    pred_test = elasticnet.predict(X_test_pca)

    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)
    print("\nElasticNet Regression (Test):")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))

def visualize_predictions():
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RidgeCV
    from sklearn.linear_model import LassoCV
    from sklearn.linear_model import ElasticNetCV

    # Create predictions for each model
    linreg = LinearRegression()
    linreg.fit(X_train_pca, y_train)
    linreg_pred = linreg.predict(X_test_pca)

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
    ridge.fit(X_train_pca, y_train) 
    ridge_pred = ridge.predict(X_test_pca)

    lasso = LassoCV(alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000, random_state=8)
    lasso.fit(X_train_pca, y_train)
    lasso_pred = lasso.predict(X_test_pca)

    elasticnet = ElasticNetCV(l1_ratio=np.linspace(0.1, 1.0, 10), alphas=np.logspace(-3, 1, 20), cv=5, max_iter=10000, random_state=8)
    elasticnet.fit(X_train_pca, y_train)
    elasticnet_pred = elasticnet.predict(X_test_pca)

    # Create subplot with all predictions
    plt.figure(figsize=(12, 8))

    plt.scatter(y_test, linreg_pred, alpha=0.5, label='Linear Regression')
    plt.scatter(y_test, ridge_pred, alpha=0.5, label='Ridge')
    plt.scatter(y_test, lasso_pred, alpha=0.5, label='Lasso') 
    plt.scatter(y_test, elasticnet_pred, alpha=0.5, label='ElasticNet')

    # Add diagonal line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='Perfect Prediction')

    plt.xlabel("Actual Score")
    plt.ylabel("Predicted Score")
    plt.title("Regression Models: Actual vs Predicted Scores (Test Set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# linear_regression_pca()
# ridge_pca()
# lasso_pca()
# elasticnet_pca()
def random_forest_raw():
    # Use X_train, X_test (raw, scaled features) instead of X_train_pca, X_test_pca
    rf = RandomForestRegressor(n_estimators=200, max_depth=None, max_features='sqrt', min_samples_split=2, min_samples_leaf=1, random_state=8)
    
    # Get IRT ability_WL feature
    irt_cols = ['item_id', 'ability_WL']
    irt_features = participant_responses[irt_cols].copy()
    irt_features = irt_features.groupby('item_id').agg({
        'ability_WL': 'mean'
    }).reset_index()
    
    # Merge with existing features (fix: merge into a new variable)
    data_with_irt_rf = data_with_irt.merge(irt_features, left_on='melody_id', right_on='item_id', how='left')
    
    # Add ability_WL to feature set
    X_with_irt = np.hstack((X, data_with_irt_rf['ability_WL'].values.reshape(-1,1)))
    X_train_with_irt = X_with_irt[X_train_indices]
    X_test_with_irt = X_with_irt[X_test_indices]
    
    rf.fit(X_train_with_irt, y_train)
    
    # Get train and test predictions
    rf_pred_train = rf.predict(X_train_with_irt)
    rf_pred_test = rf.predict(X_test_with_irt)
    
    # Calculate and print R2 scores
    print("Random Forest (Raw Features + ability_WL) Train R2:", r2_score(y_train, rf_pred_train))
    print("Random Forest (Raw Features + ability_WL) Test R2:", r2_score(y_test, rf_pred_test))

    # Create feature names for original, odd-one-out, diff and IRT features
    feature_names = (
        [f'original_{col}' for col in original_features.columns] +
        [f'odd_one_out_{col}' for col in odd_one_out_features.columns] +
        [f'diff_{col}' for col in feature_diffs.columns] +
        ['ability_WL']
    )

    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop 20 Feature Importances:")
    for f in range(20):
        print("%d. %s (%f)" % (f, feature_names[indices[f]], importances[indices[f]]))

    # Plot top 20 feature importances
    plt.figure(figsize=(15,6))
    plt.title("Top 20 Feature Importances")
    plt.bar(range(20), importances[indices[:20]])
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot predictions vs actuals for both train and test sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train set plot
    ax1.scatter(y_train, rf_pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    ax1.set_title("Random Forest: Train Set")
    
    # Test set plot
    ax2.scatter(y_test, rf_pred_test, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title("Random Forest: Test Set")
    
    plt.tight_layout()
    plt.show()

    # Assuming you have these columns in your data DataFrame
    train_ids = set(data_with_irt.iloc[X_train_indices]['item_id'])
    test_ids = set(data_with_irt.iloc[X_test_indices]['item_id'])
    overlap = train_ids.intersection(test_ids)
    print(f"Number of overlapping item_ids: {len(overlap)}")
    # Print total number of unique item IDs
    total_items = len(train_ids.union(test_ids))
    print(f"Total number of unique item_ids: {total_items}")

def visualize_distributions():
    # Plot histograms of y_train and y_test to compare their distributions
    plt.figure(figsize=(10, 5))
    plt.hist(y_train, bins=30, alpha=0.6, label='y_train')
    plt.hist(y_test, bins=30, alpha=0.6, label='y_test')
    plt.xlabel('Mean Score')
    plt.ylabel('Count')
    plt.title('Distribution of Mean Scores in Train and Test Sets')
    plt.legend()
    plt.tight_layout()
    plt.show()

def optimize_random_forest(X, y, groups):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5, None]
    }
    rf = RandomForestRegressor(random_state=8, n_jobs=-1)
    gkf = GroupKFold(n_splits=3)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=gkf.split(X, y, groups=groups),
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X, y)
    print("Best parameters:", grid_search.best_params_)
    print("Best CV R2:", grid_search.best_score_)
    return grid_search.best_estimator_

def example_optimize_random_forest():
    # Example usage
    best_rf = optimize_random_forest(X_train, y_train, groups=X_train_groups)

    # Train the best model
    best_rf.fit(X_train, y_train)

    # Get predictions and evaluate on test set
    pred_test = best_rf.predict(X_test)
    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)

    print("\nBest Random Forest (Test) with optimized params:")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))

def xgboost_raw():
    # Initialize XGBoost regressor
    xgb = XGBRegressor(random_state=8, n_jobs=-1)

    # Check if we have cached best parameters
    cache_file = 'modelling_paper/xgb_best_params.npy'
    try:
        best_params = np.load(cache_file, allow_pickle=True).item()
        print("\nLoading cached best parameters:", best_params)
        best_xgb = XGBRegressor(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(X_train, y_train)
    except:
        print("\nNo cached parameters found. Running grid search...")
        # Define expanded parameter grid
        param_grid = {'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

        # Create validation set for early stopping while preserving groups
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
        for train_idx, val_idx in gss.split(X_train, y_train, groups=X_train_groups):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            groups_tr = X_train_groups[train_idx]
        
        # Set up GroupKFold cross validation with more splits
        gkf = GroupKFold(n_splits=5)
        
        # Perform grid search with group k-fold CV
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        
        # Fit grid search
        grid_search.fit(X_tr, y_tr)
        
        print("\nXGBoost with optimized parameters:")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV R2:", round(grid_search.best_score_, 4))

        # Cache the best parameters
        np.save(cache_file, grid_search.best_params_)

        # Get best model and retrain with early stopping
        best_xgb = XGBRegressor(random_state=8, n_jobs=-1, **grid_search.best_params_)
        best_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    # Predict and evaluate on train set
    pred_train = best_xgb.predict(X_train)
    r2_train = r2_score(y_train, pred_train)

    # Predict and evaluate on test set
    pred_test = best_xgb.predict(X_test)
    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)

    print("\nTrain R-squared:", round(r2_train, 4))
    print("\nTest metrics:")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))

    # Plot predictions vs actuals for both train and test sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train set plot
    ax1.scatter(y_train, pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score") 
    ax1.set_title("XGBoost: Train Set")
    
    # Test set plot
    ax2.scatter(y_test, pred_test, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title("XGBoost: Test Set")
    
    plt.tight_layout()
    plt.show()

    # Get feature importances
    importances = best_xgb.feature_importances_
    feature_names = numeric_feature_cols
    indices = np.argsort(importances)[::-1]
    print("\nTop 20 Feature Importances:")
    for f in range(min(20, len(feature_names))):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))
    # Plot top 20 feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Most Important Features", pad=20, fontsize=14)
    
    # Create horizontal bar plot with reversed range so highest value is at top
    bars = plt.barh(range(19, -1, -1), importances[indices[:20]], color='skyblue', alpha=0.8)
    
    # Add value labels to the end of each bar
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left', va='center', fontsize=8)
    
    # Customize axes with reversed feature names
    plt.yticks(range(19, -1, -1), [feature_names[i] for i in indices[:20]], 
              ha='right', fontsize=10)
    plt.xlabel("Feature Importance Score", fontsize=12)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def ridge_raw():
    # Initialize Ridge regressor
    ridge = Ridge(random_state=8)

    # Define parameter grid
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    }

    # Set up GroupKFold cross validation
    gkf = GroupKFold(n_splits=3)
    
    # Perform grid search with group k-fold CV
    grid_search = GridSearchCV(
        ridge,
        param_grid,
        cv=gkf.split(X_train, y_train, groups=X_train_groups),
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print("\nRidge with optimized parameters:")
    print("Best parameters:", grid_search.best_params_)
    print("Best CV R2:", round(grid_search.best_score_, 4))

    # Get best model
    best_ridge = grid_search.best_estimator_

    # Predict and evaluate on train set
    pred_train = best_ridge.predict(X_train)
    r2_train = r2_score(y_train, pred_train)

    # Predict and evaluate on test set
    pred_test = best_ridge.predict(X_test)
    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)

    print("\nTrain R-squared:", round(r2_train, 4))
    print("\nTest metrics:")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))

    # Plot predictions vs actuals for both train and test sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train set plot
    ax1.scatter(y_train, pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    ax1.set_title("Ridge: Train Set")
    
    # Test set plot
    ax2.scatter(y_test, pred_test, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title("Ridge: Test Set")
    
    plt.tight_layout()
    plt.show()

    # Get feature coefficients
    coefficients = best_ridge.coef_
    feature_names = numeric_feature_cols
    indices = np.argsort(np.abs(coefficients))[::-1]
    
    print("\nTop 20 Feature Coefficients:")
    for f in range(min(20, len(feature_names))):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], coefficients[indices[f]]))

    # Plot top 20 feature coefficients
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Most Important Features", pad=20, fontsize=14)
    
    bars = plt.barh(range(19, -1, -1), np.abs(coefficients[indices[:20]]), color='skyblue', alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.yticks(range(19, -1, -1), [feature_names[i] for i in indices[:20]],
               ha='right', fontsize=10)
    plt.xlabel("Absolute Coefficient Value", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def elasticnet_raw():
    # Initialize ElasticNet regressor
    enet = ElasticNet(random_state=8)

    # Define parameter grid
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Set up GroupKFold cross validation
    gkf = GroupKFold(n_splits=3)
    
    # Perform grid search with group k-fold CV
    grid_search = GridSearchCV(
        enet,
        param_grid,
        cv=gkf.split(X_train, y_train, groups=X_train_groups),
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print("\nElasticNet with optimized parameters:")
    print("Best parameters:", grid_search.best_params_)
    print("Best CV R2:", round(grid_search.best_score_, 4))

    # Get best model
    best_enet = grid_search.best_estimator_

    # Predict and evaluate on train set
    pred_train = best_enet.predict(X_train)
    r2_train = r2_score(y_train, pred_train)

    # Predict and evaluate on test set
    pred_test = best_enet.predict(X_test)
    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)

    print("\nTrain R-squared:", round(r2_train, 4))
    print("\nTest metrics:")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))

    # Plot predictions vs actuals for both train and test sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train set plot
    ax1.scatter(y_train, pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    ax1.set_title("ElasticNet: Train Set")
    
    # Test set plot
    ax2.scatter(y_test, pred_test, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title("ElasticNet: Test Set")
    
    plt.tight_layout()
    plt.show()

    # Get feature coefficients
    coefficients = best_enet.coef_
    feature_names = numeric_feature_cols
    indices = np.argsort(np.abs(coefficients))[::-1]
    
    print("\nTop 20 Feature Coefficients:")
    for f in range(min(20, len(feature_names))):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], coefficients[indices[f]]))

    # Plot top 20 feature coefficients
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Most Important Features", pad=20, fontsize=14)
    
    bars = plt.barh(range(19, -1, -1), np.abs(coefficients[indices[:20]]), color='skyblue', alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.yticks(range(19, -1, -1), [feature_names[i] for i in indices[:20]],
               ha='right', fontsize=10)
    plt.xlabel("Absolute Coefficient Value", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def xgboost_ensemble():
    # Define parameter grids for each model in the ensemble
    # Each grid focuses on different regions of the parameter space
    param_grids = [
        # Grid 1: Conservative model (smaller trees, lower learning rate)
        {
            'n_estimators': [50, 100, 150],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.02, 0.03],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [3, 4, 5]
        },
        # Grid 2: Balanced model (medium trees, medium learning rate)
        {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.03, 0.04, 0.05],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [2, 3, 4]
        },
        # Grid 3: Aggressive model (larger trees, higher learning rate)
        {
            'n_estimators': [150, 200, 250],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.05, 0.06, 0.07],
            'subsample': [0.9, 1.0],
            'colsample_bytree': [0.9, 1.0],
            'min_child_weight': [1, 2, 3]
        },
        # Grid 4: High capacity model (many trees, careful learning)
        {
            'n_estimators': [200, 250, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.02, 0.03],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [2, 3, 4]
        },
        # Grid 5: Moderate model (balanced parameters)
        {
            'n_estimators': [100, 150, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.02, 0.03, 0.04],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [2, 3, 4]
        }
    ]

    # Create validation set for early stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, val_idx in gss.split(X_train, y_train, groups=X_train_groups):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        groups_tr = X_train_groups[train_idx]

    # Initialize list to store models and their predictions
    models = []
    train_predictions = []
    test_predictions = []
    best_params_list = []

    # Train each model in the ensemble using grid search
    for i, param_grid in enumerate(param_grids):
        print(f"\nTraining XGBoost model {i+1}/5")
        
        # Set up GroupKFold cross validation
        n_unique_groups = len(np.unique(groups_tr))
        n_splits = min(5, n_unique_groups)
        if n_splits < 2:
            raise ValueError(f"Need at least 2 unique groups for GroupKFold, got {n_unique_groups}")
        gkf = GroupKFold(n_splits=n_splits)
        
        # Initialize base model
        xgb = XGBRegressor(random_state=8, n_jobs=-1)
        
        # Perform grid search
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_tr, y_tr)
        
        # Get best parameters and model
        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        print(f"Best parameters for model {i+1}:", best_params)
        print(f"Best CV R2: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters and early stopping
        best_xgb = XGBRegressor(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Store model
        models.append(best_xgb)
        
        # Get predictions
        train_pred = best_xgb.predict(X_train)
        test_pred = best_xgb.predict(X_test)
        
        train_predictions.append(train_pred)
        test_predictions.append(test_pred)
        
        # Print individual model performance
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        print(f"Model {i+1} - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

    # Print summary of best parameters for each model
    print("\nSummary of best parameters for each model:")
    for i, params in enumerate(best_params_list):
        print(f"\nModel {i+1}:")
        for param, value in params.items():
            print(f"{param}: {value}")

    # Average predictions
    ensemble_train_pred = np.mean(train_predictions, axis=0)
    ensemble_test_pred = np.mean(test_predictions, axis=0)

    # Calculate ensemble metrics
    ensemble_train_r2 = r2_score(y_train, ensemble_train_pred)
    ensemble_test_r2 = r2_score(y_test, ensemble_test_pred)
    ensemble_test_mse = mean_squared_error(y_test, ensemble_test_pred)
    ensemble_test_rmse = np.sqrt(ensemble_test_mse)

    print("\nEnsemble Performance:")
    print("Train R-squared:", round(ensemble_train_r2, 4))
    print("Test metrics:")
    print("MSE:", round(ensemble_test_mse, 4))
    print("RMSE:", round(ensemble_test_rmse, 4))
    print("R-squared:", round(ensemble_test_r2, 4))

    # Plot predictions vs actuals for both train and test sets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train set plot
    ax1.scatter(y_train, ensemble_train_pred, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    ax1.set_title("XGBoost Ensemble: Train Set")
    
    # Test set plot
    ax2.scatter(y_test, ensemble_test_pred, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title("XGBoost Ensemble: Test Set")
    
    plt.tight_layout()
    plt.show()

    # Calculate and plot feature importances across all models
    feature_importances = np.zeros(len(numeric_feature_cols))
    for model in models:
        feature_importances += model.feature_importances_
    feature_importances /= len(models)  # Average importance across models

    # Get top 20 features
    indices = np.argsort(feature_importances)[::-1]
    print("\nTop 20 Feature Importances (Ensemble Average):")
    for f in range(min(20, len(numeric_feature_cols))):
        print("%d. %s (%f)" % (f + 1, numeric_feature_cols[indices[f]], feature_importances[indices[f]]))

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Most Important Features (Ensemble Average)", pad=20, fontsize=14)
    
    bars = plt.barh(range(19, -1, -1), feature_importances[indices[:20]], color='skyblue', alpha=0.8)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left', va='center', fontsize=8)
    
    plt.yticks(range(19, -1, -1), [numeric_feature_cols[i] for i in indices[:20]],
               ha='right', fontsize=10)
    plt.xlabel("Average Feature Importance Score", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

# best model so far is xgboost_raw()
# xgboost_raw()
# xgboost_ensemble()
# random_forest_raw()
# ridge_raw()
# elasticnet_raw()

def irtxgboost_raw():
    """XGBoost model incorporating IRT features from R analysis"""
    print("\nRunning XGBoost with IRT features...")

    # Load IRT features from R analysis
    irt_features = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=int(1e6))
    irt_features = irt_features[irt_features['test'] == 'mdt']
    
    # Get relevant IRT columns
    irt_cols = ['item_id', 'ability_WL', 'difficulty'] 
    irt_features = irt_features[irt_cols].copy()
    
    # Average ability_WL per item since it varies by participant
    irt_features = irt_features.groupby('item_id')[['ability_WL', 'difficulty']].mean().reset_index()
    
    # Merge with existing features (fix: merge into a new variable)
    data_with_irt_xgb = data_with_irt.merge(irt_features, left_on='melody_id', right_on='item_id', how='left')
    
    # Add IRT columns to feature set
    irt_feature_cols = ['ability_WL', 'difficulty']
    X_with_irt = data_with_irt_xgb[numeric_feature_cols + irt_feature_cols].values
    y = data_with_irt_xgb['score'].values
    groups = data_with_irt_xgb['melody_id'].values

    # Handle NaNs before PCA (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    X_with_irt = imputer.fit_transform(X_with_irt)

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X_with_irt, y, groups=groups):
        X_train_irt, X_test_irt = X_with_irt[train_idx], X_with_irt[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    # Impute train/test separately to avoid data leakage
    X_train_irt = imputer.fit_transform(X_train_irt)
    X_test_irt = imputer.transform(X_test_irt)

    # Apply PCA
    pca = PCA(n_components=0.95, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train_irt)
    X_test_pca = pca.transform(X_test_irt)
    print(f"Number of PCA components: {X_train_pca.shape[1]}")

    # Initialize XGBoost regressor
    xgb = XGBRegressor(random_state=8, n_jobs=-1)

    # Check if we have cached best parameters
    cache_file = 'modelling_paper/xgb_irt_best_params.npy'
    try:
        best_params = np.load(cache_file, allow_pickle=True).item()
        print("\nLoading cached best parameters:", best_params)
        best_xgb = XGBRegressor(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(X_train_irt, y_train)
    except:
        print("\nNo cached parameters found. Running grid search...")
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
        for train_idx, val_idx in gss.split(X_train_irt, y_train, groups=X_train_groups):
            X_tr, X_val = X_train_irt[train_idx], X_train_irt[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            groups_tr = X_train_groups[train_idx]

        # Set up GroupKFold cross validation
        n_unique_groups = len(np.unique(groups_tr))
        n_splits = min(5, n_unique_groups)
        if n_splits < 2:
            raise ValueError(f"Need at least 2 unique groups for GroupKFold, got {n_unique_groups}")
        gkf = GroupKFold(n_splits=n_splits)
        
        # Perform grid search
        grid_search = GridSearchCV(
            xgb,
            param_grid,
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_tr, y_tr)
        
        print("\nXGBoost with IRT features - optimized parameters:")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV R2:", round(grid_search.best_score_, 4))

        np.save(cache_file, grid_search.best_params_)

        best_xgb = XGBRegressor(random_state=8, n_jobs=-1, **grid_search.best_params_)
        best_xgb.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

    # Evaluate model
    pred_train = best_xgb.predict(X_train_irt)
    r2_train = r2_score(y_train, pred_train)

    pred_test = best_xgb.predict(X_test_irt)
    mse_test = mean_squared_error(y_test, pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, pred_test)

    print("\nTrain R-squared:", round(r2_train, 4))
    print("\nTest metrics:")
    print("MSE:", round(mse_test, 4))
    print("RMSE:", round(rmse_test, 4))
    print("R-squared:", round(r2_test, 4))

    # Plot predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.scatter(y_train, pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    ax1.set_title(f"XGBoost with IRT: Train Set (R² = {r2_train:.4f})")
    
    ax2.scatter(y_test, pred_test, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title(f"XGBoost with IRT: Test Set (R² = {r2_test:.4f})")
    
    plt.tight_layout()
    plt.show()

    # Feature importance analysis
    feature_names = numeric_feature_cols + irt_feature_cols
    importances = best_xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 Feature Importances:")
    for f in range(min(20, len(feature_names))):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

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


def permutation_importance_with_pvalues(model, X, y, feature_names, n_repeats=100, scoring='r2', random_state=8):
    """
    Compute permutation importance and empirical p-values for each feature.
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )
    importances = result.importances_mean
    stds = result.importances_std
    # Empirical p-value: fraction of times permuted importance <= 0
    p_values = np.mean(result.importances <= 0, axis=1)
    df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importances,
        'importance_std': stds,
        'empirical_p_value': p_values
    }).sort_values('importance_mean', ascending=False)
    return df

# irtxgboost_raw()

def irtxgboost_no_difficulty():
    """XGBoost regressor incorporating IRT features from R analysis, excluding difficulty"""
    print("\nRunning XGBoost Regressor with IRT features (no difficulty)...")

    # Get IRT features without difficulty
    irt_cols = ['item_id', 'ability_WL', 'score'] 
    irt_features = participant_responses[irt_cols].copy()
    irt_features = irt_features.groupby('item_id').agg({
        'ability_WL': 'mean',
        'score': 'mean'  # Changed to mean for regression
    }).reset_index()

    # Merge with existing features
    data_with_irt = features_final.merge(irt_features, left_on='melody_id', right_on='item_id', how='left')
    
    # Add oddity score from item bank
    data_with_irt = data_with_irt.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')

    # Prepare feature columns
    exclude_cols = {'melody_id', 'item_id', 'score'}  # No longer excluding oddity
    feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]
    numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    irt_feature_cols = ['ability_WL', 'oddity']  # Include oddity as a numeric feature

    # Prepare X and y
    X = data_with_irt[numeric_feature_cols].values
    y = data_with_irt['score'].values
    groups = data_with_irt['melody_id'].values

    # Handle NaNs before PCA (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    # Impute train/test separately to avoid data leakage
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Initialize XGBoost regressor
    xgb = XGBRegressor(random_state=8, n_jobs=-1)

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
    
    # Check if we have cached best parameters
    cache_file = 'modelling_paper/xgb_irt_no_difficulty_best_params.npy'
    try:
        best_params = np.load(cache_file, allow_pickle=True).item()
        print("\nLoading cached best parameters:", best_params)
        best_xgb = XGBRegressor(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(X_tr, y_tr)
        grid_search = None  # No grid search performed
    except:
        print("\nNo cached parameters found. Running grid search...")
        grid_search = GridSearchCV(
            xgb,
            param_grid, 
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_tr, y_tr)
        # Cache the best parameters
        np.save(cache_file, grid_search.best_params_)
        best_xgb = grid_search.best_estimator_

    if grid_search is not None:
        print("\nXGBoost Regressor with IRT features (no difficulty) - optimized parameters:")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV R²:", round(grid_search.best_score_, 4))
    else:
        print("\nXGBoost Regressor with IRT features (no difficulty) - loaded cached parameters.")

    # Evaluate model
    pred_train = best_xgb.predict(X_train)
    pred_test = best_xgb.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    print("\nModel Performance:")
    print("Train R²:", round(train_r2, 4))
    print("Test R²:", round(test_r2, 4))
    print("Train RMSE:", round(train_rmse, 4))
    print("Test RMSE:", round(test_rmse, 4))

    # Plot predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Train set plot
    ax1.scatter(y_train, pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    ax1.set_title(f"XGBoost with IRT: Train Set (R² = {train_r2:.4f})")
    
    # Test set plot
    ax2.scatter(y_test, pred_test, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title(f"XGBoost with IRT: Test Set (R² = {test_r2:.4f})")
    
    plt.tight_layout()
    plt.show()

    # Feature importance analysis
    feature_names = numeric_feature_cols
    importances = best_xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 Feature Importances:")
    for f in range(min(20, len(feature_names))):
        print("%d. %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

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

    # Perform combinatoric feature importance analysis
    print("\nPerforming combinatoric feature importance analysis...")

    # Get baseline performance with all features
    baseline_model = XGBRegressor(**best_xgb.get_params())
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_r2 = r2_score(y_test, baseline_pred)

    # Store feature importance scores
    feature_importance_dict = {}

    # For each feature, try removing it and measure impact
    for feature_idx in tqdm(range(len(feature_names)), desc="Analyzing feature combinations"):
        # Create mask for all features except current one
        feature_mask = np.ones(len(feature_names), dtype=bool)
        feature_mask[feature_idx] = False
        
        # Train model without this feature
        X_train_subset = X_train[:, feature_mask]
        X_test_subset = X_test[:, feature_mask]
        
        model = XGBRegressor(**best_xgb.get_params())
        model.fit(X_train_subset, y_train)
        
        # Get predictions and R2 score
        pred_test = model.predict(X_test_subset)
        r2 = r2_score(y_test, pred_test)
        
        # Calculate importance as drop in R2
        importance = baseline_r2 - r2
        feature_importance_dict[feature_names[feature_idx]] = importance

    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\nFeature importance based on R2 drop when removed:")
    for feature, importance in sorted_features[:20]:
        print(f"{feature}: {importance:.4f}")

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    features, importances = zip(*sorted_features[:20])
    
    bars = plt.barh(range(len(importances)), importances, color='lightcoral', alpha=0.8)
    
    plt.yticks(range(len(features)), features, ha='right', fontsize=10)
    plt.xlabel("Drop in R² when feature is removed")
    plt.title("Feature Importance Based on R² Impact", pad=20, fontsize=14)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=8)
    
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Perform grouped permutation importance analysis
    print("\nPerforming grouped permutation importance analysis...")

    # Define feature groups based on prefixes
    feature_groups = {
        'contour': [col for col in feature_names if 'contour' in col.lower()],
        'corpus': [col for col in feature_names if 'corpus' in col.lower()],
        'duration': [col for col in feature_names if 'duration' in col.lower()],
        'interval': [col for col in feature_names if 'interval' in col.lower()],
        'melodic_movement': [col for col in feature_names if 'melodic_movement' in col.lower()],
        'mtype': [col for col in feature_names if 'mtype' in col.lower()],
        'narmour': [col for col in feature_names if 'narmour' in col.lower()],
        'pitch': [col for col in feature_names if 'pitch' in col.lower()],
        'tonality': [col for col in feature_names if 'tonality' in col.lower() or 'key' in col.lower()],
        'ability_WL': [col for col in feature_names if col == 'ability_WL'],
        'oddity': [col for col in feature_names if col == 'oddity']  # Updated to use single oddity feature
    }

    # Print group sizes
    print("\nFeature group sizes:")
    for group, features in feature_groups.items():
        print(f"{group}: {len(features)} features")

    # Store group importance scores
    group_importance_dict = {}

    # For each group, permute all features in that group together
    for group_name, group_features in tqdm(feature_groups.items(), desc="Analyzing feature groups"):
        if not group_features:  # Skip empty groups
            continue
            
        # Get indices of features in this group
        group_indices = [feature_names.index(feat) for feat in group_features]
        
        # Create copy of test data
        X_test_permuted = X_test.copy()
        
        # Permute the features in this group
        X_test_permuted[:, group_indices] = np.random.permutation(X_test_permuted[:, group_indices])
        
        # Get predictions with permuted features
        pred_test_permuted = best_xgb.predict(X_test_permuted)
        r2_permuted = r2_score(y_test, pred_test_permuted)
        
        # Calculate importance as drop in R2
        importance = baseline_r2 - r2_permuted
        group_importance_dict[group_name] = importance

    # Sort groups by importance
    sorted_groups = sorted(group_importance_dict.items(), key=lambda x: x[1], reverse=True)

    print("\nGroup importance based on R2 drop when permuted:")
    for group, importance in sorted_groups:
        print(f"{group}: {importance:.4f}")

    # Plot group importance
    plt.figure(figsize=(12, 6))
    groups, importances = zip(*sorted_groups)
    
    bars = plt.barh(range(len(importances)), importances, color='lightseagreen', alpha=0.8)
    
    plt.yticks(range(len(groups)), groups, ha='right', fontsize=12)
    plt.xlabel("Drop in R² when group is permuted", fontsize=12)
    plt.title("Feature Group Importance Based on Permutation Impact", pad=20, fontsize=14)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}',
                ha='left', va='center', fontsize=10)
    
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Perform permutation importance with empirical p-values
    print("\nCalculating permutation importance and empirical p-values...")
    perm_df = permutation_importance_with_pvalues(
        best_xgb, X_test, y_test, feature_names, n_repeats=1000, scoring='r2', random_state=8
    )
    print(perm_df.head(20))  # Show top 20 features

    # Plot the top features with their p-values
    plt.figure(figsize=(12, 8))
    top_perm = perm_df.head(20)
    bars = plt.barh(range(len(top_perm)), top_perm['importance_mean'], color='mediumslateblue', alpha=0.8)
    plt.yticks(range(len(top_perm)), top_perm['feature'], ha='right', fontsize=10)
    plt.xlabel("Permutation Importance", fontsize=12)
    plt.title("Permutation Importance with Empirical p-values", pad=20, fontsize=14)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                 f'p={top_perm.iloc[i].empirical_p_value:.3f}',
                 ha='left', va='center', fontsize=8)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print features with p < 0.05
    print("\nFeatures with p < 0.05 (statistically significant):")
    sig_features = perm_df[perm_df['empirical_p_value'] < 0.05].sort_values('empirical_p_value')
    for _, row in sig_features.iterrows():
        print(f"{row['feature']}: p = {row['empirical_p_value']:.3f}")

# random_forest_raw()
irtxgboost_no_difficulty()

def irtxgboost_pca():
    """XGBoost regressor incorporating IRT features and PCA transformation"""
    print("\nRunning XGBoost Regressor with IRT features and PCA...")

    # Get IRT features
    irt_cols = ['item_id', 'ability_WL', 'score'] 
    irt_features = participant_responses[irt_cols].copy()
    irt_features = irt_features.groupby('item_id').agg({
        'ability_WL': 'mean',
        'score': 'mean'
    }).reset_index()

    # Merge with existing features
    data_with_irt = features_final.merge(irt_features, left_on='melody_id', right_on='item_id', how='left')
    
    # Add oddity score from item bank
    data_with_irt = data_with_irt.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')
    
    # Convert oddity to categorical and create dummy variables
    data_with_irt['oddity'] = data_with_irt['oddity'].astype('category')
    oddity_dummies = pd.get_dummies(data_with_irt['oddity'], prefix='oddity')
    data_with_irt = pd.concat([data_with_irt, oddity_dummies], axis=1)

    # Prepare feature columns
    exclude_cols = {'melody_id', 'item_id', 'score', 'oddity'}  # Exclude original oddity column
    feature_cols = [col for col in data_with_irt.columns if col not in exclude_cols]
    numeric_feature_cols = data_with_irt[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    irt_feature_cols = ['ability_WL'] + [col for col in numeric_feature_cols if col.startswith('oddity_')]  # Include oddity dummy variables

    # Prepare X and y
    X = data_with_irt[numeric_feature_cols].values
    y = data_with_irt['score'].values
    groups = data_with_irt['melody_id'].values

    # Handle NaNs before PCA (impute with mean)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    # Impute train/test separately to avoid data leakage
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=0.95, svd_solver='full')
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"Number of PCA components: {X_train_pca.shape[1]}")

    # Add ability_WL and oddity dummies back after PCA
    ability_WL_train = data_with_irt.iloc[train_idx]['ability_WL'].values.reshape(-1, 1)
    ability_WL_test = data_with_irt.iloc[test_idx]['ability_WL'].values.reshape(-1, 1)
    
    # Get oddity dummy variables
    oddity_cols = [col for col in data_with_irt.columns if col.startswith('oddity_')]
    oddity_train = data_with_irt.iloc[train_idx][oddity_cols].values
    oddity_test = data_with_irt.iloc[test_idx][oddity_cols].values
    
    # Combine PCA components with ability_WL and oddity dummies
    X_train_final = np.hstack((X_train_pca, ability_WL_train, oddity_train))
    X_test_final = np.hstack((X_test_pca, ability_WL_test, oddity_test))

    # Initialize XGBoost regressor
    xgb = XGBRegressor(random_state=8, n_jobs=-1)

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
    for train_idx, val_idx in gss.split(X_train_final, y_train, groups=X_train_groups):
        X_tr, X_val = X_train_final[train_idx], X_train_final[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        groups_tr = X_train_groups[train_idx]

    # Set up GroupKFold cross validation
    gkf = GroupKFold(n_splits=5)

    # Check if we have cached best parameters
    cache_file = 'modelling_paper/xgb_irt_pca_best_params.npy'
    try:
        best_params = np.load(cache_file, allow_pickle=True).item()
        print("\nLoading cached best parameters:", best_params)
        best_xgb = XGBRegressor(random_state=8, n_jobs=-1, **best_params)
        best_xgb.fit(X_tr, y_tr)
        grid_search = None
    except:
        print("\nNo cached parameters found. Running grid search...")
        grid_search = GridSearchCV(
            xgb,
            param_grid, 
            cv=gkf.split(X_tr, y_tr, groups=groups_tr),
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_tr, y_tr)
        np.save(cache_file, grid_search.best_params_)
        best_xgb = grid_search.best_estimator_

    if grid_search is not None:
        print("\nXGBoost Regressor with IRT features and PCA - optimized parameters:")
        print("Best parameters:", grid_search.best_params_)
        print("Best CV R²:", round(grid_search.best_score_, 4))
    else:
        print("\nXGBoost Regressor with IRT features and PCA - loaded cached parameters.")

    # Evaluate model
    pred_train = best_xgb.predict(X_train_final)
    pred_test = best_xgb.predict(X_test_final)
    
    train_r2 = r2_score(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    print("\nModel Performance:")
    print("\nTrain metrics:")
    print("RMSE:", round(train_rmse, 4))
    print("R-squared:", round(train_r2, 4))
    print("\nTest metrics:")
    print("RMSE:", round(test_rmse, 4))
    print("R-squared:", round(test_r2, 4))

    # Plot predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.scatter(y_train, pred_train, alpha=0.5)
    ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    ax1.set_xlabel("Actual Score")
    ax1.set_ylabel("Predicted Score")
    ax1.set_title(f"XGBoost with IRT + PCA: Train Set (R² = {train_r2:.4f})")
    
    ax2.scatter(y_test, pred_test, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Actual Score")
    ax2.set_ylabel("Predicted Score")
    ax2.set_title(f"XGBoost with IRT + PCA: Test Set (R² = {test_r2:.4f})")
    
    plt.tight_layout()
    plt.show()

    # Feature importance analysis for PCA components and IRT features
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

    # Create biplot of feature loadings on PC1 and PC2
    print("\nCreating biplot of feature loadings on PC1 and PC2...")
    
    # Get the feature loadings (components)
    loadings = pca.components_.T
    # Create biplot
    plt.figure(figsize=(12, 8))
    
    # Plot PC scores as scatter points
    pc_scores = pca.transform(X_test)  # Use X_test instead of X_test_irt
    plt.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.5, label='PC Scores')
    
    # Plot feature loadings as vectors
    from adjustText import adjust_text
    texts = []
    
    for i, feature in enumerate(numeric_feature_cols):  # Only use numeric_feature_cols
        plt.arrow(0, 0, 
                 loadings[i, 0] * 3, # Scale up for visibility
                 loadings[i, 1] * 3,
                 color='r', alpha=0.5)
        texts.append(plt.text(loadings[i, 0] * 3.2,
                            loadings[i, 1] * 3.2,
                            feature,
                            color='r',
                            ha='center',
                            va='center'))

    # Adjust text positions to avoid overlaps
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='r', alpha=0.5))

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance explained)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance explained)")
    plt.title("PCA Biplot: Feature Loadings on First Two Principal Components")
    
    # Add grid
    plt.grid(linestyle='--', alpha=0.3)
    
    # Center the plot around origin
    max_val = max(abs(plt.xlim()[0]), abs(plt.xlim()[1]), 
                 abs(plt.ylim()[0]), abs(plt.ylim()[1]))
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    plt.tight_layout()
    plt.show()

irtxgboost_pca()