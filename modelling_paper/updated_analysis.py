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

# 1. Load data
print("Loading original features...")
original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/testing.csv")
print("Loading odd one out features...")
odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels2.csv")
print("Loading participant responses...")
participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=int(1e6))

# Filter participant_responses to only include 'mdt' test
participant_responses = participant_responses[participant_responses['test'] == 'mdt']

# Compute mean score per melody
mean_scores = participant_responses.groupby('item_id')['score'].mean().reset_index()
mean_scores = mean_scores.rename(columns={'score': 'mean_score'})

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

# Merge with mean scores
data = features_final.merge(mean_scores, left_on='melody_id', right_on='item_id')

# Drop any rows with missing values
data = data.dropna()

# Prepare X and y
exclude_cols = {'melody_id', 'item_id', 'mean_score'}
feature_cols = [col for col in data.columns if col not in exclude_cols]

# Only keep numeric columns for modeling
numeric_feature_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [col for col in feature_cols if col not in numeric_feature_cols]
if non_numeric_cols:
    print("Dropping non-numeric columns:", non_numeric_cols)

X = data[numeric_feature_cols].values
y = data['mean_score'].values

# Train/test split by melody_id
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
groups = data['melody_id'].values
for train_idx, test_idx in gss.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train_indices, X_test_indices = train_idx, test_idx

X_train_groups = groups[X_train_indices]

# Print shapes
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 3. PCA
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
    rf.fit(X_train, y_train)
    
    # Get train and test predictions
    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)
    
    # Calculate and print R2 scores
    print("Random Forest (Raw Features) Train R2:", r2_score(y_train, rf_pred_train))
    print("Random Forest (Raw Features) Test R2:", r2_score(y_test, rf_pred_test))

    # Create feature names for original, odd-one-out, and diff features
    feature_names = (
        [f'original_{col}' for col in original_features.columns] +
        [f'odd_one_out_{col}' for col in odd_one_out_features.columns] +
        [f'diff_{col}' for col in feature_diffs.columns]
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
    train_ids = set(data.iloc[X_train_indices]['item_id'])
    test_ids = set(data.iloc[X_test_indices]['item_id'])
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
    cache_file = 'xgb_best_params.npy'
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
        gkf = GroupKFold(n_splits=3)
        
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
    
    # Merge with existing features
    data_with_irt = data.merge(irt_features, left_on='melody_id', right_on='item_id', how='left')
    
    # Add IRT columns to feature set
    irt_feature_cols = ['ability_WL', 'difficulty']
    X_with_irt = data_with_irt[numeric_feature_cols + irt_feature_cols].values
    y = data_with_irt['mean_score'].values
    groups = data_with_irt['melody_id'].values

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X_with_irt, y, groups=groups):
        X_train_irt, X_test_irt = X_with_irt[train_idx], X_with_irt[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        X_train_groups = groups[train_idx]

    # Initialize XGBoost regressor
    xgb = XGBRegressor(random_state=8, n_jobs=-1)

    # Check if we have cached best parameters
    cache_file = 'xgb_irt_best_params.npy'
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
        gkf = GroupKFold(n_splits=5)
        
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

irtxgboost_raw()
