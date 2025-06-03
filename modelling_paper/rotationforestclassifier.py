import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from RotationForest.RotationForest import RotationForest
from modelling_paper.randomforestclassifier import load_data

def rotationforest_classifier(features_final, participant_responses, item_bank):
    print("\nRunning Rotation Forest Classifier with IRT features (no manual PCA)...")

    irt_cols = ['item_id', 'ability_WL', 'score']
    irt_features = participant_responses[irt_cols].copy()
    data_with_irt = features_final.reset_index().merge(irt_features, left_on='melody_id', right_on='item_id', how='right')
    data_with_irt = data_with_irt.merge(item_bank[['item_id', 'oddity']], on='item_id', how='left')
    # Limit to 1 million items to save resources
    max_rows = int(1e4)
    if len(data_with_irt) > max_rows:
        print(f"Sampling data from {len(data_with_irt)} to {max_rows} rows to save resources")
        data_with_irt = data_with_irt.sample(n=max_rows, random_state=42)
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

    # Rotation Forest classifier
    rf = RotationForest(n_trees=100, n_features=3, sample_prop=0.5, bootstrap=True)
    print("Fitting Rotation Forest Classifier...")
    print("Training model...", end="", flush=True)
    spinner = ['|', '/', '-', '\\']
    i = 0
    while not hasattr(rf, 'trees') or len(rf.trees) < rf.n_trees:
        print(f"\rTraining model... {spinner[i]}", end="", flush=True)
        i = (i + 1) % len(spinner)
    rf.fit(X_train, y_train)
    print("\rTraining complete!      ")
    pred_train = rf.predict(X_train)
    pred_test = rf.predict(X_test)
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)

    print("\nModel Performance:")
    print("\nTrain metrics:")
    print("Accuracy:", round(train_accuracy, 4))
    print("\nTest metrics:")
    print("Accuracy:", round(test_accuracy, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

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

    feature_names = numeric_feature_cols
    print("\nFeature importances are not available for Rotation Forest in this implementation.")

    return {
        'model': rf,
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy
        },
        'predictions': {
            'train': pred_train,
            'test': pred_test
        },
        'feature_importances': None  # Not available
    }

if __name__ == "__main__":
    features_final, participant_responses, item_bank = load_data()
    results = rotationforest_classifier(features_final, participant_responses, item_bank)
    print("\nFinal Results:")
    print("Train Accuracy:", round(results['metrics']['train_accuracy'], 4))
    print("Test Accuracy:", round(results['metrics']['test_accuracy'], 4))
