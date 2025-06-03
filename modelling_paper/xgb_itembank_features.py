import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Load item bank
    item_bank = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/modelling_paper/item-bank.csv")
    item_bank = item_bank.rename(columns={'id': 'item_id'})

    # Load participant responses
    participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=1e7)
    participant_responses = participant_responses[participant_responses['test'] == 'mdt']

    # Get relevant columns from item bank
    itembank_features = item_bank[['item_id', 'displacement', 'oddity', 'in_key']].copy()

    # Get ability_WL and score from participant_responses
    irt_cols = ['item_id', 'ability_WL', 'score']
    irt_features = participant_responses[irt_cols].copy()

    # Merge on item_id
    data = irt_features.merge(itembank_features, on='item_id', how='left')

    # Convert oddity to categorical dummies, in_key to int
    data['oddity'] = data['oddity'].astype('category')
    oddity_dummies = pd.get_dummies(data['oddity'], prefix='oddity')
    data = pd.concat([data, oddity_dummies], axis=1)
    data['in_key'] = data['in_key'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0, 1: 1, 0: 0}).astype(int)

    # Prepare features
    exclude_cols = {'item_id', 'score', 'oddity'}
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    numeric_feature_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    X = data[numeric_feature_cols].values
    y = data['score'].values
    groups = data['item_id'].values

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    for train_idx, test_idx in gss.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    # Impute train/test separately
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # XGBoost classifier with specified parameters
    xgb = XGBClassifier(
        colsample_bytree=1.0,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        n_estimators=200,
        subsample=0.8,
        random_state=8,
        n_jobs=-1
    )

    # Train the model
    print("Training XGBoost model...")
    xgb.fit(X_train, y_train)

    # Evaluate
    pred_train = xgb.predict(X_train)
    pred_test = xgb.predict(X_test)
    train_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)

    # Probabilities for ROC
    if hasattr(xgb, "predict_proba"):
        y_test_proba = xgb.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = xgb.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)
    f1 = f1_score(y_test, pred_test)

    print("\nModel Performance:")
    print("Train Accuracy:", round(train_accuracy, 4))
    print("Test Accuracy:", round(test_accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("AUC:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, pred_test))

    # Confusion matrices
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f}')
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

    # Feature importances
    feature_names = numeric_feature_cols
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nTop Feature Importances:")
    for f in range(len(feature_names)):
        print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances", pad=20, fontsize=14)
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