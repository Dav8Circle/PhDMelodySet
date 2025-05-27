import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, roc_curve, auc
import time
import copy

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device to CPU only
device = torch.device('cpu')

class MelodyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        return self.layers(x) + self.skip(x)

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # First layer: input → 448
            nn.Linear(input_size, 448),
            nn.BatchNorm1d(448),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Second layer: 448 → 320
            nn.Linear(448, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Third layer: 320 → 192
            nn.Linear(320, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Fourth layer: 192 → 128
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Fifth layer: 128 → 128
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer: 128 → 1
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def calculate_class_weights(y):
    """Calculate class weights based on class distribution."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    # Balanced weighting that slightly favors positive class
    weights = np.array([1.0, 1.2]) * (total_samples / (len(class_counts) * (class_counts + 1)))
    # Normalize weights
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)

def find_optimal_threshold(model, data_loader, thresholds=None):
    """Find the optimal threshold that maximizes F1 score."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)  # Test thresholds from 0.1 to 0.9
    
    model.eval()
    all_probs = []
    all_actuals = []
    
    # Collect all predictions and actual values
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            all_probs.extend(outputs.squeeze().numpy())
            all_actuals.extend(batch_y.numpy())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_actuals = np.array(all_actuals)
    
    # Test different thresholds
    results = []
    for threshold in thresholds:
        predictions = (all_probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_actuals, predictions, average='binary')
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })
    
    # Find best threshold
    results = pd.DataFrame(results)
    best_idx = results['f1'].argmax()
    best_result = results.iloc[best_idx]
    
    # Plot precision-recall curve for different thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(results['recall'], results['precision'], 'b-', label='P-R curve')
    plt.plot(best_result['recall'], best_result['precision'], 'ro', label='Best threshold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\nBest threshold = {best_result["threshold"]:.3f} (F1 = {best_result["f1"]:.3f})')
    plt.grid(True)
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    # Print detailed results for best threshold
    print("\nBest threshold results:")
    print(f"Threshold: {best_result['threshold']:.3f}")
    print(f"F1: {best_result['f1']:.3f}")
    print(f"Precision: {best_result['precision']:.3f}")
    print(f"Recall: {best_result['recall']:.3f}")
    
    return best_result['threshold']

def evaluate_model(model, data_loader, focal_criterion, bce_criterion, threshold=0.5):
    """Evaluate model performance using F1 score and other metrics."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_actuals = []
    all_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            # Calculate combined loss
            focal_loss = focal_criterion(outputs.squeeze(), batch_y)
            bce_loss = bce_criterion(outputs.squeeze(), batch_y)
            loss = 0.7 * focal_loss + 0.3 * bce_loss
            total_loss += loss.item()
            
            probs = outputs.squeeze().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_probs.extend(probs)
            all_predictions.extend(preds)
            all_actuals.extend(batch_y.numpy())
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(all_actuals, all_predictions, average='binary')
    
    return {
        'loss': total_loss / len(data_loader),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_predictions,
        'actuals': all_actuals,
        'probabilities': all_probs
    }

def main():
    # Load and prepare data
    print("Loading data...")
    original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
    miq_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")
    participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=10000000)

    # Data preprocessing
    melody_id = original_features['melody_id']
    original_features = original_features.select_dtypes(include=[np.number])
    miq_features = miq_features.select_dtypes(include=[np.number])

    feature_diffs = miq_features - original_features
    feature_diffs = feature_diffs.drop(columns=[col for col in feature_diffs.columns if 'duration_features.tempo' in col])
    feature_diffs = feature_diffs.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale features
    scaler = StandardScaler()
    feature_diffs_scaled = scaler.fit_transform(feature_diffs)

    # Prepare target variable
    participant_responses = participant_responses[participant_responses['test'] == 'mdt']
    scores = participant_responses[['item_id', 'score']]

    # Merge features with scores
    data = pd.DataFrame(feature_diffs_scaled, columns=feature_diffs.columns)
    data['melody_id'] = melody_id
    data = data.merge(scores, left_on='melody_id', right_on='item_id')
    data = data.dropna()

    # Split features and target
    X = data.drop(['melody_id', 'item_id', 'score'], axis=1).values
    y = data['score'].values

    # First split test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # Then split remaining data into train/val
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

    # Print class distribution
    print("\nClass distribution in training set:")
    print(f"Class 0: {np.sum(y_train == 0)} samples")
    print(f"Class 1: {np.sum(y_train == 1)} samples")

    # Create datasets
    train_dataset = MelodyDataset(X_train, y_train)
    val_dataset = MelodyDataset(X_val, y_val)
    test_dataset = MelodyDataset(X_test, y_test)

    # Create weighted sampler for training
    train_labels = y_train.astype(int)
    class_weights = calculate_class_weights(y_train)
    sample_weights = torch.tensor([class_weights[label].item() for label in train_labels]).double()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )

    # Create data loaders with exact same settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,  # Matches optimization study
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model and training components
    model = BinaryClassifier(input_size=X.shape[1])
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    print(f"Class weights: {class_weights.numpy()}")
    
    # Combined loss functions with balanced implementation
    focal_criterion = FocalLoss(alpha=0.35, gamma=1.5)
    bce_criterion = nn.BCELoss(weight=torch.tensor([1.0], dtype=torch.float32))  # Neutral BCE weighting
    
    # Adjust loss combination weights to favor BCE for stability
    def combined_loss(outputs, targets, focal_criterion, bce_criterion):
        focal_loss = focal_criterion(outputs.squeeze(), targets)
        bce_loss = bce_criterion(outputs.squeeze(), targets)
        return 0.4 * focal_loss + 0.6 * bce_loss  # Give more weight to BCE loss for stability

    # Optimizer with exact same parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0016905863833757773,
        weight_decay=0.009733716793162,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler with exact same settings
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0016905863833757773,
        epochs=30,  # Matches optimization study
        steps_per_epoch=len(train_loader),
        pct_start=0.4,
        div_factor=25,
        final_div_factor=1000
    )

    # Training loop with exact same settings
    print("\nStarting training...")
    num_epochs = 30  # Matches optimization study
    best_f1 = 0
    best_balance_diff = float('inf')  # Track best precision/recall balance
    best_balanced_model = None
    best_balanced_threshold = 0.5
    patience = 7  # Matches optimization study
    patience_counter = 0
    total_start_time = time.time()
    l1_lambda = 1e-5

    # Optimized threshold range - lower range to allow more positive predictions
    threshold_min = 0.35  # Lower minimum threshold
    threshold_max = 0.55  # Lower maximum threshold

    # Modify the threshold selection criteria
    def calculate_threshold_score(tn, fp, fn, tp):
        tn_rate = tn / (tn + fp)
        tp_rate = tp / (tp + fn)
        fn_penalty = (fn / (fn + tp)) * 0.3  # Penalize false negatives
        return (tn_rate + tp_rate) / 2 - fn_penalty

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate combined loss with exact same weights
            focal_loss = focal_criterion(outputs.squeeze(), batch_y)
            bce_loss = bce_criterion(outputs.squeeze(), batch_y)
            loss = combined_loss(outputs, batch_y, focal_criterion, bce_criterion)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches

        # Validation with threshold optimization
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                probs = outputs.squeeze()
                all_probs.extend(probs.numpy())
                all_targets.extend(batch_y.numpy())
                
                # Calculate validation loss with exact same weights
                focal_loss = focal_criterion(probs, batch_y)
                bce_loss = bce_criterion(probs, batch_y)
                loss = combined_loss(outputs, batch_y, focal_criterion, bce_criterion)
                val_loss += loss.item()
                val_batches += 1

        # Optimize threshold for balanced precision/recall
        best_threshold = 0.5
        best_f1_threshold = 0
        best_precision = 0
        best_recall = 0
        best_balance_score = -float('inf')  # Track best balance score
        
        for threshold in np.arange(threshold_min, threshold_max, 0.01):
            preds = (np.array(all_probs) >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
            
            # Calculate score with emphasis on reducing false negatives
            score = calculate_threshold_score(tn, fp, fn, tp)
            
            if score > best_balance_score:
                best_balance_score = score
                best_threshold = threshold
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                best_f1_threshold = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                best_precision = precision
                best_recall = recall
                print(f"  New best balance: TN Rate={tn/(tn+fp):.4f}, TP Rate={tp/(tp+fn):.4f}, FN Rate={fn/(fn+tp):.4f}")

        current_f1 = best_f1_threshold
        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  F1 Score: {current_f1:.4f}")
        print(f"  Precision: {best_precision:.4f}")
        print(f"  Recall: {best_recall:.4f}")
        print(f"  Threshold: {best_threshold:.3f}")

        # Early stopping based on best balance
        if best_balance_score > 0.5:  # Matches optimization study
            best_balanced_model = copy.deepcopy(model)
            best_balanced_threshold = best_threshold
            patience_counter = 0
            print(f"  New best balance: Precision={best_precision:.4f}, Recall={best_recall:.4f}, Diff={recall - precision:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Prune if performance is poor
        if current_f1 < 0.55:  # Matches optimization study
            print("\nTrial pruned due to poor performance")
            break

    total_time = time.time() - total_start_time
    print(f"\nTraining completed. Total time: {total_time:.2f}s")
    
    # Save best balanced model
    print("\nSaving best balanced model...")
    torch.save(best_balanced_model.state_dict(), 'best_balanced_model.pth')
    
    # Print final training metrics
    print("\nFinal Training Metrics:")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")
    print(f"Best Threshold: {best_balanced_threshold:.3f}")
    print(f"Precision-Recall Difference: {recall - precision:.4f}")

    # Load best balanced model for final evaluation
    model.load_state_dict(torch.load('best_balanced_model.pth'))
    # Final evaluation with best balanced model and threshold
    print("\nFinal Evaluation with Best Balanced Model...")
    test_metrics = evaluate_model(best_balanced_model, test_loader, focal_criterion, bce_criterion, threshold=best_balanced_threshold)

    print("\nTest Performance:")
    print(f"Threshold: {best_balanced_threshold:.3f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(test_metrics['actuals'], test_metrics['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (threshold={best_balanced_threshold:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_metrics['actuals'], test_metrics['probabilities'])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    plt.close()

if __name__ == "__main__":
    main() 