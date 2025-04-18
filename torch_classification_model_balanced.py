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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
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

class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # First layer: input → 448 (optimized)
            nn.Linear(input_size, 448),
            nn.BatchNorm1d(448),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Second layer: 448 → 320 (optimized)
            nn.Linear(448, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Third layer: 320 → 320 (optimized)
            nn.Linear(320, 320),
            nn.BatchNorm1d(320),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Fourth layer: 320 → 256 (optimized)
            nn.Linear(320, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer: 256 → 1
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
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
    def __init__(self, alpha=0.15780446819040916, gamma=1.9048515912023805):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def combined_loss(outputs, targets, focal_criterion, bce_criterion, focal_weight=0.5887494191576673):
    """
    Combine focal loss and BCE loss with optimized weights.
    
    Args:
        outputs: Model predictions (shape: [batch_size, 1])
        targets: Ground truth labels (shape: [batch_size])
        focal_criterion: Focal loss criterion
        bce_criterion: BCE loss criterion
        focal_weight: Weight for focal loss (complement will be used for BCE)
        
    Returns:
        Combined weighted loss
    """
    # Ensure targets have the same shape as outputs
    targets = targets.unsqueeze(1)  # Add dimension to match [batch_size, 1]
    
    focal_loss = focal_criterion(outputs, targets)
    bce_loss = bce_criterion(outputs, targets)  # Now shapes match
    return focal_weight * focal_loss + (1 - focal_weight) * bce_loss

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    
    # Basic counts and rates
    metrics = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'total': total,
        
        # True rates
        'tn_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
        'tp_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Sensitivity/Recall
        
        # False rates
        'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,  # Fall-out
        'fn_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,  # Miss rate
        
        # Precision and F1
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        
        # Balanced accuracy
        'balanced_acc': ((tp / (tp + fn) if (tp + fn) > 0 else 0) + 
                        (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2
    }
    
    # Calculate percentages
    metrics.update({
        'tn_percent': (tn / total) * 100,
        'fp_percent': (fp / total) * 100,
        'fn_percent': (fn / total) * 100,
        'tp_percent': (tp / total) * 100
    })
    
    return metrics

def evaluate_model(model, data_loader, focal_criterion, bce_criterion, threshold=0.5):
    """Evaluate model performance with comprehensive metrics."""
    model.eval()
    all_predictions = []
    all_actuals = []
    all_probs = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = combined_loss(outputs, batch_y, focal_criterion, bce_criterion)
            total_loss += loss.item()
            
            probs = outputs.squeeze().numpy()
            preds = (probs >= threshold).astype(int)
            
            all_probs.extend(probs)
            all_predictions.extend(preds)
            all_actuals.extend(batch_y.numpy())
    
    # Calculate all metrics
    metrics = calculate_metrics(all_actuals, all_predictions)
    
    # Calculate balanced score with penalties
    balanced_score = (metrics['tn_rate'] + metrics['tp_rate']) / 2 - \
                    (0.4 * metrics['fn_rate']) - \
                    (0.2 * metrics['fp_rate'])
    
    metrics.update({
        'loss': total_loss / len(data_loader),
        'balanced_score': balanced_score,
        'predictions': all_predictions,
        'actuals': all_actuals,
        'probabilities': all_probs
    })
    
    return metrics

def print_metrics(metrics, prefix=""):
    """Print comprehensive metrics in a clear format."""
    print(f"\n{prefix} Metrics:")
    print("\nTrue/False Rates:")
    print(f"True Negative Rate (Specificity): {metrics['tn_rate']:.4f}")
    print(f"True Positive Rate (Sensitivity): {metrics['tp_rate']:.4f}")
    print(f"False Negative Rate (Miss Rate): {metrics['fn_rate']:.4f}")
    print(f"False Positive Rate (Fall-out): {metrics['fp_rate']:.4f}")
    
    print("\nCounts and Percentages:")
    print(f"True Negatives: {metrics['tn']} ({metrics['tn_percent']:.1f}%)")
    print(f"False Positives: {metrics['fp']} ({metrics['fp_percent']:.1f}%)")
    print(f"False Negatives: {metrics['fn']} ({metrics['fn_percent']:.1f}%)")
    print(f"True Positives: {metrics['tp']} ({metrics['tp_percent']:.1f}%)")
    
    print("\nOverall Metrics:")
    print(f"Balanced Accuracy: {metrics['balanced_acc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Balanced Score: {metrics['balanced_score']:.4f}")

def plot_confusion_matrix(metrics, threshold, save_path='confusion_matrix.png'):
    """Plot detailed confusion matrix with percentages."""
    cm = np.array([[metrics['tn'], metrics['fp']], 
                  [metrics['fn'], metrics['tp']]])
    
    # Calculate percentages
    cm_percent = cm / cm.sum() * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (threshold={threshold:.3f})\n' + \
             f'TN: {metrics["tn_percent"]:.1f}%, FP: {metrics["fp_percent"]:.1f}%\n' + \
             f'FN: {metrics["fn_percent"]:.1f}%, TP: {metrics["tp_percent"]:.1f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(metrics, save_path='roc_curve.png'):
    """Plot ROC curve with detailed metrics."""
    fpr, tpr, _ = roc_curve(metrics['actuals'], metrics['probabilities'])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

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

    scaler = StandardScaler()
    feature_diffs_scaled = scaler.fit_transform(feature_diffs)

    participant_responses = participant_responses[participant_responses['test'] == 'mdt']
    scores = participant_responses[['item_id', 'score']]

    data = pd.DataFrame(feature_diffs_scaled, columns=feature_diffs.columns)
    data['melody_id'] = melody_id
    data = data.merge(scores, left_on='melody_id', right_on='item_id')
    data = data.dropna()

    X = data.drop(['melody_id', 'item_id', 'score'], axis=1).values
    y = data['score'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create datasets and loaders
    train_dataset = MelodyDataset(X_train, y_train)
    val_dataset = MelodyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Initialize model and training components
    model = BinaryClassifier(input_size=X.shape[1])
    focal_criterion = FocalLoss()
    bce_criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0009922790814208319,
        weight_decay=0.0005533676785967943
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.0009922790814208319,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.4,
        div_factor=25,
        final_div_factor=1000
    )

    # Training loop
    print("\nStarting training...")
    num_epochs = 30
    best_score = -float('inf')
    best_model = None
    best_threshold = 0.5
    best_metrics = None
    patience = 7
    patience_counter = 0
    threshold_range = np.arange(0.38770282632483677, 0.5835734336193203, 0.01)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        batch_times = []
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_start_time = time.time()
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = combined_loss(outputs, batch_y, focal_criterion, bce_criterion)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Print progress every 10% of batches
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                avg_batch_time = np.mean(batch_times[-10:]) if len(batch_times) > 10 else batch_time
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Avg Batch Time: {avg_batch_time:.3f}s - Loss: {loss.item():.4f}")

        # Validation phase with threshold optimization
        model.eval()
        best_val_score = -float('inf')
        epoch_best_metrics = None
        epoch_best_threshold = 0.5

        for threshold in threshold_range:
            val_metrics = evaluate_model(model, val_loader, focal_criterion, bce_criterion, threshold)
            
            if val_metrics['balanced_score'] > best_val_score:
                best_val_score = val_metrics['balanced_score']
                epoch_best_metrics = val_metrics
                epoch_best_threshold = threshold

        # Calculate timing metrics
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = np.mean(batch_times)
        
        # Print epoch results with timing information
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"Epoch Duration: {epoch_time:.2f}s")
        print(f"Average Batch Time: {avg_batch_time:.3f}s")
        print(f"Training Loss: {train_loss/len(train_loader):.4f}")
        print(f"Best Threshold: {epoch_best_threshold:.3f}")
        print_metrics(epoch_best_metrics, prefix="Validation")

        # Update best model if improved
        if best_val_score > best_score:
            best_score = best_val_score
            best_model = copy.deepcopy(model)
            best_threshold = epoch_best_threshold
            best_metrics = epoch_best_metrics
            patience_counter = 0
            print(f"\nNew best model saved! Score: {best_score:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break

    # Final evaluation
    print("\nTraining completed! Evaluating final model...")
    final_metrics = evaluate_model(best_model, val_loader, focal_criterion, bce_criterion, best_threshold)
    print_metrics(final_metrics, prefix="Final")

    # Save model and create visualizations
    torch.save(best_model.state_dict(), 'best_balanced_model.pth')
    plot_confusion_matrix(final_metrics, best_threshold)
    plot_roc_curve(final_metrics)

if __name__ == "__main__":
    main() 