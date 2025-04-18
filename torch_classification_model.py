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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import time
import torch.cuda.amp

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device to CPU only
device = torch.device('cpu')

class MelodyDataset(Dataset):
    def __init__(self, X, y):
        # Convert to numpy first to ensure proper indexing
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()
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
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        return self.layers(x) + self.skip(x)

class BinaryMelodyNet(nn.Module):
    def __init__(self, input_size):
        super(BinaryMelodyNet, self).__init__()
        
        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 448),  # Best performing first layer size
            nn.BatchNorm1d(448),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Residual blocks with best performing sizes
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(448, 320, 0.3),  # Second layer
            ResidualBlock(320, 192, 0.3)   # Third layer
        ])
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
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
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.final_layers(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def plot_confusion_matrix(cm, classes=['0', '1'], title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def calculate_class_weights(y):
    """Calculate class weights based on class distribution with smoothing."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    
    # Add smoothing factor to prevent extreme weights
    smoothing = 0.1
    smoothed_counts = class_counts + smoothing * total_samples
    
    # Calculate inverse weights with smoothing
    weights = total_samples / (len(class_counts) * smoothed_counts)
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return torch.FloatTensor(weights)

def find_optimal_threshold(model, val_loader, device):
    """Find the optimal decision threshold using validation data."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            all_probs.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Try different thresholds
    thresholds = np.arange(0.3, 0.7, 0.02)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (np.array(all_probs) >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

def main():
    # Load the data
    print("Loading data...")
    original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
    miq_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")
    participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=10000000)

    # Keep melody ID column
    melody_id = original_features['melody_id']

    # Drop non-numeric columns before subtraction
    original_features = original_features.select_dtypes(include=[np.number])
    miq_features = miq_features.select_dtypes(include=[np.number])

    # Calculate differences between dataframes
    feature_diffs = miq_features - original_features

    # Find and drop zero variance columns
    zero_var_cols = feature_diffs.columns[feature_diffs.var() == 0]
    if len(zero_var_cols) > 0:
        print(f"Dropping zero variance features: {', '.join(zero_var_cols)}")
        feature_diffs = feature_diffs.drop(columns=zero_var_cols)

    # Drop tempo features
    feature_diffs = feature_diffs.drop(columns=[col for col in feature_diffs.columns if 'duration_features.tempo' in col])

    # Handle infinite and missing values
    feature_diffs = feature_diffs.replace([np.inf, -np.inf], np.nan)
    feature_diffs = feature_diffs.fillna(0)

    # Scale the features
    scaler = StandardScaler()
    feature_diffs_scaled = scaler.fit_transform(feature_diffs)

    # Prepare target variable (raw scores)
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

    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)

    # Create datasets
    train_dataset = MelodyDataset(X_train, y_train)
    val_dataset = MelodyDataset(X_val, y_val)
    test_dataset = MelodyDataset(X_test, y_test)

    # Calculate balanced class weights
    class_weights = calculate_class_weights(y_train)
    print("\nClass distribution:")
    print(f"Class 0: {np.sum(y_train == 0)} samples")
    print(f"Class 1: {np.sum(y_train == 1)} samples")
    print(f"Class weights: {class_weights.numpy()}")

    # Create data loaders with weighted sampling
    train_labels = y_train.astype(int)
    sample_weights = torch.tensor([class_weights[label] for label in train_labels]).double()
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=512,
        sampler=sampler,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=512,
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = BinaryMelodyNet(input_size=X.shape[1])
    
    # Use best performing Focal Loss parameters
    focal_criterion = FocalLoss(alpha=0.357, gamma=2.2)
    bce_criterion = nn.BCELoss(weight=class_weights[1])
    
    # Initialize threshold
    threshold = 0.5  # Default threshold
    best_threshold = 0.5
    
    # Training parameters
    num_epochs = 50
    batch_size = 512
    grad_clip = 0.5  # Reduced gradient clipping
    
    # Use best performing learning rate and weight decay
    max_lr = 0.000193  # Best performing learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=max_lr/10,
        weight_decay=0.000817,  # Best performing weight decay
        betas=(0.9, 0.999)
    )

    # Calculate steps per epoch for OneCycleLR
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * num_epochs
    
    # Use OneCycleLR scheduler with optimized parameters
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.4,  # Longer warmup
        div_factor=25,  # Lower initial learning rate
        final_div_factor=1000  # final_lr = max_lr/1000
    )

    # Training loop
    print("\nStarting training...")
    train_losses = []
    test_losses = []
    best_f1 = 0  # Changed from best_accuracy to best_f1
    patience = 10
    patience_counter = 0
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(batch_X)
            focal_loss = focal_criterion(outputs.squeeze(), batch_y)
            bce_loss = bce_criterion(outputs.squeeze(), batch_y)
            loss = 0.7 * focal_loss + 0.3 * bce_loss  # Adjusted loss weights
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] - Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}')

        # Validation phase
        if (epoch + 1) % 5 == 0:  # Find optimal threshold every 5 epochs
            threshold = find_optimal_threshold(model, val_loader, device)
            print(f"Optimal threshold: {threshold:.3f}")
            if threshold != 0.5:  # Only update if we found a better threshold
                best_threshold = threshold
        
        # Evaluate with current threshold
        model.eval()
        test_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = bce_criterion(outputs.squeeze(), batch_y)
                test_loss += loss.item()
                
                # Use best threshold for predictions
                pred = (outputs.squeeze() >= best_threshold).float()
                predictions.extend(pred.numpy())
                actuals.extend(batch_y.numpy())

        # Calculate F1 score instead of accuracy
        _, _, f1, _ = precision_recall_fscore_support(actuals, predictions, average='binary')
        
        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - total_start_time
        
        # Print progress for every epoch
        print(f'\nEpoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s - Total Time: {total_time:.2f}s')
        print(f'Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
        print(f'F1 Score: {f1:.4f}')

        # Early stopping based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_classification_model.pth')
            print(f'New best F1 score: {f1:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Total training time: {total_time:.2f}s")
                print(f"Best F1 score achieved: {best_f1:.4f}")
                break

    print(f"\nTraining completed. Total time: {(time.time() - total_start_time):.2f}s")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Time')
    plt.legend()
    plt.savefig('classification_training_curves.png')
    plt.close()

    # Final evaluation
    print("\nFinal Evaluation...")
    model.eval()
    all_predictions = []
    all_actuals = []
    all_probabilities = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            probabilities = outputs.squeeze().numpy()
            predictions = (probabilities >= best_threshold).astype(int)
            
            all_probabilities.extend(probabilities)
            all_predictions.extend(predictions)
            all_actuals.extend(batch_y.numpy())

    # Calculate metrics
    _, _, f1, _ = precision_recall_fscore_support(all_actuals, all_predictions, average='binary')
    precision, recall, _, _ = precision_recall_fscore_support(all_actuals, all_predictions, average='binary')
    auc_roc = roc_auc_score(all_actuals, all_probabilities)
    cm = confusion_matrix(all_actuals, all_predictions)

    print("\nModel Performance:")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    main() 