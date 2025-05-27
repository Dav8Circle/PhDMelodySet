import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support
import copy
import time
from datetime import datetime

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set device to CPU only
device = torch.device('cpu')

# Previous best hyperparameters
PREV_BEST = {
    'hidden_sizes': [256, 256, 128, 64],
    'dropout_rates': [0.3, 0.3, 0.3, 0.2],
    'learning_rate': 0.001,
    'batch_size': 512,
    'weight_decay': 0.02,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0
}

class MelodyDataset(Dataset):
    def __init__(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    def __init__(self, input_size, hidden_sizes, dropout_rates):
        super(BinaryMelodyNet, self).__init__()
        
        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout_rates[0])
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], dropout_rates[i+1])
            for i in range(len(hidden_sizes)-2)
        ])
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_sizes[-2], hidden_sizes[-1]),
            nn.BatchNorm1d(hidden_sizes[-1]),
            nn.GELU(),
            nn.Dropout(dropout_rates[-1]),
            nn.Linear(hidden_sizes[-1], 1),
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

def calculate_class_weights(y):
    """Calculate class weights based on class distribution."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    # Inverse frequency weighting with smoothing
    weights = total_samples / (len(class_counts) * (class_counts + 1))
    # Normalize weights
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)

def objective(trial):
    print(f"\nStarting trial {trial.number}")
    start_time = time.time()
    
    # Hyperparameters to optimize with adjusted ranges
    n_layers = trial.suggest_int("n_layers", 3, 5)
    print(f"\nModel Architecture:")
    print(f"Number of layers: {n_layers}")
    
    # Hidden sizes with wider range for first layer
    hidden_sizes = []
    for i in range(n_layers):
        if i == 0:
            # First layer with wider range
            size = trial.suggest_int(f"hidden_{i}", 256, 512, step=64)
            hidden_sizes.append(size)
            print(f"Layer {i}: {size} neurons")
        else:
            # Subsequent layers decrease in size
            prev_size = hidden_sizes[-1]
            size = trial.suggest_int(f"hidden_{i}", 
                                   max(64, prev_size - 128),
                                   prev_size,
                                   step=64)
            hidden_sizes.append(size)
            print(f"Layer {i}: {size} neurons")
    
    # Print other hyperparameters with adjusted ranges
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    focal_alpha = trial.suggest_float("focal_alpha", 0.2, 0.4)
    focal_gamma = trial.suggest_float("focal_gamma", 2.0, 3.0)
    threshold_min = trial.suggest_float("threshold_min", 0.3, 0.5)
    threshold_max = trial.suggest_float("threshold_max", 0.5, 0.7)
    
    print(f"\nTraining Parameters:")
    print(f"Learning rate: {learning_rate:.6f}")
    print(f"Weight decay: {weight_decay:.6f}")
    print(f"Focal loss alpha: {focal_alpha:.3f}")
    print(f"Focal loss gamma: {focal_gamma:.3f}")
    print(f"Threshold range: {threshold_min:.3f} - {threshold_max:.3f}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
    miq_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")
    participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=10000000)

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

    # Split data with stratification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    # Print dataset information
    print(f"\nDataset Statistics:")
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Class distribution: {class_dist}")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    print(f"Class weights: {class_weights.numpy()}")

    # Create weighted sampler
    train_labels = y_train.astype(int)
    sample_weights = torch.tensor([class_weights[label].item() for label in train_labels]).double()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )

    # Create datasets and loaders
    train_dataset = MelodyDataset(X_train, y_train)
    val_dataset = MelodyDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=PREV_BEST['batch_size'],
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=PREV_BEST['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = BinaryMelodyNet(
        input_size=X.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rates=PREV_BEST['dropout_rates']
    ).to(device)

    # Training setup with gradient clipping and learning rate schedule
    focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    bce_criterion = nn.BCELoss(weight=torch.tensor([class_weights[1].item()], dtype=torch.float32))
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.4,  # Longer warmup
        div_factor=25,  # Lower initial learning rate
        final_div_factor=1000
    )

    best_f1 = 0
    best_model = None
    patience = 7  # Increased patience
    patience_counter = 0

    # Training loop with threshold optimization
    print("\nStarting training...")
    for epoch in range(30):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            focal_loss = focal_criterion(outputs.squeeze(), batch_y)
            bce_loss = bce_criterion(outputs.squeeze(), batch_y)
            loss = 0.7 * focal_loss + 0.3 * bce_loss  # Adjusted loss weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Reduced gradient clipping
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
                
                # Calculate validation loss
                focal_loss = focal_criterion(probs, batch_y)
                bce_loss = bce_criterion(probs, batch_y)
                loss = 0.7 * focal_loss + 0.3 * bce_loss
                val_loss += loss.item()
                val_batches += 1

        # Optimize threshold within specified range
        best_threshold = 0.5
        best_f1_threshold = 0
        best_precision = 0
        best_recall = 0
        for threshold in np.arange(threshold_min, threshold_max, 0.05):
            preds = (np.array(all_probs) >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(all_targets, preds, average='binary')
            if f1 > best_f1_threshold:
                best_f1_threshold = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall

        current_f1 = best_f1_threshold
        avg_val_loss = val_loss / val_batches
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch [{epoch+1}/30] - Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  F1 Score: {current_f1:.4f}")
        print(f"  Precision: {best_precision:.4f}")
        print(f"  Recall: {best_recall:.4f}")
        print(f"  Threshold: {best_threshold:.3f}")

        # Early stopping based on F1 score
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model = copy.deepcopy(model)
            patience_counter = 0
            print(f"  New best F1 score: {best_f1:.4f}")
            print(f"  Best Precision: {best_precision:.4f}")
            print(f"  Best Recall: {best_recall:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Report F1 score
        trial.report(current_f1, epoch)
        
        # Prune if performance is poor
        if trial.should_prune() or current_f1 < 0.55:  # Increased minimum F1 threshold
            print("\nTrial pruned due to poor performance")
            raise optuna.TrialPruned()

    trial_time = time.time() - start_time
    print(f"\nTrial completed in {trial_time:.2f} seconds")
    print(f"Best F1 score: {best_f1:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")
    print(f"Best Threshold: {best_threshold:.3f}")
    
    return best_f1

def main():
    print("\nStarting optimization study...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_SEED,
            n_startup_trials=10,
            multivariate=True
        )
    )
    
    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True
    )
    
    print("\nBest trial:")
    trial = study.best_trial
    
    print(f"  F1 Score: {trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save optimization results
    results_df = study.trials_dataframe()
    results_df.to_csv(f"classification_optimization_results_{timestamp}.csv", index=False)
    
    # Save visualization plots
    try:
        import plotly
        
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"classification_optimization_history_{timestamp}.html")
        
        # Parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"classification_parameter_importance_{timestamp}.html")
        
    except (ImportError, AttributeError) as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    main() 