import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
import copy
import time
from datetime import datetime

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set device to CPU only
device = torch.device('cpu')

class MelodyDataset(Dataset):
    def __init__(self, X, y):
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

def calculate_class_weights(y):
    """Calculate class weights based on class distribution."""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    smoothing = 0.1
    smoothed_counts = class_counts + smoothing * total_samples
    weights = total_samples / (len(class_counts) * smoothed_counts)
    weights = weights / weights.sum()
    return torch.FloatTensor(weights)

def find_optimal_threshold(model, val_loader):
    """Find the optimal decision threshold using MCC."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            all_probs.extend(outputs.squeeze().numpy())
            all_labels.extend(batch_y.numpy())
    
    thresholds = np.arange(0.2, 0.8, 0.02)
    best_mcc = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (np.array(all_probs) >= threshold).astype(int)
        mcc = matthews_corrcoef(all_labels, predictions)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    
    return best_threshold, best_mcc

def objective(trial):
    print(f"\nStarting trial {trial.number}")
    start_time = time.time()
    
    # Hyperparameters to optimize
    n_layers = trial.suggest_int("n_layers", 3, 5)
    print(f"\nModel Architecture:")
    print(f"Number of layers: {n_layers}")
    
    # Hidden sizes
    hidden_sizes = []
    dropout_rates = []
    for i in range(n_layers):
        if i == 0:
            size = trial.suggest_int(f"hidden_{i}", 256, 512, step=64)
            dropout = trial.suggest_float(f"dropout_{i}", 0.2, 0.5)
        else:
            prev_size = hidden_sizes[-1]
            size = trial.suggest_int(f"hidden_{i}", 
                                   max(64, prev_size - 128),
                                   prev_size,
                                   step=64)
            dropout = trial.suggest_float(f"dropout_{i}", 0.2, 0.5)
        hidden_sizes.append(size)
        dropout_rates.append(dropout)
        print(f"Layer {i}: {size} neurons, dropout {dropout:.2f}")
    
    # Training parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    focal_alpha = trial.suggest_float("focal_alpha", 0.2, 0.3)
    focal_gamma = trial.suggest_float("focal_gamma", 1.5, 2.5)
    loss_weight = trial.suggest_float("loss_weight", 0.4, 0.6)  # Weight between focal and BCE loss
    
    print(f"\nTraining Parameters:")
    print(f"Learning rate: {learning_rate:.6f}")
    print(f"Weight decay: {weight_decay:.6f}")
    print(f"Focal loss alpha: {focal_alpha:.3f}")
    print(f"Focal loss gamma: {focal_gamma:.3f}")
    print(f"Loss weight (focal): {loss_weight:.3f}")
    
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

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

    # Create datasets and loaders
    train_dataset = MelodyDataset(X_train, y_train)
    val_dataset = MelodyDataset(X_val, y_val)

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    print(f"\nClass weights: {class_weights.numpy()}")

    # Create weighted sampler
    train_labels = y_train.astype(int)
    sample_weights = torch.tensor([class_weights[int(label)] for label in train_labels]).double()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
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

    # Initialize model
    model = BinaryMelodyNet(
        input_size=X.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rates=dropout_rates
    )

    # Training setup
    focal_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    bce_criterion = nn.BCELoss(weight=class_weights[1])
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.4,
        div_factor=25,
        final_div_factor=1000
    )

    best_mcc = -1
    best_model = None
    patience = 7
    patience_counter = 0
    threshold = 0.5

    # Training loop
    print("\nStarting training...")
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            focal_loss = focal_criterion(outputs.squeeze(), batch_y)
            bce_loss = bce_criterion(outputs.squeeze(), batch_y)
            loss = loss_weight * focal_loss + (1 - loss_weight) * bce_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches

        # Validation with threshold optimization
        model.eval()
        threshold, current_mcc = find_optimal_threshold(model, val_loader)
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/30] - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"MCC Score: {current_mcc:.4f} (threshold: {threshold:.3f})")

        # Early stopping based on MCC
        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_model = copy.deepcopy(model)
            patience_counter = 0
            print(f"New best MCC: {best_mcc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Report MCC score
        trial.report(current_mcc, epoch)
        
        # Prune if performance is poor
        if trial.should_prune() or current_mcc < 0.1:  # Minimum MCC threshold
            raise optuna.TrialPruned()

    trial_time = time.time() - start_time
    print(f"\nTrial completed in {trial_time:.2f} seconds")
    print(f"Best MCC score: {best_mcc:.4f}")
    
    return best_mcc

def main():
    print("\nStarting optimization study...")
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=10,
            interval_steps=2
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
    
    print(f"MCC Score: {trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = study.trials_dataframe()
    results_df.to_csv(f"classification_optimization_mcc_{timestamp}.csv", index=False)
    
    # Save visualization plots
    try:
        import plotly
        
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(f"classification_optimization_history_mcc_{timestamp}.html")
        
        # Parameter importance
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(f"classification_parameter_importance_mcc_{timestamp}.html")
        
    except (ImportError, AttributeError) as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    main() 