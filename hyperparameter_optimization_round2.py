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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import json
import os
import random
from datetime import datetime
import glob

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Create custom dataset
class MelodyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the neural network with configurable architecture
class ConfigurableMelodyNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates, activation_fn, use_batch_norm=False, use_residual=False):
        super(ConfigurableMelodyNet, self).__init__()
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size, dropout_rate in zip(hidden_sizes, dropout_rates):
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization if enabled
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            else:
                self.batch_norms.append(nn.LayerNorm(hidden_size))
            
            # Activation function
            self.activations.append(activation_fn())
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            # For residual connections
            if use_residual and prev_size == hidden_size:
                self.residual = True
            else:
                self.residual = False
                
            prev_size = hidden_size
        
        # Add output layer
        self.output = nn.Linear(prev_size, 1)
    
    def forward(self, x):
        for i, (layer, norm, activation, dropout) in enumerate(zip(self.layers, self.batch_norms, self.activations, self.dropouts)):
            identity = x
            x = layer(x)
            x = norm(x)
            x = activation(x)
            x = dropout(x)
            
            # Add residual connection if enabled and dimensions match
            if self.use_residual and self.residual and i > 0:
                x = x + identity
        
        x = self.output(x)
        return x

def load_and_preprocess_data():
    """Load and preprocess the data"""
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

    # Prepare target variable (mean scores)
    participant_responses = participant_responses[participant_responses['test'] == 'mdt']
    mean_scores = participant_responses.groupby('item_id')['score'].mean().reset_index()

    # Merge features with mean scores
    data = pd.DataFrame(feature_diffs_scaled, columns=feature_diffs.columns)
    data['melody_id'] = melody_id
    data = data.merge(mean_scores, left_on='melody_id', right_on='item_id')
    data = data.dropna()

    # Split features and target
    X = data.drop(['melody_id', 'item_id', 'score'], axis=1).values
    y = data['score'].values

    return X, y, feature_diffs.columns

def load_best_params():
    """Load the best parameters from the first round of optimization"""
    # Find the most recent JSON file in the optimization_results directory
    param_files = glob.glob("optimization_results/best_params_*.json")
    if not param_files:
        print("No parameter files found in optimization_results directory")
        return None
    
    # Sort by modification time (most recent first)
    param_files.sort(key=os.path.getmtime, reverse=True)
    param_file = param_files[0]
    print(f"Loading best parameters from: {param_file}")
    
    with open(param_file, 'r') as f:
        best_params = json.load(f)
    
    return best_params

def objective(trial):
    """Define the objective function for Optuna"""
    # Load data
    X, y, feature_names = load_and_preprocess_data()
    
    # Load best parameters from first round
    best_params = load_best_params()
    
    # FIXED PARAMETERS - these will remain constant
    num_layers = 3  # Fixed number of layers
    optimizer_type = 'AdamW'  # Fixed optimizer
    scheduler_type = 'ExponentialLR'  # Fixed scheduler
    use_batch_norm = False  # Fixed batch normalization
    use_residual = True  # Fixed residual connections
    activation_fn = nn.SiLU  # Fixed activation function
    
    # PARAMETERS TO OPTIMIZE - only a few key ones
    # Hidden layer sizes - use a more structured approach
    hidden_sizes = []
    prev_size = X.shape[1]
    
    for i in range(num_layers):
        # Use a factor of the previous layer size
        min_size = max(32, prev_size // 4)
        max_size = min(1024, prev_size * 2)
        hidden_size = trial.suggest_int(f'hidden_size_{i}', min_size, max_size)
        hidden_sizes.append(hidden_size)
        prev_size = hidden_size
    
    # Dropout rates - use a decreasing pattern
    dropout_rates = []
    for i in range(num_layers):
        # Higher dropout in earlier layers, lower in later layers
        min_dropout = 0.05
        max_dropout = 0.5 - (i * 0.1)  # Decreases with layer depth
        dropout_rate = trial.suggest_float(f'dropout_rate_{i}', min_dropout, max_dropout)
        dropout_rates.append(dropout_rate)
    
    # Training parameters
    batch_size = trial.suggest_int('batch_size', 8, 64)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # Learning rate scheduler parameters
    gamma = trial.suggest_float('gamma', 0.8, 0.99)
    
    # Training duration
    num_epochs = 100  # Fixed number of epochs
    patience = 20  # Fixed patience
    
    # Single train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_dataset = MelodyDataset(X_train, y_train)
    val_dataset = MelodyDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ConfigurableMelodyNet(
        input_size=X.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rates=dropout_rates,
        activation_fn=activation_fn,
        use_batch_norm=use_batch_norm,
        use_residual=use_residual
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_r2 = -float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
                predictions.extend(outputs.squeeze().numpy())
                actuals.extend(batch_y.numpy())
        
        # Calculate R2 score
        r2 = r2_score(actuals, predictions)
        
        # Update best R2 score
        if r2 > best_r2:
            best_r2 = r2
        
        # Update scheduler
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_r2

def main():
    # Create directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"optimization_results_round2_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create Optuna study
    study = optuna.create_study(direction="maximize")
    
    # Run optimization
    print("Starting hyperparameter optimization...")
    study.optimize(objective, n_trials=30, timeout=3600)  # 30 trials or 1 hour
    
    # Print best trial
    print("\nBest trial:")
    print(f"  R²: {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters
    best_params = study.best_trial.params
    with open(f"{results_dir}/best_params_{timestamp}.json", 'w') as f:
        json.dump(best_params, f, indent=4)
    
    # Save optimization history
    history = []
    for trial in study.trials:
        if trial.value is not None:
            history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params
            })
    
    with open(f"{results_dir}/optimization_history_{timestamp}.json", 'w') as f:
        json.dump(history, f, indent=4)
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f"{results_dir}/optimization_history_{timestamp}.png")
    plt.close()
    
    # Plot parameter importance
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f"{results_dir}/param_importance_{timestamp}.png")
    plt.close()
    
    # Plot parallel coordinate plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.savefig(f"{results_dir}/parallel_coordinate_{timestamp}.png")
    plt.close()
    
    # Plot slice plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(f"{results_dir}/slice_plot_{timestamp}.png")
    plt.close()
    
    print(f"\nOptimization results saved to {results_dir}/")

if __name__ == "__main__":
    main() 