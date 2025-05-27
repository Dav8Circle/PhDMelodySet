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
import os
import json
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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
    def __init__(self, input_size, hidden_sizes, dropout_rates, activation_fn):
        super(ConfigurableMelodyNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size, dropout_rate in zip(hidden_sizes, dropout_rates):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LayerNorm(hidden_size),
                activation_fn(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def load_and_preprocess_data():
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

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.shape[1]

def objective(trial):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, input_size = load_and_preprocess_data()
    
    # Create data loaders
    train_dataset = MelodyDataset(X_train, y_train)
    test_dataset = MelodyDataset(X_test, y_test)
    
    # Hyperparameters to optimize
    batch_size = trial.suggest_int('batch_size', 16, 128)
    num_layers = trial.suggest_int('num_layers', 2, 5)
    
    # Define hidden layer sizes
    hidden_sizes = []
    for i in range(num_layers):
        hidden_sizes.append(trial.suggest_int(f'hidden_size_{i}', 64, 512))
    
    # Define dropout rates
    dropout_rates = []
    for i in range(num_layers):
        dropout_rates.append(trial.suggest_float(f'dropout_rate_{i}', 0.1, 0.5))
    
    # Choose activation function
    activation_fn_name = trial.suggest_categorical('activation_fn', ['ReLU', 'GELU', 'SiLU'])
    if activation_fn_name == 'ReLU':
        activation_fn = nn.ReLU
    elif activation_fn_name == 'GELU':
        activation_fn = nn.GELU
    else:  # SiLU
        activation_fn = nn.SiLU
    
    # Learning rate and optimizer parameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ConfigurableMelodyNet(input_size, hidden_sizes, dropout_rates, activation_fn)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler_type = trial.suggest_categorical('scheduler', ['OneCycleLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'])
    
    if scheduler_type == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=30,
            steps_per_epoch=len(train_loader),
            pct_start=trial.suggest_float('pct_start', 0.1, 0.5)
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=trial.suggest_float('factor', 0.1, 0.9),
            patience=trial.suggest_int('patience', 3, 10),
            verbose=False
        )
    else:  # CosineAnnealingLR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=trial.suggest_int('T_max', 5, 20),
            eta_min=lr * 0.01
        )
    
    # Training parameters
    num_epochs = trial.suggest_int('num_epochs', 50, 200)
    patience = trial.suggest_int('patience', 10, 30)
    
    # Training loop
    best_test_loss = float('inf')
    patience_counter = 0
    best_r2 = -float('inf')
    
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
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                test_loss += loss.item()
                predictions.extend(outputs.squeeze().numpy())
                actuals.extend(batch_y.numpy())
        
        # Calculate R2 score
        r2 = r2_score(actuals, predictions)
        
        # Update scheduler
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(test_loss)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler.step()
        else:  # OneCycleLR
            scheduler.step()
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_r2 = r2
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # Report the best R2 score
    trial.report(best_r2, epoch)
    
    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return best_r2

def main():
    # Create a study object
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # Start optimization
    study.optimize(objective, n_trials=50)
    
    # Print the best trial
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value (R2): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save the best parameters to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "optimization_results"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f"{results_dir}/best_params_{timestamp}.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/optimization_history_{timestamp}.png")
    plt.close()
    
    # Plot parameter importance
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/param_importance_{timestamp}.png")
    plt.close()
    
    # Plot parallel coordinate plot
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/parallel_coordinate_{timestamp}.png")
    plt.close()
    
    print(f"Results saved to {results_dir}/")

if __name__ == "__main__":
    main() 