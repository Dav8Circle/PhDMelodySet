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
import json
import os

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

def get_feature_importance(model, X, feature_names):
    base_pred = model(torch.FloatTensor(X)).detach().numpy().mean()
    importance = []
    
    for i in range(X.shape[1]):
        X_modified = X.copy()
        X_modified[:, i] = 0
        new_pred = model(torch.FloatTensor(X_modified)).detach().numpy().mean()
        importance.append(abs(new_pred - base_pred))
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

def calculate_accuracy(predictions, actuals, tolerance=0.1):
    # Convert predictions and actuals to numpy arrays if they aren't already
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate absolute differences
    differences = np.abs(predictions - actuals)
    
    # Count predictions within tolerance
    correct_predictions = np.sum(differences <= tolerance)
    
    # Calculate accuracy
    accuracy = correct_predictions / len(predictions)
    
    return accuracy

def load_best_params():
    """Load the best parameters from the optimization results"""
    param_file = "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/optimization_results_round2/best_params_20250404_155523.json"
    print(f"Loading best parameters from: {param_file}")
    
    with open(param_file, 'r') as f:
        best_params = json.load(f)
    
    return best_params

def main():
    # Load the best hyperparameters
    best_params = load_best_params()
    print("Best hyperparameters loaded successfully")
    
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

    # Create data loaders
    train_dataset = MelodyDataset(X_train, y_train)
    test_dataset = MelodyDataset(X_test, y_test)

    # Use the batch size from the best parameters
    batch_size = best_params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract hyperparameters
    num_layers = best_params['num_layers']
    hidden_sizes = [best_params[f'hidden_size_{i}'] for i in range(num_layers)]
    dropout_rates = [best_params[f'dropout_rate_{i}'] for i in range(num_layers)]
    
    # Set activation function
    activation_fn_name = best_params['activation_fn']
    if activation_fn_name == 'ReLU':
        activation_fn = nn.ReLU
    elif activation_fn_name == 'GELU':
        activation_fn = nn.GELU
    elif activation_fn_name == 'SiLU':
        activation_fn = nn.SiLU
    elif activation_fn_name == 'LeakyReLU':
        activation_fn = lambda: nn.LeakyReLU(0.1)  # Default alpha value
    else:  # ELU
        activation_fn = nn.ELU
    
    # Initialize model with the best hyperparameters
    model = ConfigurableMelodyNet(
        input_size=X.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rates=dropout_rates,
        activation_fn=activation_fn,
        use_batch_norm=best_params['use_batch_norm'],
        use_residual=best_params['use_residual']
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    
    # Choose optimizer based on best parameters
    optimizer_type = best_params['optimizer']
    lr = best_params['lr']
    weight_decay = best_params['weight_decay']
    
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        momentum = best_params.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:  # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler_type = best_params['scheduler']
    if scheduler_type == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=best_params['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=best_params.get('pct_start', 0.3)
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=best_params.get('factor', 0.5),
            patience=best_params.get('scheduler_patience', 5),
            verbose=True
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=best_params.get('T_max', 10),
            eta_min=lr * 0.01
        )
    elif scheduler_type == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=best_params.get('gamma', 0.9)
        )
    else:
        scheduler = None
    
    # Training parameters
    num_epochs = best_params['num_epochs']
    patience = best_params['patience']
    
    # Training loop
    print("\nStarting training...")
    best_test_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    r2_scores = []

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
        r2_scores.append(r2)
        
        # Update scheduler
        if scheduler is not None:
            if scheduler_type == 'ReduceLROnPlateau':
                scheduler.step(test_loss)
            else:
                scheduler.step()
        
        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, R²: {r2:.4f}')

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model_optimised.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Plot training curves
    plt.figure(figsize=(12, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training and Test Loss Over Time', fontsize=14, pad=15)
    plt.grid(True, which="both", ls="-", alpha=0.2)  # Add grid for both major and minor ticks
    plt.legend(fontsize=10)
    
    # Plot R² scores
    plt.subplot(2, 1, 2)
    plt.plot(r2_scores, label='R² Score', linewidth=2, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('R² Score Over Time', fontsize=14, pad=15)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('training_curves_optimised.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Evaluate model
    print("\nEvaluating model...")
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().numpy())
            actuals.extend(batch_y.numpy())

    # Calculate metrics
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate accuracy with different tolerance levels
    accuracy_0_1 = calculate_accuracy(predictions, actuals, tolerance=0.05)
    accuracy_0_2 = calculate_accuracy(predictions, actuals, tolerance=0.1)
    accuracy_0_3 = calculate_accuracy(predictions, actuals, tolerance=0.2)

    print(f"\nModel Performance:")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Accuracy (±0.05): {accuracy_0_1:.4f}")
    print(f"Accuracy (±0.1): {accuracy_0_2:.4f}")
    print(f"Accuracy (±0.2): {accuracy_0_3:.4f}")

    # Plot predictions vs actuals
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.savefig('predictions_vs_actuals_optimised.png')
    plt.close()

    # Feature importance analysis
    print("\nCalculating feature importance...")
    feature_importance = get_feature_importance(model, X_test, feature_diffs.columns)

    # Plot top 10 most important features
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance_optimised.png')
    plt.close()

    # Save feature importance to CSV
    feature_importance.to_csv('feature_importance_optimised.csv', index=False)
    
    # Save the hyperparameters used
    with open('hyperparameters_used.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print("\nHyperparameters saved to 'hyperparameters_used.json'")

if __name__ == "__main__":
    main() 