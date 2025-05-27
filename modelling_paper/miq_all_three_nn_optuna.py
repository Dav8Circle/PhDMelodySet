import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna

# Set random seeds
torch.manual_seed(8)
np.random.seed(8)

# 1. Load data
print("Loading original features...")
original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
print("Loading odd one out features...")
odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")
print("Loading participant responses...")
participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=int(1e6))

# Filter participant_responses to only include 'mdt' test
participant_responses = participant_responses[participant_responses['test'] == 'mdt']

# Get IRT features
irt_features = participant_responses[['item_id', 'ability_WL', 'difficulty']].copy()
irt_features = irt_features.groupby('item_id')[['ability_WL', 'difficulty']].mean().reset_index()

# Compute mean score per melody
mean_scores = participant_responses.groupby('item_id')['score'].mean().reset_index()
mean_scores = mean_scores.rename(columns={'score': 'mean_score'})

# Drop duration features from original and odd-one-out features
duration_cols_orig = [col for col in original_features.columns if 'duration_features.' in col]
duration_cols_ooo = [col for col in odd_one_out_features.columns if 'duration_features.' in col]
print("Dropping duration features from original:", duration_cols_orig)
print("Dropping duration features from odd-one-out:", duration_cols_ooo)
original_features = original_features.drop(columns=duration_cols_orig)
odd_one_out_features = odd_one_out_features.drop(columns=duration_cols_ooo)

# Merge features on melody_id
if 'melody_id' not in original_features.columns:
    raise ValueError('melody_id column missing from original_features')
if 'melody_id' not in odd_one_out_features.columns:
    raise ValueError('melody_id column missing from odd_one_out_features')

features_merged = original_features.merge(odd_one_out_features, on='melody_id', suffixes=('_orig', '_ooo'))
# Only use numeric columns for difference calculation
orig_num = features_merged.filter(regex='_orig$').select_dtypes(include=[np.number]).copy()
ooo_num = features_merged.filter(regex='_ooo$').select_dtypes(include=[np.number]).copy()
feature_diffs = ooo_num.values - orig_num.values
feature_diffs = pd.DataFrame(feature_diffs, columns=[col.replace('_orig', '') + '_diff' for col in orig_num.columns])
# Drop any columns with zero variance
non_zero_var_cols = feature_diffs.columns[feature_diffs.var() != 0]
feature_diffs = feature_diffs[non_zero_var_cols]

# Concatenate all features
features_final = pd.concat([features_merged, feature_diffs], axis=1)

# Drop zero variance columns before merging with mean scores
zero_var_cols = [col for col in features_final.columns if features_final[col].nunique() == 1]
print("Dropping zero variance columns:", zero_var_cols)
features_final = features_final.drop(columns=zero_var_cols)
# Drop duration features which are constant across the dataset
duration_cols = [col for col in features_final.columns if 'duration' in col.lower()]
print("Dropping duration columns:", duration_cols)
features_final = features_final.drop(columns=duration_cols)

# Merge with mean scores and IRT features
data = features_final.merge(mean_scores, left_on='melody_id', right_on='item_id')
data = data.merge(irt_features, on='item_id', how='left')

# Drop any rows with missing values
data = data.dropna()

# Prepare X and y
exclude_cols = {'melody_id', 'item_id', 'mean_score'}
feature_cols = [col for col in data.columns if col not in exclude_cols]

# Only keep numeric columns for modeling
numeric_feature_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [col for col in feature_cols if col not in numeric_feature_cols]
if non_numeric_cols:
    print("Dropping non-numeric columns:", non_numeric_cols)

X = data[numeric_feature_cols].values
y = data['mean_score'].values

# Train/test split by melody_id
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
groups = data['melody_id'].values
for train_idx, test_idx in gss.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train_indices, X_test_indices = train_idx, test_idx

# Create a fixed validation split from training data
val_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=456)
train_groups = groups[train_idx]
for train_val_idx, val_idx in val_gss.split(X_train, y_train, groups=train_groups):
    X_train_final, X_val = X_train[train_val_idx], X_train[val_idx]
    y_train_final, y_val = y_train[train_val_idx], y_train[val_idx]

# Scale features
scaler = StandardScaler()
X_train_final_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

class MelodyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class OptimizedMelodyNet(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rates):
        super(OptimizedMelodyNet, self).__init__()
        layers = []
        prev_size = input_size
        
        for size, dropout in zip(layer_sizes, dropout_rates):
            layers.extend([
                nn.Linear(prev_size, size),
                nn.BatchNorm1d(size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = size
            
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def objective(trial):
    # Hyperparameters to optimize
    n_layers = trial.suggest_int('n_layers', 2, 3)
    layer_sizes = []
    dropout_rates = []
    
    # First layer
    layer_sizes.append(trial.suggest_int('layer_0_size', 256, 384))
    dropout_rates.append(trial.suggest_float('dropout_0', 0.2, 0.3))
    
    # Middle layers
    for i in range(1, n_layers):
        prev_size = layer_sizes[-1]
        layer_sizes.append(trial.suggest_int(f'layer_{i}_size', prev_size//2, prev_size))
        dropout_rates.append(trial.suggest_float(f'dropout_{i}', 0.2, 0.3))
    
    # Final layer
    layer_sizes.append(trial.suggest_int(f'layer_{n_layers}_size', 64, 128))
    dropout_rates.append(trial.suggest_float(f'dropout_{n_layers}', 0.1, 0.2))
    
    batch_size = trial.suggest_int('batch_size', 32, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Create datasets and dataloaders
    train_dataset = MelodyDataset(X_train_final_scaled, y_train_final)
    val_dataset = MelodyDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedMelodyNet(X_train.shape[1], layer_sizes, dropout_rates).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_r2 = -float('inf')
    best_model_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(150):
        # Training phase
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
        # Validation phase
        model.eval()
        val_preds = []
        val_true = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                val_preds.extend(y_pred.cpu().numpy())
                val_true.extend(y_batch.numpy())
                
        val_r2 = r2_score(val_true, np.array(val_preds).squeeze())
        scheduler.step(val_r2)
        
        if val_r2 > best_r2:
            best_r2 = val_r2
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Restore best model state
    model.load_state_dict(best_model_state)
    return best_r2

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

# Train final model with best parameters
best_params = study.best_params
layer_sizes = [best_params[f'layer_{i}_size'] for i in range(best_params['n_layers'] + 1)]
dropout_rates = [best_params[f'dropout_{i}'] for i in range(best_params['n_layers'] + 1)]

# Create final datasets and dataloaders
train_dataset = MelodyDataset(X_train_final_scaled, y_train_final)
test_dataset = MelodyDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

# Initialize and train final model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model = OptimizedMelodyNet(X_train.shape[1], layer_sizes, dropout_rates).to(device)
criterion = nn.MSELoss()
optimizer = Adam(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Train final model
print("\nTraining final model...")
best_r2 = -float('inf')
best_model_state = None
patience = 20
patience_counter = 0
train_losses = []
train_r2s = []

for epoch in range(150):
    final_model.train()
    epoch_loss = 0
    epoch_preds = []
    epoch_true = []
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = final_model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_preds.extend(y_pred.detach().cpu().numpy())
        epoch_true.extend(y_batch.cpu().numpy())
        
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    epoch_r2 = r2_score(epoch_true, np.array(epoch_preds).squeeze())
    train_r2s.append(epoch_r2)
    scheduler.step(epoch_r2)
    
    if epoch_r2 > best_r2:
        best_r2 = epoch_r2
        best_model_state = final_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Restore best model state
final_model.load_state_dict(best_model_state)

# Evaluate final model
final_model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_pred_batch = final_model(X_batch)
        y_true.extend(y_batch.numpy())
        y_pred.extend(y_pred_batch.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred).squeeze()

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print("\nFinal Test Metrics:")
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R-squared:", round(r2, 4))

# Plot training metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses)
ax1.set_title('Training Loss Over Time')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(train_r2s)
ax2.set_title('Training R² Over Time')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('R²')

plt.tight_layout()
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Optimized Neural Network: Actual vs Predicted Scores (Test Set)')
plt.show()

# Feature importance analysis using integrated gradients
def integrated_gradients(model, X, steps=50):
    model.eval()
    X = torch.FloatTensor(X).to(device)
    
    # Get baseline (zeros)
    baseline = torch.zeros_like(X)
    
    # Calculate importance scores
    importance = torch.zeros_like(X)
    
    for step in range(steps):
        # Interpolate between baseline and input
        x_step = baseline + (X - baseline) * (step / steps)
        x_step.requires_grad_(True)
        
        # Forward pass
        y_step = model(x_step)
        
        # Sum the outputs to get a scalar
        y_step_sum = y_step.sum()
        
        # Backward pass
        y_step_sum.backward()
        
        # Accumulate gradients
        if x_step.grad is not None:
            importance += x_step.grad.detach()
        
        # Clear gradients
        x_step.grad = None
    
    # Average gradients
    importance = importance / steps
    
    # Calculate final importance scores
    importance = torch.abs(importance * (X - baseline))
    importance = importance.mean(dim=0)
    
    return importance.cpu().numpy()

# Calculate feature importance
print("\nCalculating feature importance...")
feature_importance = integrated_gradients(final_model, X_test_scaled)

# Plot top 20 most important features
plt.figure(figsize=(12, 8))
plt.title("Top 20 Most Important Features", pad=20, fontsize=14)

# Get indices of top 20 features
top_indices = np.argsort(feature_importance)[-20:][::-1]
top_features = [numeric_feature_cols[i] for i in top_indices]
top_importance = feature_importance[top_indices]

# Create horizontal bar plot
bars = plt.barh(range(19, -1, -1), top_importance, color='skyblue', alpha=0.8)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}',
            ha='left', va='center', fontsize=8)

plt.yticks(range(19, -1, -1), top_features, ha='right', fontsize=10)
plt.xlabel("Feature Importance Score", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Print top 20 feature importances
print("\nTop 20 Feature Importances:")
for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
    print(f"{i+1}. {feature}: {importance:.4f}") 