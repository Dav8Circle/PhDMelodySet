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
torch.manual_seed(123)
np.random.seed(123)

# Load and prepare data (same as miq_all_three_nn.py)
print("Loading original features...")
original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
print("Loading odd one out features...")
odd_one_out_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")
print("Loading participant responses...")
participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=int(1e6))

# Data preprocessing (same as before)
participant_responses = participant_responses[participant_responses['test'] == 'mdt']
mean_scores = participant_responses.groupby('item_id')['score'].mean().reset_index()
mean_scores = mean_scores.rename(columns={'score': 'mean_score'})

duration_cols_orig = [col for col in original_features.columns if 'duration_features.' in col]
duration_cols_ooo = [col for col in odd_one_out_features.columns if 'duration_features.' in col]
original_features = original_features.drop(columns=duration_cols_orig)
odd_one_out_features = odd_one_out_features.drop(columns=duration_cols_ooo)

features_merged = original_features.merge(odd_one_out_features, on='melody_id', suffixes=('_orig', '_ooo'))
orig_num = features_merged.filter(regex='_orig$').select_dtypes(include=[np.number]).copy()
ooo_num = features_merged.filter(regex='_ooo$').select_dtypes(include=[np.number]).copy()
feature_diffs = ooo_num.values - orig_num.values
feature_diffs = pd.DataFrame(feature_diffs, columns=[col.replace('_orig', '') + '_diff' for col in orig_num.columns])
non_zero_var_cols = feature_diffs.columns[feature_diffs.var() != 0]
feature_diffs = feature_diffs[non_zero_var_cols]

features_final = pd.concat([features_merged, feature_diffs], axis=1)
zero_var_cols = [col for col in features_final.columns if features_final[col].nunique() == 1]
features_final = features_final.drop(columns=zero_var_cols)
duration_cols = [col for col in features_final.columns if 'duration' in col.lower()]
features_final = features_final.drop(columns=duration_cols)

data = features_final.merge(mean_scores, left_on='melody_id', right_on='item_id')
data = data.dropna()

exclude_cols = {'melody_id', 'item_id', 'mean_score'}
feature_cols = [col for col in data.columns if col not in exclude_cols]
numeric_feature_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

X = data[numeric_feature_cols].values
y = data['mean_score'].values

# Train/test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
groups = data['melody_id'].values
for train_idx, test_idx in gss.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
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
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout)
            ])
            prev_size = size
            
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def objective(trial):
    # Hyperparameters to optimize
    n_layers = trial.suggest_int('n_layers', 2, 4)
    layer_sizes = []
    dropout_rates = []
    
    for i in range(n_layers):
        layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 32, 512))
        dropout_rates.append(trial.suggest_float(f'dropout_{i}', 0.1, 0.5))
    
    batch_size = trial.suggest_int('batch_size', 16, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    # Create datasets and dataloaders
    train_dataset = MelodyDataset(X_train_scaled, y_train)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)
    
    # Initialize model and training components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OptimizedMelodyNet(X_train.shape[1], layer_sizes, dropout_rates).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=trial.suggest_loguniform('weight_decay', 1e-6, 1e-3))
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    best_r2 = -float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
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
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    return best_r2

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Increased number of trials

# Print best parameters
print("Best trial:")
trial = study.best_trial
print("  R2 Score: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Train final model with best parameters
best_params = study.best_params
layer_sizes = [best_params[f'layer_{i}_size'] for i in range(best_params['n_layers'])]
dropout_rates = [best_params[f'dropout_{i}'] for i in range(best_params['n_layers'])]

# Create final datasets and dataloaders
train_dataset = MelodyDataset(X_train_scaled, y_train)
test_dataset = MelodyDataset(X_test_scaled, y_test)
train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# Initialize and train final model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
final_model = OptimizedMelodyNet(X_train.shape[1], layer_sizes, dropout_rates).to(device)
criterion = nn.MSELoss()
optimizer = Adam(final_model.parameters(), lr=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

# Train final model
print("\nTraining final model...")
best_r2 = -float('inf')
patience = 20  # Increased patience for final training
patience_counter = 0
train_losses = []
train_r2s = []

for epoch in range(300):  # Increased epochs for final training
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
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/300], Loss: {avg_loss:.4f}, R2: {epoch_r2:.4f}')
    
    if epoch_r2 > best_r2:
        best_r2 = epoch_r2
        patience_counter = 0
        # Save best model state
        torch.save(final_model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Load best model state
final_model.load_state_dict(torch.load('best_model.pth'))

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
