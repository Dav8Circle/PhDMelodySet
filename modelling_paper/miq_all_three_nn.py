import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
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

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create PyTorch Dataset and DataLoader
class MelodyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets
train_dataset = MelodyDataset(X_train_scaled, y_train)
test_dataset = MelodyDataset(X_test_scaled, y_test)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network architecture
class MelodyNet(nn.Module):
    def __init__(self, input_size):
        super(MelodyNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.network(x)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MelodyNet(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization

# Training loop with early stopping
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100, patience=10):
    model.train()
    train_losses = []
    train_true = []
    train_pred = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_true = []
        epoch_pred = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_true.extend(y_batch.cpu().numpy())
            epoch_pred.extend(y_pred.squeeze().detach().cpu().numpy())
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        train_true = epoch_true
        train_pred = epoch_pred
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
    
    return train_losses, train_true, train_pred

# Train the model
print("\nTraining the neural network...")
train_losses, train_true, train_pred = train_model(model, train_loader, criterion, optimizer, device, num_epochs=200, patience=20)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred_batch = model(X_batch)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(y_pred_batch.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).squeeze()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return y_true, y_pred, mse, rmse, r2

# Evaluate on test set
print("\nEvaluating the model...")
y_true, y_pred, mse, rmse, r2 = evaluate_model(model, test_loader, device)

print("\nTest metrics:")
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R-squared:", round(r2, 4))

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Training set plot
ax1.scatter(train_true, train_pred, alpha=0.5, color='blue')
ax1.plot([min(train_true), max(train_true)], [min(train_true), max(train_true)], 'k--')
ax1.set_xlabel('Actual Score')
ax1.set_ylabel('Predicted Score')
ax1.set_title('Training Set: Actual vs Predicted Scores')

# Test set plot
ax2.scatter(y_true, y_pred, alpha=0.5, color='red')
ax2.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')
ax2.set_xlabel('Actual Score')
ax2.set_ylabel('Predicted Score')
ax2.set_title('Test Set: Actual vs Predicted Scores')

plt.tight_layout()
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
feature_importance = integrated_gradients(model, X_test_scaled)

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
