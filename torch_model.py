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

# Define the neural network
class MelodyNet(nn.Module):
    def __init__(self, input_size):
        super(MelodyNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # Skip connection handling different sizes
        self.skip = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features)
        ) if in_features != out_features else nn.Identity()
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.block(x) + self.skip(x))

class EnhancedMelodyNet(nn.Module):
    def __init__(self, input_size):
        super(EnhancedMelodyNet, self).__init__()
        
        # Initial layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 416),
            nn.LayerNorm(416),
            nn.GELU(),
            nn.Dropout(0.242)
        )
        
        # Hidden layer
        self.hidden_layer = nn.Sequential(
            nn.Linear(416, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.146)
        )
        
        # Final prediction layer with dropout
        self.final_layer = nn.Sequential(
            nn.Dropout(0.402),
            nn.Linear(320, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.final_layer(x)
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

def main():
    # Load the data
    print("Loading data...")
    original_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv")
    miq_features = pd.read_csv("/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv")
    participant_responses = pd.read_csv("/Users/davidwhyatt/Downloads/miq_trials.csv", nrows=1e7)

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = EnhancedMelodyNet(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.00452, weight_decay=8.58e-5)

    # Add learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.00452,
        epochs=30,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    best_test_loss = float('inf')
    patience = 20
    patience_counter = 0

    # Training loop
    print("\nStarting training...")
    num_epochs = 1000
    train_losses = []
    test_losses = []

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
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                test_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

        # After calculating test_loss
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('Training and Test Loss Over Time', fontsize=14, pad=15)
    plt.grid(True, which="both", ls="-", alpha=0.2)  # Add grid for both major and minor ticks
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
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

    # Plot predictions vs actuals with R² in title
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values (R² = {r2:.4f})')
    plt.savefig('predictions_vs_actuals.png')
    plt.close()

    # Plot R² vs Loss
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Test Loss Over Time')
    plt.grid(True, alpha=0.2)
    plt.legend()

    plt.subplot(1, 2, 2)
    # Calculate R² for each epoch using test predictions
    epoch_r2s = []
    model.eval()
    with torch.no_grad():
        for epoch in range(len(test_losses)):  # Use same number of points as losses
            epoch_preds = []
            epoch_acts = []
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                epoch_preds.extend(outputs.squeeze().numpy())
                epoch_acts.extend(batch_y.numpy())
            epoch_r2 = r2_score(epoch_acts, epoch_preds)
            epoch_r2s.append(epoch_r2)
    
    plt.plot(epoch_r2s, label='Test R²', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('Test R² Over Time')
    plt.grid(True, alpha=0.2)
    plt.ylim(-0.1, 1.0)
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Feature importance analysis
    print("\nCalculating feature importance...")
    feature_importance = get_feature_importance(model, X_test, feature_diffs.columns)

    # Plot top 10 most important features
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Save feature importance to CSV
    feature_importance.to_csv('feature_importance.csv', index=False)

if __name__ == "__main__":
    main()
