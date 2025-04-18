import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import sys

# Import the model architecture from torch_model.py
from torch_model import EnhancedMelodyNet, MelodyDataset, get_feature_importance

# Try to import visualization libraries
try:
    from torchviz import make_dot
    import graphviz
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization libraries not available. Install with: pip install torchviz graphviz")

def visualize_model(model, input_size, save_path='model_architecture'):
    """
    Visualize the model architecture using torchviz
    
    Args:
        model: PyTorch model
        input_size: Number of input features
        save_path: Path to save the visualization
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping model visualization.")
        return
    
    # Create a dummy input
    dummy_input = torch.randn(1, input_size)
    
    # Generate the computation graph
    y = model(dummy_input)
    
    # Create the visualization
    dot = make_dot(y, params=dict(model.named_parameters()))
    
    # Save the visualization
    dot.render(save_path, format='png', cleanup=True)
    print(f"Model architecture visualization saved to {save_path}.png")

def visualize_model_layers(model, save_path='model_layers.png'):
    """
    Create a simple visualization of model layers using matplotlib
    
    Args:
        model: PyTorch model
        save_path: Path to save the visualization
    """
    # Extract layer information
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU, nn.GELU, nn.Dropout, nn.LayerNorm)):
            layers.append((name, module))
    
    # Create a simple visualization
    plt.figure(figsize=(12, 8))
    
    # Plot each layer
    for i, (name, layer) in enumerate(layers):
        # Determine layer type and color
        if isinstance(layer, nn.Linear):
            color = 'lightblue'
            layer_type = f"Linear({layer.in_features}, {layer.out_features})"
        elif isinstance(layer, nn.ReLU):
            color = 'lightgreen'
            layer_type = "ReLU"
        elif isinstance(layer, nn.GELU):
            color = 'lightgreen'
            layer_type = "GELU"
        elif isinstance(layer, nn.Dropout):
            color = 'salmon'
            layer_type = f"Dropout({layer.p})"
        elif isinstance(layer, nn.LayerNorm):
            color = 'lightyellow'
            layer_type = f"LayerNorm({layer.normalized_shape})"
        else:
            color = 'lightgray'
            layer_type = layer.__class__.__name__
        
        # Plot the layer
        plt.barh(i, 1, color=color, edgecolor='black')
        plt.text(0.05, i, f"{name}: {layer_type}", va='center')
    
    plt.yticks(range(len(layers)), [name for name, _ in layers])
    plt.xlabel('Layer')
    plt.title('Model Architecture')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Model layers visualization saved to {save_path}")

def load_model(model_path, input_size):
    """
    Load a trained model from a .pth file
    
    Args:
        model_path (str): Path to the saved model file
        input_size (int): Number of input features
        
    Returns:
        model: Loaded PyTorch model
    """
    model = EnhancedMelodyNet(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    return model

def prepare_data(original_features_path, miq_features_path, participant_responses_path):
    """
    Prepare data in the same way as the training script
    
    Args:
        original_features_path (str): Path to original features CSV
        miq_features_path (str): Path to MIQ features CSV
        participant_responses_path (str): Path to participant responses CSV
        
    Returns:
        X: Feature matrix
        y: Target values
        feature_names: Names of the features
        scaler: Fitted StandardScaler
    """
    # Load the data
    print("Loading data...")
    original_features = pd.read_csv(original_features_path)
    miq_features = pd.read_csv(miq_features_path)
    participant_responses = pd.read_csv(participant_responses_path, nrows=10000000)

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
    feature_names = data.drop(['melody_id', 'item_id', 'score'], axis=1).columns.tolist()
    
    return X, y, feature_names, scaler

def predict(model, X, batch_size=32):
    """
    Make predictions using the loaded model
    
    Args:
        model: Loaded PyTorch model
        X: Feature matrix
        batch_size (int): Batch size for prediction
        
    Returns:
        predictions: Array of predictions
    """
    # Create dataset and dataloader
    dataset = MelodyDataset(X, np.zeros(len(X)))  # Dummy y values
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for batch_X, _ in dataloader:
            outputs = model(batch_X)
            predictions.extend(outputs.squeeze().numpy())
    
    return np.array(predictions)

def evaluate_predictions(predictions, actuals):
    """
    Evaluate model predictions
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate accuracy with different tolerance levels
    def calculate_accuracy(pred, act, tolerance=0.1):
        differences = np.abs(pred - act)
        correct_predictions = np.sum(differences <= tolerance)
        return correct_predictions / len(pred)
    
    accuracy_0_1 = calculate_accuracy(predictions, actuals, tolerance=0.05)
    accuracy_0_2 = calculate_accuracy(predictions, actuals, tolerance=0.1)
    accuracy_0_3 = calculate_accuracy(predictions, actuals, tolerance=0.2)
    
    metrics = {
        'R-squared': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Accuracy (±0.05)': accuracy_0_1,
        'Accuracy (±0.1)': accuracy_0_2,
        'Accuracy (±0.2)': accuracy_0_3
    }
    
    return metrics

def plot_predictions(predictions, actuals, save_path='predictions_vs_actuals.png'):
    """
    Plot predictions vs actuals
    
    Args:
        predictions: Array of predictions
        actuals: Array of actual values
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual Values (R² = {r2_score(actuals, predictions):.4f})')
    plt.savefig(save_path)
    plt.close()

def main():
    # Paths to data files
    original_features_path = "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/original_mel_miq_mels.csv"
    miq_features_path = "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/miq_mels.csv"
    participant_responses_path = "/Users/davidwhyatt/Downloads/miq_trials.csv"
    model_path = "best_model.pth"
    
    # Prepare data
    X, y, feature_names, scaler = prepare_data(
        original_features_path, 
        miq_features_path, 
        participant_responses_path
    )
    
    # Load model
    input_size = X.shape[1]
    model = load_model(model_path, input_size)
    
    # Visualize model architecture
    print("\nVisualizing model architecture...")
    visualize_model(model, input_size)
    visualize_model_layers(model)
    
    # Make predictions
    predictions = predict(model, X)
    
    # Evaluate predictions
    metrics = evaluate_predictions(predictions, y)
    
    # Print metrics
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions
    plot_predictions(predictions, y)
    
    # Feature importance analysis
    print("\nCalculating feature importance...")
    feature_importance = get_feature_importance(model, X, feature_names)
    
    # Plot top 10 most important features
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Save feature importance to CSV
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    print("\nAnalysis complete. Results saved to:")
    print("- model_architecture.png")
    print("- model_layers.png")
    print("- predictions_vs_actuals.png")
    print("- feature_importance.png")
    print("- feature_importance.csv")

if __name__ == "__main__":
    main() 