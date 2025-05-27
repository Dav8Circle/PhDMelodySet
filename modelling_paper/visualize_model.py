import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Import the model architecture from torch_model.py
from torch_model import EnhancedMelodyNet

# Try to import visualization libraries
try:
    from torchviz import make_dot
    import graphviz
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization libraries not available. Install with: pip install torchviz graphviz")

def visualize_model(model, input_size, save_path='model_architecture.png'):
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
    
    try:
        # Create a dummy input
        dummy_input = torch.randn(1, input_size)
        
        # Generate the computation graph
        y = model(dummy_input)
        
        # Create the visualization
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        # Save the visualization
        dot.render(save_path, format='png', cleanup=True)
        print(f"Model architecture visualization saved to {save_path}.png")
    except Exception as e:
        print(f"Error creating model architecture visualization: {e}")
        print("This might be because Graphviz is not installed on your system.")
        print("You can install it with: brew install graphviz (on macOS) or apt-get install graphviz (on Linux)")
        print("Falling back to alternative visualization method...")
        visualize_model_alternative(model, input_size, save_path)

def visualize_model_alternative(model, input_size, save_path='model_architecture_alternative.png'):
    """
    Alternative visualization method that creates a neural network diagram
    similar to traditional NN visualizations, with efficient rendering
    
    Args:
        model: PyTorch model
        input_size: Number of input features
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 8))
    
    # Define layer structure
    layer_structure = []
    
    # Input layer
    layer_structure.append(('input', input_size))
    
    # Get hidden layers
    hidden_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hidden_layers.append((name, module.out_features))
    
    # Add hidden layers and output layer
    for layer in hidden_layers[:-1]:  # All except last
        layer_structure.append(layer)
    
    # Add output layer (last linear layer)
    layer_structure.append(('output', 1))
    
    # Calculate positions
    num_layers = len(layer_structure)
    layer_positions = np.linspace(0, 1, num_layers)
    
    # Colors for different types of layers
    colors = {
        'input': 'royalblue',
        'hidden': 'forestgreen',
        'output': 'orange'
    }
    
    # Function to get representative nodes for connections
    def get_representative_nodes(size, max_nodes=5):
        if size <= max_nodes:
            return list(range(size))
        else:
            # Return first, last, and some middle nodes
            step = (size - 1) / (max_nodes - 1)
            return [int(i * step) for i in range(max_nodes)]
    
    # Draw connections first (so they appear behind nodes)
    for i in range(len(layer_structure) - 1):
        current_layer_size = layer_structure[i][1]
        next_layer_size = layer_structure[i + 1][1]
        
        current_y = np.linspace(0.2, 0.8, current_layer_size)
        next_y = np.linspace(0.2, 0.8, next_layer_size)
        
        # Get representative nodes for each layer
        current_nodes = get_representative_nodes(current_layer_size)
        next_nodes = get_representative_nodes(next_layer_size)
        
        # Draw representative connections
        for ci in current_nodes:
            for ni in next_nodes:
                plt.plot([layer_positions[i], layer_positions[i + 1]], 
                        [current_y[ci], next_y[ni]], 
                        'gray', alpha=0.1, zorder=1)
    
    # Draw nodes
    for i, (layer_name, layer_size) in enumerate(layer_structure):
        # Calculate y positions for this layer's nodes
        y_positions = np.linspace(0.2, 0.8, layer_size)
        
        # Determine color based on layer type
        if i == 0:
            color = colors['input']
        elif i == len(layer_structure) - 1:
            color = colors['output']
        else:
            color = colors['hidden']
        
        # Get representative nodes
        if layer_size > 10:
            # Draw fewer nodes for large layers
            shown_indices = get_representative_nodes(layer_size, 7)
            # Draw dots to indicate more nodes
            mid_y = (y_positions[shown_indices[-1]] + y_positions[shown_indices[-2]]) / 2
            spacing = (y_positions[1] - y_positions[0]) / 2
            plt.plot([layer_positions[i]], [mid_y], 'k.')
            plt.plot([layer_positions[i]], [mid_y + spacing], 'k.')
            plt.plot([layer_positions[i]], [mid_y - spacing], 'k.')
        else:
            shown_indices = range(layer_size)
        
        # Plot visible nodes
        for idx in shown_indices:
            plt.scatter(layer_positions[i], y_positions[idx], 
                       c=color, s=400, 
                       edgecolor='black', 
                       linewidth=1, 
                       zorder=2)
    
    # Add layer labels
    plt.text(layer_positions[0], 0.1, f'Input Layer\n({input_size} nodes)', 
             ha='center', va='top')
    plt.text(layer_positions[-1], 0.1, 'Output Layer\n(1 node)', 
             ha='center', va='top')
    
    # Add hidden layer labels
    for i in range(1, len(layer_structure) - 1):
        plt.text(layer_positions[i], 0.1, 
                f'Hidden Layer {i}\n({layer_structure[i][1]} nodes)', 
                ha='center', va='top')
    
    # Set plot properties
    plt.xlim(-0.1, 1.1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Neural Network Architecture', pad=20)
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Neural network diagram saved to {save_path}")

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

def print_model_summary(model):
    """
    Print a summary of the model architecture
    
    Args:
        model: PyTorch model
    """
    print("\nModel Summary:")
    print("=" * 50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 50)
    
    # Print layer information
    print("\nLayer Information:")
    print("=" * 50)
    print(f"{'Layer Name':<30} {'Type':<20} {'Output Shape':<20} {'Parameters':<15}")
    print("-" * 85)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.ReLU, nn.GELU, nn.Dropout, nn.LayerNorm)):
            if isinstance(module, nn.Linear):
                params = f"{module.in_features * module.out_features + module.out_features:,}"
                output_shape = f"(*, {module.out_features})"
            elif isinstance(module, nn.Dropout):
                params = "0"
                output_shape = "(*, *)"
            elif isinstance(module, nn.LayerNorm):
                params = f"{sum(module.normalized_shape) * 2:,}"
                output_shape = "(*, *)"
            else:
                params = "0"
                output_shape = "(*, *)"
            
            print(f"{name:<30} {module.__class__.__name__:<20} {output_shape:<20} {params:<15}")
    
    print("=" * 50)

# Main script execution
if __name__ == "__main__":
    # Set the model path directly
    model_path = "/Users/davidwhyatt/Documents/GitHub/PhDMelodySet/best_model.pth"
    input_size = 66  # Updated input size to match the saved model
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = EnhancedMelodyNet(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Print model summary
    print_model_summary(model)
    
    # Visualize model
    print("\nVisualizing model architecture...")
    visualize_model(model, input_size)
    visualize_model_layers(model)
    
    print("\nVisualization complete. Results saved to:")
    print("- model_architecture.png or model_architecture_alternative.png")
    print("- model_layers.png") 