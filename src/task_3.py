import torch

# Ensure deterministic execution for verification
#torch.manual_seed(42)

def generate_symmetry_dataset():
    """
    Generates all 64 possible binary vectors of length 6.
    Target is 1.0 if the vector is symmetric around its center, else 0.0.
    """
    # Generate all numbers from 0 to 63, converted to 6-bit binary tensors
    combinations = torch.arange(64, dtype=torch.int32).unsqueeze(1)
    shifts = torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.int32)
    X = ((combinations >> shifts) & 1).float()
    
    # Check if the first 3 elements match the reverse of the last 3 elements
    Y = (X[:, 0:3] == X[:, 3:6].flip(dims=[1])).all(dim=1).float().unsqueeze(1)
    
    return X, Y

def train_symmetry_detector():
    # 1. Dataset setup
    X, Y = generate_symmetry_dataset()
    num_cases = X.shape[0] # 64 cases
    
    # 2. Network Layout: 6 Input -> 2 Hidden -> 1 Output
    # We include biases by augmenting the inputs/hidden layers with an extra 1.0 unit
    # Weight shapes: (inputs + 1, outputs)
    W_hidden = torch.empty(6 + 1, 2).uniform_(-0.3, 0.3)  # Random weights between -0.3 and 0.3
    W_output = torch.empty(2 + 1, 1).uniform_(-0.3, 0.3)
    
    # Initialize velocities (momentum buffers) to 0
    V_hidden = torch.zeros_like(W_hidden)
    V_output = torch.zeros_like(W_output)
    
    # Hyperparameters from the paper
    epochs = 5000
    epsilon = 0.1  # Learning rate
    alpha = 0.9    # Momentum decay factor
    
    print(f"Starting batch gradient descent for {epochs} epochs...")
    print(f"Hyperparameters: epsilon={epsilon}, alpha={alpha}\n")
    
    for epoch in range(1, epochs + 1):
        # --- FORWARD PASS ---
        # Augment inputs with a column of 1s to handle biases implicitly
        bias_input = torch.ones(num_cases, 1)
        X_augmented = torch.cat([X, bias_input], dim=1)
        
        # Calculate Hidden Layer activation
        net_hidden = torch.mm(X_augmented, W_hidden)
        out_hidden = 1.0 / (1.0 + torch.exp(-net_hidden))
        
        # Augment Hidden outputs with a column of 1s for output bias
        out_hidden_augmented = torch.cat([out_hidden, bias_input], dim=1)
        
        # Calculate Output Layer activation
        net_output = torch.mm(out_hidden_augmented, W_output)
        out_final = 1.0 / (1.0 + torch.exp(-net_output))
        
        # --- ERROR COMPUTATION (Total Squared Error) ---
        error = 0.5 * torch.sum((out_final - Y) ** 2)
        
        # --- BACKWARD PASS (Batch Gradient Accumulation) ---
        # 1. Output layer error derivatives & deltas
        dE_dy_output = out_final - Y
        delta_output = dE_dy_output * (out_final * (1.0 - out_final))
        
        # Output layer gradients (summed over all 64 cases)
        grad_W_output = torch.mm(out_hidden_augmented.t(), delta_output)
        
        # 2. Hidden layer error derivatives & deltas
        # Backpropagate through the weights (excluding the bias term weight for error pass)
        dE_dy_hidden = torch.mm(delta_output, W_output[:-1, :].t())
        delta_hidden = dE_dy_hidden * (out_hidden * (1.0 - out_hidden))
        
        # Hidden layer gradients (summed over all 64 cases)
        grad_W_hidden = torch.mm(X_augmented.t(), delta_hidden)
        
        # --- OPTIMIZATION PROTOCOL (Gradient Descent with Momentum) ---
        V_output = -epsilon * grad_W_output + alpha * V_output
        W_output += V_output
        
        V_hidden = -epsilon * grad_W_hidden + alpha * V_hidden
        W_hidden += V_hidden
        
        # Print progress metrics periodically
        if epoch % 200 == 0 or epoch == 1 or epoch == 5000:
            predictions = (out_final > 0.5).float()
            accuracy = (predictions == Y).float().mean() * 100
            print(f"Epoch {epoch:4d} | Total Error: {error.item():8.4f} | Accuracy: {accuracy.item():6.2f}%")
            
    # --- VERIFICATION REPORTING ---
    print("\n" + "="*50)
    print("VERIFICATION RESULTS (Post 5000 Epochs)")
    print("="*50)
    
    # Format weights cleanly for readability
    print("\nLearned Weights for Hidden Units (Rows 0-5 = Inputs, Row 6 = Bias):")
    print("Format: [Weight to Hidden Unit 1, Weight to Hidden Unit 2]")
    for i in range(W_hidden.shape[0]):
        label = f"Input {i}" if i < 6 else "Bias Unit"
        print(f"  {label:<12}: [{W_hidden[i, 0].item():7.2f}, {W_hidden[i, 1].item():7.2f}]")
        
    print(f"\nFinal Output Layer Weights (including bias):\n{W_output.flatten().tolist()}")

if __name__ == "__main__":
    train_symmetry_detector()