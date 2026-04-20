
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Global device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Optimizes CUDA kernel selection for faster training

class PrunableLinear(nn.Module):
    """
    Custom Linear layer with learnable gate parameters for weight pruning.
    The gate values are constrained to [0, 1] via a sigmoid activation.
    """
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Learnable gate scores
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Weight initialization
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        # Initialize gate scores to 2.0 so gates start near 1 (active)
        # sigmoid(2.0) ~ 0.88. This prevents premature pruning at the start of training.
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, x):
        # Apply sigmoid to gate scores to get gates in [0, 1]
        gates = torch.sigmoid(self.gate_scores)
        
        # Optimization: During inference, apply a hard threshold to achieve *true* zeros
        # This completely shuts down the pruned connections in the forward pass
        if not self.training:
            gates = (gates >= 0.01).float() * gates
            
        # Apply pruning mask
        masked_weight = self.weight * gates
        
        return F.linear(x, masked_weight, self.bias)

    def get_sparsity_penalty(self):
        """Returns the L1 norm of the gates for regularization."""
        # Optimization: sigmoid output is strictly positive, so .abs() is computationally redundant
        return torch.sigmoid(self.gate_scores).sum()

class SelfPruningNet(nn.Module):
    """
    Feed-forward network architecture designed 
    Uses PrunableLinear layers to allow end-to-end learned sparsity.
    """
    def __init__(self, input_dim=3072, hidden_dims=[1024, 512, 256], num_classes=10):
        super(SelfPruningNet, self).__init__()
        
        layers = []
        curr_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                PrunableLinear(curr_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            curr_dim = h_dim
            
        layers.append(PrunableLinear(curr_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def get_total_sparsity_loss(self):
        """Aggregates sparsity penalties from all prunable layers."""
        total_loss = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                total_loss += m.get_sparsity_penalty()
        return total_loss

def get_data_loaders(batch_size=128):
    """Prepares CIFAR-10 training and testing data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Optimization: pin_memory=True speeds up host-to-device (CPU to GPU) memory transfers
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, lambda_reg):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        

        cls_loss = criterion(outputs, targets)
        sparsity_loss = model.get_total_sparsity_loss()
        loss = cls_loss + lambda_reg * sparsity_loss
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(loader), 100. * correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def get_model_stats(model, threshold=1e-2):
    """Calculates sparsity level and retrieves all gate values."""
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)
    
    all_gates = np.array(all_gates)
    sparsity = np.mean(all_gates < threshold) * 100
    return sparsity, all_gates

def run_experiment(lambda_val, epochs=30):
    print(f"\n>>> Starting Experiment: Lambda = {lambda_val}")
    
    train_loader, test_loader = get_data_loaders()
    model = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion, lambda_val)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Train Acc: {acc:.2f}%")

    test_acc = evaluate(model, test_loader)
    sparsity, gates = get_model_stats(model)
    duration = time.time() - start_time

    print(f"Finished in {duration/60:.2f}m | Test Acc: {test_acc:.2f}% | Sparsity: {sparsity:.2f}%")
    
    # Save gate distribution plot
    plt.figure()
    plt.hist(gates, bins=100, color='teal', alpha=0.7)
    plt.title(f'Gate Distribution (Lambda={lambda_val})')
    plt.xlabel('Gate Value')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.savefig(f'results_lambda_{lambda_val}.png')
    plt.close()

    return {"lambda": lambda_val, "accuracy": test_acc, "sparsity": sparsity}

if __name__ == "__main__":
    # Hyperparameters for the study
    lambdas = [0.0001, 0.001, 0.01]
    results = []

    for l in lambdas:
        res = run_experiment(l)
        results.append(res)

    print("\n" + "="*40)
    print(f"{'Lambda':<10} | {'Test Acc (%)':<15} | {'Sparsity (%)':<15}")
    print("-" * 40)
    for r in results:
        print(f"{r['lambda']:<10} | {r['accuracy']:<15.2f} | {r['sparsity']:<15.2f}")
    print("="*40)