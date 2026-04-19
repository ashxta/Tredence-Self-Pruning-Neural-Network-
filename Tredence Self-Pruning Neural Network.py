import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")          


# PART 1 — PrunableLinear Layer
class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that attaches a learnable gate scalar
    to every weight.  During the forward pass:

        gates          = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_weights = weight ⊙ gates                element-wise product
        output         = input @ pruned_weights.T + bias

    The L1 penalty on `gates` during training pushes many gates toward 0,
    effectively removing (pruning) those weight connections.

    Gradients flow through both `weight` and `gate_scores` because all
    operations are differentiable PyTorch primitives.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (detached, on CPU) for analysis."""
        return torch.sigmoid(self.gate_scores).detach().cpu()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below `threshold` (i.e. effectively pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()

# Network definition


class SelfPruningNet(nn.Module):
    """
    3-hidden-layer feed-forward network built entirely from PrunableLinear
    layers.  CIFAR-10 images are 32×32×3 = 3072-dimensional when flattened.

    Architecture: 3072 → 512 → 256 → 128 → 10
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512,  256)
        self.fc3 = PrunableLinear(256,  128)
        self.fc4 = PrunableLinear(128,  10)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # flatten spatial dims
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)                   

    def prunable_layers(self):
        """Iterator over all PrunableLinear layers in the model."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

# PART 2 — Sparsity Regularisation Loss

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of all gate values across every PrunableLinear layer.

    Why L1 encourages sparsity:
        The gradient of |g| w.r.t. g is ±1 (constant pull toward 0),
        unlike L2 whose gradient shrinks as g→0, leaving residual values.
        This constant gradient is what drives gates all the way to zero.

    Returns a scalar tensor so it can be added directly to the CE loss.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.abs().sum()   
    return total



# PART 3 — Data loading

def get_cifar10_loaders(batch_size: int = 256):
    """Download CIFAR-10 and return train / test DataLoaders."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                            download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader



def train_one_epoch(model, loader, optimizer, lam: float, device):
    model.train()
    total_ce = total_sp = total_correct = total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits  = model(images)
        ce_loss = F.cross_entropy(logits, labels)
        sp_loss = sparsity_loss(model)

        # Total loss = classification loss + lamba × sparsity regularisation
        loss = ce_loss + lam * sp_loss

        loss.backward()
        optimizer.step()

        # Accumulate stats
        total_ce      += ce_loss.item() * images.size(0)
        total_sp      += sp_loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_ce  = total_ce  / total_samples
    avg_sp  = total_sp  / total_samples
    acc     = 100.0 * total_correct / total_samples
    return avg_ce, avg_sp, acc



@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return test accuracy (%) on the given loader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return 100.0 * correct / total


def model_sparsity(model: SelfPruningNet, threshold: float = 1e-2) -> float:
    """Global sparsity: fraction of all weights with gate < threshold."""
    pruned = total = 0
    for layer in model.prunable_layers():
        gates  = layer.get_gates()
        pruned += (gates < threshold).sum().item()
        total  += gates.numel()
    return 100.0 * pruned / total


def collect_all_gates(model: SelfPruningNet) -> np.ndarray:
    """Concatenate all gate values from every prunable layer into 1-D array."""
    parts = []
    for layer in model.prunable_layers():
        parts.append(layer.get_gates().view(-1).numpy())
    return np.concatenate(parts)



def run_experiment(lam: float, epochs: int, train_loader, test_loader,
                   device, verbose: bool = True):
    """Train one model for a given λ and return (test_acc, sparsity_pct, gates)."""

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  λ = {lam:.4f}  |  training for {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        ce, sp, tr_acc = train_one_epoch(model, train_loader, optimizer, lam, device)
        scheduler.step()
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"CE={ce:.4f}  SP={sp:.4f}  Train Acc={tr_acc:.1f}%")

    test_acc  = evaluate(model, test_loader, device)
    spar      = model_sparsity(model)
    all_gates = collect_all_gates(model)

    print(f"\n  → Test Accuracy : {test_acc:.2f}%")
    print(f"  → Sparsity Level: {spar:.2f}%  (gates < 1e-2)")
    return test_acc, spar, all_gates, model


def plot_gate_distribution(all_gates: np.ndarray, lam: float, save_path: str):
    """
    Plot histogram of final gate values.
    A successful run shows a large spike at 0 and a cluster away from 0.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_gates, bins=100, color="#2563EB", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gate Value", fontsize=12)
    ax.set_ylabel("Count",      fontsize=12)
    ax.set_title(f"Gate Value Distribution  (λ = {lam})", fontsize=13, fontweight="bold")
    ax.axvline(1e-2, color="red", linestyle="--", linewidth=1.2, label="Prune threshold (1e-2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Plot saved → {save_path}]")



if __name__ == "__main__":


    EPOCHS      = 30        
    BATCH_SIZE  = 256
    LAMBDAS     = [1e-4, 1e-3, 5e-3]   # low / medium / high sparsity pressure
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}")

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results      = []
    best_acc     = -1
    best_gates   = None
    best_lam     = None

    for lam in LAMBDAS:
        acc, spar, gates, model = run_experiment(
            lam, EPOCHS, train_loader, test_loader, DEVICE)
        results.append((lam, acc, spar))

        # Save gate plot for each Lambda
        plot_gate_distribution(gates, lam,
                               f"gate_distribution_lambda_{lam}.png")

        # Track best model by accuracy for detailed plot
        if acc > best_acc:
            best_acc   = acc
            best_gates = gates
            best_lam   = lam

    # Summary table
    print("\n")
    print("┌─────────────┬───────────────────┬──────────────────────┐")
    print("│   Lambda    │   Test Accuracy   │   Sparsity Level     │")
    print("├─────────────┼───────────────────┼──────────────────────┤")
    for lam, acc, spar in results:
        print(f"│  {lam:<11}│  {acc:>7.2f}%          │  {spar:>7.2f}%             │")
    print("└─────────────┴───────────────────┴──────────────────────┘")

    print(f"\nBest model: λ={best_lam}  |  Accuracy={best_acc:.2f}%")
    print("Gate-distribution plots saved as PNG files in the working directory.")
