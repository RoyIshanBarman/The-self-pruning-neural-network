
import os
import time
import random
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Logging — timestamps make long runs much easier to follow
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":

    torch.backends.cudnn.benchmark = True

log.info(f"Running on: {device}")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# The "Prunable" Linear Layer
# ---------------------------------------------------------------------------
class PrunableLinear(nn.Module):


    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # The weight and bias are standard — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # gate_scores is the new thing. Same shape as weight.
        # After sigmoid, each score becomes the gate for the corresponding weight.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Kaiming init for weights — good default for ReLU networks
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # Start all gates near-open (sigmoid(2.0) ≈ 0.88) so the network
        # can learn useful representations before pruning pressure kicks in
        nn.init.constant_(self.gate_scores, 2.0)

        # We'll store the computed gates here after each forward pass so
        # the sparsity penalty function can reuse them without a second sigmoid
        self._cached_gates: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute gates once and cache — the penalty function reads from here
        # so we never run sigmoid twice on the same tensor in one training step
        gates = torch.sigmoid(self.gate_scores)
        self._cached_gates = gates

        # At inference time we apply a hard cutoff: any gate below 0.01 becomes
        # exactly zero. This gives us true sparsity (no tiny near-zero multiplies)
        # but we only do this outside training so it doesn't affect the gradient.
        if not self.training:
            gates = (gates >= 0.01).float() * gates

        # Mask the weights and run the standard linear operation
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_sparsity_penalty(self) -> torch.Tensor:
        """
        Returns the sum of all current gate values for this layer.

        This is the L1 term we add to the loss. Because sigmoid output is
        always positive, the sum IS the L1 norm (no need for .abs()).
        We reuse the cached gates from the most recent forward pass to avoid
        computing sigmoid a second time per training step.
        """
        if self._cached_gates is None:
            # Fallback in case someone calls this before a forward pass
            return torch.sigmoid(self.gate_scores).sum()
        return self._cached_gates.sum()

    def get_layer_sparsity(self, threshold: float = 1e-2) -> dict:
        """
        Returns a breakdown of this specific layer's sparsity and gate stats.
        Useful for the per-layer analysis in the report.
        """
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores).cpu().numpy().flatten()

        n_total   = len(gates)
        n_pruned  = int(np.sum(gates < threshold))
        sparsity  = n_pruned / n_total * 100.0

        return {
            "shape"     : (self.out_features, self.in_features),
            "n_total"   : n_total,
            "n_pruned"  : n_pruned,
            "sparsity"  : sparsity,
            "gate_mean" : float(gates.mean()),
            "gate_min"  : float(gates.min()),
        }


# ---------------------------------------------------------------------------
# The Network
# ---------------------------------------------------------------------------
class SelfPruningNet(nn.Module):
    """
    A simple feed-forward network for CIFAR-10 classification.

    All fully-connected layers are PrunableLinear, so the whole network
    participates in the self-pruning process. BatchNorm and Dropout are kept
    as-is — they're standard stabilisers, not part of the gating mechanism.

    Architecture (default):
        Flatten(3072) → PrunableLinear(1024) → BN → ReLU → Dropout
                      → PrunableLinear(512)  → BN → ReLU → Dropout
                      → PrunableLinear(256)  → BN → ReLU → Dropout
                      → PrunableLinear(10)   [logits]
    """

    def __init__(
        self,
        input_dim:   int       = 3072,    # 32×32×3 for CIFAR-10
        hidden_dims: list[int] = None,
        num_classes: int       = 10,
        dropout:     float     = 0.2,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        layers    = []
        curr_dim  = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                PrunableLinear(curr_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            curr_dim = h_dim

        # Final classification head — no activation, loss function handles that
        layers.append(PrunableLinear(curr_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CIFAR-10 images come in as (B, 3, 32, 32) — flatten to (B, 3072)
        x = x.view(x.size(0), -1)
        return self.model(x)

    def get_total_sparsity_loss(self) -> torch.Tensor:
        """
        Sums up the gate L1 penalties from every PrunableLinear layer and
        normalises by the total number of gates.

        Normalisation matters: without it, a larger network (more parameters)
        would produce a proportionally bigger penalty, making lambda values
        architecture-dependent. After normalising, lambda has a consistent
        meaning: "what fraction of gates should I push to zero?" regardless
        of how wide the network is.
        """
        total_penalty = torch.tensor(0.0, device=next(self.parameters()).device)
        total_gates   = 0

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total_penalty += module.get_sparsity_penalty()
                total_gates   += module.gate_scores.numel()

        # Avoid division by zero on a degenerate (empty) network
        return total_penalty / max(total_gates, 1)

    def get_all_layer_stats(self, threshold: float = 1e-2) -> list[dict]:
        """
        Returns per-layer sparsity info — handy for the detailed report table.
        Each entry corresponds to one PrunableLinear layer in order.
        """
        stats = []
        for i, module in enumerate(self.modules()):
            if isinstance(module, PrunableLinear):
                layer_stats = module.get_layer_sparsity(threshold)
                layer_stats["layer_idx"] = i
                stats.append(layer_stats)
        return stats


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_data_loaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """
    Builds the CIFAR-10 train and test loaders.

    The training set gets a few augmentations (horizontal flip, random crop)
    to give the model more variety without needing more data. The test set
    gets none — we want a clean, unmodified evaluation.

    The normalisation values (mean/std per channel) are the standard CIFAR-10
    statistics computed over the full training set.
    """

    # Training augmentation — small and cheap, makes a noticeable difference
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    # Test transform — just normalise, no augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # On Windows, PyTorch multiprocessing workers communicate via shared file
    # mappings. Under memory pressure this triggers error 1455 (paging file
    # exhausted). Setting num_workers=0 keeps data loading in the main process
    # and completely avoids shared memory — safe and correct on all platforms.
    # On Linux/Mac we can safely use parallel workers for speed.
    import platform
    n_workers = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)

    # pin_memory only helps when there is a CUDA GPU; skip it on CPU-only runs
    # to avoid a noisy UserWarning from PyTorch.
    pin_mem = torch.cuda.is_available()

    # persistent_workers requires num_workers > 0
    persist = n_workers > 0

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_mem,
        persistent_workers=persist,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    lambda_reg: float,
    warmup:     bool = False,
) -> tuple[float, float]:
    """
    Runs one full pass over the training data.

    The total loss is:
        L_total = L_crossentropy + lambda * L_sparsity

    L_sparsity is the normalised sum of all gate values. Minimising it pushes
    gates toward zero. lambda controls how hard we push — higher lambda means
    more aggressive pruning, usually at the cost of some accuracy.

    The 'warmup' flag lets us skip the sparsity penalty entirely for the first
    few epochs. This gives the network time to learn useful features before we
    start pressuring it to prune.
    """
    model.train()

    running_loss = 0.0
    n_correct    = 0
    n_total      = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()

        outputs   = model(inputs)
        cls_loss  = criterion(outputs, targets)

        # Skip sparsity pressure during warmup — let the weights settle first
        if warmup:
            loss = cls_loss
        else:
            sparsity_loss = model.get_total_sparsity_loss()
            loss          = cls_loss + lambda_reg * sparsity_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(dim=1)
        n_total   += targets.size(0)
        n_correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * n_correct / n_total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Runs inference over the test set and returns accuracy (%).

    We use @torch.no_grad() because we don't need gradients here —
    it halves memory usage and speeds things up a bit.
    """
    model.eval()

    n_correct = 0
    n_total   = 0

    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs      = model(inputs)
        _, predicted = outputs.max(dim=1)

        n_total   += targets.size(0)
        n_correct += predicted.eq(targets).sum().item()

    return 100.0 * n_correct / n_total


# ---------------------------------------------------------------------------
# Metrics and visualisation
# ---------------------------------------------------------------------------
def get_model_stats(model: nn.Module, threshold: float = 1e-2) -> tuple[float, np.ndarray]:
    """
    Collects all gate values across the network and computes the global
    sparsity level (i.e. what fraction of gates are effectively zero).

    A gate is considered "pruned" if its value is below the threshold.
    The default of 0.01 matches the hard cutoff applied at inference time.
    """
    all_gates = []

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).cpu().numpy().flatten()
                all_gates.extend(gates)

    all_gates = np.array(all_gates)
    sparsity  = float(np.mean(all_gates < threshold) * 100.0)

    return sparsity, all_gates


def save_gate_distribution_plot(
    gates:      np.ndarray,
    lambda_val: float,
    output_dir: str = ".",
) -> str:
    """
    Saves a histogram of gate values for a given lambda.

    A well-pruned model will show a big spike near 0 (dead connections)
    and a cluster of values away from 0 (connections the network kept).
    The log y-scale makes it easier to see the spike at 0 when sparsity
    is high and there are many more zeros than active gates.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(gates, bins=100, color="#1D9E75", alpha=0.75, edgecolor="none")
    ax.set_title(f"Gate value distribution  (λ = {lambda_val})", fontsize=13)
    ax.set_xlabel("Gate value", fontsize=11)
    ax.set_ylabel("Count (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    filepath = os.path.join(output_dir, f"results_lambda_{lambda_val}.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return filepath


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(
    lambda_val:   float,
    train_loader: DataLoader,
    test_loader:  DataLoader,
    epochs:       int   = 30,
    warmup_epochs: int  = 5,
    output_dir:   str   = ".",
) -> dict:
    """
    Trains one model end-to-end for a given lambda value and returns its results.

    The warmup_epochs argument delays the sparsity penalty for the first N
    epochs. This is important: if we apply L1 pressure immediately, the gates
    start closing before the weights have learned anything useful. By waiting
    a few epochs we let the network form a rough solution first, then prune.

    Returns a dict with lambda, test accuracy, sparsity, and per-layer stats.
    """
    log.info(f"Starting experiment  λ = {lambda_val}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Cosine annealing smoothly decays the learning rate over training.
    # It tends to work better than step-decay for this kind of experiment.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Track the best model we see during training, not just the final one.
    # With cosine LR the model can peak around epoch 20 and drift slightly
    # after — we want to report the best, not the last.
    best_acc   = 0.0
    best_state = None

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        # Warmup: no sparsity penalty for the first N epochs
        in_warmup = epoch <= warmup_epochs

        loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, lambda_val, warmup=in_warmup
        )
        scheduler.step()

        # Evaluate every 5 epochs (and always on epoch 1 so we have a baseline)
        if epoch == 1 or epoch % 5 == 0:
            val_acc = evaluate(model, test_loader)
            warmup_tag = "  [warmup]" if in_warmup else ""
            log.info(
                f"  Epoch {epoch:>2}/{epochs}  |  "
                f"loss {loss:.4f}  |  train {train_acc:.1f}%  |  "
                f"val {val_acc:.1f}%{warmup_tag}"
            )

            # Save a snapshot if this is the best we've seen
            if val_acc > best_acc:
                best_acc   = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore the best checkpoint before computing final stats
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test accuracy (on the restored best model)
    test_acc        = evaluate(model, test_loader)
    sparsity, gates = get_model_stats(model)
    layer_stats     = model.get_all_layer_stats()
    duration        = time.time() - start_time

    log.info(
        f"Done  λ = {lambda_val}  |  "
        f"test acc {test_acc:.2f}%  |  "
        f"sparsity {sparsity:.2f}%  |  "
        f"time {duration / 60:.1f}m"
    )

    # Save the gate histogram for this run
    plot_path = save_gate_distribution_plot(gates, lambda_val, output_dir)
    log.info(f"Gate distribution saved → {plot_path}")

    return {
        "lambda"      : lambda_val,
        "accuracy"    : test_acc,
        "sparsity"    : sparsity,
        "layer_stats" : layer_stats,
        "duration_min": duration / 60,
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
def print_summary_table(results: list[dict]) -> None:
    """Prints a formatted comparison table across all lambda experiments."""

    header = f"{'Lambda':<10}  {'Test Acc (%)':<15}  {'Sparsity (%)':<15}  {'Time (min)':<10}"
    divider = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*len(header)}")
    print(header)
    print(divider)

    for r in results:
        print(
            f"{r['lambda']:<10}  "
            f"{r['accuracy']:<15.2f}  "
            f"{r['sparsity']:<15.2f}  "
            f"{r['duration_min']:<10.1f}"
        )

    print(f"{'='*len(header)}\n")


def print_layer_breakdown(results: list[dict]) -> None:
    """
    Prints a per-layer sparsity breakdown for each experiment.
    This shows which layers the network decided to prune most aggressively.
    """
    for r in results:
        print(f"\n  Layer breakdown  (λ = {r['lambda']})")
        print(f"  {'Layer':<8}  {'Shape':<18}  {'Params':<10}  {'Pruned':<10}  {'Sparsity':<10}")
        print(f"  {'-'*60}")

        for ls in r["layer_stats"]:
            shape_str = f"{ls['shape'][0]}×{ls['shape'][1]}"
            print(
                f"  {ls['layer_idx']:<8}  "
                f"{shape_str:<18}  "
                f"{ls['n_total']:<10,}  "
                f"{ls['n_pruned']:<10,}  "
                f"{ls['sparsity']:<10.1f}%"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Fix the seed so results are reproducible across machines and re-runs
    set_seed(42)

    # Build loaders once — no reason to rebuild them for each lambda experiment
    log.info("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=128)

    # The three lambda values explore low / medium / high pruning pressure.
    # Higher lambda = more sparsity, potentially lower accuracy.
    lambdas = [0.0001, 0.001, 0.01]

    results = []
    for lam in lambdas:
        result = run_experiment(
            lambda_val    = lam,
            train_loader  = train_loader,
            test_loader   = test_loader,
            epochs        = 30,
            warmup_epochs = 5,       # hold off sparsity pressure for 5 epochs
            output_dir    = ".",
        )
        results.append(result)

    # Print the summary tables
    print_summary_table(results)
    print_layer_breakdown(results)