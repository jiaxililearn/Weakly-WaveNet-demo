"""
Training script for weakly supervised spatiotemporal anomaly detection

This script trains the AnomalyDetectionModel with graph-level (yearly) labels
and evaluates using AUC-ROC, AP (Average Precision), and loss metrics.
"""

import os
import time
import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.anomaly_detection import AnomalyDetectionModel
from utils import util


# ============================================================
# TRAINING FUNCTIONS
# ============================================================


def train_epoch(
    model, dataloader, optimizer, criterion, device, epoch, clip_grad, log_interval
):
    """
    Train for one epoch using util.DataLoader

    Returns:
        dict with metrics: loss, predictions, labels
    """
    model.train()

    train_loss = []
    all_preds = []
    all_labels = []

    dataloader.shuffle()

    for iter, (x, y) in enumerate(dataloader.get_iterator()):
        # Convert to tensors and move to device
        trainx = torch.Tensor(x).to(device)
        trainx = trainx.transpose(
            1, 3
        )  # (batch, seq_length, num_nodes, in_dim) -> (batch, in_dim, num_nodes, seq_length)
        trainy = torch.LongTensor(y).to(device)  # (batch,) - binary labels

        # Forward pass
        optimizer.zero_grad()
        output = model(trainx, return_attention_weights=False)

        # Compute loss
        logits = output["logits"]  # (batch, num_classes)
        loss = criterion(logits, trainy)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        # Record metrics
        train_loss.append(loss.item())

        # Get predictions (probability of class 1)
        probs = F.softmax(logits, dim=1)
        preds = probs[:, 1].detach().cpu().numpy()
        labels = trainy.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

        # Print progress
        if iter % log_interval == 0:
            log = "Epoch: {:03d}, Iter: {:03d}, Train Loss: {:.4f}"
            # print(log.format(epoch, iter, train_loss[-1]), flush=True)

    # Compute epoch metrics
    avg_loss = np.mean(train_loss)

    # Compute AUC and AP if we have both classes
    try:
        auc_score = roc_auc_score(all_labels, all_preds)
        ap_score = average_precision_score(all_labels, all_preds)
    except ValueError:
        # Handle case where only one class is present in batch
        auc_score = 0.0
        ap_score = 0.0

    return {
        "loss": avg_loss,
        "auc": auc_score,
        "ap": ap_score,
        "predictions": all_preds,
        "labels": all_labels,
    }


def evaluate_model(model, dataloader, criterion, device, phase="val"):
    """
    Evaluate model on validation or test set using util.DataLoader

    Args:
        model: The model to evaluate
        dataloader: util.DataLoader for val/test data
        criterion: Loss function
        device: torch device
        phase: 'val' or 'test'

    Returns:
        dict with metrics: loss, auc, ap, predictions, labels
    """
    model.eval()

    valid_loss = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for iter, (x, y) in enumerate(dataloader.get_iterator()):
            # Convert to tensors and move to device
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(
                1, 3
            )  # (batch, seq_length, num_nodes, in_dim) -> (batch, in_dim, num_nodes, seq_length)
            testy = torch.LongTensor(y).to(device)  # (batch,)

            # Forward pass
            output = model(testx, return_attention_weights=False)
            logits = output["logits"]

            # Compute loss
            loss = criterion(logits, testy)
            valid_loss.append(loss.item())

            # Get predictions
            probs = F.softmax(logits, dim=1)
            preds = probs[:, 1].cpu().numpy()
            labels = testy.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    # Compute metrics
    avg_loss = np.mean(valid_loss)

    try:
        auc_score = roc_auc_score(all_labels, all_preds)
        ap_score = average_precision_score(all_labels, all_preds)
    except ValueError:
        auc_score = 0.0
        ap_score = 0.0

    if phase == "test":
        print(f"\n{phase.upper()} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  AUC-ROC: {auc_score:.4f}")
        print(f"  AP: {ap_score:.4f}")

    return {
        "loss": avg_loss,
        "auc": auc_score,
        "ap": ap_score,
        "predictions": all_preds,
        "labels": all_labels,
    }


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, is_best=False):
    """
    Save model checkpoint

    Args:
        model: The model to save
        optimizer: The optimizer
        epoch: Current epoch
        metrics: Dict of metrics
        save_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pt")
        torch.save(state, best_path)
        print(f'Saved best model: {best_path} (AUC: {metrics["val"]["auc"]:.4f})')


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load model checkpoint

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state

    Returns:
        epoch, metrics
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    metrics = checkpoint.get("metrics", {})

    print(f"Loaded checkpoint from epoch {epoch}")
    # print(f"  Loaded TEST metrics: {metrics}")
    return epoch, metrics


def load_data(data_path, batch_size):
    """
    Load and prepare data for anomaly detection using util.load_dataset

    Returns:
        train_loader, val_loader, test_loader, scaler, class_weights
    """
    print(f"Loading data from {data_path}...")

    # Use the existing util.load_dataset function
    # This expects train.npz, val.npz, test.npz with keys 'x' and 'y'
    # x shape: (num_samples, seq_length, num_nodes, in_dim)
    # y shape: (num_samples,) for anomaly detection (binary labels)
    dataloader = util.load_dataset(data_path, batch_size, batch_size, batch_size)

    print("Data loaded:")
    print(f'  Train: {dataloader["train_loader"].size} samples')
    print(f'  Val:   {dataloader["val_loader"].size} samples')
    print(f'  Test:  {dataloader["test_loader"].size} samples')

    # output the basic statistics of datasets, especially to show mean and std of the data
    print("Dataset statistics:")
    for split in ["train", "val", "test"]:
        print("*" * 40)
        x_data = dataloader[f"x_{split}"]
        y_data = dataloader[f"y_{split}"].astype(int)
        for channel in range(dataloader[f"x_{split}"].shape[-2]):
            print(
                f"  {split.upper()} Channel {channel}: mean={dataloader[f'x_{split}'][..., channel, 0].mean():.4f}, std={dataloader[f'x_{split}'][..., channel, 0].std():.4f}"
            )
        print(
            f'  {split.upper()} y distribution: {np.bincount(y_data) if len(np.bincount(y_data)) > 1 else "All normal"}'
        )

    # Calculate class weights for handling imbalanced data
    y_train = dataloader["y_train"].astype(int)
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts)
    print("\nClass weights for imbalanced data:")
    print(f"  Class 0 (normal): {class_weights[0]:.4f}")
    print(f"  Class 1 (anomaly): {class_weights[1]:.4f}")

    return (
        dataloader["train_loader"],
        dataloader["val_loader"],
        dataloader["test_loader"],
        dataloader["scaler"],
        class_weights,
    )


# ============================================================
# MAIN TRAINING LOOP
# ============================================================


@click.command()
# Data parameters
@click.option(
    "--data",
    type=str,
    default="data/temp_dhw/",
    help="Directory containing train/val/test data",
)
@click.option(
    "--save_dir",
    type=str,
    default="checkpoints/anomaly_detection",
    help="Directory to save checkpoints",
)
# Model parameters
@click.option("--num_nodes", type=int, default=3, help="Number of nodes in the graph")
@click.option("--in_dim", type=int, default=2, help="Input feature dimension")
@click.option(
    "--temporal_length",
    type=int,
    default=51,
    help="Output feature dimension from GWNet (usually seq_length)",
)
@click.option("--seq_length", type=int, default=128, help="Input sequence length")
@click.option(
    "--num_classes", type=int, default=2, help="Number of classes (2 for binary)"
)
# Aggregation parameters
@click.option(
    "--aggregation_type",
    type=click.Choice(["simple", "spatial_temporal_attention", "hybrid"]),
    default="spatial_temporal_attention",
    help="Type of graph aggregation",
)
@click.option("--num_heads", type=int, default=4, help="Number of attention heads")
@click.option(
    "--feature_dim",
    type=int,
    default=16,
    help="Feature dimension for attention (prevents memory explosion with large out_dim)",
)
@click.option(
    "--gwnet_layers",
    type=int,
    default=6,
    help="Number of layers per block in GWNet (reduce to preserve temporal dimension)",
)
@click.option(
    "--gwnet_blocks",
    type=int,
    default=5,
    help="Number of blocks in GWNet (reduce to preserve temporal dimension)",
)
# Training parameters
@click.option("--batch_size", type=int, default=32, help="Batch size")
@click.option("--epochs", type=int, default=100, help="Number of training epochs")
@click.option("--lr", type=float, default=0.005, help="Learning rate")
@click.option("--weight_decay", type=float, default=1e-4, help="Weight decay")
@click.option("--dropout", type=float, default=0.3, help="Dropout rate")
@click.option(
    "--clip_grad",
    type=float,
    default=5.0,
    help="Gradient clipping threshold (0 to disable)",
)
# Optimizer parameters
@click.option(
    "--optimizer",
    type=click.Choice(["adam", "adamw", "sgd"]),
    default="adam",
    help="Optimizer type",
)
@click.option(
    "--lr_scheduler",
    type=click.Choice(["plateau", "step", "cosine", "none"]),
    default="plateau",
    help="Learning rate scheduler",
)
@click.option(
    "--patience",
    type=int,
    default=10,
    help="Patience for early stopping and lr scheduler",
)
# Other parameters
@click.option("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
@click.option(
    "--log_interval", type=int, default=10, help="Log interval during training"
)
@click.option(
    "--save_interval", type=int, default=10, help="Checkpoint save interval (epochs)"
)
@click.option(
    "--resume", type=str, default=None, help="Path to checkpoint to resume from"
)
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option(
    "--use_class_weights",
    type=bool,
    default=True,
    help="Use class weights to handle imbalanced data",
)
@click.option(
    "--focal_loss",
    type=bool,
    default=False,
    help="Use focal loss instead of cross entropy",
)
@click.option(
    "--focal_alpha", type=float, default=0.25, help="Focal loss alpha parameter"
)
@click.option(
    "--focal_gamma", type=float, default=2.0, help="Focal loss gamma parameter"
)
def main(
    data,
    save_dir,
    num_nodes,
    in_dim,
    temporal_length,
    seq_length,
    num_classes,
    aggregation_type,
    num_heads,
    feature_dim,
    gwnet_layers,
    gwnet_blocks,
    batch_size,
    epochs,
    lr,
    weight_decay,
    dropout,
    clip_grad,
    optimizer,
    lr_scheduler,
    patience,
    device,
    log_interval,
    save_interval,
    resume,
    seed,
    use_class_weights,
    focal_loss,
    focal_alpha,
    focal_gamma,
):
    """Train Anomaly Detection Model"""

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Setup device
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    # Load data
    train_loader, val_loader, test_loader, scaler, class_weights = load_data(
        data, batch_size
    )

    # Create model
    print("\nCreating model...")
    # gwnet(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
    gwnet_params = {
        "device": dev,
        "num_nodes": num_nodes,
        "dropout": dropout,
        "supports": None,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": in_dim,
        "out_dim": seq_length,  # GWNet out_dim (usually set to seq_length in original code)
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": gwnet_blocks,  # User-configurable to control temporal compression
        "layers": gwnet_layers,  # User-configurable to control temporal compression
    }

    # Calculate actual temporal length after GWNet processing
    # GWNet uses dilated convolutions that reduce temporal dimension
    # Rough estimate: temporal_length ≈ seq_length / (2^(blocks * layers - 1))

    print(f"  (Input: {seq_length}, Blocks: {gwnet_blocks}, Layers: {gwnet_layers})")
    print(f"  Using feature_dim={feature_dim} for attention to prevent memory issues")

    aggregation_params = {
        "temporal_length": temporal_length,
        "num_nodes": num_nodes,
        "out_dim": seq_length,
        "num_heads": num_heads,
        "dropout": dropout,
        "feature_dim": feature_dim,  # Reduced feature dimension for attention
    }

    model = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type=aggregation_type,
        aggregation_params=aggregation_params,
        num_classes=num_classes,
    ).to(dev)

    # Reinitialize classifier with stronger initialization
    # This helps prevent constant outputs
    for name, module in model.named_modules():
        if "classifier" in name and isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                # Initialize bias to favor normal class slightly
                # This prevents saturating at 0.5
                nn.init.constant_(module.bias, 0)
                if module.out_features == 2:  # Final layer
                    module.bias.data[0] = 0.5  # Normal class
                    module.bias.data[1] = -0.5  # Anomaly class

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    # Create learning rate scheduler
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=patience
        )
    elif lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    elif lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = None

    # Loss function (Cross Entropy for classification)
    if focal_loss:
        print(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        from torch.autograd import Variable

        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, num_classes=2):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.num_classes = num_classes

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction="none")
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()

        criterion = FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, num_classes=num_classes
        )
    elif use_class_weights:
        weight_tensor = torch.FloatTensor(class_weights).to(dev)
        print(f"Using weighted Cross Entropy Loss with weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        print("Using standard Cross Entropy Loss (not recommended for imbalanced data)")
        criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if specified
    start_epoch = 1
    best_auc = 0.0
    if resume:
        start_epoch, metrics = load_checkpoint(resume, model, opt)
        best_auc = metrics.get("auc", 0.0)
        start_epoch += 1

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...\n")
    patience_counter = 0

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, opt, criterion, dev, epoch, clip_grad, log_interval
        )

        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, dev, phase="val")

        # Learning rate scheduling
        if scheduler is not None:
            if lr_scheduler == "plateau":
                scheduler.step(val_metrics["auc"])
            else:
                scheduler.step()

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(
            f'{epoch:3d} | Train Loss: {train_metrics["loss"]:.4f} | Val Loss: {val_metrics["loss"]:.4f}, AUC: {val_metrics["auc"]:.4f}, AP: {val_metrics["ap"]:.4f} | Time: {epoch_time:.1f}s, LR: {opt.param_groups[0]["lr"]:.6f}'
        )

        # Save checkpoint
        is_best = val_metrics["auc"] > best_auc
        if is_best:
            best_auc = val_metrics["auc"]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % save_interval == 0 or is_best:
            save_checkpoint(
                model,
                opt,
                epoch,
                {"train": train_metrics, "val": val_metrics},
                save_dir,
                is_best,
            )

        # Early stopping
        if patience_counter >= patience * 2:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

        # print("-" * 80)

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)

    # Load best model
    best_model_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model)

    test_metrics = evaluate_model(model, test_loader, criterion, dev, phase="test")

    # Save test results
    results_path = os.path.join(save_dir, "test_results.pt")
    torch.save(test_metrics, results_path)
    print(f"\nTest results saved to {results_path}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
