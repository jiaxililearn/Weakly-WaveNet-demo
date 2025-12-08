"""
Calibrate model probabilities using temperature scaling

Your model has good AUC (0.87) but probabilities cluster around 0.517.
Temperature scaling fixes this while preserving ranking performance.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F


class TemperatureScaling(nn.Module):
    """
    Temperature scaling layer for probability calibration

    Divides logits by temperature T before softmax:
    p = softmax(logits / T)

    T > 1: Makes predictions less confident (spreads probabilities)
    T < 1: Makes predictions more confident (concentrates probabilities)
    """

    def __init__(self, init_temperature=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * init_temperature)

    def forward(self, logits):
        return logits / self.temperature


def calibrate_on_validation(model, val_loader, device):
    """
    Learn optimal temperature on validation set

    Args:
        model: Trained AnomalyDetectionModel
        val_loader: Validation data loader
        device: torch device

    Returns:
        optimal_temperature: Best temperature value
    """
    model.eval()

    # Collect all logits and labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader.get_iterator():
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.LongTensor(y).to(device)

            output = model(testx, return_attention_weights=False)
            logits = output["logits"]

            all_logits.append(logits)
            all_labels.append(testy)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    # Optimize temperature
    temperature_layer = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([temperature_layer.temperature], lr=0.01, max_iter=50)

    criterion = nn.CrossEntropyLoss()

    def eval_loss():
        optimizer.zero_grad()
        scaled_logits = temperature_layer(all_logits)
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval_loss)

    optimal_temp = temperature_layer.temperature.item()
    print(f"Optimal temperature: {optimal_temp:.4f}")

    # Show before/after calibration
    with torch.no_grad():
        original_probs = F.softmax(all_logits, dim=1)[:, 1].cpu().numpy()
        calibrated_probs = (
            F.softmax(all_logits / optimal_temp, dim=1)[:, 1].cpu().numpy()
        )

        labels_np = all_labels.cpu().numpy()

        print("\nBefore calibration:")
        print(f"  Prob range: [{original_probs.min():.4f}, {original_probs.max():.4f}]")
        print(f"  Normal mean: {original_probs[labels_np==0].mean():.4f}")
        print(f"  Anomaly mean: {original_probs[labels_np==1].mean():.4f}")

        print("\nAfter calibration:")
        print(
            f"  Prob range: [{calibrated_probs.min():.4f}, {calibrated_probs.max():.4f}]"
        )
        print(f"  Normal mean: {calibrated_probs[labels_np==0].mean():.4f}")
        print(f"  Anomaly mean: {calibrated_probs[labels_np==1].mean():.4f}")

        # AUC should stay the same
        auc_before = roc_auc_score(labels_np, original_probs)
        auc_after = roc_auc_score(labels_np, calibrated_probs)
        print(f"\nAUC before: {auc_before:.4f}, AUC after: {auc_after:.4f}")

    return optimal_temp


def evaluate_calibrated(model, test_loader, device, temperature):
    """Evaluate model with calibrated probabilities"""
    model.eval()

    all_probs_uncalibrated = []
    all_probs_calibrated = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader.get_iterator():
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.LongTensor(y).to(device)

            output = model(testx, return_attention_weights=False)
            logits = output["logits"]

            # Uncalibrated
            probs_uncal = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            # Calibrated
            probs_cal = F.softmax(logits / temperature, dim=1)[:, 1].cpu().numpy()

            all_probs_uncalibrated.extend(probs_uncal)
            all_probs_calibrated.extend(probs_cal)
            all_labels.extend(testy.cpu().numpy())

    all_probs_uncalibrated = np.array(all_probs_uncalibrated)
    all_probs_calibrated = np.array(all_probs_calibrated)
    all_labels = np.array(all_labels)

    # Compute metrics
    auc_uncal = roc_auc_score(all_labels, all_probs_uncalibrated)
    auc_cal = roc_auc_score(all_labels, all_probs_calibrated)

    ap_uncal = average_precision_score(all_labels, all_probs_uncalibrated)
    ap_cal = average_precision_score(all_labels, all_probs_calibrated)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print("\nUncalibrated:")
    print(f"  AUC: {auc_uncal:.4f}")
    print(f"  AP:  {ap_uncal:.4f}")
    print(
        f"  Prob range: [{all_probs_uncalibrated.min():.4f}, {all_probs_uncalibrated.max():.4f}]"
    )

    print("\nCalibrated:")
    print(f"  AUC: {auc_cal:.4f}")
    print(f"  AP:  {ap_cal:.4f}")
    print(
        f"  Prob range: [{all_probs_calibrated.min():.4f}, {all_probs_calibrated.max():.4f}]"
    )

    return {
        "temperature": temperature,
        "calibrated_probs": all_probs_calibrated,
        "uncalibrated_probs": all_probs_uncalibrated,
        "labels": all_labels,
    }


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from models.anomaly_detection import AnomalyDetectionModel
    from training.train_anomaly_detection_model import load_data, load_checkpoint

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model config (same as your training)
    gwnet_params = {
        "device": device,
        "num_nodes": 3,
        "dropout": 0.3,
        "supports": None,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": 2,
        "out_dim": 128,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 5,
        "layers": 6,
    }

    aggregation_params = {
        "temporal_length": 51,
        "num_nodes": 3,
        "out_dim": 128,
        "num_heads": 4,
        "dropout": 0.3,
        "feature_dim": 16,
    }

    # Load model
    model = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type="spatial_temporal_attention",
        aggregation_params=aggregation_params,
        num_classes=2,
    ).to(device)

    load_checkpoint("checkpoints/anomaly_detection/best_model.pt", model)

    # Load data
    train_loader, val_loader, test_loader, scaler, class_weights = load_data(
        "data/temp_dhw/", batch_size=32
    )

    print("Step 1: Finding optimal temperature on validation set...")
    temperature = calibrate_on_validation(model, val_loader, device)

    print("\nStep 2: Evaluating on test set...")
    results = evaluate_calibrated(model, test_loader, device, temperature)

    # Save calibrated model info
    torch.save(
        {"temperature": temperature, "results": results},
        "checkpoints/anomaly_detection/calibration.pt",
    )

    print(f"\n✓ Saved calibration info to checkpoints/anomaly_detection/calibration.pt")
    print(
        f"\nTo use calibrated probabilities, divide logits by {temperature:.4f} before softmax"
    )
