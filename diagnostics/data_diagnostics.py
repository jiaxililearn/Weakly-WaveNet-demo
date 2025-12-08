"""
Data Quality Diagnostics for Anomaly Detection

Run these checks to understand if your data quality is limiting performance.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_data_quality(data_path, save_dir="./diagnostics"):
    """Comprehensive data quality checks"""
    import os

    os.makedirs(save_dir, exist_ok=True)

    # Load data
    train_data = np.load(f"{data_path}/train.npz")
    val_data = np.load(f"{data_path}/val.npz")
    test_data = np.load(f"{data_path}/test.npz")

    train_x, train_y = train_data["x"], train_data["y"]
    val_x, val_y = val_data["x"], val_data["y"]
    test_x, test_y = test_data["x"], test_data["y"]

    logger.info("=" * 60)
    logger.info("DATA QUALITY DIAGNOSTICS")
    logger.info("=" * 60)

    # 1. CLASS BALANCE
    logger.info("\n1. CLASS BALANCE CHECK")
    logger.info(
        f"Train - Normal: {np.sum(train_y == 0)}, Anomaly: {np.sum(train_y == 1)}"
    )
    logger.info(f"Val   - Normal: {np.sum(val_y == 0)}, Anomaly: {np.sum(val_y == 1)}")
    logger.info(
        f"Test  - Normal: {np.sum(test_y == 0)}, Anomaly: {np.sum(test_y == 1)}"
    )

    train_ratio = np.sum(train_y == 1) / len(train_y)
    test_ratio = np.sum(test_y == 1) / len(test_y)
    logger.info(f"Anomaly ratio - Train: {train_ratio:.3f}, Test: {test_ratio:.3f}")

    if train_ratio < 0.05 or train_ratio > 0.5:
        logger.warning("⚠️  SEVERE CLASS IMBALANCE - Consider class weights!")

    # 2. DATA DISTRIBUTION SIMILARITY
    logger.info("\n2. TRAIN/TEST DISTRIBUTION SIMILARITY")
    train_mean = train_x.mean(axis=(0, 2, 3))  # Mean per feature
    test_mean = test_x.mean(axis=(0, 2, 3))
    train_std = train_x.std(axis=(0, 2, 3))
    test_std = test_x.std(axis=(0, 2, 3))

    mean_diff = np.abs(train_mean - test_mean).mean()
    std_diff = np.abs(train_std - test_std).mean()

    logger.info(f"Mean difference: {mean_diff:.4f}")
    logger.info(f"Std difference: {std_diff:.4f}")

    if mean_diff > 0.5:
        logger.warning("⚠️  LARGE DISTRIBUTION SHIFT between train/test!")

    # 3. ANOMALY SEPARABILITY (T-SNE visualization needed)
    logger.info("\n3. FEATURE STATISTICS")
    normal_samples = train_x[train_y == 0]
    anomaly_samples = train_x[train_y == 1]

    normal_var = normal_samples.var(axis=0).mean()
    anomaly_var = anomaly_samples.var(axis=0).mean()

    logger.info(f"Normal variance: {normal_var:.4f}")
    logger.info(f"Anomaly variance: {anomaly_var:.4f}")
    logger.info(f"Variance ratio: {anomaly_var / normal_var:.4f}")

    # 4. TEMPORAL PATTERNS
    logger.info("\n4. TEMPORAL PATTERN CHECK")
    normal_temporal_mean = normal_samples.mean(axis=(0, 2))  # (seq_len,)
    anomaly_temporal_mean = anomaly_samples.mean(axis=(0, 2))

    temporal_diff = np.abs(normal_temporal_mean - anomaly_temporal_mean).mean()
    logger.info(f"Temporal pattern difference: {temporal_diff:.4f}")

    if temporal_diff < 0.1:
        logger.warning(
            "⚠️  WEAK TEMPORAL SEPARATION - Anomalies may not have distinct temporal patterns!"
        )

    # 5. SPATIAL PATTERNS (across nodes)
    logger.info("\n5. SPATIAL PATTERN CHECK")
    normal_spatial_mean = normal_samples.mean(axis=(0, 1))  # (num_nodes, features)
    anomaly_spatial_mean = anomaly_samples.mean(axis=(0, 1))

    spatial_diff = np.abs(normal_spatial_mean - anomaly_spatial_mean).mean()
    logger.info(f"Spatial pattern difference: {spatial_diff:.4f}")

    if spatial_diff < 0.1:
        logger.warning(
            "⚠️  WEAK SPATIAL SEPARATION - Anomalies may not have distinct spatial patterns!"
        )

    # 6. LABEL NOISE ESTIMATION
    logger.info("\n6. POTENTIAL LABEL NOISE")
    # Calculate intra-class variance vs inter-class variance
    normal_centroid = normal_samples.mean(axis=0)
    anomaly_centroid = anomaly_samples.mean(axis=0)

    intra_normal = ((normal_samples - normal_centroid) ** 2).mean()
    intra_anomaly = ((anomaly_samples - anomaly_centroid) ** 2).mean()
    inter_class = ((normal_centroid - anomaly_centroid) ** 2).mean()

    logger.info(f"Intra-class variance (normal): {intra_normal:.4f}")
    logger.info(f"Intra-class variance (anomaly): {intra_anomaly:.4f}")
    logger.info(f"Inter-class variance: {inter_class:.4f}")

    separability = inter_class / (intra_normal + intra_anomaly + 1e-8)
    logger.info(f"Separability ratio: {separability:.4f}")

    if separability < 0.1:
        logger.warning(
            "⚠️  VERY LOW SEPARABILITY - Data may have high label noise or classes overlap!"
        )

    # 7. MISSING DATA / OUTLIERS
    logger.info("\n7. DATA QUALITY ISSUES")
    nan_count = np.isnan(train_x).sum() + np.isnan(test_x).sum()
    inf_count = np.isinf(train_x).sum() + np.isinf(test_x).sum()

    logger.info(f"NaN values: {nan_count}")
    logger.info(f"Inf values: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        logger.warning("⚠️  DATA CORRUPTION DETECTED!")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS:")
    logger.info("=" * 60)

    if separability < 0.1:
        logger.info("❌ Data quality is likely the issue - consider:")
        logger.info("   - Verify label correctness")
        logger.info("   - Add more informative features")
        logger.info("   - Use semi-supervised learning")
    elif train_ratio < 0.1:
        logger.info("⚠️  Try class balancing techniques")
    else:
        logger.info("✓ Data looks reasonable - focus on model improvements")

    return {
        "separability": separability,
        "train_ratio": train_ratio,
        "temporal_diff": temporal_diff,
        "spatial_diff": spatial_diff,
    }


def analyze_model_predictions(model, test_loader, device, save_dir="./diagnostics"):
    """Analyze where the model is failing"""
    import os

    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for iter, (x, y) in enumerate(test_loader.get_iterator()):
            # Convert to tensors and move to device
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(
                1, 3
            )  # (batch, seq_length, num_nodes, in_dim) -> (batch, in_dim, num_nodes, seq_length)
            testy = torch.LongTensor(y).to(device)  # (batch,)

            # Forward pass
            output = model(testx, return_attention_weights=False)
            logits = output["logits"]

            # Get predictions
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds_probs = probs[:, 1].cpu().numpy()
            preds_hard = (preds_probs > 0.5).astype(int)
            labels = testy.cpu().numpy()

            all_labels.extend(labels)
            all_probs.extend(preds_probs)
            all_preds.extend(preds_hard)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    logger.info("\n" + "=" * 60)
    logger.info("MODEL PREDICTION ANALYSIS")
    logger.info("=" * 60)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"TN={cm[0,0]}, FP={cm[0,1]}")
    logger.info(f"FN={cm[1,0]}, TP={cm[1,1]}")

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    logger.info(f"\nDetailed Metrics:")
    logger.info(f"Sensitivity (Recall): {sensitivity:.3f}")
    logger.info(f"Specificity: {specificity:.3f}")
    logger.info(f"PPV (Precision): {ppv:.3f}")
    logger.info(f"NPV: {npv:.3f}")

    # Prediction distribution analysis
    logger.info(f"\nPrediction Score Distribution:")
    logger.info(
        f"Normal samples - Mean: {all_probs[all_labels==0].mean():.3f}, Std: {all_probs[all_labels==0].std():.3f}"
    )
    logger.info(
        f"Anomaly samples - Mean: {all_probs[all_labels==1].mean():.3f}, Std: {all_probs[all_labels==1].std():.3f}"
    )

    # Check for model bias
    if all_probs.mean() < 0.3:
        logger.warning("⚠️  Model is biased toward NORMAL class")
    elif all_probs.mean() > 0.7:
        logger.warning("⚠️  Model is biased toward ANOMALY class")

    # Plot ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[0].plot(fpr, tpr, linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].grid(True, alpha=0.3)

    # PR curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    axes[1].plot(recall, precision, linewidth=2)
    axes[1].axhline(
        y=all_labels.mean(), color="k", linestyle="--", alpha=0.3, label="Random"
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_curves.png", dpi=150, bbox_inches="tight")
    logger.info(f"\n✓ Saved curves to {save_dir}/model_curves.png")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    # Run data diagnostics
    stats = diagnose_data_quality("./data/temp_dhw")

    print("\n" + "=" * 60)
    print("Run this with your trained model:")
    print("=" * 60)
    print(
        """
from data_diagnostics import analyze_model_predictions
from anomaly_detection import AnomalyDetectionModel
import torch

# Load your model
model = AnomalyDetectionModel(...)
model.load_state_dict(torch.load('checkpoints/anomaly_detection/best_model.pt'))

# Analyze predictions
analyze_model_predictions(model, test_loader, device='cuda')
    """
    )
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from models.anomaly_detection import AnomalyDetectionModel
    from training.train_anomaly_detection_model import load_data, load_checkpoint

    # Setup device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    num_nodes = 3
    dropout = 0.3
    in_dim = 2
    seq_length = 128
    num_classes = 2
    gwnet_blocks = 5  # User-configurable
    gwnet_layers = 6  # User-configurable
    feature_dim = 16  # Reduced feature dimension for attention to prevent memory issues
    temporal_length = 51
    num_heads = 4
    batch_size = 32
    aggregation_type = "spatial_temporal_attention"

    data = "data/temp_dhw/"

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

    # model.load_state_dict(torch.load("checkpoints/anomaly_detection/best_model.pt"))
    best_model_path = "checkpoints/anomaly_detection/best_model.pt"
    load_checkpoint(best_model_path, model)

    train_loader, val_loader, test_loader, scaler, class_weights = load_data(
        data, batch_size
    )
    # Analyze predictions
    analyze_model_predictions(model, test_loader, device="cuda")
