"""
Example: How to use different attention mechanisms for anomaly detection

This script demonstrates:
1. How to instantiate models with different attention types
2. How to compare their performance
3. How to visualize attention weights
"""

import torch
import torch.nn as nn
import numpy as np
from anomaly_detection import AnomalyDetectionModel
from graph_aggregation import SimpleGraphAggregation, SpatialTemporalAttention
from attention_utils import (
    visualize_attention_weights,
    visualize_pooling_attention,
    compare_attention_methods,
)


def create_models(device, num_nodes=207, seq_length=12, in_dim=2, out_dim=12):
    """
    Create different model variants for comparison

    Returns:
        dict of {model_name: model}
    """
    # Common GWNet parameters
    gwnet_params = {
        "device": device,
        "num_nodes": num_nodes,
        "dropout": 0.3,
        "supports": None,
        "gcn_bool": True,
        "addaptadj": True,
        "aptinit": None,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "residual_channels": 32,
        "dilation_channels": 32,
        "skip_channels": 256,
        "end_channels": 512,
        "kernel_size": 2,
        "blocks": 4,
        "layers": 7,
    }

    # Common aggregation parameters
    base_agg_params = {
        "out_dim": out_dim,
        "num_nodes": num_nodes,
        "seq_length": seq_length,
    }

    models = {}

    # ================================================
    # 1. BASELINE: Simple Mean Pooling
    # ================================================
    models["baseline_mean"] = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type="simple",
        aggregation_params={"out_dim": out_dim, "pooling_type": "mean"},
        num_classes=2,
    )

    # ================================================
    # 2. BASELINE: Simple Max Pooling
    # ================================================
    models["baseline_max"] = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type="simple",
        aggregation_params={"out_dim": out_dim, "pooling_type": "max"},
        num_classes=2,
    )

    # ================================================
    # 3. PROPOSED: Spatial-Temporal Attention (4 heads)
    # ================================================
    models["attention_4heads"] = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type="spatial_temporal_attention",
        aggregation_params={**base_agg_params, "num_heads": 4, "dropout": 0.3},
        num_classes=2,
    )

    # ================================================
    # 4. PROPOSED: Spatial-Temporal Attention (8 heads)
    # ================================================
    models["attention_8heads"] = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type="spatial_temporal_attention",
        aggregation_params={**base_agg_params, "num_heads": 8, "dropout": 0.3},
        num_classes=2,
    )

    # ================================================
    # 5. HYBRID: Multi-Head + Additive Pooling
    # ================================================
    models["hybrid"] = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type="hybrid",
        aggregation_params={**base_agg_params, "num_heads": 4, "dropout": 0.3},
        num_classes=2,
    )

    # Move all models to device
    for name, model in models.items():
        models[name] = model.to(device)

    return models


def example_forward_pass(device):
    """
    Demonstrate a forward pass with attention visualization
    """
    print("=" * 80)
    print("EXAMPLE: Forward Pass with Attention Visualization")
    print("=" * 80)

    # Create a sample input
    batch_size = 4
    in_dim = 2
    num_nodes = 207
    seq_length = 12

    # Random input (in practice, this comes from your dataloader)
    x = torch.randn(batch_size, in_dim, num_nodes, seq_length).to(device)

    # Create model with attention
    gwnet_params = {
        "device": device,
        "num_nodes": num_nodes,
        "dropout": 0.3,
        "supports": None,
        "gcn_bool": True,
        "addaptadj": True,
        "in_dim": in_dim,
        "out_dim": 12,
    }

    agg_params = {
        "out_dim": 12,
        "num_nodes": num_nodes,
        "seq_length": seq_length,
        "num_heads": 4,
        "dropout": 0.3,
    }

    model = AnomalyDetectionModel(
        gwnet_params=gwnet_params,
        aggregation_type="spatial_temporal_attention",
        aggregation_params=agg_params,
        num_classes=2,
    ).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, return_attention_weights=True)

    # Print output shapes
    print("\nOutput shapes:")
    print(f"  Detection score: {output['detection_score'].shape}")
    print(f"  Logits: {output['logits'].shape}")
    print(f"  Probabilities: {output['probabilities'].shape}")
    print(f"  Graph representation: {output['representation'].shape}")

    if "spatial_attention" in output:
        print(f"  Spatial attention: {output['spatial_attention'].shape}")
        print(f"  Temporal attention: {output['temporal_attention'].shape}")

    # Print some predictions
    print("\nPredictions (first 3 samples):")
    for i in range(min(3, batch_size)):
        prob_anomaly = output["detection_score"][i, 0].item()
        prob_normal = output["probabilities"][i, 0].item()
        print(
            f"  Sample {i}: P(Anomaly)={prob_anomaly:.4f}, P(Normal)={prob_normal:.4f}"
        )

    # Visualize attention (if available)
    if "spatial_attention" in output:
        print("\nVisualizing attention weights for sample 0...")
        visualize_attention_weights(
            output["spatial_attention"],
            output["temporal_attention"],
            sample_idx=0,
            save_path="example_attention_weights.png",
        )

    return output


def compare_model_sizes(models_dict):
    """
    Compare parameter counts of different models
    """
    print("\n" + "=" * 80)
    print("MODEL SIZE COMPARISON")
    print("=" * 80)

    results = []
    for name, model in models_dict.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        results.append(
            {
                "model": name,
                "total_params": total_params,
                "trainable_params": trainable_params,
            }
        )

        print(f"\n{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

    print("=" * 80)

    return results


def main():
    """
    Main demonstration
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ================================================
    # 1. Create all model variants
    # ================================================
    print("Creating model variants...")
    models = create_models(device)
    print(f"Created {len(models)} model variants\n")

    # ================================================
    # 2. Compare model sizes
    # ================================================
    model_sizes = compare_model_sizes(models)

    # ================================================
    # 3. Example forward pass with visualization
    # ================================================
    output = example_forward_pass(device)

    # ================================================
    # 4. Print summary
    # ================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Attention Mechanisms")
    print("=" * 80)
    print("\nImplemented attention types:")
    print("  1. Multi-Head Self-Attention (Spatial)")
    print("     - Captures node-to-node relationships")
    print("     - Identifies which nodes are correlated during anomalies")
    print()
    print("  2. Multi-Head Self-Attention (Temporal)")
    print("     - Captures time-to-time relationships")
    print("     - Identifies which timesteps are related")
    print()
    print("  3. Additive Attention (for pooling)")
    print("     - Learns importance weights for final aggregation")
    print("     - More interpretable than self-attention")
    print()
    print("Usage for your research:")
    print("  - Use models['baseline_mean'] as baseline")
    print("  - Use models['attention_4heads'] as your proposed method")
    print("  - Compare AUROC, AUPRC, F1 scores")
    print("  - Visualize attention to show which nodes/times matter")
    print("=" * 80)


if __name__ == "__main__":
    main()
