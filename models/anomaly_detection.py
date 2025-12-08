"""
Anomaly Detection Model for Spatiotemporal Graphs

This module contains the complete anomaly detection model that combines
GWNet feature extraction with graph aggregation mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from model import gwnet
from graph_aggregation import (
    SimpleGraphAggregation,
    SpatialTemporalAttention,
    HybridGraphAttention,
)
import logging

logger = logging.getLogger(__name__)


class AnomalyDetectionModel(nn.Module):
    """
    Complete model: GWNet + Graph Aggregation for Anomaly Detection

    Usage:
        model = AnomalyDetectionModel(
            gwnet_params=dict(device=device, num_nodes=207, ...),
            aggregation_type='spatial_temporal_attention',  # or 'simple', 'hybrid'
            aggregation_params=dict(temporal_length=12, num_nodes=207, seq_length=12)
        )

        output = model(x)  # x: (batch, in_dim, num_nodes, seq_length)
        detection_score = output['detection_score']  # (batch, 1) probability
    """

    def __init__(
        self,
        gwnet_params,
        aggregation_type="spatial_temporal_attention",
        aggregation_params=None,
        num_classes=2,
    ):
        """
        Args:
            gwnet_params: Dict of parameters for gwnet
            aggregation_type: 'simple', 'spatial_temporal_attention', or 'hybrid'
            aggregation_params: Dict of parameters for aggregation module
            num_classes: Number of classes (2 for binary anomaly detection)
        """
        super(AnomalyDetectionModel, self).__init__()

        # Feature extraction: GWNet
        self.gwnet = gwnet(**gwnet_params)

        # Graph aggregation
        if aggregation_params is None:
            raise ValueError("aggregation_params cannot be None")

        if aggregation_type == "simple":
            self.aggregation = SimpleGraphAggregation(
                out_dim=aggregation_params["temporal_length"],
                pooling_type=aggregation_params.get("pooling_type", "mean"),
            )
        elif aggregation_type == "spatial_temporal_attention":
            self.aggregation = SpatialTemporalAttention(**aggregation_params)
        elif aggregation_type == "hybrid":
            self.aggregation = HybridGraphAttention(**aggregation_params)
        else:
            raise ValueError(f"Unknown aggregation type: {aggregation_type}")

        # Classification head
        # Use feature_dim if available (for attention modules), otherwise temporal_length
        classifier_input_dim = aggregation_params.get(
            "feature_dim", aggregation_params["temporal_length"]
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_input_dim // 2, num_classes),
        )

        self.aggregation_type = aggregation_type

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch_size, in_dim, num_nodes, seq_length)
            return_attention_weights: If True, returns attention weights

        Returns:
            dict with:
            - logits: (batch_size, num_classes) raw predictions
            - detection_score: (batch_size, 1) probability of anomaly (class 1)
            - representation: (batch_size, temporal_length) graph-level features
            - (optional) attention weights if return_attention_weights=True
        """
        logger.debug(f"Input shape: {x.shape}")
        # Pad input for GWNet
        input = nn.functional.pad(x, (1, 0, 0, 0))
        logger.debug(f"Input shape after padding: {input.shape}")
        # Extract spatiotemporal features
        # GWNet outputs: (batch, sequence_length, num_nodes, temporal_length)
        # where temporal_length depends on receptive field and input length
        features = self.gwnet(input)
        logger.debug(f"Features shape from GWNet: {features.shape}")
        # Features is already in correct format: (batch, temporal_length, num_nodes, seq_length)
        # No transpose needed!

        # Aggregate to graph level
        if self.aggregation_type == "simple":
            graph_repr = self.aggregation(features)
            agg_output = {"representation": graph_repr}
        else:
            agg_output = self.aggregation(features, return_attention_weights)
            graph_repr = agg_output["representation"]

        # Classification
        logits = self.classifier(graph_repr)  # (batch, num_classes)

        # Probability of anomaly (class 1)
        probs = F.softmax(logits, dim=1)
        detection_score = probs[:, 1:2]  # (batch, 1)

        # Prepare output
        output = {
            "logits": logits,
            "detection_score": detection_score,
            "probabilities": probs,
            "representation": graph_repr,
        }

        # Add attention weights if available
        if return_attention_weights and self.aggregation_type != "simple":
            output.update(
                {k: v for k, v in agg_output.items() if k not in ["representation"]}
            )

        return output
