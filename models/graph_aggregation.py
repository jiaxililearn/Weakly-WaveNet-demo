"""
Graph Aggregation Modules for Anomaly Detection

This module contains various graph-level aggregation mechanisms that convert
node-level spatiotemporal features to graph-level representations.

Implemented methods:
1. SimpleGraphAggregation - Baseline pooling methods
2. SpatialTemporalAttention - Parallel multi-head attention (Phase 1)
3. AdditiveAttentionPooling - Bahdanau-style attention
4. HybridGraphAttention - Combination of multi-head + additive attention

Note: AnomalyDetectionModel has been moved to anomaly_detection.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# ============================================================
# GRAPH AGGREGATION MODULES
# ============================================================


class SimpleGraphAggregation(nn.Module):
    """
    Baseline: Simple pooling-based aggregation

    Aggregates (batch, out_dim, num_nodes, seq_length) to (batch, out_dim)
    using basic pooling operations.
    """

    def __init__(self, out_dim, pooling_type="mean"):
        super(SimpleGraphAggregation, self).__init__()
        self.pooling_type = pooling_type
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim // 2, out_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, out_dim, num_nodes, seq_length)
        Returns:
            graph_repr: (batch_size, out_dim)
        """
        if self.pooling_type == "mean":
            pooled = x.mean(dim=[2, 3])
        elif self.pooling_type == "max":
            pooled = x.max(dim=2)[0].max(dim=2)[0]
        elif self.pooling_type == "sum":
            pooled = x.sum(dim=[2, 3])
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return self.fc(pooled)


class SpatialTemporalAttention(nn.Module):
    """
    FIXED: Sequential Spatial-Temporal Attention

    Architecture: Temporal-then-Spatial (T→S)
    1. First: Temporal attention across time steps (for each node independently)
    2. Then: Spatial attention across nodes (using temporally-refined features)

    This creates proper interaction: spatial attention sees temporally-processed features.

    Input: (batch, out_dim, num_nodes, temporal_length)
    Example: (32, 365, 3, 51)

    Processing:
    - Project out_dim (365) → feature_dim (128) to reduce dimensionality
    - Temporal attention: Each node attends over its own time series
    - Spatial attention: Nodes attend to each other with temporal context
    - Output: (batch, feature_dim) graph-level representation
    """

    def __init__(
        self,
        num_nodes,
        out_dim,
        temporal_length,
        num_heads=4,
        dropout=0.3,
        feature_dim=128,
    ):
        super(SpatialTemporalAttention, self).__init__()
        self.num_nodes = num_nodes  # 3
        self.out_dim = out_dim  # 365 (channel dimension from GWNet)
        self.temporal_length = temporal_length  # 51 (compressed time)
        self.feature_dim = feature_dim  # Reduced dimension (e.g., 128)

        # Project out_dim → feature_dim (365 → 128)
        self.feature_proj = nn.Linear(self.out_dim, feature_dim)

        # Step 1: TEMPORAL ATTENTION
        # For each node, attend over its time series
        # Input per node: (batch * num_nodes, temporal_length, feature_dim)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(feature_dim)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.temporal_ffn_norm = nn.LayerNorm(feature_dim)

        # Step 2: SPATIAL ATTENTION
        # Nodes attend to each other using temporally-refined features
        # Input: (batch, num_nodes, feature_dim)
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(feature_dim)
        self.spatial_ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.spatial_ffn_norm = nn.LayerNorm(feature_dim)

        # Final graph-level pooling
        self.graph_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, out_dim, num_nodes, temporal_length)
               Example: (32, 365, 3, 51)

        Returns:
            dict with:
            - representation: (batch, feature_dim) graph-level features
            - temporal_attention: optional attention weights per node
            - spatial_attention: optional attention weights across nodes
        """
        batch_size = x.size(0)

        # Step 0: Projection
        # (batch, out_dim, num_nodes, temporal_length) → (batch, num_nodes, temporal_length, feature_dim)
        x = x.permute(0, 2, 3, 1)  # (32, 3, 51, 365)
        x = self.feature_proj(x)  # (32, 3, 51, 128)

        # ================================================
        # Step 1: TEMPORAL ATTENTION (for each node)
        # ================================================
        # Reshape: (batch * num_nodes, temporal_length, feature_dim)
        x_temporal = x.view(
            batch_size * self.num_nodes, self.temporal_length, self.feature_dim
        )
        # (96, 51, 128) = (32*3, 51, 128)

        # Self-attention over time steps
        temporal_attn_out, temporal_weights = self.temporal_attention(
            x_temporal, x_temporal, x_temporal, need_weights=return_attention_weights
        )

        # Residual + norm
        x_temporal = self.temporal_norm(x_temporal + temporal_attn_out)

        # FFN
        x_temporal = self.temporal_ffn_norm(x_temporal + self.temporal_ffn(x_temporal))

        # Pool over time: (batch * num_nodes, feature_dim)
        temporal_pooled = x_temporal.mean(dim=1)  # (96, 128)

        # Reshape back: (batch, num_nodes, feature_dim)
        temporal_pooled = temporal_pooled.view(
            batch_size, self.num_nodes, self.feature_dim
        )
        # (32, 3, 128)

        # ================================================
        # Step 2: SPATIAL ATTENTION (across nodes)
        # ================================================
        # Input: (batch, num_nodes, feature_dim)
        # Each node is now represented by its temporally-aggregated features

        spatial_attn_out, spatial_weights = self.spatial_attention(
            temporal_pooled,
            temporal_pooled,
            temporal_pooled,
            need_weights=return_attention_weights,
        )

        # Residual + norm
        x_spatial = self.spatial_norm(temporal_pooled + spatial_attn_out)

        # FFN
        x_spatial = self.spatial_ffn_norm(x_spatial + self.spatial_ffn(x_spatial))
        # (32, 3, 128)

        # ================================================
        # Step 3: GRAPH-LEVEL POOLING
        # ================================================
        # Pool over nodes: (batch, feature_dim)
        graph_repr = x_spatial.mean(dim=1)  # (32, 128)
        graph_repr = self.graph_pool(graph_repr)  # (32, 128)

        # Return results
        result = {
            "representation": graph_repr,
            "node_representations": x_spatial,  # (batch, num_nodes, feature_dim)
        }

        if return_attention_weights:
            # temporal_weights: (batch*num_nodes, temporal_length, temporal_length)
            # Reshape to (batch, num_nodes, temporal_length, temporal_length)
            if temporal_weights is not None:
                temporal_weights = temporal_weights.view(
                    batch_size,
                    self.num_nodes,
                    self.temporal_length,
                    self.temporal_length,
                )
            result["temporal_attention"] = temporal_weights  # (32, 3, 51, 51)
            result["spatial_attention"] = spatial_weights  # (32, 3, 3)

        return result


class AdditiveAttentionPooling(nn.Module):
    """
    Alternative: Additive (Bahdanau) Attention for Pooling

    Attention Type: Additive Attention (also called concat attention)
    - Simpler than multi-head attention
    - More interpretable (single attention score per node/time)
    - Good for final aggregation step

    Computes attention as: score = v^T * tanh(W * h)
    """

    def __init__(self, feature_dim, hidden_dim=128):
        super(AdditiveAttentionPooling, self).__init__()

        # Attention mechanism parameters
        self.W = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, dim):
        """
        Args:
            x: Input tensor
            dim: Dimension to pool over (e.g., 2 for nodes, 3 for time)

        Returns:
            pooled: Attention-weighted pooling
            attention_weights: Weights for each element
        """
        # Compute attention scores
        # scores = v^T * tanh(W * x)
        attn_scores = self.v(torch.tanh(self.W(x.transpose(1, dim))))

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted sum
        pooled = (x.transpose(1, dim) * attn_weights).sum(dim=1).transpose(1, 0)

        return pooled, attn_weights.squeeze(-1)


class HybridGraphAttention(nn.Module):
    """
    FIXED: Hybrid Sequential + Additive Attention

    Combines two approaches:
    1. SpatialTemporalAttention: Sequential T→S multi-head attention
    2. AdditiveAttentionPooling: Interpretable weighted pooling on node representations

    The sequential attention provides deep feature processing, while additive pooling
    adds interpretability by learning attention weights over the refined node embeddings.
    """

    def __init__(
        self,
        num_nodes,
        out_dim,
        temporal_length,
        num_heads=4,
        dropout=0.3,
        feature_dim=128,
    ):
        super(HybridGraphAttention, self).__init__()
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.temporal_length = temporal_length
        self.feature_dim = feature_dim

        # Sequential spatial-temporal attention (T→S)
        self.spatial_temporal_attn = SpatialTemporalAttention(
            num_nodes=num_nodes,
            out_dim=out_dim,
            temporal_length=temporal_length,
            num_heads=num_heads,
            dropout=dropout,
            feature_dim=feature_dim,
        )

        # Additive attention for interpretable node pooling
        # Operates on node_representations: (batch, num_nodes, feature_dim)
        self.additive_pool = AdditiveAttentionPooling(
            feature_dim=feature_dim, hidden_dim=feature_dim // 2
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention_weights=False):
        """
        Args:
            x: (batch, out_dim, num_nodes, temporal_length)
               Example: (32, 365, 3, 51)

        Returns:
            dict with:
            - representation: (batch, feature_dim) final graph representation
            - node_representations: (batch, num_nodes, feature_dim) per-node features
            - attention weights if requested
        """
        batch_size = x.size(0)

        # Step 1: Apply sequential spatial-temporal attention
        st_output = self.spatial_temporal_attn(x, return_attention_weights)
        st_repr = st_output["representation"]  # (batch, feature_dim)
        node_repr = st_output["node_representations"]  # (batch, num_nodes, feature_dim)

        # Step 2: Apply additive attention pooling over nodes
        # This provides interpretable weights showing which nodes are most important
        additive_repr, additive_weights = self.additive_pool(
            node_repr, dim=1
        )  # (batch, feature_dim), (batch, num_nodes)

        # Step 3: Fuse both representations
        # Combines deep sequential processing with interpretable pooling
        combined = torch.cat([st_repr, additive_repr], dim=1)  # (batch, feature_dim*2)
        graph_repr = self.fusion(combined)  # (batch, feature_dim)

        # Build result dictionary
        result = {
            "representation": graph_repr,
            "node_representations": node_repr,
            "st_representation": st_repr,  # From sequential attention
            "additive_representation": additive_repr,  # From additive pooling
            "additive_weights": additive_weights,  # Node importance weights
        }

        if return_attention_weights:
            result["spatial_attention"] = st_output.get("spatial_attention")
            result["temporal_attention"] = st_output.get("temporal_attention")

        return result


# ============================================================
# NOTE: AnomalyDetectionModel has been moved to anomaly_detection.py
# Import it using:
#   from anomaly_detection import AnomalyDetectionModel
# ============================================================
