"""
Attention-based pooling modules for graph-level predictions.

All modules follow the same interface:
    Input: (batch, channels, nodes, time)
    Output: (pooled_features, attention_weights)
        - pooled_features: (batch, channels)
        - attention_weights: variable shape depending on mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalAttentionPool(nn.Module):
    """
    Multi-head attention pooling over spatial-temporal features.

    Uses self-attention to learn which (node, timestep) combinations
    are most important for graph-level prediction.

    Args:
        in_channels: Number of input feature channels
        num_heads: Number of attention heads (default: 4)
        hidden_dim: Hidden dimension for Q/K projections (default: 64)
    """
    def __init__(self, in_channels, num_heads=4, hidden_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # Multi-head attention over flattened spatial-temporal positions
        self.attention = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Query vector for pooling (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, in_channels))

    def forward(self, x):
        """
        Args:
            x: (batch, channels, nodes, time)
        Returns:
            pooled: (batch, channels)
            attn_weights: (batch, 1, nodes*time) - attention weights for interpretability
        """
        b, c, n, t = x.shape

        # Reshape to (batch, nodes*time, channels)
        x_flat = x.permute(0, 2, 3, 1).reshape(b, n * t, c)

        # Expand query for batch
        query = self.query.expand(b, -1, -1)  # (batch, 1, channels)

        # Query attends to all spatial-temporal positions
        pooled, attn_weights = self.attention(query, x_flat, x_flat)
        # pooled: (batch, 1, channels)
        # attn_weights: (batch, 1, nodes*time)

        pooled = pooled.squeeze(1)  # (batch, channels)

        return pooled, attn_weights


class GraphLevelGRU(nn.Module):
    """
    GRU-based temporal aggregation after spatial encoding.

    Processes the temporal sequence with a GRU to capture temporal dynamics,
    then uses the final hidden state as the graph representation.

    Args:
        in_channels: Number of input feature channels
        num_nodes: Number of nodes in the graph
        hidden_dim: GRU hidden dimension (default: 128)
        num_layers: Number of GRU layers (default: 2)
    """
    def __init__(self, in_channels, num_nodes, hidden_dim=128, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU processes temporal sequence
        # Input: flattened spatial features (channels * nodes) at each timestep
        self.gru = nn.GRU(
            input_size=in_channels * num_nodes,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        # Project GRU hidden state back to original channel dimension
        self.projection = nn.Linear(hidden_dim, in_channels)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, nodes, time)
        Returns:
            pooled: (batch, channels)
            hidden_states: (batch, time, hidden_dim) - temporal hidden states for analysis
        """
        b, c, n, t = x.shape

        # Reshape: (batch, time, channels*nodes)
        # Flatten spatial dimension, keep temporal sequence
        x_seq = x.permute(0, 3, 1, 2).reshape(b, t, c * n)

        # Process with GRU
        output, hidden = self.gru(x_seq)
        # output: (batch, time, hidden_dim) - all timestep outputs
        # hidden: (num_layers, batch, hidden_dim) - final hidden states

        # Use final layer's hidden state as graph representation
        graph_repr = hidden[-1]  # (batch, hidden_dim)

        # Project back to original channel dimension
        pooled = self.projection(graph_repr)  # (batch, channels)

        return pooled, output


class SetTransformerPool(nn.Module):
    """
    Set Transformer pooling with learnable seed vectors.

    Treats spatial-temporal positions as a set and uses seed-based attention
    for permutation-invariant pooling.

    Reference: "Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks" (ICML 2019)

    Args:
        in_channels: Number of input feature channels
        num_seeds: Number of learnable seed vectors (default: 16)
        num_heads: Number of attention heads (default: 4)
    """
    def __init__(self, in_channels, num_seeds=16, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_seeds = num_seeds
        self.num_heads = num_heads

        # Learnable seed vectors (induced set)
        self.seeds = nn.Parameter(torch.randn(num_seeds, in_channels))
        nn.init.xavier_uniform_(self.seeds)

        # Multi-head attention block (seeds attend to input set)
        self.mab = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer norm for stability
        self.ln1 = nn.LayerNorm(in_channels)
        self.ln2 = nn.LayerNorm(in_channels)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_channels * 4, in_channels)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, nodes, time)
        Returns:
            pooled: (batch, channels)
            attn_weights: (batch, num_seeds, nodes*time) - seed attention patterns
        """
        b, c, n, t = x.shape

        # Reshape to set: (batch, nodes*time, channels)
        x_set = x.permute(0, 2, 3, 1).reshape(b, n * t, c)

        # Expand seeds for batch
        seeds = self.seeds.unsqueeze(0).expand(b, -1, -1)  # (batch, num_seeds, channels)

        # Seeds attend to input set (Multihead Attention Block)
        attended, attn_weights = self.mab(seeds, x_set, x_set)
        # attended: (batch, num_seeds, channels)
        # attn_weights: (batch, num_seeds, nodes*time)

        # Residual connection + layer norm
        seeds_updated = self.ln1(seeds + attended)

        # Feed-forward network with residual
        seeds_ffn = self.ffn(seeds_updated)
        seeds_final = self.ln2(seeds_updated + seeds_ffn)

        # Aggregate seeds (mean pooling over seed dimension)
        pooled = seeds_final.mean(dim=1)  # (batch, channels)

        return pooled, attn_weights


# Utility function to create attention module based on type
def create_attention_module(attention_type, in_channels, num_nodes=None):
    """
    Factory function to create attention pooling modules.

    Args:
        attention_type: One of ['temporal_mha', 'gru', 'set_transformer']
        in_channels: Number of feature channels
        num_nodes: Number of nodes (required for 'gru')

    Returns:
        Attention module instance
    """
    if attention_type == 'temporal_mha':
        return TemporalAttentionPool(in_channels)
    elif attention_type == 'gru':
        if num_nodes is None:
            raise ValueError("num_nodes must be specified for GRU attention")
        return GraphLevelGRU(in_channels, num_nodes)
    elif attention_type == 'set_transformer':
        return SetTransformerPool(in_channels)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
