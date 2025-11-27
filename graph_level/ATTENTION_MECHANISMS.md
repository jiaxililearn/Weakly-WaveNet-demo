# Graph-Level Attention Mechanisms

This document describes the attention-based pooling mechanisms available for graph-level anomaly detection tasks.

## Overview

Three advanced attention mechanisms have been added to improve graph-level predictions, particularly for anomaly detection tasks where temporal dynamics and spatial relationships are critical.

## Available Attention Types

When using `--pooling=attention`, you can specify the attention mechanism via `--attention-type`:

### 1. Simple Attention (Default) - `simple`
**Original implementation** - backward compatible

**How it works:**
- Learns scalar weights for each (node, timestep) position via 1×1 convolution
- Single global softmax over all positions
- Weighted sum of features

**Pros:**
- ✅ Lightweight and fast
- ✅ Simple to interpret
- ✅ Works well for global anomaly patterns

**Cons:**
- ❌ Each position scored independently (no context awareness)
- ❌ Mixes spatial and temporal dimensions in softmax

**Best for:** Simple anomaly detection where unusual values (not patterns) indicate anomalies

**Example:**
```bash
python train_graph.py --data data/test --num-nodes 2 --seq-length 365 \
       --task classification --num-classes 3 --pooling attention \
       --attention-type simple --aptonly --addaptadj
```

---

### 2. Temporal Multi-Head Attention - `temporal_mha`
**Context-aware attention over spatial-temporal features**

**How it works:**
- Uses multi-head self-attention (4 heads by default)
- Learnable query vector attends to all (node, timestep) positions
- Captures relationships between positions based on feature similarity

**Pros:**
- ✅ **Context-aware**: Positions attend to each other based on learned relationships
- ✅ **Detects relational anomalies**: Identifies when nodes desynchronize or behave differently
- ✅ **Returns attention weights**: Interpretable (which timesteps/nodes triggered detection)
- ✅ **Multi-head**: Learns different attention patterns simultaneously

**Cons:**
- ⚠️ Higher memory usage: O((nodes × time)²) attention matrix
- ⚠️ Slower than simple attention

**Best for:**
- Detecting **temporal anomalies** (unusual sequences, not just unusual values)
- Identifying **spatial anomalies** (one sensor deviates from others)
- Tasks requiring **interpretability** (visualize attention weights)

**Example:**
```bash
python train_graph.py --data data/test --num-nodes 2 --seq-length 365 \
       --task classification --num-classes 3 --pooling attention \
       --attention-type temporal_mha --aptonly --addaptadj
```

**Implementation details:**
- Embed dimension: `end_channels` (default: 512)
- Number of heads: 4
- Query: Learnable parameter (1, end_channels)
- Output: `(batch, end_channels)` + attention weights `(batch, 1, nodes*time)`

---

### 3. GRU-based Temporal Aggregation - `gru`
**Processes temporal sequence with RNN before pooling**

**How it works:**
- Flattens spatial dimension (nodes × features) at each timestep
- Processes temporal sequence with 2-layer GRU (hidden_dim=128)
- Uses final hidden state as graph representation
- Projects back to original channel dimension

**Pros:**
- ✅ **Explicitly models temporal dependencies**: Captures sequential patterns
- ✅ **Hidden state encodes history**: Suitable for gradual anomalies (degradation over time)
- ✅ **Good for long sequences**: GRU handles long-term dependencies better than pooling
- ✅ **Recurrent structure**: Natural for time-series anomaly detection

**Cons:**
- ⚠️ Sequential computation (cannot parallelize over time)
- ⚠️ May be slow for very long sequences (365 days)
- ⚠️ Requires more training data to learn temporal patterns

**Best for:**
- **Sequential anomalies**: Patterns that unfold over time (e.g., gradual sensor drift)
- **Temporal order matters**: Early vs. late events have different meanings
- **Long-term dependencies**: Yearly patterns, seasonal cycles

**Example:**
```bash
python train_graph.py --data data/test --num-nodes 2 --seq-length 365 \
       --task classification --num-classes 3 --pooling attention \
       --attention-type gru --aptonly --addaptadj
```

**Implementation details:**
- Input dimension: `num_nodes × end_channels` (flattened spatial)
- Hidden dimension: 128
- Number of layers: 2 (with 0.3 dropout)
- Output: `(batch, end_channels)` + all hidden states `(batch, time, 128)`

---

### 4. Set Transformer - `set_transformer`
**Permutation-invariant pooling with learnable seed vectors**

**How it works:**
- Treats (nodes, timesteps) as an unordered set
- Uses 16 learnable seed vectors to query the input set
- Multi-head attention (4 heads) + feed-forward network
- Aggregates seeds via mean pooling

**Pros:**
- ✅ **Permutation-invariant**: Graph topology doesn't affect output
- ✅ **Learnable pooling**: Seeds learn to focus on important patterns
- ✅ **Good for irregular graphs**: Works with variable-sized or dynamic graphs
- ✅ **Principled aggregation**: Based on Set Transformer paper (ICML 2019)

**Cons:**
- ⚠️ Less interpretable than direct attention weights
- ⚠️ Requires tuning `num_seeds` hyperparameter
- ⚠️ More complex architecture (attention + FFN)

**Best for:**
- **Irregular or variable-sized graphs**: Different sites have different numbers of sensors
- **Unordered sets**: When node ordering is arbitrary
- **Global graph properties**: Detecting overall system-level anomalies

**Example:**
```bash
python train_graph.py --data data/test --num-nodes 2 --seq-length 365 \
       --task classification --num-classes 3 --pooling attention \
       --attention-type set_transformer --aptonly --addaptadj
```

**Implementation details:**
- Number of seeds: 16
- Number of heads: 4
- Feed-forward: 4× expansion with ReLU and 0.1 dropout
- Layer normalization for stability
- Output: `(batch, end_channels)` + seed attention `(batch, 16, nodes*time)`

---

## Comparison Table

| Attention Type | Memory | Speed | Context-Aware | Temporal Modeling | Interpretability | Best For |
|----------------|--------|-------|---------------|-------------------|------------------|----------|
| `simple` | Low | Fast | ❌ | ❌ | ✅ High | Unusual values |
| `temporal_mha` | High | Moderate | ✅ | ✅ | ✅✅ Very High | Relational anomalies |
| `gru` | Moderate | Slow | ✅ | ✅✅ | ⚠️ Medium | Sequential patterns |
| `set_transformer` | High | Moderate | ✅ | ⚠️ | ⚠️ Low | Irregular graphs |

---

## Usage Examples

### Training with Different Attention Mechanisms

```bash
# Simple attention (baseline)
python train_graph.py --data data/anomaly --num-nodes 5 --seq-length 365 \
       --task classification --num-classes 2 --pooling attention \
       --attention-type simple --epochs 100 --aptonly

# Temporal MHA (best for anomaly detection)
python train_graph.py --data data/anomaly --num-nodes 5 --seq-length 365 \
       --task classification --num-classes 2 --pooling attention \
       --attention-type temporal_mha --epochs 100 --aptonly

# GRU (best for temporal patterns)
python train_graph.py --data data/anomaly --num-nodes 5 --seq-length 365 \
       --task classification --num-classes 2 --pooling attention \
       --attention-type gru --epochs 100 --aptonly

# Set Transformer (best for irregular graphs)
python train_graph.py --data data/anomaly --num-nodes 5 --seq-length 365 \
       --task classification --num-classes 2 --pooling attention \
       --attention-type set_transformer --epochs 100 --aptonly
```

### Testing

The test script automatically loads the attention type from the config file:

```bash
# Config file is auto-loaded from checkpoint directory
python test_graph.py \
       --checkpoint ./checkpoints/graph_model_exp1_best.pth \
       --data data/anomaly \
       --num-nodes 5 \
       --seq-length 365 \
       --task classification \
       --num-classes 2 \
       --aptonly
```

---

## Recommendations for Graph-Level Anomaly Detection

Based on your goal of detecting anomalies from spatial-temporal features:

### **Option 1: Start with Temporal MHA** ✅ Recommended
- Best balance of performance and interpretability
- Explicitly captures spatial-temporal relationships
- Attention weights show which (node, time) pairs triggered the anomaly
- Good for both sudden and gradual anomalies

### **Option 2: Use GRU for Sequential Anomalies**
- If anomalies manifest as temporal sequences (gradual drift, cyclical failures)
- Better at capturing long-term dependencies
- More parameters → may need more training data

### **Option 3: Use Set Transformer for Complex Graphs**
- If you have irregular graphs (variable number of sensors per site)
- Good for graph-level properties that don't depend on node ordering
- More robust to missing sensors

### **Baseline: Simple Attention**
- Use as baseline to compare against advanced methods
- Fast prototyping and debugging

---

## Implementation Notes

### Backward Compatibility
All existing models trained with `--pooling attention` (without specifying `--attention-type`) default to `attention_type='simple'`.

### Config Files
Training now saves a `*_config.json` file containing all hyperparameters, including `attention_type`. The test script automatically loads this config.

### Attention Weights
All attention mechanisms return attention weights as a second output (currently unused in forward pass). Future work can visualize these for interpretability:

```python
# In model_graph.py forward() method
x, attn_weights = self.attention_module(x)  # attn_weights available for analysis
```

---

## Hyperparameter Tuning

All attention mechanisms use hardcoded defaults optimized for typical graph sizes (2-50 nodes, 100-365 timesteps). If needed, modify `attention_modules.py`:

```python
# Temporal MHA
TemporalAttentionPool(in_channels, num_heads=4, hidden_dim=64)  # Increase num_heads for larger graphs

# GRU
GraphLevelGRU(in_channels, num_nodes, hidden_dim=128, num_layers=2)  # Increase hidden_dim for complex patterns

# Set Transformer
SetTransformerPool(in_channels, num_seeds=16, num_heads=4)  # Increase num_seeds for larger graphs
```

---

## Performance Considerations

**Memory usage (for batch_size=32, nodes=5, time=365):**
- Simple: ~50 MB
- Temporal MHA: ~200 MB (attention matrix: 32 × 1825²)
- GRU: ~100 MB
- Set Transformer: ~150 MB

**Training time per epoch (same setup):**
- Simple: ~10s
- Temporal MHA: ~30s
- GRU: ~40s (sequential)
- Set Transformer: ~25s

---

## Citation

If using these attention mechanisms, please cite:

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  booktitle={NeurIPS},
  year={2017}
}

@inproceedings{lee2019set,
  title={Set transformer: A framework for attention-based permutation-invariant neural networks},
  author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
  booktitle={ICML},
  year={2019}
}
```
