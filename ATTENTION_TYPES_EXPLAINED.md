# Attention Mechanisms for Graph Anomaly Detection - Comprehensive Guide

## 📚 Overview

This document explains the attention mechanisms implemented for your weakly supervised spatiotemporal graph anomaly detection task.

---

## 🎯 Three Types of Attention Implemented

### **1. Multi-Head Self-Attention (Spatial)**

**Purpose**: Identify which nodes are related during anomalous events

**How it works**:
```
Input: Features for all nodes at all timesteps
Query: "For node i, which other nodes should I pay attention to?"
Output: Weighted combination of all node features

Mathematical form:
Attention(Q, K, V) = softmax(QK^T / √d) V

Where:
- Q (Query): What node i is looking for
- K (Key): What each node offers
- V (Value): The actual information from each node
```

**Why Multi-Head?**
- Each head learns different relationships
- Head 1 might learn: "nearby nodes in network topology"
- Head 2 might learn: "nodes with similar traffic patterns"
- Head 3 might learn: "nodes that anomaly together"
- Head 4 might learn: "upstream/downstream dependencies"

**Example Interpretation**:
```python
# If spatial_attention[0, node_45, node_67] = 0.8
# This means: "Node 45 pays strong attention to node 67"
# Implication: "When analyzing node 45, node 67's state is important"
```

---

### **2. Multi-Head Self-Attention (Temporal)**

**Purpose**: Identify which time periods are related

**How it works**:
```
Input: Features for all timesteps across all nodes
Query: "For timestep t, which other timesteps should I consider?"
Output: Weighted combination of all temporal features

Same formula as spatial, but over time dimension instead of nodes
```

**Why important for anomaly detection?**
- Anomalies often have temporal patterns
- Example: Morning rush hour anomaly (7-9am) might be related to evening rush (5-7pm)
- Captures lag effects: "Anomaly at t=8 depends on what happened at t=5"

**Example Interpretation**:
```python
# If temporal_attention[0, timestep_8, timestep_5] = 0.7
# This means: "Timestep 8 pays attention to timestep 5"
# Implication: "What happened 3 hours ago affects current state"
```

---

### **3. Additive Attention (Bahdanau Attention)**

**Purpose**: Learn importance weights for final pooling

**How it works**:
```
Mathematical form:
score_i = v^T * tanh(W * h_i)
attention_i = exp(score_i) / Σ exp(score_j)

Where:
- h_i: feature vector for node/timestep i
- W, v: learnable parameters
- attention_i: importance weight (sums to 1)
```

**Difference from self-attention**:
- Self-attention: each item attends to all others (pairwise)
- Additive attention: each item gets a single importance score (global)

**Why use this?**
- More interpretable: single score per node/timestep
- Computationally cheaper: O(n) vs O(n²)
- Good for final aggregation step

**Example Interpretation**:
```python
# If spatial_pool_weights = [0.001, 0.003, 0.15, 0.002, ...]
# This means: "Node 2 (weight=0.15) is much more important than others"
# Implication: "Focus your investigation on node 2"
```

---

## 🔄 How They Work Together

### **Architecture Flow**:

```
Input: (batch, in_dim, num_nodes, seq_length)
          ↓
    [GWNet Feature Extraction]
          ↓
Features: (batch, out_dim, num_nodes, seq_length)
          ↓
          ├─────────────────────┬─────────────────────┐
          ↓                     ↓                     ↓
   [Spatial Path]        [Temporal Path]      [Optional: Cross-attention]
          ↓                     ↓
   Reshape to:           Reshape to:
   (batch, nodes,        (batch, time,
    out_dim*time)         out_dim*nodes)
          ↓                     ↓
   [Multi-Head           [Multi-Head
    Self-Attention]       Self-Attention]
          ↓                     ↓
   Spatial_repr          Temporal_repr
   (batch, out_dim*time) (batch, out_dim*nodes)
          ↓                     ↓
          └─────────────────────┴─────────────────────┐
                                ↓
                         [Concatenate]
                                ↓
                         [Fusion Network]
                                ↓
                    Graph Representation
                    (batch, out_dim)
                                ↓
                         [Classifier]
                                ↓
                    Detection Score (batch, 1)
```

---

## 📊 Comparison: When to Use Which?

| Attention Type | Complexity | Interpretability | Use Case |
|---------------|------------|------------------|----------|
| **Multi-Head Self (Spatial)** | O(n²) | Medium | Capture node relationships |
| **Multi-Head Self (Temporal)** | O(t²) | Medium | Capture temporal dependencies |
| **Additive Pooling** | O(n) | High | Final aggregation, get importance scores |
| **Hybrid (Both)** | O(n² + t²) | High | Best performance + interpretability |

Where:
- n = number of nodes (207 for METR-LA)
- t = sequence length (12 in your case)

---

## 🎓 For Your Paper

### **Baseline Methods to Compare Against**:

1. **Simple Pooling** (weakest baseline)
   - Just mean/max over nodes and time
   - No learned aggregation
   - Expected AUROC: ~0.70-0.75

2. **Separate Spatial + Temporal Pooling** (medium baseline)
   - Pool separately then concatenate
   - No attention, just learned fusion
   - Expected AUROC: ~0.75-0.78

3. **Your Method: Parallel Spatial-Temporal Attention**
   - Multi-head attention for both dimensions
   - Parallel processing (no information bottleneck)
   - Expected AUROC: ~0.80-0.85

4. **Ablation: Only Spatial Attention**
   - Shows importance of temporal component
   - Expected AUROC: ~0.77-0.80

5. **Ablation: Only Temporal Attention**
   - Shows importance of spatial component
   - Expected AUROC: ~0.76-0.79

---

## 💡 Key Insights for Anomaly Detection

### **Why Parallel Attention Matters**:

```
Sequential Approach (problematic):
Nodes → [Attention] → Single vector → [Attention over time] → Output
         ↑                                                      ↑
    Information loss                               More information loss

Parallel Approach (your method):
Nodes ──┐
        ├── [Spatial Attention] ──┐
        └── [Temporal Attention] ──┤── [Fusion] ── Output
Time ───┘                          ↑
                            Preserves both views!
```

### **Interpretability for Operators**:

When model detects anomaly, you can show:
1. **Spatial attention**: "These 5 nodes are critical"
2. **Temporal attention**: "Focus on these 3 time periods"
3. **Combined**: "Node 45 at timestep 8 is the key anomalous event"

This is crucial for:
- Regulatory compliance
- Operator trust
- Actionable insights

---

## 🚀 Next Steps

After implementing Phase 1 (Attention), you'll add:

### **Phase 2: Evidential Uncertainty**
- Extends attention with uncertainty quantification
- Helps identify: "I'm 90% sure node 45 is anomalous, but only 40% sure about node 67"
- Critical for weak supervision (yearly labels → monthly localization)

### **Phase 3: Counterfactual Explanations**
- Extends attention + uncertainty with intervention analysis
- Answers: "IF we fixed node 45, anomaly probability drops from 0.9 to 0.2"
- Provides actionable recommendations

---

## 📖 Mathematical Details

### **Multi-Head Attention Formula**:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O

where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Parameters per head:
- W^Q_i: (d_model, d_k) - Query projection
- W^K_i: (d_model, d_k) - Key projection  
- W^V_i: (d_model, d_v) - Value projection
- W^O: (h*d_v, d_model) - Output projection

For your model:
- d_model = out_dim * seq_length (for spatial)
- d_model = out_dim * num_nodes (for temporal)
- h = 4 (number of heads)
- d_k = d_v = d_model / h
```

### **Additive Attention Formula**:

```
score_i = v^T * tanh(W * h_i + b)
α_i = exp(score_i) / Σ_j exp(score_j)
output = Σ_i α_i * h_i

Parameters:
- W: (hidden_dim, feature_dim)
- v: (1, hidden_dim)
- b: (hidden_dim,) - bias

Computational cost: O(n * feature_dim * hidden_dim)
```

---

## 🎨 Visualization Examples

### **What Good Attention Looks Like**:

**Spatial Attention (Anomaly Sample)**:
```
        Node 0  Node 1  Node 2  ...  Node 45  ...  Node 207
Node 0   0.05    0.03    0.02  ...   0.01   ...    0.01
Node 1   0.02    0.06    0.04  ...   0.02   ...    0.01
Node 45  0.01    0.02    0.01  ...   0.42   ...    0.03  ← High self-attention!
...
Node 207 0.01    0.01    0.01  ...   0.08   ...    0.05

Interpretation: Node 45 has high self-attention (0.42) and 
other nodes attend to it (column 45 has higher values)
→ Node 45 is central to the anomaly!
```

**Temporal Attention (Anomaly Sample)**:
```
        t=0   t=1   t=2  ...  t=8   ...  t=12
t=0     0.10  0.08  0.05 ...  0.01  ...  0.01
t=1     0.08  0.12  0.09 ...  0.02  ...  0.01
t=8     0.01  0.02  0.03 ...  0.35  ...  0.15  ← Peak attention!
...
t=12    0.01  0.01  0.02 ...  0.12  ...  0.08

Interpretation: t=8 has high self-attention (0.35) and 
t=12 still attends to t=8 (0.12)
→ Anomaly occurred at t=8 and has lasting effects!
```

---

## ✅ Implementation Checklist

- [x] Multi-head self-attention (spatial)
- [x] Multi-head self-attention (temporal)
- [x] Additive attention pooling
- [x] Hybrid architecture
- [x] Simple pooling baselines
- [x] Anomaly detection model wrapper
- [x] Attention visualization utilities
- [x] Example usage scripts

---

## 🔧 How to Use

See `example_attention_usage.py` for complete examples.

**Quick start**:

```python
from model import AnomalyDetectionModel

# Create model
model = AnomalyDetectionModel(
    gwnet_params={...},
    aggregation_type='spatial_temporal_attention',
    aggregation_params={
        'out_dim': 12,
        'num_nodes': 207,
        'seq_length': 12,
        'num_heads': 4
    }
)

# Forward pass
output = model(x, return_attention_weights=True)

# Get results
detection_prob = output['detection_score']  # (batch, 1)
spatial_attn = output['spatial_attention']   # (batch, nodes, nodes)
temporal_attn = output['temporal_attention'] # (batch, time, time)
```

---

**Questions? Check `example_attention_usage.py` or `attention_utils.py` for more examples!**
