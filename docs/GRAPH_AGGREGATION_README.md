# Graph Aggregation Modules - Quick Reference

## File Organization

The graph aggregation functionality has been organized as follows:

### **`graph_aggregation.py`** - Graph Aggregation Methods
Contains graph-level aggregation mechanisms:
- `SimpleGraphAggregation` - Baseline pooling methods
- `SpatialTemporalAttention` - Parallel multi-head attention (your main method)
- `AdditiveAttentionPooling` - Bahdanau-style attention
- `HybridGraphAttention` - Combination approach

### **`anomaly_detection.py`** - Complete Detection Model
Contains the complete model:
- `AnomalyDetectionModel` - Wrapper combining GWNet + aggregation + classifier

### **`model.py`** - Core GWNet
Contains the original GWNet architecture for feature extraction.

### **`attention_utils.py`** - Visualization & Analysis
Utility functions for:
- Visualizing attention weights
- Analyzing attention patterns
- Comparing different methods

### **`example_attention_usage.py`** - Examples
Complete usage examples demonstrating how to use all attention mechanisms.

## Quick Import Guide

```python
# Import complete anomaly detection model
from anomaly_detection import AnomalyDetectionModel

# Import graph aggregation modules (if using separately)
from graph_aggregation import (
    SimpleGraphAggregation,
    SpatialTemporalAttention,
    AdditiveAttentionPooling,
    HybridGraphAttention
)

# Import visualization utilities
from attention_utils import (
    visualize_attention_weights,
    visualize_pooling_attention,
    compare_attention_methods
)

# Import GWNet (if needed separately)
from model import gwnet
```

## Usage Example

```python
from anomaly_detection import AnomalyDetectionModel

# Create model
model = AnomalyDetectionModel(
    gwnet_params={
        'device': device,
        'num_nodes': 207,
        'in_dim': 2,
        'out_dim': 12,
        # ... other gwnet params
    },
    aggregation_type='spatial_temporal_attention',
    aggregation_params={
        'out_dim': 12,
        'num_nodes': 207,
        'seq_length': 12,
        'num_heads': 4,
        'dropout': 0.3
    },
    num_classes=2
)

# Forward pass
output = model(x, return_attention_weights=True)
detection_score = output['detection_score']
spatial_attention = output['spatial_attention']
temporal_attention = output['temporal_attention']
```

## Running Examples

```bash
# Run complete example
python example_attention_usage.py
```

This will:
1. Create all model variants
2. Compare model sizes
3. Run forward passes
4. Visualize attention weights
5. Save figures

## Documentation

See `ATTENTION_TYPES_EXPLAINED.md` for comprehensive documentation on:
- Attention mechanism theory
- Mathematical formulations
- Interpretation guidelines
- Publication strategies
