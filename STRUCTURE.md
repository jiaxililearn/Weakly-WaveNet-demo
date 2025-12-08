# Project Organization

```
Weakly-WaveNet-demo/
├── models/              # Model architectures
│   ├── model.py                    # Original GWNet model
│   ├── graph_aggregation.py        # Graph-level aggregation modules
│   ├── attention_utils.py          # Attention mechanism utilities
│   └── anomaly_detection.py        # Anomaly detection wrapper
│
├── training/            # Training scripts
│   ├── train_anomaly_detection_model.py  # Main training script
│   ├── train.py                          # Original GWNet training
│   └── calibrate_model.py                # Probability calibration
│
├── scripts/             # Utility scripts
│   ├── generate_training_data.py   # Data preparation
│   └── example_attention_usage.py  # Usage examples
│
├── utils/               # Helper functions
│   ├── util.py         # Data loading utilities
│   └── engine.py       # Training engine (original)
│
├── diagnostics/         # Analysis and debugging tools
│   ├── data_diagnostics.py        # Data quality checks
│   ├── improvement_strategies.py  # Model improvement helpers
│   └── model_curves.png          # Diagnostic plots
│
├── docs/                # Documentation
│   ├── README.md
│   ├── TRAINING_GUIDE.md
│   ├── ATTENTION_TYPES_EXPLAINED.md
│   ├── GRAPH_AGGREGATION_README.md
│   └── PUBLIC_DATASETS.md
│
├── data/                # Datasets
│   ├── METR-LA/
│   ├── PEMS-BAY/
│   └── temp_dhw/
│
├── checkpoints/         # Saved models
│   └── anomaly_detection/
│
├── fig/                 # Figures and plots
│
├── test.py             # Main testing script
├── gwnet_demo.ipynb    # Demo notebooks
└── gwnet_graph_demo.ipynb
```

## Quick Start

### Training Anomaly Detection Model
```bash
python training/train_anomaly_detection_model.py --data data/temp_dhw/ --epochs 100
```

### Data Diagnostics
```bash
python diagnostics/data_diagnostics.py
```

### Calibrate Model Probabilities
```bash
python training/calibrate_model.py
```

## Import Structure

After reorganization, update imports in your code:

```python
# Models
from models.anomaly_detection import AnomalyDetectionModel
from models.graph_aggregation import SpatialTemporalAttention, HybridGraphAttention
from models.model import gwnet

# Utils
from utils.util import load_dataset
from utils.engine import trainer

# Diagnostics
from diagnostics.data_diagnostics import diagnose_data_quality
```
