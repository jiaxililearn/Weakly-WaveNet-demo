# Graph-Level Prediction for Multi-Site Time Series

Graph WaveNet adapted for **multi-site anomaly detection**: predict site-level labels from daily sensor time series.

**Use Case**: 200 reef sites × 2 sensors × 365 days → anomaly classification (2-3 classes)

---

## Quick Start

### 1. Generate Synthetic Test Data

```bash
python prepare_multisite_data.py \
    --synthetic \
    --output_dir ../data/test \
    --num_sites 200 \
    --num_sensors_per_site 2 \
    --num_years 3 \
    --seq_length 365 \
    --task classification \
    --num_classes 3
```

Creates 600 graphs (200 sites × 3 years) for training.

### 2. Train Model

```bash
python train_graph.py \
    --data ../data/test \
    --num-nodes 2 \
    --seq-length 365 \
    --task classification \
    --num-classes 3 \
    --aptonly \
    --addaptadj \
    --epochs 100 \
    --batch-size 32
```

### 3. Test Model

```bash
python test_graph.py \
    --checkpoint ../checkpoints/graph_model_exp1_best.pth \
    --data ../data/test \
    --num-nodes 2 \
    --seq-length 365 \
    --task classification \
    --num-classes 3 \
    --aptonly
```

---

## Using Your Own Data

### Data Format Required

**Option 1: CSV files (one per site)**

Structure:
```
sites_data/
├── site_0.csv
├── site_1.csv
...
└── site_199.csv

labels.csv
```

**site_X.csv format:**
```csv
date,sensor_0,sensor_1
2021-01-01,45.2,52.1
2021-01-02,47.1,53.2
...
```

**labels.csv format:**
```csv
site_id,year,label
0,0,0
0,1,2
0,2,1
1,0,0
1,1,1
...
```

Labels should be integers: 0, 1, 2 (for 3-class classification)

### Prepare Your Data

```bash
python prepare_multisite_data.py \
    --sites_dir sites_data/ \
    --labels_file labels.csv \
    --output_dir ../data/my_sites \
    --num_sites 200 \
    --num_sensors_per_site 2 \
    --seq_length 365
```

### Train

```bash
python train_graph.py \
    --data ../data/my_sites \
    --num-nodes 2 \
    --seq-length 365 \
    --task classification \
    --num-classes 3 \
    --aptonly \
    --addaptadj \
    --epochs 100
```

---

## Key Parameters

### Required
- `--num-nodes` - Sensors per site (e.g., 2)
- `--seq-length` - Days in sequence (e.g., 365)
- `--data` - Path to prepared data
- `--task` - `classification` or `regression`
- `--num-classes` - Number of classes (for classification)

### Recommended
- `--aptonly` - Learn graph structure from data (no predefined adjacency)
- `--addaptadj` - Enable adaptive adjacency learning
- `--pooling` - Aggregation method: `mean`, `max`, `attention` (default: `mean`)

### Training
- `--batch-size` - Default: 32
- `--learning-rate` - Default: 0.001
- `--epochs` - Default: 100
- `--nhid` - Hidden dimension: 32, 64, 128

---

## Architecture

**Input**: `(batch, features, nodes, time)` = `(batch, 2, 2, 365)`
- 2 features: sensor value + time-of-day
- 2 nodes: 2 sensors per site
- 365 timesteps: daily data for 1 year

**Processing**:
1. Temporal convolutions (WaveNet with dilations: 1, 2, 4, 8...)
2. Graph convolutions (spatial aggregation between sensors)
3. Adaptive adjacency learning (learns sensor relationships)

**Output**: `(batch, num_classes)` = `(batch, 3)`
- Global pooling aggregates across nodes and time
- Final FC layers predict site-level class

**Key Insight**: Each site is treated as an independent 2-node graph. Model learns patterns that generalize across all sites.

---

## Files

- `model_graph.py` - Model with global pooling
- `prepare_multisite_data.py` - Data preparation
- `train_graph.py` - Training script
- `test_graph.py` - Testing/evaluation
- `README.md` - This file

---

## Examples

### 2-Class Classification (Normal/Anomaly)

```bash
# Data prep
python prepare_multisite_data.py \
    --synthetic \
    --output_dir ../data/binary \
    --num_sites 200 \
    --num_sensors_per_site 2 \
    --num_years 3 \
    --task classification \
    --num_classes 2

# Train
python train_graph.py \
    --data ../data/binary \
    --num-nodes 2 \
    --task classification \
    --num-classes 2 \
    --aptonly
```

### Regression Task

```bash
# Data prep
python prepare_multisite_data.py \
    --synthetic \
    --output_dir ../data/regression \
    --num_sites 200 \
    --num_sensors_per_site 2 \
    --num_years 3 \
    --task regression

# Train (no --num-classes needed)
python train_graph.py \
    --data ../data/regression \
    --num-nodes 2 \
    --task regression \
    --aptonly
```

### Attention Pooling

```bash
python train_graph.py \
    --data ../data/my_sites \
    --num-nodes 2 \
    --task classification \
    --num-classes 3 \
    --pooling attention \
    --aptonly
```

Attention pooling learns which temporal periods are most important for classification.

---

## Troubleshooting

**Out of memory?**
```bash
python train_graph.py ... --batch-size 8 --nhid 16
```

**Model not learning?**
```bash
python train_graph.py ... --learning-rate 0.0001 --epochs 200
```

**Check data format:**
```python
import numpy as np
data = np.load('../data/my_sites/train.npz')
print(f"x: {data['x'].shape}")  # Should be (samples, 365, 2, 2)
print(f"y: {data['y'].shape}")  # Should be (samples,)
print(f"Labels: {np.unique(data['y'])}")  # Should be [0, 1, 2]
```

**NumPy version error:**
```bash
pip install "numpy<2"
```

---

## Design Rationale

**Why treat sites independently?**

Each reef site has its own baseline ecosystem. The model learns:
- "What is normal for THIS site?"
- "Is THIS site deviating from its own patterns?"

NOT: "Is this site different from other sites?" (which would require connecting all sites in one graph)

**Benefits:**
- ✅ 600 training samples (200 sites × 3 years)
- ✅ Each site's unique baseline is learned
- ✅ Robust to site-specific conditions
- ✅ Predictions are site-specific and actionable
- ✅ Biologically meaningful for independent ecosystems

---

## Citation

Based on Graph WaveNet:
```bibtex
@inproceedings{wu2019graph,
  title={Graph wavenet for deep spatial-temporal graph modeling},
  author={Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle={IJCAI},
  pages={1907--1913},
  year={2019}
}
```
