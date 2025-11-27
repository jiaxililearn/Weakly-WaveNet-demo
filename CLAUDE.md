# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph WaveNet implementation for spatial-temporal graph modeling on traffic forecasting tasks (IJCAI 2019). The model combines WaveNet-style dilated temporal convolutions with graph convolutions and adaptive adjacency learning.

## Data Preparation

Download METR-LA/PEMS-BAY datasets from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g).

```bash
# Create directories
mkdir -p data/{METR-LA,PEMS-BAY}

# Process raw data (creates train.npz, val.npz, test.npz)
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

Data format: `.h5` files (raw) → `.npz` files (processed train/val/test splits with 70/10/20 ratio)

## Commands

### Training (Standard Time Series Forecasting)

**Option 1: train.py (argparse-based, original)**
```bash
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj
```

**Option 2: run.py (click-based, flexible)**
```bash
# With predefined adjacency
python run.py --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --num-nodes 207 \
              --gcn-bool --addaptadj --randomadj --adjtype doubletransition

# Adaptive adjacency only (no predefined graph)
python run.py --data data/METR-LA --num-nodes 207 --addaptadj --aptonly
```

**Key arguments**:
- `--gcn_bool/--gcn-bool`: Enable graph convolution layers
- `--addaptadj`: Enable adaptive adjacency learning (learns graph structure)
- `--randomadj`: Random initialize adaptive adj (vs. SVD initialization from predefined adj)
- `--aptonly`: Use only adaptive adjacency (no predefined graph)
- `--adjtype`: Type of predefined adjacency (doubletransition, transition, symnadj, normlap, scalap, identity)
- `--device`: Default 'cuda:3' (train.py) or 'cuda:0' (run.py)
- `--num_nodes/--num-nodes`: Default 207 (METR-LA)
- `--epochs`: Default 100
- `--save`: Model checkpoint path, default './garage/metr' (train.py) or './checkpoints/model' (run.py)

### Testing (Standard)
```bash
python test.py --checkpoint ./garage/metr_exp1_best_2.34.pth --gcn_bool --adjtype doubletransition --addaptadj

# Generates:
# - Per-horizon (1-12 steps) metrics printed to console
# - emb.pdf: Heatmap of learned adaptive adjacency matrix
# - wave.csv: Sample predictions for node 99 at horizons 3 and 12
```

### Graph-Level Tasks (Multi-Site Classification/Regression)

Located in [graph_level/](graph_level/) directory - extends Graph WaveNet for site-level predictions.

**Generate synthetic data**:
```bash
cd graph_level
python prepare_multisite_data.py --synthetic --output_dir ../data/test \
       --num_sites 200 --num_sensors_per_site 2 --num_years 3 \
       --seq_length 365 --task classification --num_classes 3
```

**Train graph-level model**:
```bash
python train_graph.py --data ../data/test --num-nodes 2 --seq-length 365 \
       --task classification --num-classes 3 --aptonly --addaptadj
```

**Test graph-level model**:
```bash
python test_graph.py --checkpoint ../checkpoints/graph_model_exp1_best.pth \
       --data ../data/test --num-nodes 2 --seq-length 365 \
       --task classification --num-classes 3 --aptonly
```

## Architecture

### Standard Time Series Forecasting ([model.py](model.py))

**`gwnet` (main model)**
- **Input**: `(batch, in_dim, num_nodes, seq_length)` - default in_dim=2 (traffic value + time_in_day)
- **Output**: `(batch, out_dim, num_nodes, 1)` - default out_dim=12 (future 12 steps)
- **Structure**:
  - Start conv: Maps input to residual_channels (default 32)
  - 4 blocks × 2 layers with dilated causal convolutions (dilation: 1,2,4,8,...)
  - Each layer: gated activation (tanh × sigmoid) + skip connection + GCN + residual
  - End convs: Aggregate skip connections → final prediction

**`gcn` (graph convolution)**
- Implements K-hop diffusion: stacks powers of adjacency matrix up to order K (default 2)
- Supports multiple support matrices (e.g., forward + backward transition)
- Applied after gated temporal convolution in each layer

**Adaptive adjacency**
- Learned via node embeddings: `nodevec1` (N×10), `nodevec2` (10×N)
- Computed as: `softmax(ReLU(nodevec1 @ nodevec2))`
- Initialization: random (--randomadj) or SVD of predefined adj
- Combined with predefined adjacency matrices during forward pass

### Graph-Level Classification/Regression ([graph_level/model_graph.py](graph_level/model_graph.py))

Extends `gwnet` for site-level predictions (vs. node-level forecasting):

**Key differences**:
- **Input**: `(batch, in_dim, num_nodes, seq_length)` - e.g., (32, 2, 2, 365) for 2 sensors over 365 days
- **Output**: `(batch, num_classes)` or `(batch, 1)` for regression - site-level label
- **Global pooling**: Aggregates temporal & spatial features (mean/max/attention)
- **Use case**: Each graph = 1 site's sensor data → predict site-level anomaly class

**Architecture flow**:
1. Same WaveNet + GCN layers as standard model
2. Global pooling: `(batch, channels, nodes, time)` → `(batch, channels)`
3. FC layers: `channels → hidden → num_classes`

**Design rationale**: Treats each site independently (not connected across sites), learns within-site patterns to detect anomalies

### Training Flow ([train.py](train.py), [engine.py](engine.py))

1. Load adjacency matrix (`.pkl` file) and apply transformation (doubletransition creates forward + backward)
2. Load train/val/test data from `.npz` files
3. Standardize using z-score (fit on training data)
4. Training loop:
   - Forward: pad input by 1 timestep, transpose to `(batch, channels, nodes, time)`
   - Loss: masked MAE (handles missing values with null_val=0.0)
   - Optimizer: Adam with gradient clipping (max_norm=5)
   - Save checkpoint every epoch
5. Testing: Load best checkpoint (lowest val loss), evaluate on 12 horizons

### Utilities ([util.py](util.py))

**Adjacency transformations** (`load_adj`):
- `doubletransition`: [D^-1·A, D^-1·A^T] - row-normalized forward + backward
- `transition`: [D^-1·A] - random walk
- `symnadj`: [D^-1/2·A·D^-1/2] - symmetric normalized
- `normlap`: Normalized Laplacian
- `scalap`: Scaled Laplacian
- `identity`: Identity matrix

**Data loader**: Custom iterator with shuffling, padding to batch size

**Metrics**: Masked MAE/MAPE/RMSE to handle missing sensor readings

## Key Design Patterns

1. **Temporal receptive field**: Grows exponentially with dilations (final: 1+2+4+8+... = 2^layers - 1 per block)
2. **Multi-scale temporal modeling**: Skip connections aggregate features from all dilation rates
3. **Spatial-temporal decoupling**: Temporal conv → GCN → residual (applied sequentially, not jointly)
4. **Adaptive + predefined graphs**: Combines domain knowledge (road network) with learned patterns
5. **Z-score normalization**: Applied only to first channel (traffic value), not time features

## Project Structure

**Main scripts**:
- [train.py](train.py) - Original training script (argparse, hardcoded paths)
- [run.py](run.py) - Flexible training script (click CLI, custom datasets)
- [test.py](test.py) - Evaluation script for saved checkpoints
- [model.py](model.py) - Core Graph WaveNet model
- [engine.py](engine.py) - Training/evaluation engine (optimizer, forward/backward pass)
- [util.py](util.py) - Data loading, adjacency transformations, metrics
- [generate_training_data.py](generate_training_data.py) - Converts `.h5` → `.npz`

**Graph-level extension** ([graph_level/](graph_level/)):
- [model_graph.py](graph_level/model_graph.py) - Graph-level model with pooling
- [train_graph.py](graph_level/train_graph.py) - Training for classification/regression
- [test_graph.py](graph_level/test_graph.py) - Testing for graph-level tasks
- [prepare_multisite_data.py](graph_level/prepare_multisite_data.py) - Multi-site data preparation

## Notes

- Default configuration assumes 2-channel input (value + time_in_day from `generate_training_data.py`)
- Model saves checkpoints as: `{save_path}_epoch_{i}_{val_loss}.pth`
- Adjacency matrix file must contain: (sensor_ids, sensor_id_to_ind, adj_mx) tuple
- Data splits: 70% train, 10% val, 20% test (chronological, not random)
- `train.py` vs `run.py`: Use `run.py` for custom datasets (better CLI); `train.py` for reproducing paper results
- Graph-level models treat each site as an independent graph (not interconnected)
