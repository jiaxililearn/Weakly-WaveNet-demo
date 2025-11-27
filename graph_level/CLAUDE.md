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

### Training
```bash
# Full training with all components
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj

# Key arguments:
# --gcn_bool: Enable graph convolution layers
# --addaptadj: Enable adaptive adjacency learning (learns graph structure)
# --randomadj: Random initialize adaptive adj (vs. SVD initialization from predefined adj)
# --aptonly: Use only adaptive adjacency (no predefined graph)
# --adjtype: Type of predefined adjacency (doubletransition, transition, symnadj, normlap, scalap, identity)
# --device: Default 'cuda:3'
# --num_nodes: Default 207 (METR-LA)
# --epochs: Default 100
# --save: Model checkpoint path, default './garage/metr'
```

### Testing
```bash
# Evaluate saved model checkpoint
python test.py --checkpoint ./garage/metr_exp1_best_2.34.pth --gcn_bool --adjtype doubletransition --addaptadj

# Generates:
# - Per-horizon (1-12 steps) metrics printed to console
# - emb.pdf: Heatmap of learned adaptive adjacency matrix
# - wave.csv: Sample predictions for node 99 at horizons 3 and 12
```

## Architecture

### Model Components ([model.py](model.py))

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

## Notes

- Default configuration assumes 2-channel input (value + time_in_day from `generate_training_data.py`)
- Model saves checkpoints as: `{save_path}_epoch_{i}_{val_loss}.pth`
- Adjacency matrix file must contain: (sensor_ids, sensor_id_to_ind, adj_mx) tuple
- Data splits: 70% train, 10% val, 20% test (chronological, not random)
