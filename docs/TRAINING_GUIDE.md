# Training Guide for Anomaly Detection Model

## 🚀 Quick Start

### Basic Training

```bash
python train_anomaly_detection_model.py \
    --data_dir data/METR-LA \
    --save_dir checkpoints/experiment_1 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001
```

### With Spatial-Temporal Attention

```bash
python train_anomaly_detection_model.py \
    --aggregation_type spatial_temporal_attention \
    --num_heads 4 \
    --dropout 0.3 \
    --save_dir checkpoints/attention_4heads
```

### Resume Training

```bash
python train_anomaly_detection_model.py \
    --resume checkpoints/experiment_1/checkpoint_epoch_50.pt \
    --epochs 150
```

---

## 📊 Metrics Explained

### Primary Metrics

1. **AUC-ROC (Area Under ROC Curve)**
   - **What**: Measures model's ability to distinguish between classes
   - **Range**: 0.5 (random) to 1.0 (perfect)
   - **Used for**: Model selection (best model saved based on this)
   - **Interpretation**: >0.8 is good, >0.9 is excellent

2. **AP (Average Precision)**
   - **What**: Summarizes precision-recall curve
   - **Range**: 0.0 to 1.0
   - **Better for**: Imbalanced datasets
   - **Interpretation**: Similar to AUC-ROC

3. **Loss (Cross Entropy)**
   - **What**: Training objective being minimized
   - **Lower is better**
   - **Used for**: Monitoring training progress

### Secondary Metrics

4. **PR-AUC (Precision-Recall AUC)**
   - Computed during validation/testing
   - Useful for imbalanced anomaly detection

---

## 🎯 Command Line Arguments

### Data Parameters
```bash
--data_dir          # Directory with train/val/test.npz files
--save_dir          # Where to save checkpoints
```

### Model Architecture
```bash
--num_nodes         # Number of graph nodes (default: 207)
--in_dim            # Input feature dimension (default: 2)
--out_dim           # GWNet output dimension (default: 12)
--seq_length        # Temporal sequence length (default: 12)
--num_classes       # Classification classes (default: 2)
```

### Aggregation Settings
```bash
--aggregation_type  # simple | spatial_temporal_attention | hybrid
--num_heads         # Number of attention heads (default: 4)
```

### Training Hyperparameters
```bash
--batch_size        # Batch size (default: 32)
--epochs            # Number of epochs (default: 100)
--lr                # Learning rate (default: 0.001)
--weight_decay      # L2 regularization (default: 1e-4)
--dropout           # Dropout rate (default: 0.3)
--clip_grad         # Gradient clipping (default: 5.0)
```

### Optimization
```bash
--optimizer         # adam | adamw | sgd
--lr_scheduler      # plateau | step | cosine | none
--patience          # For early stopping (default: 10)
```

---

## 📁 Data Format

Your data should be organized as:

```
data/METR-LA/
├── train.npz
│   ├── x: (num_samples, in_dim, num_nodes, seq_length)
│   └── y: (num_samples,) - binary labels (0 or 1)
├── val.npz
│   ├── x: (num_samples, in_dim, num_nodes, seq_length)
│   └── y: (num_samples,)
└── test.npz
    ├── x: (num_samples, in_dim, num_nodes, seq_length)
    └── y: (num_samples,)
```

### Creating Data Files

```python
import numpy as np

# Example: Create training data
X_train = np.random.randn(1000, 2, 207, 12)  # 1000 samples
y_train = np.random.randint(0, 2, 1000)       # Binary labels

np.savez('data/METR-LA/train.npz', x=X_train, y=y_train)
```

---

## 🔄 Training Workflow

### 1. **Training Phase**
- Model trains on training set
- Loss, AUC, and AP computed per batch
- Gradients clipped to prevent explosion
- Prints progress every `--log_interval` batches

### 2. **Validation Phase**
- Runs after each epoch
- Computes validation metrics
- **Best model saved** based on validation AUC
- Learning rate adjusted if using scheduler

### 3. **Checkpointing**
- Regular checkpoints saved every `--save_interval` epochs
- Best model always saved when validation AUC improves
- Checkpoints include:
  - Model weights
  - Optimizer state
  - Current epoch
  - Train/val metrics

### 4. **Early Stopping**
- Triggered if no improvement for `patience * 2` epochs
- Prevents overfitting
- Can be disabled by setting very high patience

### 5. **Final Testing**
- Loads best model automatically
- Evaluates on test set
- Saves detailed results

---

## 📈 Monitoring Training

### Expected Output

```
Epoch 1 [0/1000 (0%)]     Loss: 0.693147
Epoch 1 [320/1000 (32%)]  Loss: 0.541234
...

VAL Results:
  Loss: 0.4523
  AUC-ROC: 0.7812
  AP: 0.7645
  PR-AUC: 0.7534

Epoch 1 Summary:
  Time: 45.23s
  Train Loss: 0.5123, AUC: 0.7234, AP: 0.7123
  Val   Loss: 0.4523, AUC: 0.7812, AP: 0.7645
  Current LR: 0.001000

Saved best model: checkpoints/experiment_1/best_model.pt (AUC: 0.7812)
```

### What to Watch For

✅ **Good Signs:**
- Training loss decreasing steadily
- Validation AUC increasing
- Gap between train/val metrics reasonable (<0.1)

⚠️ **Warning Signs:**
- Training loss not decreasing → lr too low or data issues
- Validation AUC not improving → need more regularization
- Large train/val gap → overfitting, increase dropout

🛑 **Stop Training If:**
- Validation AUC decreasing → overfitting
- Loss becomes NaN → lr too high or gradient explosion
- No improvement after many epochs → try different architecture

---

## 🎓 Experiment Tracking

### Baseline Experiment

```bash
# Simple mean pooling (baseline)
python train_anomaly_detection_model.py \
    --aggregation_type simple \
    --save_dir checkpoints/baseline_mean
```

### Your Main Method

```bash
# Spatial-temporal attention (your contribution)
python train_anomaly_detection_model.py \
    --aggregation_type spatial_temporal_attention \
    --num_heads 4 \
    --save_dir checkpoints/attention_4heads
```

### Ablation Studies

```bash
# Try different number of heads
for heads in 2 4 8; do
    python train_anomaly_detection_model.py \
        --aggregation_type spatial_temporal_attention \
        --num_heads $heads \
        --save_dir checkpoints/attention_${heads}heads
done
```

---

## 💾 Checkpoint Management

### Load Best Model

```python
import torch
from anomaly_detection import AnomalyDetectionModel

# Create model
model = AnomalyDetectionModel(...)

# Load best checkpoint
checkpoint = torch.load('checkpoints/experiment_1/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Check metrics
print(checkpoint['metrics'])
```

### Resume Training

```bash
# Continue from epoch 50
python train_anomaly_detection_model.py \
    --resume checkpoints/experiment_1/checkpoint_epoch_50.pt \
    --epochs 150  # Will train until epoch 150
```

---

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
--batch_size 16

# Or reduce model size
--out_dim 8
--num_heads 2
```

### Poor Performance

```bash
# Try different learning rate
--lr 0.0001  # Lower
--lr 0.01    # Higher

# Add more regularization
--dropout 0.5
--weight_decay 1e-3

# Try different optimizer
--optimizer adamw
```

### Training Too Slow

```bash
# Increase batch size (if memory allows)
--batch_size 64

# Reduce sequence length
--seq_length 6

# Use fewer attention heads
--num_heads 2
```

---

## 📊 Results Comparison Table

After running experiments, compare like this:

| Method | AUC-ROC | AP | Loss | Params | Time/Epoch |
|--------|---------|-----|------|--------|------------|
| Baseline (mean) | 0.72 | 0.68 | 0.45 | 150K | 30s |
| Attention (4-head) | 0.82 | 0.79 | 0.38 | 180K | 45s |
| Attention (8-head) | 0.83 | 0.80 | 0.37 | 210K | 52s |
| Hybrid | 0.84 | 0.81 | 0.36 | 200K | 48s |

**For your paper:** Show that attention improves AUC by ~10% with reasonable overhead!

---

## 🚀 Next Steps

After training baseline + attention models:

1. **Phase 2**: Add evidential uncertainty
2. **Phase 3**: Add counterfactual explanations
3. **Visualize**: Use attention weights to show which nodes/times matter
4. **Paper**: Write up results showing improvement

Good luck! 🎯
