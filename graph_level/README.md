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

### Understanding the Data Model

The model expects:
- **Multiple sites** (e.g., 200 reef monitoring locations)
- **Each site** has the same number of sensors (e.g., 2 sensors: temperature & pH)
- **Each site** has continuous time series data (e.g., daily readings for multiple years)
- **Each site-year** gets one label (classification: 0/1/2, or regression: continuous value)

**Example**: 200 sites × 3 years × 365 days = 600 training graphs

### Data Format Required

**Option 1: CSV Files (Recommended for Real Data)**

Create two components:

#### 1. Site Data Files

One CSV per site in a directory:

```
sites_data/
├── site_0.csv
├── site_1.csv
├── site_2.csv
...
└── site_199.csv
```

**Each site_X.csv must have**:
- Column 1: `date` (required, used as index)
- Columns 2+: One column per sensor (e.g., `sensor_0`, `sensor_1`)
- Rows: Daily measurements in chronological order

**Example site_0.csv**:
```csv
date,sensor_0,sensor_1
2021-01-01,45.2,52.1
2021-01-02,47.1,53.2
2021-01-03,46.8,51.9
...
2023-12-31,48.3,54.0
```

**Requirements**:
- Must have exactly `num_years * seq_length` rows (e.g., 3 years × 365 days = 1095 rows)
- All sites must have the same number of sensors
- Missing dates are NOT allowed (every day must be present)
- Sensor values should be numeric (floats)

#### 2. Labels File

Single CSV with site-year labels:

**labels.csv**:
```csv
site_id,year,label
0,0,0
0,1,2
0,2,1
1,0,0
1,1,1
1,2,2
2,0,1
...
199,2,0
```

**Columns**:
- `site_id`: Integer 0 to (num_sites - 1)
- `year`: Integer 0 to (num_years - 1)
- `label`:
  - **Classification**: Integer class (e.g., 0, 1, 2 for 3 classes)
  - **Regression**: Float value (e.g., 98.5, 102.3)

**Requirements**:
- Must have `num_sites × num_years` rows (e.g., 200 × 3 = 600 labels)
- Each site must have labels for years 0, 1, 2, ... in order

---

### Preparing Your Data

#### Classification Task (3 classes)

```bash
python prepare_multisite_data.py \
    --sites_dir sites_data/ \
    --labels_file labels.csv \
    --output_dir ../data/my_sites \
    --num_sites 200 \
    --num_sensors_per_site 2 \
    --seq_length 365 \
    --task classification \
    --num_classes 3
```

#### Regression Task

```bash
python prepare_multisite_data.py \
    --sites_dir sites_data/ \
    --labels_file labels.csv \
    --output_dir ../data/my_sites \
    --num_sites 200 \
    --num_sensors_per_site 2 \
    --seq_length 365 \
    --task regression
```

**Important Parameters**:
- `--num_sites`: Total number of sites in your dataset
- `--num_sensors_per_site`: Number of sensor columns in each site CSV (excluding date)
- `--seq_length`: Days per year (365 for daily data, 52 for weekly, etc.)
- `--task`: `classification` or `regression`
- `--num_classes`: Required for classification (e.g., 3 for "healthy/warning/critical")

**Output**: Creates `train.npz`, `val.npz`, `test.npz`, `metadata.json` in the output directory

---

### Option 2: NumPy Arrays (For Programmatic Data)

If you already have data in memory (e.g., from a database or API), you can use the array-based preparation:

```python
import numpy as np
from prepare_multisite_data import prepare_multisite_data_from_array, split_and_save

# Your data shapes:
# sensor_data: (num_sites, total_days, num_sensors_per_site)
#   Example: (200, 1095, 2) for 200 sites, 3 years × 365 days, 2 sensors
# labels: (num_sites, num_years)
#   Example: (200, 3) for 200 sites, 3 years

sensor_data = np.load('my_sensor_data.npy')  # Shape: (200, 1095, 2)
labels = np.load('my_labels.npy')            # Shape: (200, 3)

# Prepare graphs
x, y = prepare_multisite_data_from_array(
    sensor_data, labels,
    num_sites=200,
    num_sensors_per_site=2,
    seq_length=365
)

# Split and save
split_and_save(x, y, output_dir='../data/my_sites')
```

---

### Understanding the Output Format

After running `prepare_multisite_data.py`, you'll have:

**train.npz, val.npz, test.npz**:
```python
import numpy as np
data = np.load('../data/my_sites/train.npz')

# data['x']: (num_graphs, seq_length, num_sensors, num_features)
#   Example: (420, 365, 2, 2) = 420 graphs, 365 days, 2 sensors, 2 features
#   Feature 0: Original sensor value
#   Feature 1: Time feature (day-of-year / 365)

# data['y']: (num_graphs,)
#   Example: (420,) = one label per graph
#   Classification: integers [0, 1, 2]
#   Regression: float values
```

**metadata.json**:
```json
{
  "num_sites": "multiple",
  "num_sensors_per_site": 2,
  "seq_length": 365,
  "num_features": 2,
  "total_graphs": 600,
  "train_samples": 420,
  "val_samples": 60,
  "test_samples": 120
}
```

**Data split** (default):
- 70% train (420 graphs)
- 10% validation (60 graphs)
- 20% test (120 graphs)
- Randomly shuffled (sites are independent)

---

### Verification

Check your prepared data:

```python
import numpy as np

# Load training data
data = np.load('../data/my_sites/train.npz')
x = data['x']
y = data['y']

print(f"Input shape: {x.shape}")   # Should be (samples, 365, 2, 2)
print(f"Output shape: {y.shape}")  # Should be (samples,)
print(f"Label range: {y.min():.2f} to {y.max():.2f}")
print(f"Unique labels: {np.unique(y)}")  # For classification: [0 1 2]

# Check for NaN or infinite values
print(f"NaN in x: {np.isnan(x).any()}")
print(f"Inf in x: {np.isinf(x).any()}")
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

## Complete Example: Reef Monitoring

Here's a complete workflow for a real-world scenario:

**Scenario**: 200 coral reef sites, 2 sensors (temperature & pH), 3 years of daily data, predict health status (healthy=0, warning=1, critical=2)

### Step 1: Organize Your Data

```bash
# Create directory structure
mkdir -p reef_data/sites_data

# Your site CSV files should look like:
# reef_data/sites_data/site_0.csv:
#   date,temperature,ph
#   2021-01-01,26.5,8.1
#   2021-01-02,26.8,8.0
#   ...
#   2023-12-31,27.2,8.2

# Your labels file:
# reef_data/labels.csv:
#   site_id,year,label
#   0,0,0
#   0,1,1
#   0,2,2
#   1,0,0
#   ...
```

### Step 2: Prepare Data

```bash
cd graph_level

python prepare_multisite_data.py \
    --sites_dir ../reef_data/sites_data \
    --labels_file ../reef_data/labels.csv \
    --output_dir ../data/reef_prepared \
    --num_sites 200 \
    --num_sensors_per_site 2 \
    --seq_length 365 \
    --task classification \
    --num_classes 3

# Output:
#   Total graphs created: 600
#   Graph shape: (600, 365, 2, 2)
#   Saved to ../data/reef_prepared/
#     train: x=(420, 365, 2, 2), y=(420,)
#     val:   x=(60, 365, 2, 2), y=(60,)
#     test:  x=(120, 365, 2, 2), y=(120,)
```

### Step 3: Train Model

```bash
python train_graph.py \
    --data ../data/reef_prepared \
    --num-nodes 2 \
    --seq-length 365 \
    --task classification \
    --num-classes 3 \
    --aptonly \
    --addaptadj \
    --batch-size 32 \
    --learning-rate 0.001 \
    --epochs 100 \
    --nhid 32 \
    --save ../checkpoints/reef_model

# Training will show:
#   Epoch 1: Train Loss: 1.0856 Acc: 0.3524 | Val Loss: 1.0234 Acc: 0.4167
#   Epoch 2: Train Loss: 0.9821 Acc: 0.4286 | Val Loss: 0.9432 Acc: 0.5000
#   ...
```

### Step 4: Evaluate

```bash
python test_graph.py \
    --checkpoint ../checkpoints/reef_model_exp1_best.pth \
    --data ../data/reef_prepared \
    --num-nodes 2 \
    --seq-length 365 \
    --task classification \
    --num-classes 3 \
    --aptonly

# Output:
#   Test Loss: 0.8234
#   Test Accuracy: 0.6583
#
#   Confusion Matrix:
#   [[35  8  2]
#    [ 7 28  5]
#    [ 3  6 26]]
```

### Step 5: Make Predictions on New Sites

```python
import torch
import numpy as np
from model_graph import gwnet_graph

# Load model
model = gwnet_graph(device=torch.device('cpu'), num_nodes=2,
                    in_dim=2, out_dim=3, task='classification')
model.load_state_dict(torch.load('../checkpoints/reef_model_exp1_best.pth'))
model.eval()

# Load new site data (shape: 365, 2)
new_site_data = pd.read_csv('new_site.csv', index_col='date').values

# Prepare input (add time feature)
time_feature = np.arange(365) / 365
x = np.stack([new_site_data,
              np.tile(time_feature[:, np.newaxis], (1, 2))], axis=-1)

# Transpose to (1, 2, 2, 365) for model input
x = torch.FloatTensor(x).unsqueeze(0).transpose(1, 3)

# Predict
with torch.no_grad():
    logits = model(x)
    predicted_class = logits.argmax(dim=1).item()

print(f"Predicted health status: {predicted_class}")  # 0, 1, or 2
```

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

**Missing site files:**
```bash
# The script will warn and skip missing sites:
# "Warning: sites_data/site_5.csv not found, skipping"
# Make sure all site files follow the naming: site_0.csv, site_1.csv, ...
```

**Wrong number of rows in site CSV:**
```bash
# Site CSV must have exactly num_years * seq_length rows
# Example: 3 years × 365 days = 1095 rows
# Check your CSV: wc -l sites_data/site_0.csv
```

**Labels mismatch:**
```bash
# If you see: "Warning: Site X has Y years of data but Z labels"
# Make sure labels.csv has one entry per site-year
# For 200 sites × 3 years = 600 rows in labels.csv
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
