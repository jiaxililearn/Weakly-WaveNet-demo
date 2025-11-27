"""
Training script for graph-level prediction using Graph WaveNet

Usage:
    # Regression task
    python train_graph.py \
        --data data/graph_level_data \
        --adjdata data/sensor_graph/adj_mx.pkl \
        --num-nodes 50 \
        --seq-length 365 \
        --task regression

    # Classification task
    python train_graph.py \
        --data data/graph_level_data \
        --adjdata data/sensor_graph/adj_mx.pkl \
        --num-nodes 50 \
        --seq-length 365 \
        --task classification \
        --num-classes 5
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import click
import time
import os
import json
import sys
sys.path.append('..')
import util
from model_graph import gwnet_graph_level


class GraphDataLoader:
    """Simple data loader for graph-level prediction"""

    def __init__(self, x, y, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(x)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]

            yield self.x[batch_indices], self.y[batch_indices]

    def __len__(self):
        return self.num_batches


def load_graph_dataset(data_dir, batch_size):
    """Load graph-level dataset"""
    data = {}

    for split in ['train', 'val', 'test']:
        split_data = np.load(os.path.join(data_dir, f'{split}.npz'))
        data[f'x_{split}'] = split_data['x']
        data[f'y_{split}'] = split_data['y']

    # Create data loaders
    data['train_loader'] = GraphDataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = GraphDataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    data['test_loader'] = GraphDataLoader(data['x_test'], data['y_test'], batch_size, shuffle=False)

    # Create scaler for input normalization (only on first feature channel)
    mean = data['x_train'][..., 0].mean()
    std = data['x_train'][..., 0].std()
    data['scaler'] = util.StandardScaler(mean, std)

    # Normalize first channel
    for split in ['train', 'val', 'test']:
        data[f'x_{split}'][..., 0] = data['scaler'].transform(data[f'x_{split}'][..., 0])

    return data


def train_epoch(model, dataloader, criterion, optimizer, device, clip):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for x, y in dataloader:
        # x: (batch, seq_length, num_nodes, features)
        # y: (batch,) for regression or (batch,) for classification

        x = torch.FloatTensor(x).to(device)
        y = torch.FloatTensor(y).to(device) if criterion.__class__.__name__ == 'MSELoss' else torch.LongTensor(y).to(device)

        # Transpose to (batch, features, num_nodes, seq_length)
        x = x.permute(0, 3, 2, 1)

        # Forward pass
        optimizer.zero_grad()
        pred = model(x).squeeze(-1)  # (batch, out_dim) -> (batch,) for regression

        # Compute loss
        loss = criterion(pred, y)

        # Backward pass
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device, task='regression'):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = torch.FloatTensor(x).to(device)
            y = torch.FloatTensor(y).to(device) if task == 'regression' else torch.LongTensor(y).to(device)

            # Transpose to (batch, features, num_nodes, seq_length)
            x = x.permute(0, 3, 2, 1)

            # Forward pass
            pred = model(x).squeeze(-1)

            # Compute loss
            loss = criterion(pred, y)

            total_loss += loss.item()
            num_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    avg_loss = total_loss / num_batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    if task == 'regression':
        mae = np.mean(np.abs(all_preds - all_labels))
        rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
        metrics = {'loss': avg_loss, 'mae': mae, 'rmse': rmse}
    else:  # classification
        pred_classes = np.argmax(all_preds, axis=1) if len(all_preds.shape) > 1 else (all_preds > 0).astype(int)
        accuracy = np.mean(pred_classes == all_labels)
        metrics = {'loss': avg_loss, 'accuracy': accuracy}

    return metrics


@click.command()
# Data arguments
@click.option('--data', required=True, type=click.Path(exists=True),
              help='Path to data directory containing train.npz, val.npz, test.npz')
@click.option('--adjdata', default=None, type=click.Path(),
              help='Path to adjacency matrix pickle file')
@click.option('--num-nodes', required=True, type=int,
              help='Number of nodes/sensors')
@click.option('--seq-length', default=365, type=int,
              help='Sequence length')
@click.option('--in-dim', default=2, type=int,
              help='Input feature dimension')
# Task parameters
@click.option('--task', default='regression', type=click.Choice(['regression', 'classification']),
              help='Task type')
@click.option('--num-classes', default=None, type=int,
              help='Number of classes (for classification)')
# Model architecture
@click.option('--gcn-bool', is_flag=True, default=True,
              help='Enable graph convolution layers')
@click.option('--addaptadj', is_flag=True, default=True,
              help='Enable adaptive adjacency learning')
@click.option('--aptonly', is_flag=True,
              help='Use only adaptive adjacency')
@click.option('--randomadj', is_flag=True,
              help='Randomly initialize adaptive adjacency')
@click.option('--adjtype', default='doubletransition',
              type=click.Choice(['doubletransition', 'transition', 'symnadj',
                                'normlap', 'scalap', 'identity']),
              help='Adjacency matrix transformation type')
@click.option('--pooling', default='mean', type=click.Choice(['mean', 'max', 'attention']),
              help='Pooling method for graph-level readout')
@click.option('--attention-type', default='simple',
              type=click.Choice(['simple', 'temporal_mha', 'gru', 'set_transformer']),
              help='Attention mechanism type (only used when --pooling=attention)')
# Model hyperparameters
@click.option('--nhid', default=32, type=int,
              help='Number of hidden units')
@click.option('--dropout', default=0.3, type=float,
              help='Dropout rate')
@click.option('--blocks', default=4, type=int,
              help='Number of WaveNet blocks')
@click.option('--layers', default=2, type=int,
              help='Number of layers per block')
# Training hyperparameters
@click.option('--batch-size', default=32, type=int,
              help='Batch size')
@click.option('--learning-rate', default=0.001, type=float,
              help='Learning rate')
@click.option('--weight-decay', default=0.0001, type=float,
              help='Weight decay')
@click.option('--epochs', default=100, type=int,
              help='Number of epochs')
@click.option('--clip', default=5.0, type=float,
              help='Gradient clipping threshold')
# Device and output
@click.option('--device', default='cuda:0', type=str,
              help='Device')
@click.option('--save', default='./checkpoints/graph_model', type=str,
              help='Save path')
@click.option('--expid', default=1, type=int,
              help='Experiment ID')
@click.option('--seed', default=None, type=int,
              help='Random seed')
def main(data, adjdata, num_nodes, seq_length, in_dim, task, num_classes,
         gcn_bool, addaptadj, aptonly, randomadj, adjtype, pooling, attention_type,
         nhid, dropout, blocks, layers, batch_size, learning_rate,
         weight_decay, epochs, clip, device, save, expid, seed):
    """Train Graph WaveNet for graph-level prediction"""

    # Validation
    if task == 'classification' and num_classes is None:
        raise click.UsageError("--num-classes is required for classification task")

    if not aptonly and adjdata is None:
        raise click.UsageError("--adjdata is required when not using --aptonly")

    # Create save directory
    save_dir = os.path.dirname(save)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        click.echo(f"Created checkpoint directory: {save_dir}")

    # Set random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        click.echo(f"Random seed set to: {seed}")

    # Setup device
    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    click.echo(f"Using device: {device_obj}")

    # Load adjacency matrix
    if aptonly:
        click.echo("Using adaptive adjacency only")
        supports = None
        adjinit = None
    else:
        click.echo(f"Loading adjacency matrix from: {adjdata}")
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adjdata, adjtype)
        supports = [torch.tensor(i).to(device_obj) for i in adj_mx]
        click.echo(f"Adjacency type: {adjtype}, support matrices: {len(supports)}")

        if randomadj:
            adjinit = None
            click.echo("Using random initialization for adaptive adjacency")
        else:
            adjinit = supports[0]
            click.echo("Using SVD initialization for adaptive adjacency")

    # Load dataset
    click.echo(f"\nLoading dataset from: {data}")
    dataloader = load_graph_dataset(data, batch_size)

    click.echo(f"Train samples: {len(dataloader['x_train'])}")
    click.echo(f"Val samples:   {len(dataloader['x_val'])}")
    click.echo(f"Test samples:  {len(dataloader['x_test'])}")
    click.echo(f"Input shape:   {dataloader['x_train'].shape}")
    click.echo(f"Label shape:   {dataloader['y_train'].shape}")

    # Output dimension
    out_dim = 1 if task == 'regression' else num_classes

    # Initialize model
    click.echo("\nInitializing model...")
    click.echo(f"Pooling: {pooling}" + (f" (attention type: {attention_type})" if pooling == 'attention' else ""))
    model = gwnet_graph_level(
        device=device_obj,
        num_nodes=num_nodes,
        dropout=dropout,
        supports=supports,
        gcn_bool=gcn_bool,
        addaptadj=addaptadj,
        aptinit=adjinit,
        in_dim=in_dim,
        out_dim=out_dim,
        residual_channels=nhid,
        dilation_channels=nhid,
        skip_channels=nhid * 8,
        end_channels=nhid * 16,
        blocks=blocks,
        layers=layers,
        pooling=pooling,
        attention_type=attention_type,
        task=task
    ).to(device_obj)

    # Loss function
    if task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    click.echo(f"Total parameters: {total_params:,}")
    click.echo(f"Trainable parameters: {trainable_params:,}")

    # Training loop
    click.echo("\n" + "="*70)
    click.echo("Starting training...")
    click.echo("="*70 + "\n")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}

    for epoch in range(1, epochs + 1):
        # Training
        t1 = time.time()
        train_loss = train_epoch(model, dataloader['train_loader'], criterion,
                                optimizer, device_obj, clip)
        t2 = time.time()

        # Validation
        val_metrics = evaluate(model, dataloader['val_loader'], criterion, device_obj, task)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_metrics'].append(val_metrics)

        # Logging
        if task == 'regression':
            log = 'Epoch {:03d} | Train Loss: {:.4f} | Val Loss: {:.4f} MAE: {:.4f} RMSE: {:.4f} | Time: {:.1f}s'
            click.echo(log.format(epoch, train_loss, val_metrics['loss'],
                                 val_metrics['mae'], val_metrics['rmse'], t2 - t1))
        else:
            log = 'Epoch {:03d} | Train Loss: {:.4f} | Val Loss: {:.4f} Acc: {:.4f} | Time: {:.1f}s'
            click.echo(log.format(epoch, train_loss, val_metrics['loss'],
                                 val_metrics['accuracy'], t2 - t1))

        # Save checkpoint
        checkpoint_path = f"{save}_epoch_{epoch}_{val_metrics['loss']:.4f}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            click.echo(f"  → New best validation loss: {val_metrics['loss']:.4f}")

    # Test best model
    click.echo("\n" + "="*70)
    click.echo("Testing best model...")
    click.echo("="*70)

    best_epoch = np.argmin(history['val_loss']) + 1
    best_model_path = f"{save}_epoch_{best_epoch}_{history['val_loss'][best_epoch-1]:.4f}.pth"
    click.echo(f"Loading best model from epoch {best_epoch}")

    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, dataloader['test_loader'], criterion, device_obj, task)

    if task == 'regression':
        click.echo(f"Test Loss: {test_metrics['loss']:.4f}")
        click.echo(f"Test MAE:  {test_metrics['mae']:.4f}")
        click.echo(f"Test RMSE: {test_metrics['rmse']:.4f}")
    else:
        click.echo(f"Test Loss:     {test_metrics['loss']:.4f}")
        click.echo(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    # Save final model
    final_path = f"{save}_exp{expid}_best.pth"
    torch.save(model.state_dict(), final_path)
    click.echo(f"\nBest model saved to: {final_path}")

    # Save model configuration
    config_path = f"{save}_exp{expid}_config.json"
    config = {
        'num_nodes': num_nodes,
        'seq_length': seq_length,
        'in_dim': in_dim,
        'task': task,
        'num_classes': num_classes,
        'pooling': pooling,
        'attention_type': attention_type,
        'nhid': nhid,
        'dropout': dropout,
        'blocks': blocks,
        'layers': layers,
        'gcn_bool': gcn_bool,
        'addaptadj': addaptadj,
        'aptonly': aptonly,
        'adjtype': adjtype,
        'out_dim': out_dim
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    click.echo(f"Model config saved to: {config_path}")

    # Save training history
    history_path = f"{save}_exp{expid}_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy types to native Python types
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_metrics': [{k: float(v) for k, v in m.items()} for m in history['val_metrics']],
            'test_metrics': {k: float(v) for k, v in test_metrics.items()}
        }
        json.dump(history_serializable, f, indent=2)
    click.echo(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")
