"""
Testing/Inference script for graph-level prediction

Usage:
    python test_graph.py \
        --checkpoint ./checkpoints/graph_model_exp1_best.pth \
        --data data/graph_level_data \
        --adjdata data/sensor_graph/adj_mx.pkl \
        --num-nodes 50 \
        --seq-length 365 \
        --task regression
"""

import torch
import numpy as np
import click
import os
import json
import sys
sys.path.append('..')
import util
from model_graph import gwnet_graph_level
from train_graph import GraphDataLoader, load_graph_dataset


def predict(model, dataloader, device, task='regression'):
    """Make predictions on a dataset"""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = torch.FloatTensor(x).to(device)

            # Transpose to (batch, features, num_nodes, seq_length)
            x = x.permute(0, 3, 2, 1)

            # Forward pass
            pred = model(x).squeeze(-1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return all_preds, all_labels


def compute_metrics(preds, labels, task='regression'):
    """Compute evaluation metrics"""
    if task == 'regression':
        mae = np.mean(np.abs(preds - labels))
        rmse = np.sqrt(np.mean((preds - labels) ** 2))
        mse = np.mean((preds - labels) ** 2)

        # Compute R² score
        ss_res = np.sum((labels - preds) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mse': mse,
            'r2': r2
        }
    else:  # classification
        pred_classes = np.argmax(preds, axis=1) if len(preds.shape) > 1 else (preds > 0).astype(int)
        accuracy = np.mean(pred_classes == labels)

        # Per-class accuracy
        num_classes = len(np.unique(labels))
        per_class_acc = {}
        for c in range(num_classes):
            mask = labels == c
            if mask.sum() > 0:
                per_class_acc[f'class_{c}_acc'] = np.mean(pred_classes[mask] == c)

        metrics = {
            'accuracy': accuracy,
            **per_class_acc
        }

    return metrics


@click.command()
# Model checkpoint
@click.option('--checkpoint', required=True, type=click.Path(exists=True),
              help='Path to model checkpoint')
# Data arguments
@click.option('--data', required=True, type=click.Path(exists=True),
              help='Path to data directory')
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
# Model architecture (must match training)
@click.option('--gcn-bool', is_flag=True, default=True,
              help='Enable graph convolution layers')
@click.option('--addaptadj', is_flag=True, default=True,
              help='Enable adaptive adjacency learning')
@click.option('--aptonly', is_flag=True,
              help='Use only adaptive adjacency')
@click.option('--adjtype', default='doubletransition',
              type=click.Choice(['doubletransition', 'transition', 'symnadj',
                                'normlap', 'scalap', 'identity']),
              help='Adjacency matrix transformation type')
@click.option('--pooling', default='mean', type=click.Choice(['mean', 'max', 'attention']),
              help='Pooling method')
@click.option('--attention-type', default='simple',
              type=click.Choice(['simple', 'temporal_mha', 'gru', 'set_transformer']),
              help='Attention mechanism type (only used when --pooling=attention)')
@click.option('--nhid', default=32, type=int,
              help='Number of hidden units')
@click.option('--dropout', default=0.3, type=float,
              help='Dropout rate')
@click.option('--blocks', default=4, type=int,
              help='Number of WaveNet blocks')
@click.option('--layers', default=2, type=int,
              help='Number of layers per block')
# Other
@click.option('--batch-size', default=32, type=int,
              help='Batch size')
@click.option('--device', default='cuda:0', type=str,
              help='Device')
@click.option('--split', default='test', type=click.Choice(['train', 'val', 'test']),
              help='Data split to evaluate')
@click.option('--save-predictions', default=None, type=str,
              help='Path to save predictions')
def main(checkpoint, data, adjdata, num_nodes, seq_length, in_dim, task, num_classes,
         gcn_bool, addaptadj, aptonly, adjtype, pooling, attention_type, nhid, dropout, blocks, layers,
         batch_size, device, split, save_predictions):
    """Test Graph WaveNet model"""

    # Try to load config from checkpoint directory
    config_path = checkpoint.replace('_best.pth', '_config.json')
    if os.path.exists(config_path):
        click.echo(f"Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Override CLI args with config values if not explicitly provided
        # This ensures consistency with training
        attention_type = config.get('attention_type', attention_type)
        pooling = config.get('pooling', pooling)
        click.echo(f"Loaded attention_type from config: {attention_type}")
        click.echo(f"Loaded pooling from config: {pooling}")

    # Validation
    if task == 'classification' and num_classes is None:
        raise click.UsageError("--num-classes is required for classification task")

    if not aptonly and adjdata is None:
        raise click.UsageError("--adjdata is required when not using --aptonly")

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
        adjinit = supports[0]

    # Load dataset
    click.echo(f"\nLoading dataset from: {data}")
    dataloader = load_graph_dataset(data, batch_size)

    # Get the appropriate split
    if split == 'train':
        test_loader = dataloader['train_loader']
        x_data = dataloader['x_train']
        y_data = dataloader['y_train']
    elif split == 'val':
        test_loader = dataloader['val_loader']
        x_data = dataloader['x_val']
        y_data = dataloader['y_val']
    else:
        test_loader = dataloader['test_loader']
        x_data = dataloader['x_test']
        y_data = dataloader['y_test']

    click.echo(f"Evaluating on {split} set: {len(x_data)} samples")

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

    # Load checkpoint
    click.echo(f"Loading checkpoint from: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device_obj))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    click.echo(f"Total parameters: {total_params:,}")

    # Make predictions
    click.echo("\nMaking predictions...")
    preds, labels = predict(model, test_loader, device_obj, task)

    # Compute metrics
    click.echo("\n" + "="*70)
    click.echo(f"Evaluation Results on {split.upper()} set:")
    click.echo("="*70)

    metrics = compute_metrics(preds, labels, task)

    for metric_name, metric_value in metrics.items():
        click.echo(f"{metric_name:20s}: {metric_value:.6f}")

    click.echo("="*70)

    # Save predictions
    if save_predictions:
        save_dir = os.path.dirname(save_predictions)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        results = {
            'predictions': preds.tolist(),
            'labels': labels.tolist(),
            'metrics': {k: float(v) for k, v in metrics.items()}
        }

        with open(save_predictions, 'w') as f:
            json.dump(results, f, indent=2)

        click.echo(f"\nPredictions saved to: {save_predictions}")

    # Show some example predictions
    if task == 'regression':
        click.echo("\nSample predictions:")
        click.echo("-" * 50)
        click.echo(f"{'Index':<10} {'True':<15} {'Predicted':<15} {'Error':<15}")
        click.echo("-" * 50)
        num_examples = min(10, len(preds))
        for i in range(num_examples):
            error = abs(preds[i] - labels[i])
            click.echo(f"{i:<10} {labels[i]:<15.4f} {preds[i]:<15.4f} {error:<15.4f}")
    else:
        click.echo("\nSample predictions:")
        click.echo("-" * 50)
        click.echo(f"{'Index':<10} {'True':<15} {'Predicted':<15} {'Correct':<15}")
        click.echo("-" * 50)
        num_examples = min(10, len(preds))
        pred_classes = np.argmax(preds, axis=1) if len(preds.shape) > 1 else (preds > 0).astype(int)
        for i in range(num_examples):
            correct = "✓" if pred_classes[i] == labels[i] else "✗"
            click.echo(f"{i:<10} {int(labels[i]):<15} {int(pred_classes[i]):<15} {correct:<15}")


if __name__ == "__main__":
    main()
