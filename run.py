"""
run.py - Flexible training script for Graph WaveNet with custom datasets

Usage examples:
    # Basic training with your own data
    python run.py --data /path/to/your/data --adjdata /path/to/adj.pkl --num-nodes 207

    # Training with all graph components
    python run.py --data /path/to/your/data --adjdata /path/to/adj.pkl --num-nodes 207 \
                  --gcn-bool --addaptadj --randomadj --adjtype doubletransition

    # Training with only adaptive adjacency (no predefined graph)
    python run.py --data /path/to/your/data --num-nodes 207 --addaptadj --aptonly

    # Custom hyperparameters
    python run.py --data /path/to/your/data --adjdata /path/to/adj.pkl --num-nodes 207 \
                  --batch-size 32 --learning-rate 0.0001 --epochs 200 --nhid 64
"""

import torch
import numpy as np
import click
import time
import os
import util
from engine import trainer


def train_epoch(engine, dataloader, device, print_every):
    """Train for one epoch"""
    train_loss = []
    train_mape = []
    train_rmse = []

    dataloader["train_loader"].shuffle()

    for iter_num, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
        trainx = torch.Tensor(x).to(device)
        trainx = trainx.transpose(1, 3)  # [batch, feature, nodes, time]
        trainy = torch.Tensor(y).to(device)
        trainy = trainy.transpose(1, 3)

        metrics = engine.train(trainx, trainy[:, 0, :, :])
        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])

        if iter_num % print_every == 0:
            log = "Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}"
            print(
                log.format(iter_num, train_loss[-1], train_mape[-1], train_rmse[-1]),
                flush=True,
            )

    return np.mean(train_loss), np.mean(train_mape), np.mean(train_rmse)


def validate(engine, dataloader, device):
    """Validate on validation set"""
    valid_loss = []
    valid_mape = []
    valid_rmse = []

    for iter_num, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
        valx = torch.Tensor(x).to(device)
        valx = valx.transpose(1, 3)
        valy = torch.Tensor(y).to(device)
        valy = valy.transpose(1, 3)

        metrics = engine.eval(valx, valy[:, 0, :, :])
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])

    return np.mean(valid_loss), np.mean(valid_mape), np.mean(valid_rmse)


def test(engine, dataloader, device, scaler, horizon):
    """Test on test set and report per-horizon metrics"""
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter_num, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    # Per-horizon evaluation
    amae = []
    amape = []
    armse = []

    print("\n" + "=" * 70)
    print("Per-Horizon Test Results:")
    print("=" * 70)

    for i in range(horizon):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Horizon {:2d} | MAE: {:.4f} | MAPE: {:.4f} | RMSE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    print("=" * 70)
    log = "Average     | MAE: {:.4f} | MAPE: {:.4f} | RMSE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    print("=" * 70)

    return np.mean(amae), np.mean(amape), np.mean(armse)


@click.command()
# Data arguments
@click.option(
    "--data",
    required=True,
    type=click.Path(exists=True),
    help="Path to data directory containing train.npz, val.npz, test.npz",
)
@click.option(
    "--adjdata",
    default=None,
    type=click.Path(),
    help="Path to adjacency matrix pickle file (required if not using --aptonly)",
)
@click.option(
    "--num-nodes", required=True, type=int, help="Number of nodes/sensors in your graph"
)
@click.option(
    "--in-dim",
    default=2,
    type=int,
    help="Input feature dimension (default: 2 for value + time_in_day)",
)
# Graph convolution arguments
@click.option("--gcn-bool", is_flag=True, help="Enable graph convolution layers")
@click.option("--addaptadj", is_flag=True, help="Enable adaptive adjacency learning")
@click.option(
    "--aptonly", is_flag=True, help="Use only adaptive adjacency (no predefined graph)"
)
@click.option(
    "--randomadj",
    is_flag=True,
    help="Randomly initialize adaptive adj (vs. SVD initialization)",
)
@click.option(
    "--adjtype",
    default="doubletransition",
    type=click.Choice(
        ["doubletransition", "transition", "symnadj", "normlap", "scalap", "identity"]
    ),
    help="Type of adjacency matrix transformation",
)
# Model hyperparameters
@click.option(
    "--seq-length",
    default=12,
    type=int,
    help="Input sequence length (timesteps to look back)",
)
@click.option(
    "--horizon",
    default=12,
    type=int,
    help="Prediction horizon (timesteps to predict forward)",
)
@click.option(
    "--nhid", default=32, type=int, help="Number of hidden units (residual_channels)"
)
@click.option("--dropout", default=0.3, type=float, help="Dropout rate")
# Training hyperparameters
@click.option("--batch-size", default=64, type=int, help="Batch size for training")
@click.option(
    "--learning-rate", default=0.001, type=float, help="Initial learning rate"
)
@click.option(
    "--weight-decay",
    default=0.0001,
    type=float,
    help="Weight decay (L2 regularization)",
)
@click.option("--epochs", default=100, type=int, help="Number of training epochs")
@click.option("--clip", default=5.0, type=float, help="Gradient clipping threshold")
# Device and output
@click.option(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use (e.g., cuda:0, cuda:1, cpu)",
)
@click.option(
    "--save",
    default="./checkpoints/model",
    type=str,
    help="Path prefix for saving model checkpoints",
)
@click.option(
    "--expid", default=1, type=int, help="Experiment ID for tracking multiple runs"
)
# Logging
@click.option(
    "--print-every",
    default=50,
    type=int,
    help="Print training metrics every N iterations",
)
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
def main(
    data,
    adjdata,
    num_nodes,
    in_dim,
    gcn_bool,
    addaptadj,
    aptonly,
    randomadj,
    adjtype,
    seq_length,
    horizon,
    nhid,
    dropout,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    clip,
    device,
    save,
    expid,
    print_every,
    seed,
):
    """Graph WaveNet for custom datasets"""

    # Validation
    if not aptonly and adjdata is None:
        raise click.UsageError("--adjdata is required when not using --aptonly")

    # Check for required data files
    for split in ["train", "val", "test"]:
        filepath = os.path.join(data, f"{split}.npz")
        if not os.path.exists(filepath):
            raise click.BadParameter(f"Missing required file: {filepath}")

    # Create checkpoint directory
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
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device_obj}")

    # Load adjacency matrix
    if aptonly:
        click.echo("Using adaptive adjacency only (no predefined graph)")
        sensor_ids, sensor_id_to_ind = None, None
        supports = None
        adjinit = None
    else:
        click.echo(f"Loading adjacency matrix from: {adjdata}")
        sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(adjdata, adjtype)
        supports = [torch.tensor(i).to(device_obj) for i in adj_mx]
        click.echo(f"Adjacency type: {adjtype}")
        click.echo(f"Number of support matrices: {len(supports)}")

        if randomadj:
            adjinit = None
            click.echo("Using random initialization for adaptive adjacency")
        else:
            adjinit = supports[0]
            click.echo("Using SVD initialization for adaptive adjacency")

    # Load dataset
    click.echo(f"\nLoading dataset from: {data}")
    dataloader = util.load_dataset(data, batch_size, batch_size, batch_size)
    scaler = dataloader["scaler"]

    click.echo(f"Train samples: {len(dataloader['x_train'])}")
    click.echo(f"Val samples:   {len(dataloader['x_val'])}")
    click.echo(f"Test samples:  {len(dataloader['x_test'])}")
    click.echo(f"Input shape:   {dataloader['x_train'].shape}")
    click.echo(f"Output shape:  {dataloader['y_train'].shape}")

    # Print configuration
    click.echo("\n" + "=" * 70)
    click.echo("Configuration:")
    click.echo("=" * 70)
    config = {
        "data": data,
        "adjdata": adjdata,
        "num_nodes": num_nodes,
        "in_dim": in_dim,
        "gcn_bool": gcn_bool,
        "addaptadj": addaptadj,
        "aptonly": aptonly,
        "randomadj": randomadj,
        "adjtype": adjtype,
        "seq_length": seq_length,
        "horizon": horizon,
        "nhid": nhid,
        "dropout": dropout,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "clip": clip,
        "device": device,
        "save": save,
        "expid": expid,
        "print_every": print_every,
        "seed": seed,
    }
    for key, value in config.items():
        click.echo(f"{key:20s}: {value}")
    click.echo("=" * 70 + "\n")

    # Initialize model
    click.echo("Initializing model...")
    engine = trainer(
        scaler=scaler,
        in_dim=in_dim,
        seq_length=horizon,
        num_nodes=num_nodes,
        nhid=nhid,
        dropout=dropout,
        lrate=learning_rate,
        wdecay=weight_decay,
        device=device_obj,
        supports=supports,
        gcn_bool=gcn_bool,
        addaptadj=addaptadj,
        aptinit=adjinit,
    )

    # Update clip value
    engine.clip = clip

    # Count parameters
    total_params = sum(p.numel() for p in engine.model.parameters())
    trainable_params = sum(
        p.numel() for p in engine.model.parameters() if p.requires_grad
    )
    click.echo(f"Total parameters: {total_params:,}")
    click.echo(f"Trainable parameters: {trainable_params:,}")

    # Training loop
    click.echo("\n" + "=" * 70)
    click.echo("Starting training...")
    click.echo("=" * 70 + "\n")

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Training
        t1 = time.time()
        train_loss, train_mape, train_rmse = train_epoch(
            engine, dataloader, device_obj, print_every
        )
        t2 = time.time()
        train_time.append(t2 - t1)

        # Validation
        s1 = time.time()
        val_loss, val_mape, val_rmse = validate(engine, dataloader, device_obj)
        s2 = time.time()
        val_time.append(s2 - s1)

        his_loss.append(val_loss)

        # Logging
        log = (
            "Epoch: {:03d} | Train Loss: {:.4f} MAPE: {:.4f} RMSE: {:.4f} | "
            "Val Loss: {:.4f} MAPE: {:.4f} RMSE: {:.4f} | Time: {:.1f}s"
        )
        click.echo(
            log.format(
                epoch,
                train_loss,
                train_mape,
                train_rmse,
                val_loss,
                val_mape,
                val_rmse,
                t2 - t1,
            )
        )

        # Save checkpoint
        checkpoint_path = f"{save}_epoch_{epoch}_{val_loss:.2f}.pth"
        torch.save(engine.model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            click.echo(f"  → New best validation loss: {val_loss:.4f}")

    click.echo(f"\nAverage training time: {np.mean(train_time):.2f}s/epoch")
    click.echo(f"Average validation time: {np.mean(val_time):.2f}s")

    # Load best model and test
    click.echo("\n" + "=" * 70)
    click.echo("Testing best model...")
    click.echo("=" * 70)

    bestid = np.argmin(his_loss)
    best_model_path = f"{save}_epoch_{bestid+1}_{his_loss[bestid]:.2f}.pth"
    click.echo(f"Loading best model from epoch {bestid+1}")
    click.echo(f"Best validation loss: {his_loss[bestid]:.4f}")

    engine.model.load_state_dict(torch.load(best_model_path))
    test_mae, test_mape, test_rmse = test(
        engine, dataloader, device_obj, scaler, horizon
    )

    # Save final best model
    final_path = f"{save}_exp{expid}_best_{his_loss[bestid]:.2f}.pth"
    torch.save(engine.model.state_dict(), final_path)
    click.echo(f"\nBest model saved to: {final_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")
