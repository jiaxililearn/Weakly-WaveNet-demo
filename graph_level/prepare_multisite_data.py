"""
Data preparation for multi-site scenario

Supports loading from pandas pickle file with nested DataFrames.
Each site contains time-series sensor data, used to predict site-level labels.
"""

import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm


def split_and_save(x, y, output_dir, train_ratio=0.7, val_ratio=0.1):
    """Split and save data"""
    num_samples = len(x)

    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)

    # Random shuffle (sites are independent)
    indices = np.random.permutation(num_samples)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]

    x_train, y_train = x[train_indices], y[train_indices]
    x_val, y_val = x[val_indices], y[val_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    os.makedirs(output_dir, exist_ok=True)

    np.savez_compressed(os.path.join(output_dir, 'train.npz'), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(output_dir, 'val.npz'), x=x_val, y=y_val)
    np.savez_compressed(os.path.join(output_dir, 'test.npz'), x=x_test, y=y_test)

    print(f"\nSaved to {output_dir}/")
    print(f"  train: x={x_train.shape}, y={y_train.shape}")
    print(f"  val:   x={x_val.shape}, y={y_val.shape}")
    print(f"  test:  x={x_test.shape}, y={y_test.shape}")

    # Save metadata
    import json
    metadata = {
        'num_sites': 'multiple',
        'num_sensors_per_site': x.shape[2],
        'seq_length': x.shape[1],
        'num_features': x.shape[3],
        'total_graphs': num_samples,
        'train_samples': len(x_train),
        'val_samples': len(x_val),
        'test_samples': len(x_test)
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def prepare_multisite_data_from_pickle(pickle_file, data_cols, label_col,
                                        seq_length=365):
    """
    Prepare data from pandas pickle file

    Args:
        pickle_file: Path to pandas pickle file
        data_cols: List of column names from site_data dataframe to use as sensor features
                   e.g., ['temperature', 'missing']
        label_col: Column name for label (e.g., 'bleached_flag')
        seq_length: Number of days per year (default 365, handles leap years by truncation)

    Expected schema:
        Main pickle DataFrame columns: site, date_start, date_end, site_data, [label_col]
        site_data (nested DataFrame) columns: time, [data_cols...], site

    Returns:
        x: (num_graphs, seq_length, num_sensors, num_features)
           where num_features = len(data_cols) + 1 (data + time-of-year)
        y: (num_graphs,)
    """
    print(f"Loading pickle file: {pickle_file}")
    df = pd.read_pickle(pickle_file)

    print(f"Loaded {len(df)} site records")
    print(f"Using data columns: {data_cols}")
    print(f"Using label column: {label_col}")

    all_graphs_x = []
    all_graphs_y = []

    num_sensors = len(data_cols)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sites"):
        site_id = row['site']
        site_data_df = row['site_data']
        label = row[label_col]

        # Get total days available
        total_days = len(site_data_df)

        if total_days < seq_length:
            tqdm.write(f"Warning: Site {site_id} has only {total_days} days, skipping")
            continue

        # Extract sensor data columns (first seq_length days only)
        sensor_data = site_data_df[data_cols].iloc[:seq_length].values  # (seq_length, num_sensors)

        # Add feature dimension
        sensor_data_expanded = np.expand_dims(sensor_data, axis=-1)  # (365, num_sensors, 1)

        # Add time-of-year feature
        time_feature = np.arange(seq_length) / seq_length  # (365,)
        time_feature = np.tile(time_feature[:, np.newaxis, np.newaxis],
                               (1, num_sensors, 1))  # (365, num_sensors, 1)

        data_with_time = np.concatenate([sensor_data_expanded, time_feature],
                                       axis=-1)  # (365, num_sensors, 2)

        all_graphs_x.append(data_with_time)
        all_graphs_y.append(label)

    all_graphs_x = np.array(all_graphs_x, dtype=np.float32)
    all_graphs_y = np.array(all_graphs_y, dtype=np.float32)

    print(f"\nTotal graphs created: {len(all_graphs_x)}")
    print(f"Graph shape: {all_graphs_x.shape}")  # (num_graphs, 365, num_sensors, 2)
    print(f"Labels shape: {all_graphs_y.shape}")  # (num_graphs,)

    return all_graphs_x, all_graphs_y


def prepare_multisite_data_from_array(sensor_data, labels, num_sites,
                                       num_sensors_per_site, seq_length):
    """
    Prepare data from arrays (used by synthetic data generator)

    Args:
        sensor_data: (num_sites, total_days, num_sensors_per_site)
        labels: (num_sites, num_years)
        num_sites: Number of sites
        num_sensors_per_site: Number of sensors per site
        seq_length: Days per year

    Returns:
        x: (num_graphs, seq_length, num_sensors, 2)
        y: (num_graphs,)
    """
    print(f"Processing array data: {sensor_data.shape}")

    all_graphs_x = []
    all_graphs_y = []

    num_years = sensor_data.shape[1] // seq_length

    for site_id in tqdm(range(num_sites), desc="Processing synthetic sites"):
        site_data = sensor_data[site_id]  # (total_days, num_sensors_per_site)
        site_labels = labels[site_id]  # (num_years,)

        for year_id in range(num_years):
            start_day = year_id * seq_length
            end_day = start_day + seq_length

            yearly_data = site_data[start_day:end_day, :]  # (365, num_sensors)

            # Add features
            yearly_data_expanded = np.expand_dims(yearly_data, axis=-1)  # (365, num_sensors, 1)

            time_feature = np.arange(seq_length) / seq_length
            time_feature = np.tile(time_feature[:, np.newaxis, np.newaxis],
                                   (1, num_sensors_per_site, 1))

            yearly_data_with_time = np.concatenate([yearly_data_expanded, time_feature],
                                                   axis=-1)  # (365, num_sensors, 2)

            all_graphs_x.append(yearly_data_with_time)
            all_graphs_y.append(site_labels[year_id])

    all_graphs_x = np.array(all_graphs_x, dtype=np.float32)
    all_graphs_y = np.array(all_graphs_y, dtype=np.float32)

    print(f"Total graphs created: {len(all_graphs_x)}")
    print(f"Graph shape: {all_graphs_x.shape}")

    return all_graphs_x, all_graphs_y


def generate_synthetic_multisite_data(num_sites, num_sensors_per_site,
                                      num_years, seq_length, task='regression',
                                      num_classes=None):
    """Generate synthetic multi-site data for testing"""
    print(f"Generating synthetic data:")
    print(f"  Sites: {num_sites}")
    print(f"  Sensors per site: {num_sensors_per_site}")
    print(f"  Years: {num_years}")
    print(f"  Days per year: {seq_length}")
    print(f"  Task: {task}")

    total_days = num_years * seq_length

    # Generate sensor data
    sensor_data = np.random.randn(num_sites, total_days,
                                  num_sensors_per_site).astype(np.float32)

    # Generate labels based on task type
    if task == 'regression':
        labels = np.random.randn(num_sites, num_years).astype(np.float32) * 10 + 100
        print(f"  Labels: continuous values (regression)")
    elif task == 'classification':
        if num_classes is None:
            raise ValueError("num_classes must be specified for classification task")
        labels = np.random.randint(0, num_classes, size=(num_sites, num_years))
        print(f"  Labels: {num_classes} classes (classification)")
    else:
        raise ValueError(f"Unknown task: {task}")

    return sensor_data, labels


def main():
    parser = argparse.ArgumentParser(description='Prepare multi-site data')

    parser.add_argument('--pickle_file', type=str, default=None,
                        help='Path to pandas pickle file')
    parser.add_argument('--data_cols', type=str, nargs='+', default=None,
                        help='Column names from site_data to use as features (e.g., temperature missing)')
    parser.add_argument('--label_col', type=str, default=None,
                        help='Column name for label (e.g., bleached_flag)')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    parser.add_argument('--num_sites', type=int, default=200,
                        help='Number of sites (for synthetic data)')
    parser.add_argument('--num_sensors_per_site', type=int, default=2,
                        help='Number of sensors per site (for synthetic data)')
    parser.add_argument('--num_years', type=int, default=3,
                        help='Number of years (for synthetic data)')
    parser.add_argument('--seq_length', type=int, default=365,
                        help='Days per year')

    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic data')

    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Task type (for synthetic data)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (for classification)')

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    args = parser.parse_args()

    # Validation
    if args.synthetic and args.task == 'classification' and args.num_classes is None:
        parser.error("--num_classes is required for classification task")

    if args.pickle_file:
        if not args.data_cols or not args.label_col:
            parser.error("--data_cols and --label_col are required when using --pickle_file")
        print("Loading from pandas pickle file...")
        x, y = prepare_multisite_data_from_pickle(
            args.pickle_file, args.data_cols, args.label_col, args.seq_length
        )
    elif args.synthetic:
        print("Generating synthetic multi-site data...")
        sensor_data, labels = generate_synthetic_multisite_data(
            args.num_sites, args.num_sensors_per_site,
            args.num_years, args.seq_length,
            args.task, args.num_classes
        )
        x, y = prepare_multisite_data_from_array(
            sensor_data, labels, args.num_sites,
            args.num_sensors_per_site, args.seq_length
        )
    else:
        raise ValueError("Either use --pickle_file or --synthetic")

    # Split and save
    split_and_save(x, y, args.output_dir, args.train_ratio, args.val_ratio)

    print("\nData preparation complete!")
    print(f"Each graph has {x.shape[2]} nodes (sensors)")
    print(f"Total graphs: {len(x)}")


if __name__ == "__main__":
    main()
