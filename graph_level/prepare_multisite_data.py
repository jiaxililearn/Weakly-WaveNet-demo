"""
Data preparation for multi-site scenario

Scenario:
- 200 sites (each site is an independent graph)
- 2 sensors per site (nodes in each graph)
- Multiple years of daily data
- One label per site per year

Example structure:
    Site 0: sensor_0, sensor_1 → label for year 0, label for year 1, ...
    Site 1: sensor_0, sensor_1 → label for year 0, label for year 1, ...
    ...
"""

import numpy as np
import pandas as pd
import os
import argparse


def prepare_multisite_data_from_csv(sites_dir, labels_file, output_dir,
                                     num_sites, num_sensors_per_site, seq_length):
    """
    Prepare data when each site has its own CSV file

    Expected structure:
        sites_dir/
            site_0.csv  # Columns: date, sensor_0, sensor_1
            site_1.csv
            ...
            site_199.csv

        labels_file:  # CSV with columns: site_id, year, label
            site_id,year,label
            0,0,100.5
            0,1,105.2
            0,2,98.3
            1,0,110.2
            ...
    """
    print("Loading labels...")
    labels_df = pd.read_csv(labels_file)

    all_graphs_x = []
    all_graphs_y = []

    print(f"Processing {num_sites} sites...")

    for site_id in range(num_sites):
        # Load this site's sensor data
        site_file = os.path.join(sites_dir, f'site_{site_id}.csv')

        if not os.path.exists(site_file):
            print(f"Warning: {site_file} not found, skipping")
            continue

        site_df = pd.read_csv(site_file, index_col='date', parse_dates=True)
        site_data = site_df.values  # (days, num_sensors_per_site)

        # Get labels for this site
        site_labels = labels_df[labels_df['site_id'] == site_id].sort_values('year')['label'].values

        # Split into yearly windows
        num_years = len(site_data) // seq_length

        if num_years != len(site_labels):
            print(f"Warning: Site {site_id} has {num_years} years of data but {len(site_labels)} labels")
            num_years = min(num_years, len(site_labels))

        for year_id in range(num_years):
            start_day = year_id * seq_length
            end_day = start_day + seq_length

            yearly_data = site_data[start_day:end_day, :]  # (365, 2)

            # Add feature dimension and time feature
            yearly_data_expanded = np.expand_dims(yearly_data, axis=-1)  # (365, 2, 1)

            # Add time-of-day feature (simple day-of-year normalization)
            time_feature = np.arange(seq_length) / seq_length  # (365,)
            time_feature = np.tile(time_feature[:, np.newaxis, np.newaxis],
                                   (1, num_sensors_per_site, 1))  # (365, 2, 1)

            yearly_data_with_time = np.concatenate([yearly_data_expanded, time_feature],
                                                   axis=-1)  # (365, 2, 2)

            all_graphs_x.append(yearly_data_with_time)
            all_graphs_y.append(site_labels[year_id])

        if (site_id + 1) % 50 == 0:
            print(f"  Processed {site_id + 1}/{num_sites} sites...")

    all_graphs_x = np.array(all_graphs_x, dtype=np.float32)
    all_graphs_y = np.array(all_graphs_y, dtype=np.float32)

    print(f"\nTotal graphs created: {len(all_graphs_x)}")
    print(f"Graph shape: {all_graphs_x.shape}")  # (num_graphs, 365, 2, 2)
    print(f"Labels shape: {all_graphs_y.shape}")  # (num_graphs,)

    return all_graphs_x, all_graphs_y


def prepare_multisite_data_from_array(sensor_data, labels, num_sites,
                                       num_sensors_per_site, seq_length):
    """
    Prepare data when you have arrays

    Args:
        sensor_data: (num_sites, total_days, num_sensors_per_site)
        labels: (num_sites, num_years)
    """
    print(f"Processing array data: {sensor_data.shape}")

    all_graphs_x = []
    all_graphs_y = []

    num_years = sensor_data.shape[1] // seq_length

    for site_id in range(num_sites):
        site_data = sensor_data[site_id]  # (total_days, num_sensors_per_site)
        site_labels = labels[site_id]  # (num_years,)

        for year_id in range(num_years):
            start_day = year_id * seq_length
            end_day = start_day + seq_length

            yearly_data = site_data[start_day:end_day, :]  # (365, 2)

            # Add features
            yearly_data_expanded = np.expand_dims(yearly_data, axis=-1)  # (365, 2, 1)

            time_feature = np.arange(seq_length) / seq_length
            time_feature = np.tile(time_feature[:, np.newaxis, np.newaxis],
                                   (1, num_sensors_per_site, 1))

            yearly_data_with_time = np.concatenate([yearly_data_expanded, time_feature],
                                                   axis=-1)  # (365, 2, 2)

            all_graphs_x.append(yearly_data_with_time)
            all_graphs_y.append(site_labels[year_id])

    all_graphs_x = np.array(all_graphs_x, dtype=np.float32)
    all_graphs_y = np.array(all_graphs_y, dtype=np.float32)

    print(f"Total graphs created: {len(all_graphs_x)}")
    print(f"Graph shape: {all_graphs_x.shape}")

    return all_graphs_x, all_graphs_y


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

    parser.add_argument('--sites_dir', type=str, default=None,
                        help='Directory containing site CSV files')
    parser.add_argument('--labels_file', type=str, default=None,
                        help='CSV file with labels')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    parser.add_argument('--num_sites', type=int, default=200,
                        help='Number of sites')
    parser.add_argument('--num_sensors_per_site', type=int, default=2,
                        help='Number of sensors per site')
    parser.add_argument('--num_years', type=int, default=3,
                        help='Number of years (for synthetic data)')
    parser.add_argument('--seq_length', type=int, default=365,
                        help='Days per year')

    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic data')

    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Task type')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Number of classes (for classification)')

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)

    args = parser.parse_args()

    # Validation
    if args.task == 'classification' and args.num_classes is None:
        parser.error("--num_classes is required for classification task")

    if args.synthetic:
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
    elif args.sites_dir and args.labels_file:
        print("Loading from CSV files...")
        x, y = prepare_multisite_data_from_csv(
            args.sites_dir, args.labels_file, args.output_dir,
            args.num_sites, args.num_sensors_per_site, args.seq_length
        )
    else:
        raise ValueError("Either use --synthetic or provide --sites_dir and --labels_file")

    # Split and save
    split_and_save(x, y, args.output_dir, args.train_ratio, args.val_ratio)

    print("\nData preparation complete!")
    print(f"Each graph has {args.num_sensors_per_site} nodes (sensors)")
    print(f"Total graphs: {len(x)} ({args.num_sites} sites × {args.num_years} years)")


if __name__ == "__main__":
    main()
