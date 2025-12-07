"""
Utility functions for attention visualization and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_attention_weights(spatial_attn, temporal_attn, sample_idx=0, 
                                 save_path=None, node_names=None):
    """
    Visualize spatial and temporal attention weights
    
    Args:
        spatial_attn: (batch, num_nodes, num_nodes) or (batch, num_heads, num_nodes, num_nodes)
        temporal_attn: (batch, seq_length, seq_length) or (batch, num_heads, seq_length, seq_length)
        sample_idx: Which sample to visualize
        save_path: Path to save figure
        node_names: Optional list of node names for axis labels
    """
    # Handle multi-head attention (average over heads)
    if spatial_attn.dim() == 4:
        spatial_attn = spatial_attn.mean(dim=1)
    if temporal_attn.dim() == 4:
        temporal_attn = temporal_attn.mean(dim=1)
    
    # Extract single sample
    spatial_weights = spatial_attn[sample_idx].detach().cpu().numpy()
    temporal_weights = temporal_attn[sample_idx].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ================================================
    # Spatial Attention Heatmap
    # ================================================
    sns.heatmap(spatial_weights, cmap='YlOrRd', ax=axes[0], 
                cbar_kws={'label': 'Attention Weight'})
    axes[0].set_xlabel('Node (Key)')
    axes[0].set_ylabel('Node (Query)')
    axes[0].set_title('Spatial Attention: Node-to-Node Relationships')
    
    # ================================================
    # Temporal Attention Heatmap
    # ================================================
    sns.heatmap(temporal_weights, cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': 'Attention Weight'})
    axes[1].set_xlabel('Timestep (Key)')
    axes[1].set_ylabel('Timestep (Query)')
    axes[1].set_title('Temporal Attention: Time-to-Time Relationships')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("Attention Statistics")
    print("="*60)
    print(f"Spatial attention range: [{spatial_weights.min():.4f}, {spatial_weights.max():.4f}]")
    print(f"Temporal attention range: [{temporal_weights.min():.4f}, {temporal_weights.max():.4f}]")
    print(f"Spatial attention sparsity: {(spatial_weights < 0.01).mean()*100:.2f}% near zero")
    print(f"Temporal attention sparsity: {(temporal_weights < 0.01).mean()*100:.2f}% near zero")
    print("="*60)


def visualize_pooling_attention(spatial_pool_weights, temporal_pool_weights, 
                                 sample_idx=0, save_path=None):
    """
    Visualize additive attention pooling weights
    
    Args:
        spatial_pool_weights: (batch, num_nodes) - importance of each node
        temporal_pool_weights: (batch, seq_length) - importance of each timestep
        sample_idx: Which sample to visualize
        save_path: Path to save figure
    """
    spatial_weights = spatial_pool_weights[sample_idx].detach().cpu().numpy()
    temporal_weights = temporal_pool_weights[sample_idx].detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # ================================================
    # Spatial Pooling Weights
    # ================================================
    axes[0].bar(range(len(spatial_weights)), spatial_weights, color='steelblue')
    axes[0].set_xlabel('Node ID')
    axes[0].set_ylabel('Attention Weight')
    axes[0].set_title('Spatial Pooling: Node Importance')
    axes[0].axhline(y=1.0/len(spatial_weights), color='r', linestyle='--', 
                    label='Uniform (no attention)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ================================================
    # Temporal Pooling Weights
    # ================================================
    axes[1].plot(range(len(temporal_weights)), temporal_weights, 
                 marker='o', linewidth=2, markersize=8, color='darkgreen')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('Temporal Pooling: Timestep Importance')
    axes[1].axhline(y=1.0/len(temporal_weights), color='r', linestyle='--',
                    label='Uniform (no attention)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Find top-k important nodes/timesteps
    top_k = 5
    top_nodes = np.argsort(spatial_weights)[-top_k:][::-1]
    top_times = np.argsort(temporal_weights)[-top_k:][::-1]
    
    print("\n" + "="*60)
    print(f"Top {top_k} Important Nodes (for this sample):")
    print("="*60)
    for i, node_idx in enumerate(top_nodes):
        print(f"  {i+1}. Node {node_idx:3d} - Weight: {spatial_weights[node_idx]:.4f}")
    
    print("\n" + "="*60)
    print(f"Top {top_k} Important Timesteps (for this sample):")
    print("="*60)
    for i, time_idx in enumerate(top_times):
        print(f"  {i+1}. Timestep {time_idx:2d} - Weight: {temporal_weights[time_idx]:.4f}")
    print("="*60)


def analyze_attention_diversity(attention_weights, attention_type='spatial'):
    """
    Analyze diversity of attention patterns
    
    Args:
        attention_weights: (batch, seq, seq) attention matrix
        attention_type: 'spatial' or 'temporal'
    
    Returns:
        dict with diversity metrics
    """
    # Handle multi-head (average first)
    if attention_weights.dim() == 4:
        attention_weights = attention_weights.mean(dim=1)
    
    weights = attention_weights.detach().cpu().numpy()
    batch_size = weights.shape[0]
    
    # Compute entropy (higher = more diverse attention)
    entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1).mean()
    
    # Compute sparsity (percentage of weights < threshold)
    sparsity = (weights < 0.01).mean()
    
    # Compute max attention (how much focuses on single item)
    max_attention = weights.max(axis=-1).mean()
    
    metrics = {
        'entropy': entropy,
        'sparsity': sparsity,
        'max_attention': max_attention,
        'mean_weight': weights.mean(),
        'std_weight': weights.std()
    }
    
    print(f"\n{attention_type.upper()} Attention Analysis:")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("="*60)
    
    return metrics


def compare_attention_methods(models_dict, data_loader, device):
    """
    Compare different attention mechanisms
    
    Args:
        models_dict: Dict of {model_name: model}
        data_loader: DataLoader for evaluation
        device: torch.device
    
    Returns:
        Comparison results
    """
    results = {}
    
    for model_name, model in models_dict.items():
        model.eval()
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(device)
                
                # Forward pass
                output = model(x, return_attention_weights=True)
                
                # Analyze attention if available
                if 'spatial_attention' in output:
                    spatial_metrics = analyze_attention_diversity(
                        output['spatial_attention'], 'spatial'
                    )
                    temporal_metrics = analyze_attention_diversity(
                        output['temporal_attention'], 'temporal'
                    )
                    
                    results[model_name] = {
                        'spatial': spatial_metrics,
                        'temporal': temporal_metrics
                    }
                
                break  # Only use first batch for comparison
    
    return results


def plot_attention_comparison(results_dict, save_path=None):
    """
    Plot comparison of different attention mechanisms
    
    Args:
        results_dict: Dict from compare_attention_methods
        save_path: Path to save figure
    """
    metrics = ['entropy', 'sparsity', 'max_attention']
    model_names = list(results_dict.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, metric in enumerate(metrics):
        spatial_values = [results_dict[m]['spatial'][metric] for m in model_names]
        temporal_values = [results_dict[m]['temporal'][metric] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[i].bar(x - width/2, spatial_values, width, label='Spatial', alpha=0.8)
        axes[i].bar(x + width/2, temporal_values, width, label='Temporal', alpha=0.8)
        
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
