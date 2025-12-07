"""
Model Improvement Strategies for Anomaly Detection

Based on AUC=0.71, AP=0.44, try these approaches in order:
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# STRATEGY 1: CLASS BALANCING (Quick Win)
# ============================================================


def get_class_weights(train_labels):
    """Compute balanced class weights"""
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    return torch.FloatTensor(class_weights)


def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss - Better for imbalanced data
    Focuses on hard examples
    """
    ce_loss = nn.CrossEntropyLoss(reduction="none")(outputs, targets)
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


# ============================================================
# STRATEGY 2: REGULARIZATION & ARCHITECTURE TWEAKS
# ============================================================


class ImprovedAnomalyDetectionModel(nn.Module):
    """Enhanced version with better regularization"""

    def __init__(self, gwnet, aggregation_module, num_classes=2, dropout=0.5):
        super().__init__()
        self.gwnet = gwnet
        self.aggregation = aggregation_module

        feature_dim = 128  # From aggregation output

        # Stronger regularization
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Label smoothing
        self.label_smoothing = 0.1

    def forward(self, x):
        gwnet_out = self.gwnet(x)
        agg_out = self.aggregation(gwnet_out)
        graph_repr = agg_out["representation"]
        logits = self.classifier(graph_repr)
        return logits


# ============================================================
# STRATEGY 3: ENSEMBLE METHODS
# ============================================================


class EnsembleAnomalyDetector(nn.Module):
    """Ensemble multiple aggregation methods"""

    def __init__(self, gwnet, aggregations, num_classes=2):
        super().__init__()
        self.gwnet = gwnet
        self.aggregations = nn.ModuleList(aggregations)

        # Each aggregation outputs feature_dim (128)
        ensemble_dim = 128 * len(aggregations)

        self.fusion = nn.Sequential(
            nn.Linear(ensemble_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        gwnet_out = self.gwnet(x)

        # Get representations from all aggregation methods
        representations = []
        for agg in self.aggregations:
            agg_out = agg(gwnet_out)
            representations.append(agg_out["representation"])

        # Concatenate and fuse
        combined = torch.cat(representations, dim=1)
        logits = self.fusion(combined)
        return logits


# ============================================================
# STRATEGY 4: CURRICULUM LEARNING
# ============================================================


class CurriculumTrainer:
    """Train on easy examples first, then hard ones"""

    def __init__(self, model, train_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.sample_difficulties = None

    def compute_sample_difficulties(self):
        """
        Score each sample by loss
        Higher loss = harder sample
        """
        self.model.eval()
        difficulties = []

        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = nn.CrossEntropyLoss(reduction="none")(output, target)
                difficulties.extend(loss.cpu().numpy())

        self.sample_difficulties = np.array(difficulties)
        return self.sample_difficulties

    def get_curriculum_subset(self, difficulty_percentile):
        """
        Get subset of data up to difficulty_percentile
        E.g., 50 = easiest 50% of samples
        """
        threshold = np.percentile(self.sample_difficulties, difficulty_percentile)
        easy_indices = np.where(self.sample_difficulties <= threshold)[0]
        return easy_indices


# ============================================================
# STRATEGY 5: DATA AUGMENTATION
# ============================================================


class TemporalAugmentation:
    """Augment time series data"""

    @staticmethod
    def add_noise(x, noise_level=0.05):
        """Add Gaussian noise"""
        noise = torch.randn_like(x) * noise_level
        return x + noise

    @staticmethod
    def time_shift(x, max_shift=5):
        """Randomly shift in time"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return torch.cat([x[:, :, :, shift:], x[:, :, :, :shift]], dim=3)
        elif shift < 0:
            return torch.cat([x[:, :, :, shift:], x[:, :, :, :shift]], dim=3)
        return x

    @staticmethod
    def magnitude_warp(x, sigma=0.2):
        """Warp magnitude with smooth random curve"""
        batch, features, nodes, time = x.shape
        warp = 1 + torch.randn(batch, 1, 1, time) * sigma
        warp = torch.nn.functional.interpolate(
            warp, size=time, mode="linear", align_corners=True
        )
        return x * warp

    @staticmethod
    def augment(x, apply_noise=True, apply_shift=True, apply_warp=True):
        """Apply random augmentations"""
        if apply_noise and np.random.random() > 0.5:
            x = TemporalAugmentation.add_noise(x)
        if apply_shift and np.random.random() > 0.5:
            x = TemporalAugmentation.time_shift(x)
        if apply_warp and np.random.random() > 0.5:
            x = TemporalAugmentation.magnitude_warp(x)
        return x


# ============================================================
# STRATEGY 6: CONTRASTIVE PRE-TRAINING
# ============================================================


class ContrastivePretrainer:
    """
    Pre-train GWNet with contrastive learning
    Learn better representations before classification
    """

    def __init__(self, model, temperature=0.5):
        self.model = model
        self.temperature = temperature

    def contrastive_loss(self, z1, z2):
        """
        InfoNCE loss for contrastive learning
        z1, z2: (batch, feature_dim) - two augmented views
        """
        batch_size = z1.shape[0]

        # Normalize
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature

        # Positive pairs on diagonal
        labels = torch.arange(batch_size).to(z1.device)

        loss = nn.CrossEntropyLoss()(sim_matrix, labels)
        return loss

    def pretrain_step(self, x):
        """One pre-training step"""
        # Create two augmented views
        x1 = TemporalAugmentation.augment(x)
        x2 = TemporalAugmentation.augment(x)

        # Get representations
        out1 = self.model.gwnet(x1)
        out2 = self.model.gwnet(x2)

        z1 = self.model.aggregation(out1)["representation"]
        z2 = self.model.aggregation(out2)["representation"]

        return self.contrastive_loss(z1, z2)


# ============================================================
# RECOMMENDED TRAINING PIPELINE
# ============================================================


def recommended_training_pipeline():
    """
    Best practices based on your current results
    """
    print(
        """
RECOMMENDED IMPROVEMENT PIPELINE:
==================================

QUICK WINS (Try first - 1-2 hours):
1. ✓ Add class weights to loss function
2. ✓ Use Focal Loss instead of CrossEntropy
3. ✓ Increase dropout to 0.5-0.6
4. ✓ Add BatchNorm layers
5. ✓ Enable data augmentation

MEDIUM EFFORT (If quick wins don't work - 1 day):
6. ✓ Ensemble multiple aggregation methods
7. ✓ Implement curriculum learning
8. ✓ Try different attention configurations

ADVANCED (If data quality is poor - 2-3 days):
9. ✓ Contrastive pre-training
10. ✓ Semi-supervised learning
11. ✓ Active learning to fix labels

EXAMPLE TRAINING CODE:
======================

# 1. Quick win - Focal Loss + Class Weights
from improvement_strategies import focal_loss, get_class_weights

class_weights = get_class_weights(train_labels)
criterion = lambda outputs, targets: focal_loss(outputs, targets, alpha=0.25, gamma=2.0)

# 2. Data augmentation
from improvement_strategies import TemporalAugmentation

for data, target in train_loader:
    if np.random.random() > 0.5:  # 50% augmentation
        data = TemporalAugmentation.augment(data)
    
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

# 3. Ensemble (if single model plateaus)
from improvement_strategies import EnsembleAnomalyDetector
from graph_aggregation import (
    SpatialTemporalAttention,
    HybridGraphAttention,
    SimpleGraphAggregation
)

aggregations = [
    SpatialTemporalAttention(num_nodes=3, out_dim=365, temporal_length=51),
    HybridGraphAttention(num_nodes=3, out_dim=365, temporal_length=51),
    SimpleGraphAggregation(out_dim=365, pooling_type='mean')
]

ensemble_model = EnsembleAnomalyDetector(gwnet, aggregations)

EXPECTED IMPROVEMENTS:
======================
- Focal Loss + Class Weights: +0.05-0.10 AUC
- Data Augmentation: +0.02-0.05 AUC
- Ensemble: +0.03-0.08 AUC
- Contrastive Pre-training: +0.05-0.15 AUC (if data quality is poor)

Target: AUC 0.80-0.85, AP 0.55-0.65
    """
    )


if __name__ == "__main__":
    recommended_training_pipeline()
