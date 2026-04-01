"""Dual ENAP utilities — feature extraction, normalisation, data loading."""

import torch
import torch.nn as nn
import numpy as np
import pickle


class NatureCNN(nn.Module):
    """Image + state encoder (identical to original — used at inference time)."""

    def __init__(self, sample_obs):
        super().__init__()
        extractors = {}
        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]

        cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = cnn(
                sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()
            ).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2) / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


def normalize_feature(feat, mean, std):
    """Apply Z-score normalization."""
    return (feat - mean) / (std + 1e-8)


def load_trajectories(pkl_path):
    """Load dual episodes, cluster centres, and normalization stats."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    episodes = payload["episodes"]
    cluster_centers = payload["cluster_centers"]
    action_norm_stats = payload["action_norm_stats"]
    feature_norm_stats = payload["feature_norm_stats"]

    print(f"Loading {len(episodes)} episodes")
    print(f"Loaded cluster_centers with shape {cluster_centers.shape}")
    return episodes, cluster_centers, action_norm_stats, feature_norm_stats
