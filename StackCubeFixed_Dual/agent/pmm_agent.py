"""Dual ENAP — ResidualMLP for feature prediction.

Original: attention over action_list using obs-delta query → predict action.
Dual:     attention over feature_list using action-delta query → predict feature.

Formula:
  1. prototype = attn_select(feature_list, query=actual_action - cluster_center)
  2. feature_raw = prototype + MLP(prototype, delta_action)
  3. clamp to feature_list AABB
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class ResidualMLP(nn.Module):
    """Attention-based prototype selection + residual refinement (dual)."""

    def __init__(self, f_dim, a_dim, hidden=1024, d_attn=32, tau=0.1):
        super().__init__()
        self.f_dim = f_dim
        self.a_dim = a_dim
        self.tau = tau

        self.query_proj = nn.Linear(a_dim, d_attn)
        self.key_proj = nn.Linear(f_dim, d_attn)

        self.net = nn.Sequential(
            nn.Linear(f_dim + a_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, f_dim),
        )
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.net[-1].bias)

        self.logstd = nn.Parameter(torch.ones(1, f_dim) * -0.5)

    def _select_prototype(self, feature_list, delta_action):
        """Straight-through attention: hard forward, soft gradient.

        Returns: prototype (B, f_dim), aabb_min (B, f_dim), aabb_max (B, f_dim)
        """
        B = len(feature_list)
        device = delta_action.device

        lengths = [np.asarray(f, dtype=np.float32).shape[0] for f in feature_list]
        max_len = max(lengths)
        padded = torch.zeros(B, max_len, self.f_dim, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i, f in enumerate(feature_list):
            arr = torch.as_tensor(np.asarray(f, dtype=np.float32), device=device)
            n = arr.shape[0]
            padded[i, :n] = arr
            mask[i, :n] = True

        mask_3d = mask.unsqueeze(-1).expand_as(padded)
        aabb_min = padded.masked_fill(~mask_3d, float('inf')).min(dim=1).values
        aabb_max = padded.masked_fill(~mask_3d, float('-inf')).max(dim=1).values

        q = self.query_proj(delta_action)
        k = self.key_proj(padded)
        scores = (q.unsqueeze(1) * k).sum(-1)
        scores = scores.masked_fill(~mask, -1e9)

        weights_soft = torch.softmax(scores / self.tau, dim=-1)
        idx_hard = scores.argmax(dim=-1)
        weights_hard = torch.nn.functional.one_hot(
            idx_hard, num_classes=max_len
        ).float()
        weights = weights_hard - weights_soft.detach() + weights_soft

        prototype = (weights.unsqueeze(-1) * padded).sum(dim=1)
        return prototype, aabb_min, aabb_max

    def forward(self, feature_list, action_cluster_center, actual_action):
        """
        feature_list         : list[B] of arrays, each (N_i, f_dim)
        action_cluster_center: (B, a_dim) — 8-D action cluster centre
        actual_action        : (B, a_dim) — 8-D actual (normalised) action
        Returns              : (B, f_dim)
        """
        delta = actual_action - action_cluster_center
        prototype, aabb_min, aabb_max = self._select_prototype(
            feature_list, delta
        )
        raw_feature = self.net(torch.cat([prototype, delta], dim=-1))  + prototype

        # return raw_feature
        return torch.clamp(raw_feature, aabb_min - 1e-5, aabb_max + 1e-5)

    def get_feature_mean(self, feature_list, action_cluster_center, actual_action):
        return self.forward(feature_list, action_cluster_center, actual_action)

    def get_log_prob(self, feature_list, action_cluster_center, actual_action,
                     feature_true):
        feature_mean = self.forward(
            feature_list, action_cluster_center, actual_action
        )
        logstd = torch.clamp(self.logstd, min=-2.0, max=2.0).expand_as(
            feature_mean
        )
        return Normal(feature_mean, torch.exp(logstd)).log_prob(
            feature_true
        ).sum(dim=-1)
