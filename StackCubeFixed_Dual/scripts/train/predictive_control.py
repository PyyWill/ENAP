"""Predictive control: chain original PMM (feat→act) with dual PMM (act→feat).

Autoregressive rollout:
  f_0 (real) → pmm1 → a_0 → pmm2 → f_1 → pmm1 → a_1 → pmm2 → ... → f_{N-1}

Run from StackCubeFixed_Dual/:
  python scripts/train/predictive_control.py
"""

import os
import sys
import pickle
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DUAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ORIG_ROOT = os.path.abspath(os.path.join(DUAL_ROOT, "..", "StackCubeFixed"))

ORIG_PMM_PKL = os.path.join(ORIG_ROOT, "results", "pmm_residual_training.pkl")
ORIG_RESID_PT = os.path.join(ORIG_ROOT, "results", "checkpoints", "residual_net.pt")
ORIG_DATA_PKL = os.path.join(ORIG_ROOT, "data", "episodes_with_states.pkl")

DUAL_PMM_PKL = os.path.join(DUAL_ROOT, "results", "pmm_residual_training.pkl")
DUAL_RESID_PT = os.path.join(DUAL_ROOT, "results", "checkpoints", "residual_net.pt")
DUAL_DATA_PKL = os.path.join(DUAL_ROOT, "data", "episodes_with_states.pkl")

PLOT_DIR = os.path.join(DUAL_ROOT, "results", "plot", "predictive_control")

# ---------------------------------------------------------------------------
# Import helpers — load classes from specific file paths to avoid collisions
# ---------------------------------------------------------------------------
def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Dual imports (add to sys.path so agent.* resolves to Dual's)
if DUAL_ROOT not in sys.path:
    sys.path.insert(0, DUAL_ROOT)

from agent.pmm_class import PMM as DualPMM
from agent.pmm_agent import ResidualMLP as DualResidualMLP

# Original imports via importlib (reuses Dual's agent.utils for NatureCNN — identical)
_orig_pmm_mod = _import_module("orig_pmm_class", os.path.join(ORIG_ROOT, "agent", "pmm_class.py"))
_orig_agent_mod = _import_module("orig_pmm_agent", os.path.join(ORIG_ROOT, "agent", "pmm_agent.py"))
OrigPMM = _orig_pmm_mod.PMM
OrigResidualMLP = _orig_agent_mod.ResidualMLP


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
FEATURE_CLAMP = 5.0


def normalize(feat, mean, std):
    return (feat - mean) / (std + 1e-8)


def denormalize(feat, mean, std):
    return feat * (std + 1e-8) + mean


def orig_to_dual(feat_orig, o_stats, d_stats, rgb_dim):
    """orig Z-score → raw → dual Z-score + clamp ±5."""
    raw_rgb = denormalize(feat_orig[:rgb_dim], o_stats['rgb_mean'], o_stats['rgb_std'])
    raw_pos = denormalize(feat_orig[rgb_dim:], o_stats['pos_mean'], o_stats['pos_std'])
    d_rgb = np.clip(normalize(raw_rgb, d_stats['rgb_mean'], d_stats['rgb_std']),
                    -FEATURE_CLAMP, FEATURE_CLAMP)
    d_pos = np.clip(normalize(raw_pos, d_stats['pos_mean'], d_stats['pos_std']),
                    -FEATURE_CLAMP, FEATURE_CLAMP)
    return np.concatenate([d_rgb, d_pos]).astype(np.float32)


def dual_to_orig(feat_dual, d_stats, o_stats, rgb_dim):
    """dual Z-score → raw → orig Z-score (no clamp)."""
    raw_rgb = denormalize(feat_dual[:rgb_dim], d_stats['rgb_mean'], d_stats['rgb_std'])
    raw_pos = denormalize(feat_dual[rgb_dim:], d_stats['pos_mean'], d_stats['pos_std'])
    o_rgb = normalize(raw_rgb, o_stats['rgb_mean'], o_stats['rgb_std'])
    o_pos = normalize(raw_pos, o_stats['pos_mean'], o_stats['pos_std'])
    return np.concatenate([o_rgb, o_pos]).astype(np.float32)


def find_valid_state_idx(current_q, state_idx, pmm_delta, cluster_centers, query_vec):
    """Return state_idx if (current_q, state_idx) is valid; otherwise fall back
    to the nearest valid state_idx by distance to cluster_centers."""
    if (current_q, state_idx) in pmm_delta:
        return state_idx
    valid = [s for (q, s) in pmm_delta if q == current_q]
    if not valid:
        return state_idx
    dists = [np.linalg.norm(query_vec - cluster_centers[s]) for s in valid]
    return valid[int(np.argmin(dists))]


def disambiguate(pmm_obj, current_q, x, h):
    """Transition: single edge → follow; multiple → cosine with rep_embeddings."""
    dist = pmm_obj.pmm['delta'].get((current_q, x))
    if dist is None:
        return current_q
    if len(dist) == 1:
        return next(iter(dist))
    rep_emb = pmm_obj.pmm.get('rep_embeddings', [])
    best_q, best_sim = current_q, -2.0
    h_norm = h / (np.linalg.norm(h) + 1e-12)
    for q_cand in dist:
        if q_cand < len(rep_emb):
            r = rep_emb[q_cand]
            sim = np.dot(h_norm, r / (np.linalg.norm(r) + 1e-12))
            if sim > best_sim:
                best_sim, best_q = sim, q_cand
    return best_q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    device = "cpu"

    # ── 1. Load PMMs ─────────────────────────────────────────────────
    pmm1 = OrigPMM()
    pmm1.load_pmm(ORIG_PMM_PKL)

    pmm2 = DualPMM()
    pmm2.load_pmm(DUAL_PMM_PKL)

    # ── 2. Load Residual networks & norm stats ─────────────────────────
    ckpt1 = torch.load(ORIG_RESID_PT, map_location=device, weights_only=False)
    o_stats = ckpt1['norm_stats']
    rgb_dim = len(o_stats['rgb_mean'])
    feat_dim = rgb_dim + len(o_stats['pos_mean'])
    a_dim = pmm1.a_dim
    resid1 = OrigResidualMLP(feat_dim, a_dim, hidden=256).to(device)
    resid1.load_state_dict(ckpt1['model'])
    resid1.eval()

    ckpt2 = torch.load(DUAL_RESID_PT, map_location=device, weights_only=False)
    a_norm_stats = ckpt2['action_norm_stats']
    a_mean, a_std = a_norm_stats['mean'], a_norm_stats['std']
    resid2 = DualResidualMLP(feat_dim, a_dim, hidden=1024).to(device)
    resid2.load_state_dict(ckpt2['model'])
    resid2.eval()

    with open(DUAL_DATA_PKL, "rb") as f:
        dual_payload = pickle.load(f)
    d_stats = dual_payload["feature_norm_stats"]

    # ── 3. Load episode 0 — build GT in both spaces ──────────────────
    with open(ORIG_DATA_PKL, "rb") as f:
        orig_payload = pickle.load(f)
    episode = orig_payload["episodes"][0]
    N = len(episode)

    gt_feats_orig = []   # orig Z-score (no clamp)  — pmm1's space
    gt_feats_dual = []   # dual Z-score (clamp ±5)  — pmm2's space
    gt_actions = []      # raw actions
    for step in episode:
        rgb_raw = step['rgb_feature'].reshape(-1)
        pos_raw = step['pos_feature'].reshape(-1)
        o_rgb = normalize(rgb_raw, o_stats['rgb_mean'], o_stats['rgb_std'])
        o_pos = normalize(pos_raw, o_stats['pos_mean'], o_stats['pos_std'])
        d_rgb = np.clip(normalize(rgb_raw, d_stats['rgb_mean'], d_stats['rgb_std']),
                        -FEATURE_CLAMP, FEATURE_CLAMP)
        d_pos = np.clip(normalize(pos_raw, d_stats['pos_mean'], d_stats['pos_std']),
                        -FEATURE_CLAMP, FEATURE_CLAMP)
        gt_feats_orig.append(np.concatenate([o_rgb, o_pos]).astype(np.float32))
        gt_feats_dual.append(np.concatenate([d_rgb, d_pos]).astype(np.float32))
        gt_actions.append(step['action'].reshape(-1).astype(np.float32))
    gt_feats_orig = np.array(gt_feats_orig)  # (N, feat_dim)
    gt_feats_dual = np.array(gt_feats_dual)  # (N, feat_dim)
    gt_actions = np.array(gt_actions)         # (N, a_dim)

    pmm1_cc = pmm1.cluster_centers   # in orig space
    pmm2_cc = pmm2.cluster_centers   # in normalised action space

    # ── 4. Initialise PMM states ─────────────────────────────────────
    q1 = 0
    h1 = np.zeros(pmm1.rnn_hidden, dtype=np.float32)

    q2 = 0
    h2 = np.zeros(pmm2.rnn_hidden, dtype=np.float32)
    h2 = pmm2.rnn.forward_step(gt_feats_dual[0], 0, h2)
    q2 = disambiguate(pmm2, q2, 0, h2)

    # ── 5. Autoregressive loop ───────────────────────────────────────
    pred_actions = []
    pred_feats_dual = []
    current_feat_orig = gt_feats_orig[0]

    with torch.no_grad():
        for k in range(N - 1):
            # ── pmm1: feature (orig space) → action (raw) ────────
            obs_state_idx = int(np.argmin(
                np.linalg.norm(pmm1_cc - current_feat_orig, axis=1)))
            obs_state_idx = find_valid_state_idx(
                q1, obs_state_idx, pmm1.pmm['delta'],
                pmm1_cc, current_feat_orig)

            action_list = [pmm1.predict_list(q1, obs_state_idx)]
            cc1_t = torch.from_numpy(
                pmm1_cc[obs_state_idx]).float().unsqueeze(0)
            feat_t = torch.from_numpy(
                current_feat_orig).float().unsqueeze(0)
            action_pred = resid1.get_action_mean(
                action_list, cc1_t, feat_t).numpy()[0]
            pred_actions.append(action_pred)

            h1 = pmm1.rnn.forward_step(action_pred, obs_state_idx, h1)
            q1 = disambiguate(pmm1, q1, obs_state_idx, h1)

            # ── pmm2: action (norm) → feature (dual space) ───────
            action_norm = normalize(
                action_pred, a_mean, a_std).astype(np.float32)
            act_state_idx = int(np.argmin(
                np.linalg.norm(pmm2_cc - action_norm, axis=1)))
            act_state_idx = find_valid_state_idx(
                q2, act_state_idx, pmm2.pmm['delta'],
                pmm2_cc, action_norm)

            feat_list = [pmm2.predict_list(q2, act_state_idx)]
            cc2_t = torch.from_numpy(
                pmm2_cc[act_state_idx]).float().unsqueeze(0)
            act_t = torch.from_numpy(action_norm).float().unsqueeze(0)
            feat_pred_dual = resid2.get_feature_mean(
                feat_list, cc2_t, act_t).numpy()[0]
            pred_feats_dual.append(feat_pred_dual)

            h2 = pmm2.rnn.forward_step(feat_pred_dual, act_state_idx, h2)
            q2 = disambiguate(pmm2, q2, act_state_idx, h2)

            current_feat_orig = dual_to_orig(
                feat_pred_dual, d_stats, o_stats, rgb_dim)

    pred_actions = np.array(pred_actions)
    pred_feats_dual = np.array(pred_feats_dual)

    gt_actions_cmp = gt_actions[:N - 1]
    gt_fd_cmp = gt_feats_dual[1:]

    l2_dual = np.linalg.norm(gt_fd_cmp - pred_feats_dual, axis=1)
    cos_dual = (np.sum(gt_fd_cmp * pred_feats_dual, axis=1)
                / (np.linalg.norm(gt_fd_cmp, axis=1)
                   * np.linalg.norm(pred_feats_dual, axis=1) + 1e-12))

    # ── 6. Plots (all comparisons in DUAL space) ─────────────────────
    ts = np.arange(N - 1)

    # 6a. Feature quantitative: L2 + cosine (dual space)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(ts, l2_dual, 'o-', ms=3, lw=1.5, color='tab:red')
    axes[0].set_ylabel('L2 Error')
    axes[0].set_title('Predicted vs GT Feature (dual space) — L2 Error')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ts, cos_dual, 's-', ms=3, lw=1.5, color='tab:blue')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_xlabel('Timestep')
    axes[1].set_title('Predicted vs GT Feature (dual space) — Cosine Similarity')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_quantitative.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 6b. Feature qualitative: heatmap in dual space
    diff = np.abs(gt_fd_cmp - pred_feats_dual)
    vmin = min(gt_fd_cmp.min(), pred_feats_dual.min())
    vmax = max(gt_fd_cmp.max(), pred_feats_dual.max())

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    im0 = axes[0].imshow(gt_fd_cmp.T, aspect='auto', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[0].set_ylabel('Feature Dim')
    axes[0].set_title('GT Feature (dual space, clamped ±5)')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(pred_feats_dual.T, aspect='auto', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[1].set_ylabel('Feature Dim')
    axes[1].set_title('Predicted Feature (dual space)')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    im2 = axes[2].imshow(diff.T, aspect='auto', cmap='hot')
    axes[2].set_ylabel('Feature Dim'); axes[2].set_xlabel('Timestep')
    axes[2].set_title('|Difference|')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_qualitative.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 6c. Action comparison: PCA 1-D
    pca = PCA(n_components=1, random_state=42)
    combined = np.vstack([gt_actions_cmp, pred_actions])
    pca.fit(combined)

    gt_1d = pca.transform(gt_actions_cmp).flatten()
    pred_1d = pca.transform(pred_actions).flatten()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, gt_1d, 'o-', label='GT', ms=3, lw=1.5)
    ax.plot(ts, pred_1d, '^-', label='Predicted', ms=3, lw=1.5, alpha=0.8)
    ax.set_xlabel('Timestep'); ax.set_ylabel('PCA Component 1')
    ax.set_title('Predicted vs GT Action (PCA 1-D)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "action_comparison.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
