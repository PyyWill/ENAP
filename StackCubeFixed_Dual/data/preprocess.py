#!/usr/bin/env python3
"""Dual ENAP preprocessing: cluster actions (instead of observations).

Shifts the (f_t, a_t) pairing to (a_t, f_{t+1}), adding:
  - a dummy action a_{-1} at the start of each episode
  - a dummy feature f_{T+1} at the end of each episode
Then clusters real actions via Z-score + HDBSCAN to produce discrete
state symbols m (action modes).
"""

import os
import argparse
import pickle
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLUSTERING_DIR = os.path.join(SCRIPT_DIR, "clustering")

DUMMY_ACTION_LABEL = -2


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_dual_episodes(pkl_path):
    """Load trajectories and build dual (a_t, f_{t+1}) episodes.

    For an original episode with T steps [s_0, ..., s_{T-1}]:
      dual_step_0:   dummy action  +  f_0        (from s_0)
      dual_step_t:   a_{t-1}       +  f_t        (t = 1..T-1)
      dual_step_T:   a_{T-1}       +  dummy feat

    Returns (all_real_actions, dual_episodes, metadata).
    """
    data = load_pickle(pkl_path)
    assert isinstance(data, dict), f"Expected dict pkl, got {type(data)}"
    episodes = data["episodes"]
    metadata = data.get("metadata", None)

    all_real_actions = []
    all_rgb_feats, all_pos_feats = [], []
    sample_rgb_img = None

    for ep in episodes:
        for step in ep:
            act = step["action"]
            if act.ndim > 1:
                act = act.reshape(-1)
            all_real_actions.append(act)

            rf = step.get("rgb_feature")
            if rf is not None:
                all_rgb_feats.append(rf.reshape(-1))
            pf = step.get("pos_feature")
            if pf is not None:
                all_pos_feats.append(pf.reshape(-1))
            if sample_rgb_img is None and "rgb" in step:
                sample_rgb_img = step["rgb"]

    all_real_actions = np.array(all_real_actions)
    all_rgb_feats = np.array(all_rgb_feats)
    all_pos_feats = np.array(all_pos_feats)

    act_mean = all_real_actions.mean(axis=0)
    act_std = all_real_actions.std(axis=0) + 1e-8
    dummy_action = act_mean + 10 * act_std

    dummy_rgb_feature = all_rgb_feats.mean(0) + 10 * (all_rgb_feats.std(0) + 1e-8)
    dummy_pos_feature = all_pos_feats.mean(0) + 10 * (all_pos_feats.std(0) + 1e-8)
    dummy_rgb = (np.full_like(sample_rgb_img, 128, dtype=np.uint8)
                 if sample_rgb_img is not None else None)

    def _rgb(step):
        img = step.get("rgb")
        if img is not None and img.ndim == 3 and img.shape[2] == 6:
            img = img[:, :, :3]
        return img

    dual_episodes = []
    for episode in tqdm(episodes, desc="Building dual episodes"):
        T = len(episode)
        if T == 0:
            continue
        dual_ep = []

        dual_ep.append({
            "action": dummy_action.copy(),
            "rgb_feature": episode[0].get("rgb_feature"),
            "pos_feature": episode[0].get("pos_feature"),
            "rgb": _rgb(episode[0]),
            "is_dummy_action": True,
            "is_dummy_feature": False,
        })

        for t in range(1, T):
            act = episode[t - 1]["action"]
            if act.ndim > 1:
                act = act.reshape(-1)
            dual_ep.append({
                "action": act,
                "rgb_feature": episode[t].get("rgb_feature"),
                "pos_feature": episode[t].get("pos_feature"),
                "rgb": _rgb(episode[t]),
                "is_dummy_action": False,
                "is_dummy_feature": False,
            })

        act_last = episode[T - 1]["action"]
        if act_last.ndim > 1:
            act_last = act_last.reshape(-1)
        dual_ep.append({
            "action": act_last,
            "rgb_feature": dummy_rgb_feature.copy(),
            "pos_feature": dummy_pos_feature.copy(),
            "rgb": dummy_rgb.copy() if dummy_rgb is not None else None,
            "is_dummy_action": False,
            "is_dummy_feature": True,
        })

        dual_episodes.append(dual_ep)

    # Per-modality Z-score stats (for combined_feature and inverse transform)
    rgb_mean = all_rgb_feats.mean(axis=0)
    rgb_std = all_rgb_feats.std(axis=0) + 1e-8
    pos_mean = all_pos_feats.mean(axis=0)
    pos_std = all_pos_feats.std(axis=0) + 1e-8
    feat_norm_stats = {
        "rgb_mean": rgb_mean, "rgb_std": rgb_std,
        "pos_mean": pos_mean, "pos_std": pos_std,
    }

    return all_real_actions, dual_episodes, metadata, feat_norm_stats


def hdbscan_cluster(X):
    n = X.shape[0]
    min_cluster_size = max(10, int(round(0.06 * n)))
    min_samples = max(2, int(round(0.3 * min_cluster_size)))
    print(f"HDBSCAN: n={n}, min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.05,
        metric="euclidean",
    )
    return clusterer.fit_predict(X)


# ── visualisation helpers ────────────────────────────────────────────

def _build_colormap(labels):
    """Unified label → RGBA colour mapping used by all plot functions."""
    uniq = np.unique(labels)
    cluster_labs = sorted(uniq[uniq != -1])
    n = len(cluster_labs)
    if n <= 10:
        pal = np.vstack([plt.cm.Set2(np.linspace(0, 1, 8)),
                         plt.cm.Dark2(np.linspace(0, 1, 8))])[:n]
    elif n <= 20:
        pal = np.vstack([plt.cm.Set2(np.linspace(0, 1, 8)),
                         plt.cm.Dark2(np.linspace(0, 1, 8)),
                         plt.cm.tab10(np.linspace(0, 1, 10))])[:n]
    else:
        pal = plt.cm.turbo(np.linspace(0.1, 0.9, n))
    cmap = {lab: pal[i] for i, lab in enumerate(cluster_labs)}
    cmap[-1] = np.array([0.4, 0.4, 0.4, 1.0])
    return cmap


def _scatter_ax(ax, Y, labels, cmap, skip_noise=False):
    """Scatter clusters on a single axis using the shared colormap."""
    for lab in sorted(cmap):
        if skip_noise and lab == -1:
            continue
        mask = labels == lab
        if not mask.any():
            continue
        if lab == -1:
            ax.scatter(Y[mask, 0], Y[mask, 1], s=8, color=cmap[-1],
                       label=f"noise (-1, n={mask.sum()})", alpha=0.1,
                       edgecolors="none", marker="x")
        else:
            ax.scatter(Y[mask, 0], Y[mask, 1], s=15, color=cmap[lab],
                       label=f"m{lab} (n={mask.sum()})", alpha=0.9,
                       edgecolors="gray", linewidths=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, ncol=2, frameon=False,
              loc="upper right")


def visualize_clusters(actions_norm, labels, cmap, out_png):
    """PCA-2D scatter plot of action clusters."""
    Y = PCA(n_components=2, random_state=0).fit_transform(actions_norm)
    n_clusters = sum(1 for l in cmap if l >= 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), dpi=140)
    _scatter_ax(ax1, Y, labels, cmap)
    ax1.set_title(f"All action clusters (PCA 2D) — {n_clusters} clusters",
                  fontsize=12)
    _scatter_ax(ax2, Y, labels, cmap, skip_noise=True)
    ax2.set_title("Action clusters only (PCA 2D)", fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def save_cluster_arrows(actions_norm, labels, cmap, out_dir, n_samples=16):
    """Per-cluster arrow plot: 8-D action → PCA 2-D vector from origin."""
    Y = PCA(n_components=2, random_state=0).fit_transform(actions_norm)

    for lab in tqdm(sorted(cmap), desc="Saving cluster arrow plots"):
        mask = labels == lab
        if not mask.any():
            continue
        Yc = Y[mask]
        nc = len(Yc)
        ns = min(n_samples, nc)
        sel = (np.arange(nc) if ns == nc
               else np.linspace(0, nc - 1, ns, dtype=int))

        fig, ax = plt.subplots(figsize=(8, 8))
        color = cmap[lab][:3]
        max_r = 0.0
        for idx in sel:
            dx, dy = Yc[idx]
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.8, alpha=0.7))
            max_r = max(max_r, np.hypot(dx, dy))

        lim = max(max_r * 1.3, 0.1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.3)
        ax.grid(True, alpha=0.2)

        name = f"m{lab}" if lab != -1 else "noise"
        ax.set_title(f"Cluster {name} — {ns} action arrows (total n={nc})",
                     fontsize=12)
        ax.set_xlabel("PCA dim 1"); ax.set_ylabel("PCA dim 2")

        fname = (f"cluster_{lab}_arrows.png" if lab != -1
                 else "cluster_noise_arrows.png")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()


# ── cluster centres & state labelling ────────────────────────────────

def compute_cluster_centers(actions_norm, labels):
    unique_labels = sorted(np.unique(labels))
    centers = [actions_norm[labels == lab].mean(axis=0) for lab in unique_labels]
    return np.array(centers), unique_labels


def add_states_to_dual_episodes(dual_episodes, real_labels, label_to_idx,
                                num_states):
    """Add one-hot state (action-cluster label) to every dual step."""
    dummy_idx = label_to_idx[DUMMY_ACTION_LABEL]
    real_ptr = 0
    for dual_ep in tqdm(dual_episodes, desc="Adding states"):
        for step in dual_ep:
            state = np.zeros(num_states, dtype=np.float32)
            if step.get("is_dummy_action", False):
                state[dummy_idx] = 1.0
            else:
                state[label_to_idx[int(real_labels[real_ptr])]] = 1.0
                real_ptr += 1
            step["state"] = state
    assert real_ptr == len(real_labels), \
        f"Mismatch: consumed {real_ptr} labels but have {len(real_labels)}"
    return dual_episodes


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Dual ENAP preprocessing (action clustering)")
    ap.add_argument("--pkl", type=str, default="trajectories.pkl")
    ap.add_argument("--n-samples", type=int, default=16,
                    help="Number of sample arrows per cluster image")
    ap.add_argument("--output-pkl", type=str, default="",
                    help="Output pkl path "
                         "(default: <script_dir>/episodes_with_states.pkl)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(CLUSTERING_DIR, exist_ok=True)

    # 1. Build dual episodes ──────────────────────────────────────────
    real_actions, dual_episodes, metadata, feat_norm_stats = \
        build_dual_episodes(args.pkl)
    a_dim = real_actions.shape[1]
    print(f"Real actions: {real_actions.shape}  (action_dim={a_dim})")
    print(f"Dual episodes: {len(dual_episodes)}  "
          f"(steps in first: {len(dual_episodes[0])})")

    # 2. Z-score normalise ────────────────────────────────────────────
    act_mean = real_actions.mean(axis=0)
    act_std = real_actions.std(axis=0) + 1e-8
    actions_norm = (real_actions - act_mean) / act_std

    # 3. HDBSCAN on normalised 8-D actions ────────────────────────────
    real_labels = hdbscan_cluster(actions_norm)
    np.save(os.path.join(CLUSTERING_DIR, "action_cluster_labels.npy"),
            real_labels)

    uniq_real = np.unique(real_labels)
    n_clusters = int((uniq_real >= 0).sum())
    has_noise = -1 in uniq_real

    all_unique = sorted(set(uniq_real.tolist()) | {DUMMY_ACTION_LABEL})
    label_to_idx = {lab: idx for idx, lab in enumerate(all_unique)}
    num_states = len(all_unique)

    print(f"\nClustering result: {len(real_labels)} actions → "
          f"{n_clusters} clusters")
    print(f"Total states (incl. noise + dummy): {num_states}")
    for lab in sorted(uniq_real):
        cnt = int((real_labels == lab).sum())
        name = "Noise (-1)" if lab == -1 else f"Cluster m{lab}"
        print(f"  {name}: {cnt}")
    print(f"  Dummy action (m_start): 1 per episode × "
          f"{len(dual_episodes)} episodes")

    # 4. Visualise ────────────────────────────────────────────────────
    cmap = _build_colormap(real_labels)
    visualize_clusters(actions_norm, real_labels, cmap,
                       os.path.join(CLUSTERING_DIR,
                                    "action_clusters_hdbscan.png"))
    save_cluster_arrows(actions_norm, real_labels, cmap, CLUSTERING_DIR,
                        n_samples=args.n_samples)

    # 5. Cluster centres (normalised action space) ────────────────────
    centers_real, center_labels_real = compute_cluster_centers(
        actions_norm, real_labels)

    dummy_center = np.full(a_dim, 10.0, dtype=np.float32)
    all_center_labels = [DUMMY_ACTION_LABEL] + list(center_labels_real)
    cluster_centers = np.vstack([dummy_center[None], centers_real])

    print(f"Cluster centres shape: {cluster_centers.shape}")

    # 6. Add state one-hots ───────────────────────────────────────────
    add_states_to_dual_episodes(dual_episodes, real_labels,
                                label_to_idx, num_states)

    # 7. Add combined_feature (per-modality Z-score → clamp → concat) ─
    FEATURE_CLAMP = 5.0
    rm, rs = feat_norm_stats["rgb_mean"], feat_norm_stats["rgb_std"]
    pm, ps = feat_norm_stats["pos_mean"], feat_norm_stats["pos_std"]
    for dual_ep in tqdm(dual_episodes, desc="Adding combined features"):
        for step in dual_ep:
            rgb_norm = np.clip((step["rgb_feature"].reshape(-1) - rm) / rs,
                               -FEATURE_CLAMP, FEATURE_CLAMP)
            pos_norm = np.clip((step["pos_feature"].reshape(-1) - pm) / ps,
                               -FEATURE_CLAMP, FEATURE_CLAMP)
            step["combined_feature"] = np.concatenate(
                [rgb_norm, pos_norm]).astype(np.float32)
    cf_dim = dual_episodes[0][0]["combined_feature"].shape[0]
    print(f"Added combined_feature (dim={cf_dim}, clamp=±{FEATURE_CLAMP}) "
          f"to all steps")

    # 8. Save ─────────────────────────────────────────────────────────
    output_pkl = (args.output_pkl
                  or os.path.join(SCRIPT_DIR, "episodes_with_states.pkl"))
    output_data = {
        "episodes": dual_episodes,
        "num_clusters": n_clusters,
        "num_states": num_states,
        "cluster_centers": cluster_centers,
        "cluster_labels": all_center_labels,
        "action_norm_stats": {"mean": act_mean, "std": act_std},
        "feature_norm_stats": feat_norm_stats,
        "dummy_action_label": DUMMY_ACTION_LABEL,
    }
    if metadata is not None:
        output_data["metadata"] = metadata

    with open(output_pkl, "wb") as f:
        pickle.dump(output_data, f)

    print(f"\nSaved {len(dual_episodes)} dual episodes → {output_pkl}")
    print(f"States: {num_states}  "
          f"(dummy=1, noise={'1' if has_noise else '0'}, "
          f"clusters={n_clusters})")
