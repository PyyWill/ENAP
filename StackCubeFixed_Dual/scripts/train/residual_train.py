"""Dual ENAP — Residual network training (predict features from PMM edges).

Original: PMM edge stores actions  → residual refines action prediction.
Dual:     PMM edge stores features → residual refines feature prediction.
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.pmm_class import PMM
from agent.pmm_agent import ResidualMLP
from agent.utils import normalize_feature, load_trajectories


# ---------------------------------------------------------------------------
# Node transition helpers (mirrors pmm_class.replay_assign)
# ---------------------------------------------------------------------------
def compute_episode_rnn_embeddings(episode, pmm):
    """Run the PMM's RNN over a dual episode and return per-step hidden states."""
    features = np.stack([s['combined_feature'] for s in episode], axis=0)
    state_indices = np.array([int(np.argmax(s['state'])) for s in episode])
    return pmm.rnn.forward_trajectory(features, state_indices)


def update_pmm_node(pmm, current_q, x, z_t):
    """Transition using embedding-based disambiguation."""
    dist = pmm.pmm['delta'].get((current_q, x))
    if dist is None:
        raise KeyError(f"PMM has no edge for (q={current_q}, x={x})")
    if len(dist) == 1:
        return next(iter(dist))

    rep_embeddings = pmm.pmm.get('rep_embeddings', [])
    best_q, best_sim = None, -2.0
    if z_t is not None and rep_embeddings:
        z_norm = z_t / (np.linalg.norm(z_t) + 1e-12)
        for q_cand in dist:
            if q_cand < len(rep_embeddings):
                z_rep = rep_embeddings[q_cand]
                sim = np.dot(z_norm, z_rep / (np.linalg.norm(z_rep) + 1e-12))
                if sim > best_sim:
                    best_sim = sim
                    best_q = q_cand
    if best_q is None:
        raise RuntimeError("PMM multi-edge disambiguation failed")
    return best_q


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_valid_state_idx(current_q, state_idx, action_norm, pmm, cluster_centers):
    """Find a valid state_idx for current_q by nearest action cluster centre."""
    if (current_q, state_idx) in pmm.pmm['delta']:
        return state_idx
    valid_states = [s for (q, s) in pmm.pmm['delta'].keys() if q == current_q]
    assert valid_states, f"PMM node q={current_q} has no outgoing edges"
    dists = [np.linalg.norm(action_norm - cluster_centers[s])
             for s in valid_states]
    return valid_states[int(np.argmin(dists))]


def collect_episode_data(episodes, pmm, action_norm_stats, cluster_centers):
    """Flatten episodes into per-step dicts with PMM node tracking."""
    a_mean, a_std = action_norm_stats['mean'], action_norm_stats['std']
    all_data = []
    for episode in episodes:
        if not episode:
            continue
        h_seq = compute_episode_rnn_embeddings(episode, pmm)
        current_q = 0
        for t, step in enumerate(episode):
            state_idx = int(np.argmax(step['state']))
            action_raw = step['action'].reshape(-1).astype(np.float32)
            action_norm = normalize_feature(action_raw, a_mean, a_std)

            state_idx = find_valid_state_idx(
                current_q, state_idx, action_norm, pmm, cluster_centers)

            all_data.append({
                'current_q': current_q,
                'state_idx': state_idx,
                'feature_true': step['combined_feature'].astype(np.float32),
                'action_norm': action_norm.astype(np.float32),
            })
            current_q = update_pmm_node(pmm, current_q, state_idx, h_seq[t])
    return all_data


def process_batch(batch_data, pmm, cluster_centers_t, node_to_valid, device):
    """Build tensors for one mini-batch."""
    feature_lists, cc_list = [], []
    for d in batch_data:
        curr_q, s_idx = d['current_q'], d['state_idx']

        if (curr_q, s_idx) not in pmm.pmm['delta']:
            valid_s = node_to_valid.get(curr_q)
            assert valid_s, f"PMM node q={curr_q} has no outgoing edges"
            act_t = torch.from_numpy(d['action_norm']).to(device)
            dists = torch.norm(
                cluster_centers_t[valid_s] - act_t.unsqueeze(0), dim=1)
            s_idx = valid_s[torch.argmin(dists).item()]

        feature_lists.append(pmm.predict_list(curr_q, s_idx))
        cc_list.append(cluster_centers_t[s_idx])

    cc_batch = torch.stack(cc_list)
    act_batch = torch.from_numpy(
        np.array([d['action_norm'] for d in batch_data])).to(device)
    feat_true_batch = torch.from_numpy(
        np.array([d['feature_true'] for d in batch_data])).to(device)
    return feature_lists, cc_batch, act_batch, feat_true_batch


def evaluate_model(model, eval_data, pmm, cluster_centers_t, node_to_valid,
                   device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(eval_data), 256):
            batch = eval_data[i:i + 256]
            fl, cc, act, ft = process_batch(
                batch, pmm, cluster_centers_t, node_to_valid, device)
            log_prob = model.get_log_prob(fl, cc, act, ft)
            losses.append(-log_prob.mean().item())
    model.train()
    return np.mean(losses) if losses else float('inf')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_residual_network(train_episodes, eval_episodes, cluster_centers,
                           pmm, action_norm_stats,
                           epochs=400, lr=3e-4, device="cuda", model=None,
                           results_dir="results",
                           checkpoint_dir="results/checkpoints"):
    first_step = next(ep[0] for ep in train_episodes if ep)
    f_dim = len(first_step['combined_feature'])
    a_dim = len(first_step['action'].reshape(-1))
    print(f"Feature dim: {f_dim}, Action dim: {a_dim}")

    print("Pre-processing episodes …")
    all_train = collect_episode_data(
        train_episodes, pmm, action_norm_stats, cluster_centers)
    all_eval = collect_episode_data(
        eval_episodes, pmm, action_norm_stats, cluster_centers)
    print(f"  train steps: {len(all_train)}, eval steps: {len(all_eval)}")

    cluster_centers_t = torch.from_numpy(cluster_centers).float().to(device)
    node_to_valid = {}
    for q in pmm.pmm['Q']:
        vs = [s for (node, s) in pmm.pmm['delta'].keys() if node == q]
        if vs:
            node_to_valid[q] = vs

    if model is None:
        model = ResidualMLP(f_dim, a_dim, hidden=1024).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_eval_loss = float('inf')
    train_hist, eval_hist = [], []

    pbar = tqdm(range(epochs), desc="Training Residual")
    for epoch in pbar:
        np.random.shuffle(all_train)
        epoch_losses = []
        bs = 512
        for i in range(0, len(all_train), bs):
            batch = all_train[i:i + bs]
            fl, cc, act, ft = process_batch(
                batch, pmm, cluster_centers_t, node_to_valid, device)

            act = act + torch.randn_like(act) * 0.01

            log_prob = model.get_log_prob(fl, cc, act, ft)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        eval_loss = evaluate_model(
            model, all_eval, pmm, cluster_centers_t, node_to_valid, device)
        scheduler.step()

        avg_train = np.mean(epoch_losses)
        train_hist.append(avg_train)
        eval_hist.append(eval_loss)
        pbar.set_postfix({
            'train': f"{avg_train:.4f}",
            'eval': f"{eval_loss:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        if optimizer.param_groups[0]["lr"] < 1e-7:
            print("Learning rate below 1e-7, stopping.")
            break

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(
                {'model': model.state_dict(),
                 'action_norm_stats': action_norm_stats},
                os.path.join(checkpoint_dir, "residual_net.pt"))

    plot_dir = os.path.join(results_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    if train_hist:
        plt.figure(figsize=(10, 6))
        x = range(1, len(train_hist) + 1)
        plt.plot(x, train_hist, 'o-', label='Train', lw=2, ms=4)
        plt.plot(x, eval_hist, 's-', label='Eval', lw=2, ms=4)
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.title('Residual Training Loss Curves')
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "loss_curves.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved loss curves → {plot_dir}/loss_curves.png")

    return model, plot_dir


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def _cosine_sim_batch(a, b):
    """Row-wise cosine similarity between two (T, D) arrays."""
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1) + 1e-12
    nb = np.linalg.norm(b, axis=1) + 1e-12
    return dot / (na * nb)


def visualize_features(test_episodes, cluster_centers, pmm, model,
                       action_norm_stats, device="cuda", n_vis=3,
                       plot_dir="results/plot"):
    """Quantitative (L2, cosine) + Qualitative (heatmap) visualisation."""
    model.eval()
    a_mean, a_std = action_norm_stats['mean'], action_norm_stats['std']

    selected = np.random.choice(
        len(test_episodes), min(n_vis, len(test_episodes)), replace=False)

    for ep_idx in selected:
        episode = test_episodes[ep_idx]
        true_feats, pmm_feats, res_feats = [], [], []

        h_seq = compute_episode_rnn_embeddings(episode, pmm)
        current_q = 0

        with torch.no_grad():
            for t, step in enumerate(episode):
                state_idx = int(np.argmax(step['state']))
                action_raw = step['action'].reshape(-1).astype(np.float32)
                action_norm = normalize_feature(action_raw, a_mean, a_std)

                state_idx = find_valid_state_idx(
                    current_q, state_idx, action_norm, pmm, cluster_centers)

                pmm_feat = pmm.predict(current_q, state_idx)
                f_list = [pmm.predict_list(current_q, state_idx)]

                current_q = update_pmm_node(
                    pmm, current_q, state_idx, h_seq[t])

                cc_t = torch.from_numpy(
                    cluster_centers[state_idx]).float().to(device).unsqueeze(0)
                act_t = torch.from_numpy(
                    action_norm).float().to(device).unsqueeze(0)
                pred = model.get_feature_mean(
                    f_list, cc_t, act_t).cpu().numpy()[0]

                true_feats.append(step['combined_feature'])
                pmm_feats.append(pmm_feat)
                res_feats.append(pred)

        true_arr = np.array(true_feats)
        pmm_arr = np.array(pmm_feats)
        res_arr = np.array(res_feats)
        T = len(true_feats)
        ts = np.arange(T)

        # ── Quantitative: L2 error + cosine similarity ──
        l2_pmm = np.linalg.norm(true_arr - pmm_arr, axis=1)
        l2_res = np.linalg.norm(true_arr - res_arr, axis=1)
        cos_pmm = _cosine_sim_batch(true_arr, pmm_arr)
        cos_res = _cosine_sim_batch(true_arr, res_arr)

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(ts, l2_pmm, 's-', label='PMM', alpha=0.7, ms=3)
        axes[0].plot(ts, l2_res, '^-', label='Residual', alpha=0.7, ms=3)
        axes[0].set_ylabel('L2 Error')
        axes[0].set_title(f'Episode {ep_idx} — Per-step L2 Error')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(ts, cos_pmm, 's-', label='PMM', alpha=0.7, ms=3)
        axes[1].plot(ts, cos_res, '^-', label='Residual', alpha=0.7, ms=3)
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_xlabel('Timestep')
        axes[1].set_title(f'Episode {ep_idx} — Per-step Cosine Similarity')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f"ep{ep_idx}_quantitative.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # ── Qualitative: 2-D heatmap (True / Predicted / |Diff|) ──
        diff_arr = np.abs(true_arr - res_arr)
        vmin = min(true_arr.min(), res_arr.min())
        vmax = max(true_arr.max(), res_arr.max())

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        im0 = axes[0].imshow(true_arr.T, aspect='auto', cmap='viridis',
                              vmin=vmin, vmax=vmax)
        axes[0].set_ylabel('Feature Dim')
        axes[0].set_title(f'Episode {ep_idx} — True Feature')
        plt.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(res_arr.T, aspect='auto', cmap='viridis',
                              vmin=vmin, vmax=vmax)
        axes[1].set_ylabel('Feature Dim')
        axes[1].set_title('Predicted Feature (Residual)')
        plt.colorbar(im1, ax=axes[1], shrink=0.8)

        im2 = axes[2].imshow(diff_arr.T, aspect='auto', cmap='hot')
        axes[2].set_ylabel('Feature Dim')
        axes[2].set_xlabel('Timestep')
        axes[2].set_title('|Difference|')
        plt.colorbar(im2, ax=axes[2], shrink=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir,
                                 f"ep{ep_idx}_qualitative.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved visualisations → {plot_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", type=str,
                        default="data/episodes_with_states.pkl")
    parser.add_argument("--rnn-weights", type=str,
                        default="results/checkpoints/rnn_pretrained.pt")
    parser.add_argument("--pmm-pkl", type=str, default=None,
                        help="Load existing PMM .pkl instead of rebuilding")
    parser.add_argument("--trained-pt", type=str, default=None,
                        help="Resume residual_net.pt if provided")
    parser.add_argument("--cos-tau", type=float, default=0.6)
    parser.add_argument("--error-threshold", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()

    results_dir = "results"
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    episodes, cluster_centers, action_norm_stats, _ = \
        load_trajectories(args.input_pkl)
    assert action_norm_stats is not None, \
        "PKL missing action_norm_stats — regenerate with preprocess.py"

    train_episodes = episodes[:int(len(episodes) * 0.9)]
    test_episodes = episodes[int(len(episodes) * 0.9):]
    print(f"Train: {len(train_episodes)}, Test: {len(test_episodes)}")

    # ── PMM ──
    if args.pmm_pkl:
        pmm = PMM()
        pmm.load_pmm(args.pmm_pkl)
    else:
        pmm_train = train_episodes[:10]
        pmm_trajs = [[{'state': s['state'],
                        'combined_feature': s['combined_feature']}
                       for s in ep] for ep in pmm_train]
        pmm = PMM(cos_tau_row=args.cos_tau,
                  error_threshold=args.error_threshold,
                  use_tqdm=True, max_inner_iters=1000)
        pmm.learn_pmm(pmm_trajs, rnn_weights_path=args.rnn_weights,
                      cluster_centers=cluster_centers)
        pmm.save_dot(os.path.join(results_dir,
                                  "pmm_residual_training.dot"))
        pmm.save_pmm(os.path.join(results_dir,
                                  "pmm_residual_training.pkl"))
    print(f"PMM: {len(pmm.pmm['Q'])} nodes, "
          f"{sum(len(d) for d in pmm.pmm['delta'].values())} edges")

    # ── Residual training ──
    resume_model = None
    if args.trained_pt and os.path.exists(args.trained_pt):
        f_dim = len(train_episodes[0][0]['combined_feature'])
        a_dim = len(train_episodes[0][0]['action'].reshape(-1))
        resume_model = ResidualMLP(f_dim, a_dim, hidden=1024).to(device)
        ckpt = torch.load(args.trained_pt, map_location=device,
                          weights_only=False)
        resume_model.load_state_dict(ckpt['model'])
        print(f"Resumed residual weights from {args.trained_pt}")

    model, plot_dir = train_residual_network(
        train_episodes, test_episodes, cluster_centers, pmm,
        action_norm_stats, epochs=args.epochs, device=device,
        model=resume_model, results_dir=results_dir,
        checkpoint_dir=checkpoint_dir)

    visualize_features(test_episodes, cluster_centers, pmm, model,
                       action_norm_stats, device=device,
                       plot_dir=plot_dir)
    print(f"Done. Checkpoints → {checkpoint_dir}, plots → {plot_dir}")


if __name__ == "__main__":
    main()
