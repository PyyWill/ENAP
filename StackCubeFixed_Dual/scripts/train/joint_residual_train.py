"""Joint horizon-H fine-tuning of both ResidualMLPs.

Unrolls the chain  f_t → pmm1/resid1 → a_t → pmm2/resid2 → f_{t+1} → …
for H steps.  PMM lookups are detached; gradients flow through the continuous
ResidualMLP chain.  Both resid1 and resid2 are fine-tuned from pre-trained
checkpoints.

Run from StackCubeFixed_Dual/:
  python scripts/train/joint_residual_train.py [--epochs 200] [--lr 1e-4]
"""

import os
import sys
import pickle
import argparse
import importlib.util

import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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

JOINT_DIR = os.path.join(DUAL_ROOT, "results", "joint")

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------
def _import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if DUAL_ROOT not in sys.path:
    sys.path.insert(0, DUAL_ROOT)

from agent.pmm_class import PMM as DualPMM
from agent.pmm_agent import ResidualMLP as DualResidualMLP

_orig_pmm_mod = _import_module(
    "orig_pmm_class", os.path.join(ORIG_ROOT, "agent", "pmm_class.py"))
_orig_agent_mod = _import_module(
    "orig_pmm_agent", os.path.join(ORIG_ROOT, "agent", "pmm_agent.py"))
OrigPMM = _orig_pmm_mod.PMM
OrigResidualMLP = _orig_agent_mod.ResidualMLP

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
FEATURE_CLAMP = 5.0


def normalize_np(feat, mean, std):
    return (feat - mean) / (std + 1e-8)


def find_valid_state_idx(current_q, state_idx, pmm_delta, cluster_centers,
                         query_vec):
    if (current_q, state_idx) in pmm_delta:
        return state_idx
    valid = [s for (q, s) in pmm_delta if q == current_q]
    if not valid:
        return state_idx
    dists = [np.linalg.norm(query_vec - cluster_centers[s]) for s in valid]
    return valid[int(np.argmin(dists))]


def disambiguate(pmm_obj, current_q, x, h):
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


def dual_to_orig_torch(feat_dual, d2o, rgb_dim):
    """Differentiable dual→orig space conversion.
    d2o: dict of torch tensors {d_rgb_mean, d_rgb_std, ..., o_rgb_mean, ...}
    """
    f_rgb = feat_dual[:, :rgb_dim]
    f_pos = feat_dual[:, rgb_dim:]
    raw_rgb = f_rgb * (d2o['d_rgb_std'] + 1e-8) + d2o['d_rgb_mean']
    raw_pos = f_pos * (d2o['d_pos_std'] + 1e-8) + d2o['d_pos_mean']
    o_rgb = (raw_rgb - d2o['o_rgb_mean']) / (d2o['o_rgb_std'] + 1e-8)
    o_pos = (raw_pos - d2o['o_pos_mean']) / (d2o['o_pos_std'] + 1e-8)
    return torch.cat([o_rgb, o_pos], dim=-1)


def compute_nll(pred_mean, logstd_param, target):
    """Per-dimension mean NLL, averaged over batch."""
    logstd = torch.clamp(logstd_param, min=-2.0, max=2.0).expand_as(pred_mean)
    return -Normal(pred_mean, torch.exp(logstd)).log_prob(target) \
            .mean(dim=-1).mean()


# ---------------------------------------------------------------------------
# Episode pre-processing
# ---------------------------------------------------------------------------
def preprocess_episode(episode, pmm1, pmm2, o_stats, d_stats, a_mean, a_std,
                       rgb_dim):
    """Build GT arrays + PMM state trajectories for one original episode."""
    T = len(episode)
    feat_dim = rgb_dim + len(o_stats['pos_mean'])
    a_dim = len(a_mean)

    gt_fo = np.zeros((T, feat_dim), dtype=np.float32)
    gt_fd = np.zeros((T, feat_dim), dtype=np.float32)
    gt_act = np.zeros((T, a_dim), dtype=np.float32)
    gt_act_norm = np.zeros((T, a_dim), dtype=np.float32)

    for t, step in enumerate(episode):
        rgb_raw = step['rgb_feature'].reshape(-1)
        pos_raw = step['pos_feature'].reshape(-1)
        gt_fo[t] = np.concatenate([
            normalize_np(rgb_raw, o_stats['rgb_mean'], o_stats['rgb_std']),
            normalize_np(pos_raw, o_stats['pos_mean'], o_stats['pos_std']),
        ])
        gt_fd[t] = np.concatenate([
            np.clip(normalize_np(rgb_raw, d_stats['rgb_mean'],
                                 d_stats['rgb_std']),
                    -FEATURE_CLAMP, FEATURE_CLAMP),
            np.clip(normalize_np(pos_raw, d_stats['pos_mean'],
                                 d_stats['pos_std']),
                    -FEATURE_CLAMP, FEATURE_CLAMP),
        ])
        a = step['action'].reshape(-1).astype(np.float32)
        gt_act[t] = a
        gt_act_norm[t] = normalize_np(a, a_mean, a_std)

    pmm1_cc = pmm1.cluster_centers
    pmm2_cc = pmm2.cluster_centers

    # pmm1 GT trajectory
    q1_seq = np.zeros(T, dtype=np.int64)
    h1_seq = np.zeros((T, pmm1.rnn_hidden), dtype=np.float32)
    q1, h1 = 0, np.zeros(pmm1.rnn_hidden, dtype=np.float32)
    for t in range(T):
        q1_seq[t] = q1
        h1_seq[t] = h1
        s = int(np.argmin(np.linalg.norm(pmm1_cc - gt_fo[t], axis=1)))
        s = find_valid_state_idx(q1, s, pmm1.pmm['delta'], pmm1_cc, gt_fo[t])
        h1 = pmm1.rnn.forward_step(gt_act[t], s, h1)
        q1 = disambiguate(pmm1, q1, s, h1)

    # pmm2 GT trajectory (with dummy init)
    q2_seq = np.zeros(T, dtype=np.int64)
    h2_seq = np.zeros((T, pmm2.rnn_hidden), dtype=np.float32)
    q2, h2 = 0, np.zeros(pmm2.rnn_hidden, dtype=np.float32)
    h2 = pmm2.rnn.forward_step(gt_fd[0], 0, h2)
    q2 = disambiguate(pmm2, q2, 0, h2)
    for t in range(T):
        q2_seq[t] = q2
        h2_seq[t] = h2
        if t < T - 1:
            s = int(np.argmin(
                np.linalg.norm(pmm2_cc - gt_act_norm[t], axis=1)))
            s = find_valid_state_idx(
                q2, s, pmm2.pmm['delta'], pmm2_cc, gt_act_norm[t])
            h2 = pmm2.rnn.forward_step(gt_fd[t + 1], s, h2)
            q2 = disambiguate(pmm2, q2, s, h2)

    return {
        'gt_fo': gt_fo, 'gt_fd': gt_fd,
        'gt_act': gt_act, 'gt_act_norm': gt_act_norm,
        'q1': q1_seq, 'h1': h1_seq,
        'q2': q2_seq, 'h2': h2_seq,
        'T': T,
    }


def build_windows(ep_data_list, H):
    """Enumerate all valid (episode_idx, start_t) pairs."""
    windows = []
    for i, ep in enumerate(ep_data_list):
        for t in range(ep['T'] - H):
            windows.append((i, t))
    return windows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_joint(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = args.horizon
    gamma = args.gamma
    os.makedirs(JOINT_DIR, exist_ok=True)

    # ── Load PMMs (frozen) ────────────────────────────────────────────
    pmm1 = OrigPMM()
    pmm1.load_pmm(ORIG_PMM_PKL)
    pmm2 = DualPMM()
    pmm2.load_pmm(DUAL_PMM_PKL)

    # ── Load pre-trained ResidualMLPs ─────────────────────────────────
    ckpt1 = torch.load(ORIG_RESID_PT, map_location=device, weights_only=False)
    o_stats = ckpt1['norm_stats']
    rgb_dim = len(o_stats['rgb_mean'])
    feat_dim = rgb_dim + len(o_stats['pos_mean'])
    a_dim = pmm1.a_dim

    resid1 = OrigResidualMLP(feat_dim, a_dim, hidden=256).to(device)
    resid1.load_state_dict(ckpt1['model'])

    ckpt2 = torch.load(DUAL_RESID_PT, map_location=device, weights_only=False)
    a_norm_stats = ckpt2['action_norm_stats']
    a_mean_np, a_std_np = a_norm_stats['mean'], a_norm_stats['std']
    resid2 = DualResidualMLP(feat_dim, a_dim, hidden=1024).to(device)
    resid2.load_state_dict(ckpt2['model'])

    with open(DUAL_DATA_PKL, "rb") as f:
        d_stats = pickle.load(f)["feature_norm_stats"]

    # tensors for differentiable normalize / dual_to_orig
    a_mean_t = torch.from_numpy(a_mean_np).float().to(device)
    a_std_t = torch.from_numpy(a_std_np).float().to(device)
    d2o = {k: torch.from_numpy(v).float().to(device) for k, v in {
        'd_rgb_mean': d_stats['rgb_mean'], 'd_rgb_std': d_stats['rgb_std'],
        'd_pos_mean': d_stats['pos_mean'], 'd_pos_std': d_stats['pos_std'],
        'o_rgb_mean': o_stats['rgb_mean'], 'o_rgb_std': o_stats['rgb_std'],
        'o_pos_mean': o_stats['pos_mean'], 'o_pos_std': o_stats['pos_std'],
    }.items()}

    pmm1_cc = pmm1.cluster_centers
    pmm2_cc = pmm2.cluster_centers

    # ── Pre-process episodes ──────────────────────────────────────────
    with open(ORIG_DATA_PKL, "rb") as f:
        episodes = pickle.load(f)["episodes"]
    split = int(len(episodes) * 0.9)
    train_eps_raw, eval_eps_raw = episodes[:split], episodes[split:]

    print(f"Pre-processing {len(train_eps_raw)} train + "
          f"{len(eval_eps_raw)} eval episodes …")
    train_data = [preprocess_episode(ep, pmm1, pmm2, o_stats, d_stats,
                                     a_mean_np, a_std_np, rgb_dim)
                  for ep in tqdm(train_eps_raw, desc="train")]
    eval_data = [preprocess_episode(ep, pmm1, pmm2, o_stats, d_stats,
                                    a_mean_np, a_std_np, rgb_dim)
                 for ep in tqdm(eval_eps_raw, desc="eval")]

    train_windows = build_windows(train_data, H)
    eval_windows = build_windows(eval_data, H)
    print(f"  train windows: {len(train_windows)}, "
          f"eval windows: {len(eval_windows)}")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = optim.Adam(
        list(resid1.parameters()) + list(resid2.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    def run_batch(windows, ep_data_list, train=True):
        """Run one batch of windows and return mean loss."""
        B = len(windows)
        if B == 0:
            return 0.0

        # gather GT tensors for the full horizon
        fo_anchor = []
        act_gt_all = [[] for _ in range(H)]
        fd_gt_all = [[] for _ in range(H)]
        q1_init, h1_init, q2_init, h2_init = [], [], [], []

        for (ei, t) in windows:
            ep = ep_data_list[ei]
            fo_anchor.append(ep['gt_fo'][t])
            for k in range(H):
                act_gt_all[k].append(ep['gt_act'][t + k])
                fd_gt_all[k].append(ep['gt_fd'][t + k + 1])
            q1_init.append(int(ep['q1'][t]))
            h1_init.append(ep['h1'][t].copy())
            q2_init.append(int(ep['q2'][t]))
            h2_init.append(ep['h2'][t].copy())

        feat_orig = torch.from_numpy(
            np.array(fo_anchor)).float().to(device)           # (B, feat_dim)
        act_gt_t = [torch.from_numpy(np.array(a)).float().to(device)
                     for a in act_gt_all]                      # H × (B, a_dim)
        fd_gt_t = [torch.from_numpy(np.array(f)).float().to(device)
                    for f in fd_gt_all]                        # H × (B, feat_dim)

        q1_b, h1_b = list(q1_init), list(h1_init)
        q2_b, h2_b = list(q2_init), list(h2_init)

        total_loss = torch.tensor(0.0, device=device)

        for k in range(H):
            # ── pmm1 lookup (detached) ────────────────────────────
            action_lists, cc1_list = [], []
            obs_s = []
            feat_np = feat_orig.detach().cpu().numpy()
            for i in range(B):
                s = int(np.argmin(
                    np.linalg.norm(pmm1_cc - feat_np[i], axis=1)))
                s = find_valid_state_idx(
                    q1_b[i], s, pmm1.pmm['delta'], pmm1_cc, feat_np[i])
                obs_s.append(s)
                action_lists.append(pmm1.predict_list(q1_b[i], s))
                cc1_list.append(pmm1_cc[s])
            cc1_t = torch.from_numpy(
                np.array(cc1_list)).float().to(device)

            # ── resid1 forward (grad flows) ───────────────────────
            a_pred = resid1.forward(action_lists, cc1_t, feat_orig)
            nll_a = compute_nll(a_pred, resid1.logstd, act_gt_t[k])

            # ── action normalize (differentiable) ─────────────────
            a_norm = (a_pred - a_mean_t) / (a_std_t + 1e-8)

            # ── pmm2 lookup (detached) ────────────────────────────
            feat_lists, cc2_list = [], []
            act_s = []
            a_norm_np = a_norm.detach().cpu().numpy()
            for i in range(B):
                s = int(np.argmin(
                    np.linalg.norm(pmm2_cc - a_norm_np[i], axis=1)))
                s = find_valid_state_idx(
                    q2_b[i], s, pmm2.pmm['delta'], pmm2_cc, a_norm_np[i])
                act_s.append(s)
                feat_lists.append(pmm2.predict_list(q2_b[i], s))
                cc2_list.append(pmm2_cc[s])
            cc2_t = torch.from_numpy(
                np.array(cc2_list)).float().to(device)

            # ── resid2 forward (grad flows) ───────────────────────
            f_pred_dual = resid2.forward(feat_lists, cc2_t, a_norm)
            nll_f = compute_nll(f_pred_dual, resid2.logstd, fd_gt_t[k])

            total_loss = total_loss + gamma ** k * (nll_a + nll_f)

            # ── PMM state update with predicted outputs (detached) ─
            a_det = a_pred.detach().cpu().numpy()
            f_det = f_pred_dual.detach().cpu().numpy()
            for i in range(B):
                h1_b[i] = pmm1.rnn.forward_step(
                    a_det[i], obs_s[i], h1_b[i])
                q1_b[i] = disambiguate(
                    pmm1, q1_b[i], obs_s[i], h1_b[i])
                h2_b[i] = pmm2.rnn.forward_step(
                    f_det[i], act_s[i], h2_b[i])
                q2_b[i] = disambiguate(
                    pmm2, q2_b[i], act_s[i], h2_b[i])

            # ── dual→orig for next step (differentiable) ──────────
            if k < H - 1:
                feat_orig = dual_to_orig_torch(f_pred_dual, d2o, rgb_dim)

        return total_loss / H

    # ── Training loop ─────────────────────────────────────────────────
    best_eval_loss = float('inf')
    train_hist, eval_hist = [], []
    bs = args.batch_size

    pbar = tqdm(range(args.epochs), desc="Joint fine-tuning")
    for epoch in pbar:
        resid1.train()
        resid2.train()
        np.random.shuffle(train_windows)

        epoch_losses = []
        for i in range(0, len(train_windows), bs):
            batch = train_windows[i:i + bs]
            optimizer.zero_grad()
            loss = run_batch(batch, train_data, train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(resid1.parameters()) + list(resid2.parameters()),
                max_norm=5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        resid1.eval()
        resid2.eval()
        eval_losses = []
        with torch.no_grad():
            for i in range(0, len(eval_windows), bs):
                batch = eval_windows[i:i + bs]
                loss = run_batch(batch, eval_data, train=False)
                eval_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(epoch_losses)
        avg_eval = np.mean(eval_losses) if eval_losses else float('inf')
        train_hist.append(avg_train)
        eval_hist.append(avg_eval)

        pbar.set_postfix({
            'train': f"{avg_train:.4f}",
            'eval': f"{avg_eval:.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        if avg_eval < best_eval_loss:
            best_eval_loss = avg_eval
            torch.save({
                'resid1': resid1.state_dict(),
                'resid2': resid2.state_dict(),
                'norm_stats': o_stats,
                'action_norm_stats': a_norm_stats,
            }, os.path.join(JOINT_DIR, "joint_residual.pt"))

    # ── Loss curves ───────────────────────────────────────────────────
    if train_hist:
        plt.figure(figsize=(10, 6))
        x = range(1, len(train_hist) + 1)
        plt.plot(x, train_hist, 'o-', label='Train', lw=2, ms=3)
        plt.plot(x, eval_hist, 's-', label='Eval', lw=2, ms=3)
        plt.xlabel('Epoch'); plt.ylabel('Loss (per-dim NLL)')
        plt.title(f'Joint Residual Training (H={H})')
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(JOINT_DIR, "loss_curves.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Done. Best eval loss: {best_eval_loss:.4f}")
    print(f"Checkpoint → {JOINT_DIR}/joint_residual.pt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.9)
    train_joint(parser.parse_args())


if __name__ == "__main__":
    main()
