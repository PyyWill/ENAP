#!/usr/bin/env python3
"""
Replay-filter PMM by stacking direction.

Keeps the PMM upper-level topology unchanged.  At the replay (lower) level,
filters out episodes whose stacking_direction != the desired direction,
then re-runs replay-assign + prune on the remaining episodes.

Usage:
    python scripts/train/replay_filter_direction.py \
        --pmm-pkl results/pmm_residual_training.pkl \
        --data-pkl data/episodes_with_states.pkl \
        --keep red_on_green
"""
import argparse
import os
import pickle
import sys

import numpy as np
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.pmm_class import PMM


def load_direction_for_pmm_episodes(data_pkl_path, pmm_episodes):
    """Match each PMM episode to the data pkl by action sequence,
    return per-PMM-episode stacking_direction."""
    with open(data_pkl_path, "rb") as f:
        data = pickle.load(f)
    metadata = data["metadata"]
    data_episodes = data["episodes"]

    directions = []
    for pi, pep in enumerate(pmm_episodes):
        T = pep['S'].shape[0]
        if T == 0:
            directions.append(None)
            continue
        matched = False
        for di, dep in enumerate(data_episodes):
            if len(dep) != T:
                continue
            data_actions = np.array([s['action'] for s in dep])
            if np.allclose(pep['A'], data_actions, atol=1e-5):
                directions.append(metadata[di]["stacking_direction"])
                matched = True
                break
        if not matched:
            print(f"  WARNING: PMM episode {pi} (T={T}) not matched in data pkl")
            directions.append(None)

    return directions


def replay_with_filter(pmm_obj, keep_indices):
    """Replay-assign only the kept episodes on the original PMM topology.

    Same logic as PMM.replay_assign + production_pmm_ba.replay_on_product,
    but skips episodes not in keep_indices.
    """
    delta = pmm_obj.pmm['delta']
    rep_embeddings = pmm_obj.pmm.get('rep_embeddings', [])

    edge_cache = defaultdict(list)
    qx_actions = defaultdict(list)
    keep_set = set(keep_indices)

    for epi, ep in enumerate(pmm_obj.episodes):
        if epi not in keep_set:
            continue
        T = ep['S'].shape[0]
        if T == 0:
            continue

        q = 0
        trajectory = []

        for t in range(T):
            x = int(np.argmax(ep['S'][t]))
            dist = delta.get((q, x))

            if dist is None:
                q_next = q
            elif len(dist) == 1:
                q_next = next(iter(dist))
            else:
                z_t = pmm_obj._get_step_embedding(epi, t)
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
                    best_q = max(dist, key=dist.get)
                q_next = best_q

            trajectory.append((q, x, q_next, t))
            q = q_next

        for q_src, x, q_dst, t in trajectory:
            edge_cache[(q_src, x, q_dst)].append((epi, t))
            qx_actions[(q_src, x)].append(ep['A'][t])

    return dict(edge_cache), dict(qx_actions)


def main():
    parser = argparse.ArgumentParser(
        description="Replay-filter PMM by stacking direction")
    parser.add_argument("--pmm-pkl", default="results/pmm_residual_training.pkl",
                        help="Path to the trained PMM pkl")
    parser.add_argument("--data-pkl",
                        default="data/episodes_with_states.pkl",
                        help="Path to data pkl with metadata")
    parser.add_argument("--keep", default="red_on_green",
                        choices=["red_on_green", "green_on_red"],
                        help="Which stacking_direction to keep")
    parser.add_argument("--output", default=None,
                        help="Output pkl path")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            "results", "filtered", f"pmm_filtered_{args.keep}.pkl")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- 1. Load PMM ----
    print(f"Loading PMM from {args.pmm_pkl} ...")
    pmm = PMM(use_tqdm=True)
    pmm.load_pmm(args.pmm_pkl)
    n_q = len(pmm.pmm['Q'])
    n_e = sum(len(d) for d in pmm.pmm['delta'].values())
    print(f"  {n_q} nodes, {n_e} edges, {len(pmm.episodes)} episodes, sigma={pmm.sigma}")

    # ---- 2. Match directions & filter ----
    print(f"\nMatching PMM episodes to data pkl {args.data_pkl} ...")
    directions = load_direction_for_pmm_episodes(args.data_pkl, pmm.episodes)
    for i, d in enumerate(directions):
        T = pmm.episodes[i]['S'].shape[0]
        print(f"  ep {i}: T={T}, direction={d}")

    keep_indices = [i for i, d in enumerate(directions) if d == args.keep]
    discard_indices = [i for i, d in enumerate(directions) if d != args.keep]
    print(f"  Keep '{args.keep}': {len(keep_indices)} episodes")
    print(f"  Discard: {len(discard_indices)} episodes")

    # ---- 3. Replay on original topology with filtered episodes ----
    print(f"\nReplaying on original PMM topology (filtered) ...")
    edge_cache, qx_actions = replay_with_filter(pmm, keep_indices)

    # Zero out discarded episodes
    for i in discard_indices:
        pmm.episodes[i] = {'S': np.zeros((0, pmm.s_dim)),
                           'A': np.zeros((0, pmm.a_dim))}

    pmm._edge_cache = edge_cache
    pmm._qx_actions = qx_actions

    # ---- 4. Prune empty edges / unreachable nodes ----
    print("Pruning ...")
    pmm._prune_after_replay()

    # ---- 5. Save ----
    print(f"\nSaving filtered PMM to {args.output} ...")
    pmm.save_pmm(args.output)

    dot_path = args.output.replace('.pkl', '.dot')
    pmm.save_dot(dot_path)

    n_q = len(pmm.pmm['Q'])
    n_e = sum(len(d) for d in pmm.pmm['delta'].values())
    print(f"\nFinal filtered PMM: {n_q} nodes, {n_e} edges, "
          f"{len(pmm.episodes)} episodes, sigma={pmm.sigma}")


if __name__ == "__main__":
    main()
