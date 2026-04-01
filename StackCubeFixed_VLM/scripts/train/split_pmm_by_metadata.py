#!/usr/bin/env python3
"""
Split PMM topology by metadata categories.

Given a trained PMM and a data pkl with per-episode metadata, split
non-noise clusters by the values of a metadata key (e.g., stacking_direction),
then split the graph topology following the split-instruction rules,
re-label episodes, replay, prune, and save.

Supports N-way splits (N = number of unique values of the metadata key).

Usage:
    python scripts/train/split_pmm_by_metadata.py \
        --pmm-pkl results/pmm_residual_training.pkl \
        --data-pkl data/episodes_with_states.pkl \
        --split-key stacking_direction
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.pmm_class import PMM, TrainableRNN


# -----------------------------------------------------------------------
# Data matching
# -----------------------------------------------------------------------
def match_episodes_to_metadata(pmm_episodes, data_episodes, metadata, split_key):
    """Match each PMM episode to data by action sequence.
    Returns per-PMM-episode category value (or None for empty/unmatched)."""
    categories = []
    for pi, pep in enumerate(pmm_episodes):
        T = pep['S'].shape[0]
        if T == 0:
            categories.append(None)
            continue
        matched = False
        for di, dep in enumerate(data_episodes):
            if len(dep) != T:
                continue
            data_actions = np.array([s['action'] for s in dep])
            if np.allclose(pep['A'], data_actions, atol=1e-5):
                categories.append(metadata[di][split_key])
                matched = True
                break
        if not matched:
            print(f"  WARNING: PMM episode {pi} (T={T}) not matched")
            categories.append(None)
    return categories


# -----------------------------------------------------------------------
# Symbol mapping
# -----------------------------------------------------------------------
def detect_noise_symbols(data_pkl):
    """Auto-detect noise symbol indices from data pkl's cluster_labels."""
    cluster_labels = data_pkl.get('cluster_labels', [])
    return {i for i, label in enumerate(cluster_labels) if label == -1}


def build_symbol_mapping(sigma, noise_symbols, n_categories):
    """Build old->new symbol mapping.

    Returns:
        noise_map:     {old_sym: new_sym}
        split_map:     {old_sym: [new_sym_cat_0, ..., new_sym_cat_{N-1}]}
        new_sigma:     sorted list of all new symbol indices
        new_s_dim:     total number of new symbols
        parent_symbol: {new_sym: old_sym}
    """
    new_id = 0
    noise_map = {}
    split_map = {}
    parent_symbol = {}

    for sym in sorted(sigma):
        if sym in noise_symbols:
            noise_map[sym] = new_id
            parent_symbol[new_id] = sym
            new_id += 1

    for sym in sorted(sigma):
        if sym not in noise_symbols:
            sub_syms = list(range(new_id, new_id + n_categories))
            split_map[sym] = sub_syms
            for ns in sub_syms:
                parent_symbol[ns] = sym
            new_id += n_categories

    new_sigma = list(range(new_id))
    return noise_map, split_map, new_sigma, new_id, parent_symbol


# -----------------------------------------------------------------------
# Topology split
# -----------------------------------------------------------------------
def split_topology(pmm_data, split_map, noise_map, n_categories):
    """Split PMM graph topology per split-instruction rules.

    Rules:
      1. Out-edge split: edge with split-label cX becomes N edges
         (cX_0 ... cX_{N-1}).
      2. In-edge node split: node with incoming split-label edge becomes
         N sub-nodes.
      3. Self-loop constraint: no cross-connections between sub-nodes of
         the same original node.
      4. Sub-nodes inherit all non-self-loop out-edges from the parent node.

    Returns (new_pmm_data, node_map, node_parent).
    """
    old_delta = pmm_data['delta']
    old_Q = pmm_data['Q']
    old_reps = pmm_data['reps']
    old_rep_emb = pmm_data['rep_embeddings']

    # --- Step 1: identify nodes to split ---
    nodes_to_split = set()
    for (q, x), dist in old_delta.items():
        if x in split_map:
            for q_next in dist:
                nodes_to_split.add(q_next)

    # --- Step 2: build node index mapping ---
    node_map = {}      # old_q -> [new_q, ...] (len 1 if unsplit, N if split)
    node_parent = {}   # new_q -> old_q
    next_id = 0
    for q in sorted(old_Q):
        if q in nodes_to_split:
            new_ids = list(range(next_id, next_id + n_categories))
            node_map[q] = new_ids
            for nid in new_ids:
                node_parent[nid] = q
            next_id += n_categories
        else:
            node_map[q] = [next_id]
            node_parent[next_id] = q
            next_id += 1

    # --- Step 3: build new delta ---
    new_delta = {}

    for (q, x), dist in old_delta.items():
        src_nodes = node_map[q]
        is_src_split = len(src_nodes) > 1

        if x in split_map:
            sub_syms = split_map[x]
            for src_cat, new_src in enumerate(src_nodes):
                for cat_i in range(n_categories):
                    new_x = sub_syms[cat_i]
                    new_dist = {}
                    for q_next, prob in dist.items():
                        dst_nodes = node_map[q_next]
                        if len(dst_nodes) > 1:  # destination is split
                            if q_next == q and is_src_split:
                                # Self-loop: same-category only
                                if cat_i == src_cat:
                                    new_dist[dst_nodes[cat_i]] = prob
                            else:
                                new_dist[dst_nodes[cat_i]] = prob
                        else:
                            new_dist[dst_nodes[0]] = prob
                    if new_dist:
                        new_delta[(new_src, new_x)] = new_dist
        else:
            new_x = noise_map.get(x, x)
            for _src_cat, new_src in enumerate(src_nodes):
                new_dist = {}
                for q_next, prob in dist.items():
                    dst_nodes = node_map[q_next]
                    if len(dst_nodes) > 1:
                        for cat_i in range(n_categories):
                            new_dist[dst_nodes[cat_i]] = prob / n_categories
                    else:
                        new_dist[dst_nodes[0]] = prob
                if new_dist:
                    new_delta[(new_src, new_x)] = new_dist

    # --- Step 4: build reps & rep_embeddings ---
    new_reps = []
    new_rep_emb = []
    for q in sorted(old_Q):
        old_rep = old_reps[q] if q < len(old_reps) else ()
        old_emb = old_rep_emb[q] if q < len(old_rep_emb) else np.zeros(64)
        for sub_idx, _new_q in enumerate(node_map[q]):
            cat_idx = sub_idx if len(node_map[q]) > 1 else None
            new_reps.append(_convert_rep(old_rep, cat_idx, split_map, noise_map))
            new_rep_emb.append(old_emb.copy())

    new_Q = list(range(next_id))
    return {'Q': new_Q, 'delta': new_delta,
            'reps': new_reps, 'rep_embeddings': new_rep_emb}, node_map, node_parent


def _convert_rep(old_rep, category_idx, split_map, noise_map):
    """Convert a symbolic prefix tuple to new symbol indices."""
    if category_idx is None:
        new_rep = []
        for sym in old_rep:
            sym = int(sym)
            new_rep.append(noise_map.get(sym, sym))
        return tuple(new_rep)
    new_rep = []
    for sym in old_rep:
        sym = int(sym)
        if sym in split_map:
            new_rep.append(split_map[sym][category_idx])
        elif sym in noise_map:
            new_rep.append(noise_map[sym])
        else:
            new_rep.append(sym)
    return tuple(new_rep)


# -----------------------------------------------------------------------
# Episode re-labelling
# -----------------------------------------------------------------------
def relabel_episodes(pmm_episodes, ep_categories, category_list,
                     split_map, noise_map, new_s_dim):
    """Replace each episode's S matrix with new one-hot vectors."""
    cat_to_idx = {c: i for i, c in enumerate(category_list)}
    new_episodes = []
    for epi, ep in enumerate(pmm_episodes):
        T = ep['S'].shape[0]
        if T == 0 or ep_categories[epi] is None:
            new_episodes.append({'S': np.zeros((0, new_s_dim), dtype=np.float32),
                                 'A': ep['A']})
            continue
        cat_idx = cat_to_idx[ep_categories[epi]]
        new_S = np.zeros((T, new_s_dim), dtype=np.float32)
        for t in range(T):
            old_sym = int(np.argmax(ep['S'][t]))
            if old_sym in split_map:
                new_sym = split_map[old_sym][cat_idx]
            else:
                new_sym = noise_map.get(old_sym, old_sym)
            new_S[t, new_sym] = 1.0
        new_episodes.append({'S': new_S, 'A': ep['A']})
    return new_episodes


# -----------------------------------------------------------------------
# RNN extension
# -----------------------------------------------------------------------
def extend_rnn_embeddings(pmm, new_s_dim, parent_symbol):
    """Expand the state embedding matrix so the RNN accepts new_s_dim symbols.
    Each new symbol inherits its parent's embedding row."""
    if not isinstance(pmm.rnn, TrainableRNN):
        return
    old_weight = pmm.rnn.state_embed.weight.data          # (old_s_dim, embed_dim)
    embed_dim = old_weight.shape[1]
    new_weight = torch.zeros(new_s_dim, embed_dim)
    for new_sym, old_sym in parent_symbol.items():
        if old_sym < old_weight.shape[0]:
            new_weight[new_sym] = old_weight[old_sym]
    pmm.rnn.state_embed = torch.nn.Embedding(new_s_dim, embed_dim)
    pmm.rnn.state_embed.weight.data = new_weight


# -----------------------------------------------------------------------
# Cluster centres
# -----------------------------------------------------------------------
def build_new_cluster_centers(old_centers, noise_map, split_map, new_s_dim):
    """Duplicate parent cluster centres for each split sub-symbol."""
    feat_dim = old_centers.shape[1]
    new_centers = np.zeros((new_s_dim, feat_dim), dtype=old_centers.dtype)
    for old_sym, new_sym in noise_map.items():
        new_centers[new_sym] = old_centers[old_sym]
    for old_sym, sub_syms in split_map.items():
        for ns in sub_syms:
            new_centers[ns] = old_centers[old_sym]
    return new_centers


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Split PMM topology by metadata categories")
    parser.add_argument("--pmm-pkl", default="results/pmm_residual_training.pkl",
                        help="Path to the trained PMM pkl")
    parser.add_argument("--data-pkl", default="data/episodes_with_states.pkl",
                        help="Path to data pkl with per-episode metadata")
    parser.add_argument("--split-key", default="stacking_direction",
                        help="Metadata key to split by")
    parser.add_argument("--noise-symbols", default=None,
                        help="Comma-separated noise symbol indices "
                             "(default: auto-detect from data pkl)")
    parser.add_argument("--output", default=None,
                        help="Output pkl path")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            "results", "split", f"pmm_split_{args.split_key}.pkl")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ==== 1. Load PMM ====
    print(f"Loading PMM from {args.pmm_pkl} ...")
    pmm = PMM(use_tqdm=True)
    pmm.load_pmm(args.pmm_pkl)
    n_q = len(pmm.pmm['Q'])
    n_e = sum(len(d) for d in pmm.pmm['delta'].values())
    print(f"  {n_q} nodes, {n_e} edges, {len(pmm.episodes)} episodes, "
          f"sigma={pmm.sigma}, s_dim={pmm.s_dim}")

    # ==== 2. Load data & match episodes ====
    print(f"\nLoading data from {args.data_pkl} ...")
    with open(args.data_pkl, "rb") as f:
        data = pickle.load(f)
    ep_categories = match_episodes_to_metadata(
        pmm.episodes, data['episodes'], data['metadata'], args.split_key)
    for i, cat in enumerate(ep_categories):
        T = pmm.episodes[i]['S'].shape[0]
        print(f"  ep {i}: T={T}, {args.split_key}={cat}")

    # ==== 3. Determine categories & build symbol mapping ====
    all_categories = sorted({c for c in ep_categories if c is not None})
    n_categories = len(all_categories)
    print(f"\nCategories ({n_categories}): {all_categories}")

    if args.noise_symbols is not None:
        noise_symbols = {int(x) for x in args.noise_symbols.split(',')}
    else:
        noise_symbols = detect_noise_symbols(data)
    print(f"Noise symbols: {noise_symbols}")

    noise_map, split_map, new_sigma, new_s_dim, parent_symbol = \
        build_symbol_mapping(pmm.sigma, noise_symbols, n_categories)

    print(f"\nSymbol mapping (old sigma={pmm.sigma} -> new sigma={new_sigma}):")
    for old_sym, new_sym in sorted(noise_map.items()):
        print(f"  c{old_sym} (noise)  -> c{new_sym}")
    for old_sym, sub_syms in sorted(split_map.items()):
        for i, ns in enumerate(sub_syms):
            print(f"  c{old_sym} + {all_categories[i]}  -> c{ns}")
    print(f"Parent symbol: {parent_symbol}")

    # ==== 4. Split PMM topology ====
    print(f"\nSplitting topology ...")
    split_pmm, node_map, node_parent = split_topology(
        pmm.pmm, split_map, noise_map, n_categories)
    n_q_new = len(split_pmm['Q'])
    n_e_new = sum(len(d) for d in split_pmm['delta'].values())
    print(f"  {n_q} nodes -> {n_q_new} nodes")
    print(f"  {n_e} edges -> {n_e_new} edges (before replay prune)")
    print(f"  Node mapping: {node_map}")
    print(f"  Node parent:  {node_parent}")

    if len(node_map[0]) > 1:
        print("WARNING: initial node q0 is split. Per-episode routing needed "
              "but not yet implemented. Results may be incorrect.")

    # ==== 5. Extend RNN embeddings ====
    print(f"\nExtending RNN state embeddings: {pmm.s_dim} -> {new_s_dim} ...")
    extend_rnn_embeddings(pmm, new_s_dim, parent_symbol)

    # ==== 6. Re-label episodes ====
    print("Re-labelling episodes ...")
    new_episodes = relabel_episodes(
        pmm.episodes, ep_categories, all_categories,
        split_map, noise_map, new_s_dim)

    # ==== 7. Update PMM fields ====
    pmm.pmm = split_pmm
    pmm.episodes = new_episodes
    pmm.s_dim = new_s_dim
    pmm.sigma = new_sigma
    pmm.cluster_centers = build_new_cluster_centers(
        pmm.cluster_centers, noise_map, split_map, new_s_dim)

    # ==== 8. Recompute RNN embeddings & replay ====
    print("Recomputing RNN embeddings ...")
    pmm._precompute_embeddings()

    print("Replay assign ...")
    pmm._rebuild_cache()

    # ==== 9. Prune ====
    print("Pruning ...")
    pmm._prune_after_replay()

    # ==== 10. Save ====
    print(f"\nSaving split PMM to {args.output} ...")
    pmm.save_pmm(args.output)

    # Patch the saved pkl with split metadata
    with open(args.output, "rb") as f:
        payload = pickle.load(f)
    payload['splitted'] = {
        'split_key': args.split_key,
        'categories': all_categories,
        'parent_symbol': parent_symbol,
        'node_parent': node_parent,
    }
    with open(args.output, "wb") as f:
        pickle.dump(payload, f)

    dot_path = args.output.replace('.pkl', '.dot')
    pmm.save_dot(dot_path)

    # Summary
    n_q_final = len(pmm.pmm['Q'])
    n_e_final = sum(len(d) for d in pmm.pmm['delta'].values())
    print(f"\nFinal split PMM: {n_q_final} nodes, {n_e_final} edges, "
          f"{len(pmm.episodes)} episodes, sigma={pmm.sigma}")
    print(f"Node parent: {node_parent}")
    print(f"Parent symbol: {parent_symbol}")


if __name__ == "__main__":
    main()
