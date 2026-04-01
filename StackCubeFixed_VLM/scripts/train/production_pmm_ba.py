#!/usr/bin/env python3
"""
PMM × Büchi product construction.

Two-phase approach:
  Phase 1 – High-level skeleton:
      1. Parse Büchi automaton from DOT file
      2. Build product topology (Mealy × Büchi)
         - alphabet = intersection of PMM sigma and BA symbols
         - BFS prune unreachable product states

  Phase 2 – Replay assign on product skeleton:
      3. Walk every episode through the product graph in a single pass:
         validate AND assign actions to product edges simultaneously.
         Reject entire episode if:
           a) any step has no valid transition, OR
           b) final product state is NOT in the BA accepting set
      4. Set product topology + caches on the PMM object
      5. Prune empty edges / unreachable nodes → save

Usage:
    python scripts/train/production_pmm_ba.py \\
        --pmm-pkl results/pmm_residual_training.pkl \\
        --ba-dot  results/preference/not_c1.dot
"""
import argparse
import os
import re
import sys

import numpy as np
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.pmm_class import PMM


# -----------------------------------------------------------------------
# Büchi automaton parser
# -----------------------------------------------------------------------
def parse_ba_dot(dot_path):
    """Parse a Büchi automaton from a Graphviz DOT file.

    Expected format (see results/preference/not_c1.dot):
      - Accepting nodes have ``shape=doublecircle``
      - Initial state is the first declared node
      - Edge labels are ``"cN"`` where N is an integer symbol index

    Returns dict with keys:
        states   : list[str]           – declared state names (order preserved)
        initial  : str                 – initial state name
        accepting: set[str]            – accepting state names
        delta    : dict[(str,int),str] – transition function  (state, symbol) → state
        symbols  : set[int]            – alphabet (symbols appearing on edges)
    """
    with open(dot_path) as f:
        text = f.read()

    accepting = {m.group(1) for m in
                 re.finditer(r'(\w+)\s*\[.*?shape\s*=\s*doublecircle.*?\]', text)}

    skip = {'digraph', 'node', 'edge', 'graph', 'rankdir', 'subgraph'}
    declared = []
    for m in re.finditer(r'^\s*(\w+)\s*\[', text, re.MULTILINE):
        name = m.group(1)
        if name not in skip and name not in declared:
            declared.append(name)

    delta, symbols = {}, set()
    for m in re.finditer(r'(\w+)\s*->\s*(\w+)\s*\[label\s*=\s*"c(\d+)"', text):
        src, dst, sym = m.group(1), m.group(2), int(m.group(3))
        delta[(src, sym)] = dst
        symbols.add(sym)

    return {
        'states': declared,
        'initial': declared[0] if declared else None,
        'accepting': accepting,
        'delta': delta,
        'symbols': symbols,
    }


# -----------------------------------------------------------------------
# Product topology
# -----------------------------------------------------------------------
def build_product(pmm_data, ba, pmm_sigma):
    """Build product graph  PMM × BA.

    Product states are ``(q_mealy, q_buchi)`` tuples, remapped to contiguous
    ints with the initial state guaranteed to be index 0.

    Returns dict with:
        Q, delta, reps, rep_embeddings  – standard PMM topology fields
        accepting        – set of product-node indices whose BA component ∈ F
        product_symbols  – sorted list (intersection of PMM & BA alphabets)
        initial          – int (always 0)
    """
    pmm_delta = pmm_data['delta']
    pmm_reps = pmm_data['reps']
    pmm_rep_emb = pmm_data['rep_embeddings']
    rnn_h_dim = pmm_rep_emb[0].shape[0] if pmm_rep_emb else 64

    ba_states = ba['states']
    ba_initial = ba['initial']
    ba_delta = ba['delta']
    ba_accepting = ba['accepting']

    product_symbols = set(pmm_sigma) & ba['symbols']

    # --- raw product transitions ---
    raw_delta = {}  # ((q_m, q_b), x) -> {(q_m', q_b'): prob}

    for (q_m, x), dists in pmm_delta.items():
        if x not in product_symbols:
            continue
        for q_b in ba_states:
            if (q_b, x) not in ba_delta:
                continue
            q_b_next = ba_delta[(q_b, x)]
            key = ((q_m, q_b), x)
            if key not in raw_delta:
                raw_delta[key] = {}
            for q_m_next, prob in dists.items():
                raw_delta[key][(q_m_next, q_b_next)] = prob

    # --- BFS from initial product state ---
    initial = (0, ba_initial)
    reachable = set()
    frontier = [initial]
    while frontier:
        node = frontier.pop(0)
        if node in reachable:
            continue
        reachable.add(node)
        for (src, _x), dists in raw_delta.items():
            if src == node:
                for dst in dists:
                    if dst not in reachable:
                        frontier.append(dst)

    # --- remap to contiguous ints, initial → 0 ---
    ordered = sorted(reachable)
    if initial in ordered:
        ordered.remove(initial)
        ordered.insert(0, initial)
    idx = {n: i for i, n in enumerate(ordered)}

    remapped_delta = {}
    for (src, x), dists in raw_delta.items():
        if src not in reachable:
            continue
        nd = {}
        for dst, prob in dists.items():
            if dst in reachable:
                nd[idx[dst]] = prob
        if nd:
            total = sum(nd.values())
            remapped_delta[(idx[src], x)] = {k: v / total for k, v in nd.items()}

    new_reps, new_rep_emb, product_accepting = [], [], set()
    for prod_node in ordered:
        q_m, q_b = prod_node
        new_reps.append(pmm_reps[q_m] if q_m < len(pmm_reps) else tuple())
        new_rep_emb.append(pmm_rep_emb[q_m] if q_m < len(pmm_rep_emb)
                           else np.zeros(rnn_h_dim))
        if q_b in ba_accepting:
            product_accepting.add(idx[prod_node])

    n = len(ordered)
    n_edges = sum(len(d) for d in remapped_delta.values())
    print(f"Product topology:")
    print(f"  PMM {len(pmm_data['Q'])} nodes | BA {len(ba_states)} states")
    print(f"  Alphabet (intersection): {sorted(product_symbols)}")
    print(f"  Reachable product nodes: {n} / {len(pmm_data['Q']) * len(ba_states)}")
    print(f"  Product edges: {n_edges}")
    print(f"  Accepting nodes: {len(product_accepting)} / {n}")

    return {
        'Q': list(range(n)),
        'delta': remapped_delta,
        'reps': new_reps,
        'rep_embeddings': new_rep_emb,
        'accepting': product_accepting,
        'product_symbols': sorted(product_symbols),
        'initial': 0,
    }


# -----------------------------------------------------------------------
# Phase 2: Replay assign (validate + assign in one pass)
# -----------------------------------------------------------------------
def _disambiguate(dist, z_t, rep_embeddings, fallback_q):
    """Pick destination among multiple outgoing edges via RNN cosine similarity.

    Mirrors pmm_class.replay_assign / pmm_agent._disambiguate.
    """
    if len(dist) == 1:
        return next(iter(dist))
    if z_t is not None and rep_embeddings:
        z_norm = z_t / (np.linalg.norm(z_t) + 1e-12)
        best_q, best_sim = fallback_q, -2.0
        for q_cand in dist:
            if q_cand < len(rep_embeddings):
                z_rep = rep_embeddings[q_cand]
                sim = np.dot(z_norm, z_rep / (np.linalg.norm(z_rep) + 1e-12))
                if sim > best_sim:
                    best_sim = sim
                    best_q = q_cand
        return best_q
    return max(dist, key=dist.get)


def replay_on_product(pmm_obj, product):
    """Single-pass replay: validate episodes AND assign actions on the product graph.

    For each episode, walks through the product delta step by step.
    An episode is **rejected** (entire episode discarded) if:
      1. At any step, ``(current_node, x)`` has no transition in the product, OR
      2. The final product state is NOT in the BA accepting set.

    For valid episodes, actions are recorded into edge_cache and qx_actions,
    identical to what ``pmm_class._rebuild_cache`` would produce.

    Node disambiguation: identical to ``pmm_class.replay_assign``.

    Returns:
        valid_indices : list[int]        – original episode indices that passed
        edge_cache    : dict[(q,x,q') -> list[(epi,t)]]
        qx_actions    : dict[(q,x) -> list[np.ndarray]]
    """
    delta = product['delta']
    rep_emb = product['rep_embeddings']
    accepting = product['accepting']

    edge_cache = defaultdict(list)
    qx_actions = defaultdict(list)
    valid_indices = []
    reasons = defaultdict(int)

    for epi, ep in enumerate(pmm_obj.episodes):
        T = ep['S'].shape[0]
        if T == 0:
            continue

        q = 0
        trajectory = []
        ok = True

        for t in range(T):
            x = int(np.argmax(ep['S'][t]))
            dist = delta.get((q, x))
            if dist is None:
                ok = False
                reasons['no_transition'] += 1
                break
            z_t = pmm_obj._get_step_embedding(epi, t)
            q_next = _disambiguate(dist, z_t, rep_emb, q)
            trajectory.append((q, x, q_next, t))
            q = q_next

        if ok and q not in accepting:
            ok = False
            reasons['not_accepting'] += 1

        if ok:
            valid_indices.append(epi)
            for q_src, x, q_dst, t in trajectory:
                edge_cache[(q_src, x, q_dst)].append((epi, t))
                qx_actions[(q_src, x)].append(ep['A'][t])

    rejected = len(pmm_obj.episodes) - len(valid_indices)
    print(f"Replay on product: kept {len(valid_indices)}, "
          f"rejected {rejected}  {dict(reasons)}")
    return valid_indices, dict(edge_cache), dict(qx_actions)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PMM × Büchi automaton product construction")
    parser.add_argument("--pmm-pkl", default="results/split/pmm_split_stacking_direction.pkl",
                        help="Path to the trained PMM pkl")
    parser.add_argument("--ba-dot", default="results/preference/not_c3.dot",
                        help="Path to the Büchi automaton DOT file")
    parser.add_argument("--output", default=None,
                        help="Output pkl path (default: results/product_pmm_<ba>.pkl)")
    args = parser.parse_args()

    if args.output is None:
        ba_name = os.path.splitext(os.path.basename(args.ba_dot))[0]
        args.output = os.path.join("results", "product", f"product_pmm_{ba_name}.pkl")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # ---- 1. Load PMM ----
    print(f"Loading PMM from {args.pmm_pkl} ...")
    pmm = PMM(use_tqdm=True)
    pmm.load_pmm(args.pmm_pkl)
    n_q = len(pmm.pmm['Q'])
    n_e = sum(len(d) for d in pmm.pmm['delta'].values())
    print(f"  {n_q} nodes, {n_e} edges, {len(pmm.episodes)} episodes, sigma={pmm.sigma}")

    # ---- 2. Parse BA ----
    print(f"\nParsing BA from {args.ba_dot} ...")
    ba = parse_ba_dot(args.ba_dot)
    print(f"  States: {ba['states']}, initial: {ba['initial']}")
    print(f"  Accepting: {ba['accepting']}, symbols: {sorted(ba['symbols'])}")

    # ==== Phase 1: High-level product skeleton ====
    print("\nPhase 1: Building product skeleton ...")
    product = build_product(pmm.pmm, ba, pmm.sigma)

    # ==== Phase 2: Replay assign on product skeleton ====
    print("\nPhase 2: Replay assign on product graph ...")
    valid_idx, edge_cache, qx_actions = replay_on_product(pmm, product)

    if not valid_idx:
        print("ERROR: No valid episodes remain after product filtering. "
              "Check BA accepting states and product alphabet.")
        sys.exit(1)

    # Zero out invalid episodes (preserve original indices)
    valid_set = set(valid_idx)
    for i in range(len(pmm.episodes)):
        if i not in valid_set:
            pmm.episodes[i] = {'S': np.zeros((0, pmm.s_dim)), 'A': np.zeros((0, pmm.a_dim))}

    # Set product topology + caches directly (no redundant re-replay)
    pmm.pmm = {
        'Q': product['Q'],
        'delta': product['delta'],
        'reps': product['reps'],
        'rep_embeddings': product['rep_embeddings'],
    }
    pmm.sigma = product['product_symbols']
    pmm._edge_cache = edge_cache
    pmm._qx_actions = qx_actions

    # Prune empty edges / unreachable nodes (internally remaps + rebuilds final cache)
    pmm._prune_after_replay()

    # ---- 8. Save ----
    print(f"\nSaving product PMM to {args.output} ...")
    pmm.save_pmm(args.output)

    dot_path = args.output.replace('.pkl', '.dot')
    pmm.save_dot(dot_path)

    # ---- Summary ----
    n_q = len(pmm.pmm['Q'])
    n_e = sum(len(d) for d in pmm.pmm['delta'].values())
    print(f"\nFinal product PMM: {n_q} nodes, {n_e} edges, "
          f"{len(pmm.episodes)} episodes, sigma={pmm.sigma}")


if __name__ == "__main__":
    main()
