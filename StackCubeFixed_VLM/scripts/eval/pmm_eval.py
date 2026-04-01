#!/usr/bin/env python3
"""
PMM + Residual policy evaluation on fixed StackCube setup.

Environment setup (seed, cube positions, noise) mirrors the data-generation
script generate_demonstrations_with_video.py exactly.
"""
import os
import random
import sys
import types
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro

import mani_skill.envs
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from agent.pmm_agent import PMMResidualAgent, ResidualMLP
from agent.pmm_class import PMM
from agent.utils import build_feature_indexes_from_pmm, load_trajectories


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _pick_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state", "state_dict", "model", "encoder"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt if isinstance(ckpt, dict) else {}


def _print_param_report(name, obj):
    if isinstance(obj, torch.nn.Module):
        params = list(obj.parameters())
        n, b = sum(p.numel() for p in params), sum(p.numel() * p.element_size() for p in params)
    elif isinstance(obj, dict):
        ts = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        n, b = sum(t.numel() for t in ts), sum(t.numel() * t.element_size() for t in ts)
    else:
        n, b = 0, 0
    print(f"{name}: {n/1e6:.2f}M params, {b/1024/1024:.2f} MB")


def _load_ppo_feature_net(agent, ckpt_path, device):
    print(f"Loading PPO feature_net weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("agent_state_dict", ckpt)
    feature_state = {k[len("feature_net."):]: v for k, v in state.items() if k.startswith("feature_net.")}
    if not feature_state:
        raise ValueError(f"No feature_net weights found in PPO checkpoint: {ckpt_path}")
    incompatible = agent.feature_net.load_state_dict(feature_state, strict=True)
    for p in agent.feature_net.parameters():
        p.requires_grad = False
    agent.feature_net.eval()


# ---------------------------------------------------------------------------
# Fixed cube initialisation — identical to generate_demonstrations_with_video.py
# ---------------------------------------------------------------------------
def _fixed_initialize_episode(self_env, env_idx, options):
    with torch.device(self_env.device):
        b = len(env_idx)
        self_env.table_scene.initialize(env_idx)
        xyz = torch.zeros((b, 3), device=self_env.device)
        xyz[:, 2] = 0.02
        qs = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self_env.device,
                          dtype=torch.float32).repeat(b, 1)
        xyz[:, 0], xyz[:, 1] = 0.0, -0.175
        self_env.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
        xyz[:, 0], xyz[:, 1] = 0.0, 0.2
        self_env.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))


@dataclass
class EvalArgs:
    num_episodes: int = 64
    env_id: str = "StackCube-v1"
    residual_weights: str = "results/checkpoints/residual_net.pt"
    rnn_weights: str = "results/checkpoints/rnn_pretrained.pt"
    traj_pkl: str = "data/episodes_with_states.pkl"
    pmm_pkl: str = "results/split/pmm_split_stacking_direction.pkl"
    # pmm_pkl: str = "results/product/product_pmm_not_c6.pkl"
    encoder_weights: str = "/sata1/data/yiyuan/project_backup/Project_ENAP/4_stack_cube_maniskill/runs/StackCube-v1__ppo/ckpt_531.pt"
    seed: int = 42
    max_steps: int = 80
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    control_mode: str = "pd_joint_delta_pos"


def main():
    args = tyro.cli(EvalArgs)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load checkpoints
    # ------------------------------------------------------------------
    print(f"Loading weights from {args.residual_weights}...")
    checkpoint = torch.load(args.residual_weights, map_location=args.device, weights_only=False)
    assert "model" in checkpoint, f"Checkpoint missing 'model' key. Keys: {list(checkpoint.keys())}"
    model_state = checkpoint["model"]
    assert "norm_stats" in checkpoint, f"Checkpoint missing 'norm_stats' key. Keys: {list(checkpoint.keys())}"
    norm_stats = checkpoint["norm_stats"]

    episodes, cluster_centers = load_trajectories(args.traj_pkl)
    assert cluster_centers is not None, "cluster_centers missing in trajectory pkl."

    print("Loading PMM...")
    pmm = PMM(use_tqdm=True, max_inner_iters=1000)
    pmm.load_pmm(args.pmm_pkl)
    print(f"PMM: {len(pmm.pmm['Q'])} nodes, {len(pmm.pmm['delta'])} edges")

    print("Building feature indexes for state matching...")
    feature_indexes = build_feature_indexes_from_pmm(pmm, episodes, norm_stats)

    first_step = episodes[0][0]
    a_dim = len(first_step["action"])
    feat_dim = len(norm_stats["rgb_mean"]) + len(norm_stats["pos_mean"])

    residual_net = ResidualMLP(feat_dim, a_dim, hidden=256).to(args.device)
    residual_net.load_state_dict(model_state)
    residual_net.eval()

    _print_param_report("Residual", residual_net)
    rnn_ckpt = torch.load(args.rnn_weights, map_location="cpu", weights_only=False)
    _print_param_report("RNN", _pick_state_dict(rnn_ckpt))
    enc_ckpt = torch.load(args.encoder_weights, map_location="cpu", weights_only=False)
    _print_param_report("PPO Encoder", _pick_state_dict(enc_ckpt))

    # ------------------------------------------------------------------
    # Environment — matches generate_demonstrations_with_video.py exactly
    # ------------------------------------------------------------------
    env_kwargs = dict(obs_mode="rgb", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    env = gym.make(args.env_id, num_envs=1, reconfiguration_freq=1, **env_kwargs)
    env.unwrapped._initialize_episode = types.MethodType(_fixed_initialize_episode, env.unwrapped)
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)

    base_env = env
    while hasattr(base_env, "env") or hasattr(base_env, "_env"):
        base_env = getattr(base_env, "env", getattr(base_env, "_env", base_env))

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    obs, _ = env.reset(seed=args.seed)
    if "state" in obs:
        obs["state"] = obs["state"] + torch.randn_like(obs["state"]) * 0.1

    agent = PMMResidualAgent(
        envs=env,
        sample_obs=obs,
        pmm=pmm,
        residual_net=residual_net,
        cluster_centers=pmm.cluster_centers,
        device=args.device,
        feature_indexes=feature_indexes,
        norm_stats=norm_stats,
    )
    agent.to(args.device)
    _load_ppo_feature_net(agent, args.encoder_weights, args.device)
    agent.eval()

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    print(f"\nEvaluating {args.num_episodes} episodes (seed={args.seed}, fixed cube poses)...")
    success_count = 0
    red_on_green_count = 0
    green_on_red_count = 0
    total_rewards = []

    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed)
        if "state" in obs:
            obs["state"] = obs["state"] + torch.randn_like(obs["state"]) * 0.1
        agent.reset_pmm_state()

        done = False
        ep_reward = 0.0
        ep_len = 0
        episode_success = False
        stacking_direction = "none"
        prev_cubeA_z = None
        prev_cubeB_z = None

        while not done and ep_len < args.max_steps:
            obs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu()
                elif isinstance(action, dict):
                    action = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in action.items()}

            obs, reward, terminations, truncations, infos = env.step(action)

            # Per-step state noise — matches data generation
            if "state" in obs:
                obs["state"] = obs["state"] + torch.randn_like(obs["state"]) * 0.01

            done = bool(terminations.item() or truncations.item())
            ep_len += 1
            ep_reward += reward.item()

            if not done:
                prev_cubeA_z = base_env.cubeA.pose.p[0, 2].item()
                prev_cubeB_z = base_env.cubeB.pose.p[0, 2].item()

            if "final_info" in infos:
                mask = infos.get("_final_info", False)
                if isinstance(mask, torch.Tensor):
                    mask = bool(mask.item())
                if mask and "success_once" in infos["final_info"]["episode"]:
                    episode_success = bool(infos["final_info"]["episode"]["success_once"].item())
                    if episode_success and prev_cubeA_z is not None and prev_cubeB_z is not None:
                        if prev_cubeA_z > prev_cubeB_z:
                            stacking_direction = "red_on_green"
                        elif prev_cubeB_z > prev_cubeA_z:
                            stacking_direction = "green_on_red"

        total_rewards.append(ep_reward)
        if episode_success:
            success_count += 1
            if stacking_direction == "red_on_green":
                red_on_green_count += 1
            elif stacking_direction == "green_on_red":
                green_on_red_count += 1

        status = "SUCCESS" if episode_success else "FAILED"
        print(
            f"Episode {ep+1}/{args.num_episodes}: {status} | "
            f"Reward: {ep_reward:.2f} | Length: {ep_len} | Stacking: {stacking_direction}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    success_rate = 100.0 * success_count / max(args.num_episodes, 1)
    avg_reward = float(np.mean(total_rewards)) if total_rewards else 0.0
    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Environment: {args.env_id}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    if success_count > 0:
        print("Stacking Direction (successful episodes):")
        print(f"  Red on Green: {red_on_green_count} ({red_on_green_count/success_count*100:.1f}%)")
        print(f"  Green on Red: {green_on_red_count} ({green_on_red_count/success_count*100:.1f}%)")
    print("=" * 40)

    env.close()


if __name__ == "__main__":
    main()
