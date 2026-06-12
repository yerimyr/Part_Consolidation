from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.policy import PCPolicy


DEFAULT_GENERATOR_PARAMS = dict(
    num_parts=10,
    max_num_parts=10,
    material_types=2,
    p_relative_motion=0.05,
    p_extra_edge=0.70,
    L_low=20.0,
    L_high=120.0,
    W_low=10.0,
    W_high=55.0,
    H_low=2.0,
    H_high=24.0,
    build_limit_L=360.0,
    build_limit_W=180.0,
    build_limit_H=90.0,
    p_maint_H=0.05,
    p_standard=0.02,
)


def canonical_groups(groups: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Canonical 1-based grouping representation for exact set comparison."""
    return tuple(sorted(tuple(sorted(group)) for group in groups if group))


def mask_to_group(mask: int, n: int) -> list[int]:
    return [idx + 1 for idx in range(n) if mask & (1 << idx)]


def enumerate_feasible_groups(env: PartConsolidationEnv, td) -> dict[int, list[tuple[int, list[int]]]]:
    n = int(td["num_parts"][0].item())
    groups_by_first = {idx: [] for idx in range(1, n + 1)}

    size = td["size"][0]
    build_limit = td["build_limit"][0]
    isstandard = td["isstandard"][0]
    mat_var = td["mat_var"][0]
    maint_diff = td["maint_diff"][0]
    rel_motion = td["rel_motion"][0]
    assembly_adj = td["assembly_adj"][0]

    for mask in range(1, 1 << n):
        group = mask_to_group(mask, n)
        if not env._group_feasible(
            group,
            size,
            build_limit,
            isstandard,
            mat_var,
            maint_diff,
            rel_motion,
            assembly_adj,
        ):
            continue
        groups_by_first[group[0]].append((mask, group))
    return groups_by_first


def score_partition(env: PartConsolidationEnv, groups: list[list[int]]) -> tuple[float, dict[str, float]]:
    metrics_t = env._terminal_reward_components([groups], device=env.device)
    metrics = {key: float(value[0].item()) for key, value in metrics_t.items()}
    score = env._terminal_reward_score(metrics_t)[0].item()
    return float(score), metrics


def enumerate_feasible_solutions(env: PartConsolidationEnv, td):
    n = int(td["num_parts"][0].item())
    env._reward_static_td = td.clone().to(env.device)
    groups_by_first = enumerate_feasible_groups(env, td)

    solutions: dict[tuple[tuple[int, ...], ...], dict] = {}
    best_key = None
    best_score = -float("inf")

    def backtrack(remaining_mask: int, current_groups: list[list[int]]):
        nonlocal best_key, best_score
        if remaining_mask == 0:
            key = canonical_groups(current_groups)
            score, metrics = score_partition(env, current_groups)
            solutions[key] = {"score": score, "metrics": metrics, "groups": [list(g) for g in current_groups]}
            if score > best_score:
                best_score = score
                best_key = key
            return

        first_zero_based = (remaining_mask & -remaining_mask).bit_length() - 1
        first = first_zero_based + 1
        for group_mask, group in groups_by_first[first]:
            if group_mask & remaining_mask != group_mask:
                continue
            current_groups.append(group)
            backtrack(remaining_mask ^ group_mask, current_groups)
            current_groups.pop()

    backtrack((1 << n) - 1, [])
    return solutions, best_key


def make_instance(env: PartConsolidationEnv, seed: int, device: torch.device):
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return env.reset(batch_size=1).to(device).clone()
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        np.random.set_state(np_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def load_policy(checkpoint_path: Path, env: PartConsolidationEnv, device: torch.device) -> PCPolicy:
    generator = env.generator
    policy = PCPolicy(
        node_feat_dim=generator.node_feat_dim,
        edge_feat_dim=generator.edge_feat_dim,
        emb_dim=128,
        num_message_passing=3,
        temperature=1.2,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("policy", checkpoint)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def rollout_batch(
    env: PartConsolidationEnv,
    policy: PCPolicy,
    td_single,
    batch_size: int,
    sample: bool,
    max_steps: int,
):
    td = td_single.repeat(batch_size).to(env.device)
    env._reward_static_td = td.clone()
    actions = []

    with torch.no_grad():
        for _ in range(max_steps):
            if td["done"].all():
                break
            action, _, _, _ = policy.act(td, sample=sample, epsilon=0.0)
            actions.append(action)
            td = env.step(td, action)

    if not actions:
        raise RuntimeError("No actions were generated during rollout")
    actions_t = torch.stack(actions, dim=1)
    rewards = env.reward_from_actions(actions_t)
    groups = env.actions_to_groups(actions_t, N=env.N)
    return groups, rewards.detach().cpu().tolist()


def sample_nco_solutions(
    env: PartConsolidationEnv,
    policy: PCPolicy,
    td_single,
    num_samples: int,
    chunk_size: int,
    max_steps: int,
):
    sampled: dict[tuple[tuple[int, ...], ...], dict] = {}
    best_key = None
    best_score = -float("inf")
    remaining = num_samples

    while remaining > 0:
        current = min(chunk_size, remaining)
        groups_batch, rewards = rollout_batch(env, policy, td_single, current, True, max_steps)
        for groups, reward in zip(groups_batch, rewards):
            key = canonical_groups(groups)
            item = sampled.setdefault(key, {"count": 0, "score": float(reward), "groups": groups})
            item["count"] += 1
            item["score"] = max(item["score"], float(reward))
            if float(reward) > best_score:
                best_score = float(reward)
                best_key = key
        remaining -= current

    return sampled, best_key


def write_sample_csv(path: Path, sampled: dict, feasible: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["groups", "count", "score", "in_feasible_space", "global_score"])
        writer.writeheader()
        for key, item in sorted(sampled.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
            writer.writerow(
                {
                    "groups": repr([list(group) for group in key]),
                    "count": item["count"],
                    "score": item["score"],
                    "in_feasible_space": key in feasible,
                    "global_score": feasible[key]["score"] if key in feasible else "",
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether NCO samples cover the exact feasible search space for one PC instance."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-samples", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = FPIGenerator(**DEFAULT_GENERATOR_PARAMS)
    env = PartConsolidationEnv(generator=generator, min_group_size_before_sep=1, device=str(device))
    td_single = make_instance(env, seed=args.seed, device=device)

    print("===== Instance =====")
    print(f"seed: {args.seed}")
    print(f"num_parts: {int(td_single['num_parts'][0].item())}")
    print(f"checkpoint: {args.checkpoint}")

    print("\n===== Exact Feasible Search Space =====")
    feasible, global_best_key = enumerate_feasible_solutions(env, td_single)
    global_best = feasible[global_best_key]
    print(f"total_feasible_solutions: {len(feasible)}")
    print(f"global_best_reward: {global_best['score']:.6f}")
    print(f"global_best_groups: {[list(group) for group in global_best_key]}")
    print(
        "global_best_components: "
        + ", ".join(
            f"{key}={global_best['metrics'][key]:.6f}"
            for key in ["Q_observed", "Q_expected", "Q_gamma", "num_groups"]
        )
    )

    policy = load_policy(args.checkpoint, env, device)
    max_steps = generator.max_num_parts * 2 + 4

    greedy_groups_batch, greedy_rewards = rollout_batch(env, policy, td_single, 1, False, max_steps)
    greedy_key = canonical_groups(greedy_groups_batch[0])
    greedy_reward = float(greedy_rewards[0])

    print("\n===== NCO Greedy =====")
    print(f"greedy_reward: {greedy_reward:.6f}")
    print(f"greedy_groups: {[list(group) for group in greedy_key]}")
    print(f"greedy_in_feasible_space: {greedy_key in feasible}")
    print(f"greedy_gap_to_global_best: {global_best['score'] - greedy_reward:.6f}")

    print("\n===== NCO Sampling =====")
    sampled, sampled_best_key = sample_nco_solutions(
        env=env,
        policy=policy,
        td_single=td_single,
        num_samples=args.samples,
        chunk_size=args.chunk_size,
        max_steps=max_steps,
    )
    sampled_best = sampled[sampled_best_key]
    feasible_sampled_unique = sum(1 for key in sampled if key in feasible)
    coverage = feasible_sampled_unique / max(len(feasible), 1)
    print(f"samples: {args.samples}")
    print(f"unique_sampled_solutions: {len(sampled)}")
    print(f"unique_sampled_feasible_solutions: {feasible_sampled_unique}")
    print(f"coverage_ratio: {coverage:.8f}")
    print(f"sampled_best_reward: {sampled_best['score']:.6f}")
    print(f"sampled_best_groups: {[list(group) for group in sampled_best_key]}")
    print(f"sampled_best_in_feasible_space: {sampled_best_key in feasible}")
    print(f"sampled_gap_to_global_best: {global_best['score'] - sampled_best['score']:.6f}")

    if sampled_best_key in feasible:
        sorted_scores = sorted((item["score"] for item in feasible.values()), reverse=True)
        better = sum(1 for score in sorted_scores if score > sampled_best["score"] + 1e-12)
        print(f"sampled_best_rank_among_feasible: {better + 1} / {len(feasible)}")

    if args.save_samples is not None:
        write_sample_csv(args.save_samples, sampled, feasible)
        print(f"\nsaved_sample_summary: {args.save_samples}")


if __name__ == "__main__":
    main()
