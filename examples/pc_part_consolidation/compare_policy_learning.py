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

from diagnose_search_coverage import DEFAULT_GENERATOR_PARAMS
from diagnose_search_coverage import canonical_groups
from diagnose_search_coverage import enumerate_feasible_solutions
from diagnose_search_coverage import load_policy
from diagnose_search_coverage import make_instance
from diagnose_search_coverage import rollout_batch
from diagnose_search_coverage import sample_nco_solutions
from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.policy import PCPolicy


def make_untrained_policy(env: PartConsolidationEnv, device: torch.device, seed: int, temperature: float) -> PCPolicy:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = env.generator
    policy = PCPolicy(
        node_feat_dim=generator.node_feat_dim,
        edge_feat_dim=generator.edge_feat_dim,
        emb_dim=128,
        num_message_passing=3,
        temperature=temperature,
    ).to(device)
    policy.eval()
    return policy


def rollout_uniform_mask(
    env: PartConsolidationEnv,
    td_single,
    batch_size: int,
    max_steps: int,
):
    td = td_single.repeat(batch_size).to(env.device)
    env._reward_static_td = td.clone()
    actions = []

    with torch.no_grad():
        for _ in range(max_steps):
            if td["done"].all():
                break
            mask = td["action_mask"].clone()
            no_valid = ~mask.any(dim=-1)
            if no_valid.any():
                mask[no_valid, 0] = True
            probs = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
            actions.append(action)
            td = env.step(td, action)

    if not actions:
        raise RuntimeError("No actions were generated during uniform rollout")
    actions_t = torch.stack(actions, dim=1)
    rewards = env.reward_from_actions(actions_t)
    groups = env.actions_to_groups(actions_t, N=env.N)
    return groups, rewards.detach().cpu().tolist()


def sample_uniform_solutions(
    env: PartConsolidationEnv,
    td_single,
    num_samples: int,
    chunk_size: int,
    max_steps: int,
):
    sampled: dict[tuple[tuple[int, ...], ...], dict] = {}
    best_key = None
    best_score = -float("inf")
    rewards_all: list[float] = []
    remaining = num_samples

    while remaining > 0:
        current = min(chunk_size, remaining)
        groups_batch, rewards = rollout_uniform_mask(env, td_single, current, max_steps)
        rewards_all.extend(float(reward) for reward in rewards)
        for groups, reward in zip(groups_batch, rewards):
            key = canonical_groups(groups)
            item = sampled.setdefault(key, {"count": 0, "score": float(reward), "groups": groups})
            item["count"] += 1
            item["score"] = max(item["score"], float(reward))
            if float(reward) > best_score:
                best_score = float(reward)
                best_key = key
        remaining -= current

    return sampled, best_key, rewards_all


def sample_policy_with_rewards(
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
    rewards_all: list[float] = []
    remaining = num_samples

    while remaining > 0:
        current = min(chunk_size, remaining)
        groups_batch, rewards = rollout_batch(env, policy, td_single, current, True, max_steps)
        rewards_all.extend(float(reward) for reward in rewards)
        for groups, reward in zip(groups_batch, rewards):
            key = canonical_groups(groups)
            item = sampled.setdefault(key, {"count": 0, "score": float(reward), "groups": groups})
            item["count"] += 1
            item["score"] = max(item["score"], float(reward))
            if float(reward) > best_score:
                best_score = float(reward)
                best_key = key
        remaining -= current

    return sampled, best_key, rewards_all


def summarize_samples(name: str, sampled: dict, best_key, rewards: list[float], feasible: dict | None) -> dict:
    best = sampled[best_key]
    row = {
        "method": name,
        "sample_mean_reward": float(np.mean(rewards)),
        "sample_std_reward": float(np.std(rewards)),
        "sample_best_reward": float(best["score"]),
        "unique_solutions": len(sampled),
        "best_groups": repr([list(group) for group in best_key]),
    }
    if feasible is not None:
        feasible_unique = sum(1 for key in sampled if key in feasible)
        row["unique_feasible_solutions"] = feasible_unique
        row["coverage_ratio"] = feasible_unique / max(len(feasible), 1)
        row["best_in_feasible_space"] = best_key in feasible
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Compare constraint-only random search, untrained NCO, and trained NCO on fixed PC instances."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--seed-start", type=int, default=4321)
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--init-seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--exact", action="store_true", help="Enumerate exact feasible space for each instance.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    generator = FPIGenerator(**DEFAULT_GENERATOR_PARAMS)
    env = PartConsolidationEnv(generator=generator, min_group_size_before_sep=1, device=str(device))
    max_steps = generator.max_num_parts * 2 + 4

    trained_policy = load_policy(args.checkpoint, env, device)
    untrained_policy = make_untrained_policy(env, device, seed=args.init_seed, temperature=args.temperature)

    rows = []
    for idx in range(args.instances):
        seed = args.seed_start + idx
        td_single = make_instance(env, seed=seed, device=device)

        feasible = None
        global_best_score = None
        if args.exact:
            feasible, global_best_key = enumerate_feasible_solutions(env, td_single)
            global_best_score = feasible[global_best_key]["score"]

        uniform_sampled, uniform_best_key, uniform_rewards = sample_uniform_solutions(
            env=env,
            td_single=td_single,
            num_samples=args.samples,
            chunk_size=args.chunk_size,
            max_steps=max_steps,
        )
        untrained_sampled, untrained_best_key, untrained_rewards = sample_policy_with_rewards(
            env=env,
            policy=untrained_policy,
            td_single=td_single,
            num_samples=args.samples,
            chunk_size=args.chunk_size,
            max_steps=max_steps,
        )
        trained_sampled, trained_best_key, trained_rewards = sample_policy_with_rewards(
            env=env,
            policy=trained_policy,
            td_single=td_single,
            num_samples=args.samples,
            chunk_size=args.chunk_size,
            max_steps=max_steps,
        )

        for row in [
            summarize_samples("uniform_mask", uniform_sampled, uniform_best_key, uniform_rewards, feasible),
            summarize_samples("untrained_nco", untrained_sampled, untrained_best_key, untrained_rewards, feasible),
            summarize_samples("trained_nco", trained_sampled, trained_best_key, trained_rewards, feasible),
        ]:
            row["seed"] = seed
            if global_best_score is not None:
                row["global_best_reward"] = global_best_score
                row["gap_to_global_best"] = global_best_score - row["sample_best_reward"]
            rows.append(row)

        print(f"\n===== seed {seed} =====")
        if global_best_score is not None:
            print(f"global_best_reward: {global_best_score:.6f}")
        for row in rows[-3:]:
            msg = (
                f"{row['method']}: mean={row['sample_mean_reward']:.6f}, "
                f"best={row['sample_best_reward']:.6f}, unique={row['unique_solutions']}"
            )
            if "coverage_ratio" in row:
                msg += f", coverage={row['coverage_ratio']:.4f}"
            print(msg)

    print("\n===== Overall Mean =====")
    for method in ["uniform_mask", "untrained_nco", "trained_nco"]:
        method_rows = [row for row in rows if row["method"] == method]
        print(
            f"{method}: "
            f"mean_reward={np.mean([r['sample_mean_reward'] for r in method_rows]):.6f}, "
            f"best_reward={np.mean([r['sample_best_reward'] for r in method_rows]):.6f}, "
            f"unique={np.mean([r['unique_solutions'] for r in method_rows]):.2f}"
        )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved_csv: {args.csv}")


if __name__ == "__main__":
    main()
