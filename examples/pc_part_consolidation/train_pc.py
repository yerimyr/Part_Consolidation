from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.policy import PCPolicy


def tensor_device_name(x) -> str:
    return str(x.device) if torch.is_tensor(x) else "unknown"


def first_parameter_device(module: torch.nn.Module) -> str:
    return str(next(module.parameters()).device)


def rollout_episode_from_td(
    env: PartConsolidationEnv,
    policy: PCPolicy,
    td_init,
    max_steps: int,
    sample: bool = True,
    epsilon: float = 0.0,
):
    env._reward_static_td = td_init.clone().to(env.device)
    td = td_init.clone().to(env.device)

    actions = []
    logps = []
    entropies = []
    step_counts = torch.zeros((td.batch_size[0],), dtype=torch.float32, device=env.device)

    for _ in range(max_steps):
        active = ~td["done"].view(-1)
        action, logp, entropy, _ = policy.act(td, sample=sample, epsilon=epsilon)
        actions.append(action)
        logps.append(logp)
        entropies.append(entropy)

        td = env.step(td, action)
        step_counts = step_counts + active.float()

        if td["done"].all():
            break

    actions = torch.stack(actions, dim=1)
    logps = torch.stack(logps, dim=1)
    entropies = torch.stack(entropies, dim=1)

    terminal_reward = env.reward_from_actions(actions)
    total_reward = terminal_reward

    return actions, logps, entropies, terminal_reward, total_reward, td, step_counts


def make_fixed_eval_td(
    env: PartConsolidationEnv,
    batch_size: int,
    seed: int,
    device: str,
):
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return env.reset(batch_size=batch_size).to(device).clone()
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)


def group_feasible_for_instance(env: PartConsolidationEnv, td_static, batch_idx: int, group: list[int]) -> bool:
    return env._group_feasible(
        sorted(group),
        td_static["size"][batch_idx],
        td_static["build_limit"][batch_idx],
        td_static["isstandard"][batch_idx],
        td_static["mat_var"][batch_idx],
        td_static["maint_diff"][batch_idx],
        td_static["rel_motion"][batch_idx],
        td_static["assembly_adj"][batch_idx],
    )


def mutate_grouping_once(env: PartConsolidationEnv, td_static, batch_idx: int, groups: list[list[int]]):
    groups = [sorted(group) for group in groups if group]
    if not groups:
        return None

    ops = ["split", "merge"]
    random.shuffle(ops)

    for op in ops:
        if op == "merge" and len(groups) >= 2:
            pairs = [(i, j) for i in range(len(groups)) for j in range(i + 1, len(groups))]
            random.shuffle(pairs)
            for i, j in pairs:
                merged = sorted(groups[i] + groups[j])
                if not group_feasible_for_instance(env, td_static, batch_idx, merged):
                    continue
                next_groups = [g[:] for k, g in enumerate(groups) if k not in (i, j)]
                next_groups.append(merged)
                return [sorted(g) for g in next_groups]

        if op == "split":
            candidates = [idx for idx, group in enumerate(groups) if len(group) >= 2]
            random.shuffle(candidates)
            for idx in candidates:
                group = groups[idx][:]
                nodes = group[:]
                random.shuffle(nodes)
                for node in nodes:
                    left = [n for n in group if n != node]
                    right = [node]
                    if not left:
                        continue
                    if not group_feasible_for_instance(env, td_static, batch_idx, left):
                        continue
                    next_groups = [g[:] for k, g in enumerate(groups) if k != idx]
                    next_groups.extend([sorted(left), right])
                    return [sorted(g) for g in next_groups]

    return None


def reward_for_single_grouping(env: PartConsolidationEnv, td_static, batch_idx: int, groups: list[list[int]]) -> torch.Tensor:
    old_static_td = env._reward_static_td
    try:
        env._reward_static_td = td_static[batch_idx : batch_idx + 1].clone()
        metrics = env._terminal_reward_components([groups], device=td_static["W"].device)
        return metrics["Q_gamma"][0]
    finally:
        env._reward_static_td = old_static_td


def grouping_to_action_sequence(
    env: PartConsolidationEnv,
    td_single,
    groups: list[list[int]],
    max_steps: int,
):
    td = td_single.clone()
    actions = []

    for group in [sorted(g) for g in groups if len(g) >= 2]:
        base = group[0]
        pending = group[1:]
        while pending:
            group_id = td["group_id"][0]
            base_gid = int(group_id[base].item())
            matched = False

            for node in list(pending):
                node_gid = int(group_id[node].item())
                if base_gid < 0 or node_gid < 0 or base_gid == node_gid:
                    pending.remove(node)
                    matched = True
                    break

                pair = tuple(sorted((base_gid, node_gid)))
                if pair not in env.group_pair_list:
                    continue
                action = env.group_pair_list.index(pair) + 1
                if not bool(td["action_mask"][0, action].item()):
                    continue

                action_tensor = torch.tensor([action], dtype=torch.long, device=td["group_id"].device)
                actions.append(action)
                td = env.step(td, action_tensor)
                pending.remove(node)
                matched = True
                break

            if not matched:
                return None

            if len(actions) >= max_steps:
                return None

    if len(actions) < max_steps:
        actions.append(0)
    return actions


def logprob_of_action_sequences(
    env: PartConsolidationEnv,
    policy: PCPolicy,
    td_init,
    action_sequences: list[list[int]],
) -> torch.Tensor:
    device = td_init["group_id"].device
    if not action_sequences:
        return torch.empty((0,), dtype=torch.float32, device=device)

    lengths = torch.tensor([len(seq) for seq in action_sequences], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())
    action_tensor = torch.zeros((len(action_sequences), max_len), dtype=torch.long, device=device)
    for idx, seq in enumerate(action_sequences):
        action_tensor[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    td = td_init.clone()
    logp_sum = torch.zeros((len(action_sequences),), dtype=torch.float32, device=device)

    for t in range(max_len):
        active = t < lengths
        if not bool(active.any().item()):
            break

        node_emb = policy.encode(td)
        logits = policy.compute_logits(node_emb, td) / policy.temperature
        mask = td["action_mask"].clone()
        no_valid = ~mask.any(dim=-1)
        if no_valid.any():
            mask[no_valid, 0] = True
        logits = logits.masked_fill(~mask, -1e9)
        probs = torch.softmax(logits, dim=-1)

        action = action_tensor[:, t]
        chosen = probs.gather(1, action.view(-1, 1)).clamp_min(1e-12).squeeze(-1)
        logp_sum = logp_sum + torch.where(active, torch.log(chosen), torch.zeros_like(logp_sum))
        td = env.step(td, action)

    return logp_sum


def compute_mutation_auxiliary_loss(
    env: PartConsolidationEnv,
    policy: PCPolicy,
    td_static,
    sampled_actions: torch.Tensor,
    sampled_reward: torch.Tensor,
    mutation_frac: float,
    mutation_attempts: int,
    mutation_accept_worse: bool,
    mutation_worse_accept_prob: float,
    mutation_worse_weight: float,
    max_steps: int,
):
    batch_size = sampled_actions.size(0)
    selected_count = int(round(batch_size * mutation_frac))
    if selected_count <= 0 or mutation_attempts <= 0:
        zero = sampled_reward.new_tensor(0.0)
        return zero, {
            "applied_ratio": 0.0,
            "improved_ratio": 0.0,
            "reward_gain": 0.0,
            "kept_count": 0,
            "worse_kept_count": 0,
            "worse_accept_ratio": 0.0,
        }

    selected_count = min(batch_size, selected_count)
    selected = torch.randperm(batch_size, device=sampled_actions.device)[:selected_count].tolist()
    sampled_groups = env.actions_to_groups(sampled_actions, N=env.N)

    kept_indices = []
    action_sequences = []
    weights = []
    gains = []
    improved_count = 0
    worse_kept_count = 0

    with torch.no_grad():
        for batch_idx in selected:
            base_groups = sampled_groups[batch_idx]
            best_groups = None
            best_gain = None

            for _ in range(mutation_attempts):
                mutated = mutate_grouping_once(env, td_static, batch_idx, base_groups)
                if mutated is None:
                    continue
                reward_mut = reward_for_single_grouping(env, td_static, batch_idx, mutated)
                gain = reward_mut - sampled_reward[batch_idx]
                if best_gain is None or gain > best_gain:
                    best_gain = gain
                    best_groups = mutated

            if best_groups is None or best_gain is None:
                continue

            improved = bool((best_gain > 0).item())
            if improved:
                sequence_weight = best_gain.detach()
                improved_count += 1
            else:
                if not mutation_accept_worse:
                    continue
                if torch.rand((), device=sampled_actions.device).item() > mutation_worse_accept_prob:
                    continue
                sequence_weight = sampled_reward.new_tensor(float(mutation_worse_weight))
                worse_kept_count += 1

            seq = grouping_to_action_sequence(
                env=env,
                td_single=td_static[batch_idx : batch_idx + 1],
                groups=best_groups,
                max_steps=max_steps,
            )
            if seq is None:
                continue

            action_sequences.append(seq)
            kept_indices.append(batch_idx)
            weights.append(sequence_weight)
            gains.append(float(best_gain.detach().item()))

    if not action_sequences:
        zero = sampled_reward.new_tensor(0.0)
        return zero, {
            "applied_ratio": selected_count / float(batch_size),
            "improved_ratio": 0.0,
            "reward_gain": 0.0,
            "kept_count": 0,
            "worse_kept_count": 0,
            "worse_accept_ratio": 0.0,
        }

    selected_td = torch.cat([td_static[idx : idx + 1] for idx in kept_indices], dim=0)
    weight_tensor = torch.stack(weights).to(sampled_actions.device)
    logp_mut = logprob_of_action_sequences(env, policy, selected_td, action_sequences)
    loss_mut = -(weight_tensor.detach() * logp_mut).mean()

    return loss_mut, {
        "applied_ratio": selected_count / float(batch_size),
        "improved_ratio": improved_count / float(selected_count),
        "reward_gain": float(np.mean(gains)) if gains else 0.0,
        "kept_count": len(action_sequences),
        "worse_kept_count": worse_kept_count,
        "worse_accept_ratio": worse_kept_count / float(selected_count),
    }


def main():
    train_start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # =========================
    # Hyperparameters
    # =========================
    batch_size = 256
    eval_batch_size = 128
    eval_seed = 4321
    epochs = 500
    lr = 1e-4
    grad_clip = 1.0
    entropy_coef = 0.05
    temperature = 2.0
    mutation_frac = 0.10
    mutation_attempts = 1
    mutation_loss_weight = 0.30
    mutation_accept_worse = True
    mutation_worse_accept_prob = 0.20
    mutation_worse_weight = 0.02

    # =========================
    # TensorBoard
    # =========================
    log_dir = f"runs/pc_general_graph_groupcount_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard log dir:", log_dir)
    timing_csv_path = Path(log_dir) / "timing_by_epoch.csv"
    timing_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with timing_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            [
                "epoch",
                "batch_index",
                "batch_size",
                "problem_generation_and_transfer_sec",
                "sampling_and_greedy_sec",
                "learning_sec",
                "epoch_sec",
                "cumulative_problem_generation_and_transfer_sec",
                "cumulative_sampling_and_greedy_sec",
                "cumulative_learning_sec",
                "cumulative_epoch_sec",
                "cumulative_stage_sum_sec",
                "wall_clock_elapsed_sec",
                "unaccounted_wall_clock_sec",
                "generation_device",
                "problem_input_device",
                "policy_device",
                "sampling_action_device",
                "sampling_logp_device",
                "loss_device",
            ]
        )
    print("Timing CSV:", timing_csv_path)

    # =========================
    # 🔥 [추가] 모델 저장 설정
    # =========================
    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = save_dir / "best_model.pt"
    best_eval_reward = -1e9

    # =========================
    # Environment / Model
    # =========================
    generator_params = dict(
        num_parts=20,
        max_num_parts=20,
        material_types=2,
        p_relative_motion=0.10,
        p_extra_edge=0.50,
        L_low=20.0,
        L_high=120.0,
        W_low=10.0,
        W_high=55.0,
        H_low=2,
        H_high=24.0,
        build_limit_L=1000.0,
        build_limit_W=1000.0,
        build_limit_H=500.0,
        p_maint_H=0.10,
        p_standard=0.10,
    )

    gen = FPIGenerator(**generator_params)
    env = PartConsolidationEnv(
        generator=gen,
        min_group_size_before_sep=1,
        device=device,
    )

    policy = PCPolicy(
        node_feat_dim=gen.node_feat_dim,
        edge_feat_dim=gen.edge_feat_dim,
        emb_dim=128,
        num_message_passing=3,
        temperature=temperature,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    max_steps = gen.max_num_parts
    td_eval_fixed = make_fixed_eval_td(
        env=env,
        batch_size=eval_batch_size,
        seed=eval_seed,
        device=device,
    )
    print(f"Fixed eval batch: size={eval_batch_size}, seed={eval_seed}")

    # =========================
    # Training Loop
    # =========================
    cumulative_generation_time_sec = 0.0
    cumulative_sampling_time_sec = 0.0
    cumulative_learning_time_sec = 0.0
    cumulative_epoch_time_sec = 0.0
    for ep in range(1, epochs + 1):
        epoch_time_start = time.perf_counter()
        policy.train()
        policy.temperature = float(temperature)

        generation_time_start = time.perf_counter()
        td_generated = env.reset(batch_size)
        generation_device = tensor_device_name(td_generated["material"])
        td0 = td_generated.to(device)
        problem_input_device = tensor_device_name(td0["material"])
        policy_device = first_parameter_device(policy)
        generation_time_sec = time.perf_counter() - generation_time_start

        sampling_time_start = time.perf_counter()
        actions, logps, entropies, terminal_reward, total_reward, _, step_counts = rollout_episode_from_td(
            env=env,
            policy=policy,
            td_init=td0,
            max_steps=max_steps,
            sample=True,
            epsilon=0.0,
        )
        sampling_action_device = tensor_device_name(actions)
        sampling_logp_device = tensor_device_name(logps)

        # greedy baseline
        policy.eval()
        with torch.no_grad():
            _, _, _, reward_greedy, _, _, _ = rollout_episode_from_td(
                env=env,
                policy=policy,
                td_init=td0,
                max_steps=max_steps,
                sample=False,
                epsilon=0.0,
        )
        policy.train()
        sampling_time_sec = time.perf_counter() - sampling_time_start

        learning_time_start = time.perf_counter()
        advantage = total_reward - reward_greedy
        advantage_norm = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        logp_sum = logps.sum(dim=1)
        entropy_mean = entropies.mean()
        reward_metrics = env.reward_metrics_from_actions(actions)

        loss_pg = -(advantage_norm.detach() * logp_sum).mean()
        loss_mutation, mutation_stats = compute_mutation_auxiliary_loss(
            env=env,
            policy=policy,
            td_static=td0,
            sampled_actions=actions,
            sampled_reward=terminal_reward.detach(),
            mutation_frac=mutation_frac,
            mutation_attempts=mutation_attempts,
            mutation_accept_worse=mutation_accept_worse,
            mutation_worse_accept_prob=mutation_worse_accept_prob,
            mutation_worse_weight=mutation_worse_weight,
            max_steps=max_steps,
        )
        loss = loss_pg + mutation_loss_weight * loss_mutation - entropy_coef * entropy_mean
        loss_device = tensor_device_name(loss)

        if ep == 1:
            device_report = (
                "Stage 1 - problem instance generation\n"
                f"- env.reset output device: {generation_device}\n"
                f"- model input TensorDict after .to(device): {problem_input_device}\n\n"
                "Stage 2 - policy sampling\n"
                f"- policy parameter device: {policy_device}\n"
                f"- sampled action tensor device: {sampling_action_device}\n"
                f"- sampled log probability tensor device: {sampling_logp_device}\n\n"
                "Stage 3 - learning / loss update\n"
                f"- terminal reward device: {tensor_device_name(terminal_reward)}\n"
                f"- loss tensor device: {loss_device}\n"
            )
            print("\n===== DEVICE STAGE REPORT =====")
            print(device_report)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()
        learning_time_sec = time.perf_counter() - learning_time_start
        epoch_time_sec = time.perf_counter() - epoch_time_start
        cumulative_generation_time_sec += generation_time_sec
        cumulative_sampling_time_sec += sampling_time_sec
        cumulative_learning_time_sec += learning_time_sec
        cumulative_epoch_time_sec += epoch_time_sec
        cumulative_stage_sum_sec = (
            cumulative_generation_time_sec
            + cumulative_sampling_time_sec
            + cumulative_learning_time_sec
        )
        wall_clock_elapsed_sec = time.time() - train_start_time
        unaccounted_wall_clock_sec = wall_clock_elapsed_sec - cumulative_stage_sum_sec
        with timing_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(
                [
                    ep,
                    0,
                    batch_size,
                    f"{generation_time_sec:.6f}",
                    f"{sampling_time_sec:.6f}",
                    f"{learning_time_sec:.6f}",
                    f"{epoch_time_sec:.6f}",
                    f"{cumulative_generation_time_sec:.6f}",
                    f"{cumulative_sampling_time_sec:.6f}",
                    f"{cumulative_learning_time_sec:.6f}",
                    f"{cumulative_epoch_time_sec:.6f}",
                    f"{cumulative_stage_sum_sec:.6f}",
                    f"{wall_clock_elapsed_sec:.6f}",
                    f"{unaccounted_wall_clock_sec:.6f}",
                    generation_device,
                    problem_input_device,
                    policy_device,
                    sampling_action_device,
                    sampling_logp_device,
                    loss_device,
                ]
            )

        # =========================
        # 🔥 [추가] checkpoint 저장
        # =========================
        if ep % 100 == 0:
            torch.save({
                "epoch": ep,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_dir / f"pc_model_ep{ep}.pt")

        writer.add_scalar("train/reward_total", total_reward.mean().item(), ep)
        writer.add_scalar("train/reward_greedy", reward_greedy.mean().item(), ep)
        writer.add_scalar("train/episode_steps_mean", step_counts.mean().item(), ep)
        writer.add_scalar("train/loss", loss.item(), ep)
        writer.add_scalar("train/loss_pg", loss_pg.item(), ep)
        writer.add_scalar("train/loss_mutation", loss_mutation.item(), ep)
        writer.add_scalar("train/entropy", entropy_mean.item(), ep)
        writer.add_scalar("train/mutation_improved_ratio", mutation_stats["improved_ratio"], ep)
        writer.add_scalar("train/mutation_reward_gain", mutation_stats["reward_gain"], ep)
        writer.add_scalar("train/mutation_kept_count", mutation_stats["kept_count"], ep)
        writer.add_scalar("train/mutation_worse_kept_count", mutation_stats["worse_kept_count"], ep)
        writer.add_scalar("train/mutation_worse_accept_ratio", mutation_stats["worse_accept_ratio"], ep)
        writer.add_scalar("train/advantage_mean", advantage.mean().item(), ep)
        writer.add_scalar("train/advantage_std", advantage.std(unbiased=False).item(), ep)
        writer.add_scalar("train/num_groups", reward_metrics["num_groups"].mean().item(), ep)

        if ep % 10 == 0:
            policy.eval()
            with torch.no_grad():
                actions_eval, _, _, reward_eval, _, _, _ = rollout_episode_from_td(
                    env=env,
                    policy=policy,
                    td_init=td_eval_fixed,
                    max_steps=max_steps,
                    sample=False,
                    epsilon=0.0,
                )
                eval_metrics = env.reward_metrics_from_actions(actions_eval)

            avg_eval = reward_eval.mean().item()

            writer.add_scalar("eval/reward_total", avg_eval, ep)
            writer.add_scalar("eval/reward_greedy", avg_eval, ep)
            writer.add_scalar("eval/num_groups", eval_metrics["num_groups"].mean().item(), ep)

            # =========================
            # 🔥 [추가] BEST MODEL 저장
            # =========================
            if avg_eval > best_eval_reward:
                best_eval_reward = avg_eval

                torch.save({
                    "epoch": ep,
                    "policy": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_reward": best_eval_reward,
                }, best_model_path)

                print(f"🔥 BEST MODEL UPDATED @ ep {ep} | reward={avg_eval:.4f}")

            print(
                f"[{ep:5d}] "
                f"train_total={total_reward.mean().item():.4f} "
                f"eval_greedy={avg_eval:.4f} "
                f"train_feasible={reward_metrics['feasible'].mean().item():.3f} "
                f"eval_feasible={eval_metrics['feasible'].mean().item():.3f} "
                f"loss={loss.item():.4f} "
                f"loss_mut={loss_mutation.item():.4f} "
                f"mut_improve={mutation_stats['improved_ratio']:.3f} "
                f"mut_worse={mutation_stats['worse_accept_ratio']:.3f} "
                f"entropy={entropy_mean.item():.4f} "
                f"beta={entropy_coef:.4f} "
                f"temp={temperature:.3f} "
                f"avg_group_count={reward_metrics['num_groups'].mean().item():.2f} "
                f"time_gen={generation_time_sec:.3f}s "
                f"time_sampling={sampling_time_sec:.3f}s "
                f"time_learning={learning_time_sec:.3f}s "
                f"time_epoch={epoch_time_sec:.3f}s "
                f"cum_stage={cumulative_stage_sum_sec:.1f}s "
                f"wall={wall_clock_elapsed_sec:.1f}s "
                f"unaccounted={unaccounted_wall_clock_sec:.1f}s"
            )

    writer.close()
    total_train_time = time.time() - train_start_time
    print(f"Training wall time: {total_train_time:.2f}s ({total_train_time / 60:.2f} min)")


if __name__ == "__main__":
    main()
