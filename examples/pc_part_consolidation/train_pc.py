from __future__ import annotations

import argparse
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

    for _ in range(max_steps):
        action, logp, entropy, _ = policy.act(td, sample=sample, epsilon=epsilon)
        actions.append(action)
        logps.append(logp)
        entropies.append(entropy)

        td = env.step(td, action)

        if td["done"].all():
            break

    actions = torch.stack(actions, dim=1)
    logps = torch.stack(logps, dim=1)
    entropies = torch.stack(entropies, dim=1)

    terminal_reward = env.reward_from_actions(actions)
    total_reward = terminal_reward

    return actions, logps, entropies, terminal_reward, total_reward, td


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


def main():
    train_start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # =========================
    # Hyperparameters
    # =========================
    batch_size = 64
    eval_batch_size = 64
    eval_seed = 4321
    epochs = 500
    lr = 1e-4
    grad_clip = 1.0
    entropy_start = 0.05
    entropy_end = 0.005
    temperature_start = 1.5
    temperature_end = 0.9
    schedule_epochs = 300

    # Prefer on-policy exploration through entropy regularization.
    # Epsilon-greedy overwrites sampled actions and can make REINFORCE updates noisy.
    use_epsilon_exploration = False
    eps_start = 0.05
    eps_end = 0.0
    eps_decay_epochs = 1000

    # =========================
    # TensorBoard
    # =========================
    log_dir = f"runs/pc_general_graph_groupcount_{int(time.time())}"
    writer = SummaryWriter(log_dir=log_dir)
    print("TensorBoard log dir:", log_dir)
    writer.add_custom_scalars(
        {
            "Reward Components": {
                "Train weighted objective terms": [
                    "Multiline",
                    ["train/Q_observed", "train/Q_expected_penalty", "train/Q_gamma"],
                ],
                "Eval weighted objective terms": [
                    "Multiline",
                    ["eval/Q_observed", "eval/Q_expected_penalty", "eval/Q_gamma"],
                ],
            }
        }
    )

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
        num_parts=10,
        max_num_parts=10,
        material_types=2,
        p_relative_motion=0.03,
        p_extra_edge=0.80,
        L_low=20.0,
        L_high=120.0,
        W_low=10.0,
        W_high=55.0,
        H_low=2,
        H_high=24.0,
        build_limit_L=450.0,
        build_limit_W=220.0,
        build_limit_H=120.0,
        p_maint_H=0.03,
        p_standard=0.01,
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
        temperature=temperature_start,
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    max_steps = gen.max_num_parts * 2 + 4
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
    for ep in range(1, epochs + 1):
        policy.train()

        schedule_frac = min(1.0, ep / float(schedule_epochs))
        entropy_coef = entropy_start + (entropy_end - entropy_start) * schedule_frac
        temperature = temperature_start + (temperature_end - temperature_start) * schedule_frac
        policy.temperature = float(temperature)

        if use_epsilon_exploration:
            frac = min(1.0, ep / float(eps_decay_epochs))
            epsilon = eps_start + (eps_end - eps_start) * frac
        else:
            epsilon = 0.0

        td0 = env.reset(batch_size).to(device)
        actions, logps, entropies, terminal_reward, total_reward, _ = rollout_episode_from_td(
            env=env,
            policy=policy,
            td_init=td0,
            max_steps=max_steps,
            sample=True,
            epsilon=epsilon,
        )

        # greedy baseline
        policy.eval()
        with torch.no_grad():
            _, _, _, reward_greedy, _, _ = rollout_episode_from_td(
                env=env,
                policy=policy,
                td_init=td0,
                max_steps=max_steps,
                sample=False,
                epsilon=0.0,
        )
        policy.train()

        advantage = total_reward - reward_greedy
        advantage_norm = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        logp_sum = logps.sum(dim=1)
        entropy_mean = entropies.mean()
        reward_metrics = env.reward_metrics_from_actions(actions)
        train_q_observed, train_q_expected_penalty, train_q_gamma = env._terminal_reward_terms(reward_metrics)

        loss_pg = -(advantage_norm.detach() * logp_sum).mean()
        loss = loss_pg - entropy_coef * entropy_mean

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        # =========================
        # 🔥 [추가] checkpoint 저장
        # =========================
        if ep % 100 == 0:
            torch.save({
                "epoch": ep,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, save_dir / f"pc_model_ep{ep}.pt")

        groups = env.actions_to_groups(actions, N=gen.num_nodes)
        avg_group_count = float(np.mean([len(g) for g in groups]))
        avg_group_size = float(
            np.mean([np.mean([len(x) for x in g]) if len(g) > 0 else 0.0 for g in groups])
        )
        avg_terminal_reward = terminal_reward.mean().item()

        writer.add_scalar("train/reward_total", total_reward.mean().item(), ep)
        writer.add_scalar("train/reward_greedy", reward_greedy.mean().item(), ep)
        writer.add_scalar("train/loss", loss.item(), ep)
        writer.add_scalar("train/entropy", entropy_mean.item(), ep)
        writer.add_scalar("train/entropy_coef", entropy_coef, ep)
        writer.add_scalar("train/temperature", temperature, ep)
        writer.add_scalar("train/advantage_mean", advantage.mean().item(), ep)
        writer.add_scalar("train/advantage_std", advantage.std(unbiased=False).item(), ep)
        writer.add_scalar("train/epsilon", epsilon, ep)
        writer.add_scalar("train/feasible_ratio", reward_metrics["feasible"].mean().item(), ep)
        writer.add_scalar("train/infeasible_solution", reward_metrics["infeasible_solution"].mean().item(), ep)
        writer.add_scalar("train/infeasible_groups", reward_metrics["infeasible_groups"].mean().item(), ep)
        writer.add_scalar("train/num_groups", reward_metrics["num_groups"].mean().item(), ep)
        writer.add_scalar("train/internal_strength", reward_metrics["total_internal_strength"].mean().item(), ep)
        writer.add_scalar("train/normalized_internal_strength", reward_metrics["normalized_internal_strength"].mean().item(), ep)
        writer.add_scalar("train/feasible_pair_count", reward_metrics["feasible_pair_count"].mean().item(), ep)
        writer.add_scalar("train/Q_observed", train_q_observed.mean().item(), ep)
        writer.add_scalar("train/Q_expected_penalty", train_q_expected_penalty.mean().item(), ep)
        writer.add_scalar("train/Q_gamma", train_q_gamma.mean().item(), ep)

        if ep % 10 == 0:
            policy.eval()
            with torch.no_grad():
                actions_eval, _, _, reward_eval, _, _ = rollout_episode_from_td(
                    env=env,
                    policy=policy,
                    td_init=td_eval_fixed,
                    max_steps=max_steps,
                    sample=False,
                    epsilon=0.0,
                )
                eval_metrics = env.reward_metrics_from_actions(actions_eval)
                eval_q_observed, eval_q_expected_penalty, eval_q_gamma = env._terminal_reward_terms(eval_metrics)

            avg_eval = reward_eval.mean().item()

            writer.add_scalar("eval/reward_total", avg_eval, ep)
            writer.add_scalar("eval/feasible_ratio", eval_metrics["feasible"].mean().item(), ep)
            writer.add_scalar("eval/infeasible_solution", eval_metrics["infeasible_solution"].mean().item(), ep)
            writer.add_scalar("eval/infeasible_groups", eval_metrics["infeasible_groups"].mean().item(), ep)
            writer.add_scalar("eval/num_groups", eval_metrics["num_groups"].mean().item(), ep)
            writer.add_scalar("eval/internal_strength", eval_metrics["total_internal_strength"].mean().item(), ep)
            writer.add_scalar("eval/normalized_internal_strength", eval_metrics["normalized_internal_strength"].mean().item(), ep)
            writer.add_scalar("eval/feasible_pair_count", eval_metrics["feasible_pair_count"].mean().item(), ep)
            writer.add_scalar("eval/Q_observed", eval_q_observed.mean().item(), ep)
            writer.add_scalar("eval/Q_expected_penalty", eval_q_expected_penalty.mean().item(), ep)
            writer.add_scalar("eval/Q_gamma", eval_q_gamma.mean().item(), ep)

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
                f"eval_total={avg_eval:.4f} "
                f"train_feasible={reward_metrics['feasible'].mean().item():.3f} "
                f"eval_feasible={eval_metrics['feasible'].mean().item():.3f} "
                f"loss={loss.item():.4f} "
                f"entropy={entropy_mean.item():.4f} "
                f"beta={entropy_coef:.4f} "
                f"temp={temperature:.3f} "
                f"avg_group_count={avg_group_count:.2f} "
                f"avg_group_size={avg_group_size:.2f} "
                f"eps={epsilon:.3f}"
            )

    writer.close()
    total_train_time = time.time() - train_start_time
    print(f"Training wall time: {total_train_time:.2f}s ({total_train_time / 60:.2f} min)")


if __name__ == "__main__":
    main()
