import argparse

import numpy as np
import torch

from rl4co.envs.pc.evaluator import check_r3, group_feasible
from rl4co.envs.pc.generator import FPIGenerator


DEFAULT_GENERATOR_PARAMS = dict(
    num_parts=10,
    max_num_parts=10,
    topology_mode="dense_clustered",
    material_types=2,
    p_relative_motion=0.03,
    p_extra_edge=0.80,
    L_low=20.0,
    L_high=120.0,
    W_low=10.0,
    W_high=55.0,
    H_low=2.0,
    H_high=24.0,
    build_limit_L=450.0,
    build_limit_W=220.0,
    build_limit_H=120.0,
    p_maint_H=0.02,
    p_standard=0.01,
)


def np_round(arr, decimals=3):
    return np.round(arr, decimals=decimals)


def td_to_inst(td, fallback_num_parts):
    actual_num_parts = int(td.get("num_parts", torch.tensor([fallback_num_parts]))[0].item())
    return {
        "num_parts": actual_num_parts,
        "topology_id": int(td.get("topology_id", torch.tensor([-1]))[0].item()),
        "material": td["material"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "size": td["size"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "maintfreq": td["maintfreq"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "isstandard": td["isstandard"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "build_limit": td["build_limit"][0].cpu().numpy(),
        "assembly_adj": td["assembly_adj"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "mat_var": td["mat_var"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "stack_size": td["stack_size"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1, :].cpu().numpy(),
        "maint_diff": td["maint_diff"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "rel_motion": td["rel_motion"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "compat": td["compat"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "W": td["W"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
    }


def print_instance(inst, title):
    print(f"\n===== {title} =====")
    print("num_parts:")
    print(inst["num_parts"])
    print("build_limit [L, W, H]:")
    print(np_round(inst["build_limit"]))
    print("material:")
    print(inst["material"])
    print("maintfreq:")
    print(inst["maintfreq"])
    print("isstandard:")
    print(inst["isstandard"])
    print("size [L, W, H] per part:")
    print(np_round(inst["size"]))
    print("assembly_adj:")
    print(inst["assembly_adj"].astype(int))
    print("mat_var:")
    print(inst["mat_var"].astype(int))
    print("maint_diff:")
    print(inst["maint_diff"].astype(int))
    print("rel_motion:")
    print(inst["rel_motion"].astype(int))
    print("compat:")
    print(inst["compat"].astype(int))
    print("W:")
    print(np_round(inst["W"]))


def bell_number(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    stirling = [[0] * (n + 1) for _ in range(n + 1)]
    stirling[0][0] = 1
    for i in range(1, n + 1):
        for k in range(1, i + 1):
            stirling[i][k] = stirling[i - 1][k - 1] + k * stirling[i - 1][k]
    return sum(stirling[n])


def mask_to_group(mask: int, n: int) -> list[int]:
    return [idx for idx in range(n) if mask & (1 << idx)]


def enumerate_feasible_groups(inst):
    n = int(inst["num_parts"])
    feasible_groups = []
    groups_by_first = {idx: [] for idx in range(n)}

    for mask in range(1, 1 << n):
        group = mask_to_group(mask, n)
        if not group_feasible(group, inst):
            continue
        feasible_groups.append((mask, group))
        first = group[0]
        groups_by_first[first].append((mask, group))

    return feasible_groups, groups_by_first


def count_feasible_solutions(inst):
    n = int(inst["num_parts"])
    feasible_groups, groups_by_first = enumerate_feasible_groups(inst)
    total = 0

    def backtrack(remaining_mask: int, current_groups: list[list[int]]):
        nonlocal total
        if remaining_mask == 0:
            if check_r3(current_groups, inst):
                total += 1
            return

        first = (remaining_mask & -remaining_mask).bit_length() - 1
        for group_mask, group in groups_by_first[first]:
            if group_mask & remaining_mask != group_mask:
                continue
            current_groups.append(group)
            backtrack(remaining_mask ^ group_mask, current_groups)
            current_groups.pop()

    backtrack((1 << n) - 1, [])
    return total, len(feasible_groups)


def main():
    parser = argparse.ArgumentParser(description="Count exact feasible solutions for one random 10-part instance.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for generator sampling")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    generator = FPIGenerator(**DEFAULT_GENERATOR_PARAMS)
    td = generator(batch_size=1)
    inst = td_to_inst(td, DEFAULT_GENERATOR_PARAMS["num_parts"])

    print_instance(inst, "RANDOM INSTANCE")

    n = int(inst["num_parts"])
    bell = bell_number(n)
    feasible_solution_count, feasible_group_count = count_feasible_solutions(inst)
    feasible_ratio = feasible_solution_count / bell if bell > 0 else 0.0

    print("\n===== FEASIBLE SEARCH SPACE =====")
    print(f"Bell number B_{n}:")
    print(bell)
    print("Feasible group count (valid subsets that can appear as one group):")
    print(feasible_group_count)
    print("Feasible solution count (exact feasible partitions):")
    print(feasible_solution_count)
    print("Feasible solution ratio = feasible_solution_count / Bell number:")
    print(f"{feasible_ratio:.8f}")


if __name__ == "__main__":
    main()
