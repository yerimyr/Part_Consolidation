from __future__ import annotations

import torch
from tensordict import TensorDict

from rl4co.envs.pc.evaluator import DEFAULT_MODULARITY_GAMMA
from rl4co.envs.pc.generator import FPIGenerator


class PartConsolidationEnv:
    """
    General-graph Part Consolidation environment.

    Action space:
        0       : STOP (terminate with current grouping)
        1..K    : merge one pair of currently active groups

    Reward:
        terminal reward only
    """

    def __init__(
        self,
        generator: FPIGenerator | None = None,
        generator_params: dict | None = None,
        min_group_size_before_sep: int = 1,
        allow_fallback: bool = False,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.generator = generator or FPIGenerator(**(generator_params or {}))
        self.min_group_size_before_sep = int(min_group_size_before_sep)
        self.allow_fallback = bool(allow_fallback)

        self.N = self.generator.num_nodes
        self.max_parts = self.N - 1
        self.group_pair_list = [
            (i, j) for i in range(self.max_parts) for j in range(i + 1, self.max_parts)
        ]
        self.num_actions = 1 + len(self.group_pair_list)
        self.F = self.generator.node_feat_dim
        self._reward_static_td: TensorDict | None = None
        self._reward_eps = 1e-8
        self._modularity_gamma = DEFAULT_MODULARITY_GAMMA

    def reset(self, batch_size: int) -> TensorDict:
        td = self.generator(batch_size=batch_size, device=self.device)
        B = batch_size
        valid_part_mask = td.get("valid_part_mask", torch.ones((B, self.N), dtype=torch.bool, device=self.device))

        group_id = torch.full((B, self.N), -1, dtype=torch.long, device=self.device)
        for node in range(1, self.N):
            group_id[:, node] = node - 1
        group_id = torch.where(valid_part_mask, group_id, torch.full_like(group_id, -1))

        td_out = TensorDict(
            {
                **td,
                "group_id": group_id,
                "fallback_part_mask": torch.zeros((B, self.N), dtype=torch.bool, device=self.device),
                "dead_end": torch.zeros((B, 1), dtype=torch.bool, device=self.device),
                "done": torch.zeros((B, 1), dtype=torch.bool, device=self.device),
                "action_mask": torch.ones((B, self.num_actions), dtype=torch.bool, device=self.device),
            },
            batch_size=[B],
        )

        td_out["action_mask"] = self.get_action_mask(td_out)
        td_out["dead_end"] = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        self._reward_static_td = td_out.clone()
        return td_out

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        group_id = td["group_id"]
        size = td["size"]
        build_limit = td["build_limit"]
        assembly_adj = td["assembly_adj"]
        isstandard = td["isstandard"]
        mat_var = td["mat_var"]
        maint_diff = td["maint_diff"]
        rel_motion = td["rel_motion"]
        valid_part_mask = td.get("valid_part_mask", group_id.ge(0))

        B, _ = group_id.shape
        mask = torch.zeros((B, self.num_actions), dtype=torch.bool, device=group_id.device)
        mask[:, 0] = True

        for b in range(B):
            if bool(td["done"][b].item()):
                mask[b] = False
                mask[b, 0] = True
                continue

            active_groups = sorted(
                {
                    int(g)
                    for g in group_id[b, 1:][valid_part_mask[b, 1:]].tolist()
                    if int(g) >= 0
                }
            )
            group_nodes = {
                gid: torch.where((group_id[b] == gid) & valid_part_mask[b])[0].tolist()
                for gid in active_groups
            }
            group_nodes = {gid: [node for node in nodes if node > 0] for gid, nodes in group_nodes.items()}

            for action_idx, (ga, gb) in enumerate(self.group_pair_list, start=1):
                if ga not in group_nodes or gb not in group_nodes:
                    continue
                candidate = sorted(group_nodes[ga] + group_nodes[gb])
                if self._group_feasible(
                    candidate,
                    size[b],
                    build_limit[b],
                    isstandard[b],
                    mat_var[b],
                    maint_diff[b],
                    rel_motion[b],
                    assembly_adj[b],
                ):
                    mask[b, action_idx] = True

        return mask

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        B = td.batch_size[0]
        action = action.long().view(B)

        group_id = td["group_id"].clone()
        done = td["done"].clone()

        td2 = td.clone()
        for b in range(B):
            if bool(done[b].item()):
                continue
            a = int(action[b].item())
            if a == 0:
                done[b, 0] = True
                continue
            if 1 <= a <= len(self.group_pair_list):
                ga, gb = self.group_pair_list[a - 1]
                group_id[b, group_id[b] == gb] = ga

        td2["group_id"] = group_id

        td2["action_mask"] = self.get_action_mask(td2)
        no_feasible_merge = ~td2["action_mask"][:, 1:].any(dim=-1, keepdim=True)
        td2["dead_end"] = torch.zeros_like(done)
        td2["done"] = done | no_feasible_merge
        return td2

    def reward_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        raw = self.reward_metrics_from_actions(actions)
        if self._reward_static_td is None:
            raise RuntimeError("reward_from_actions called before env.reset")
        return self._terminal_reward_score(raw)

    def _terminal_reward_score(self, raw: dict[str, torch.Tensor]) -> torch.Tensor:
        return raw["Q_gamma"]

    def _terminal_reward_terms(self, raw: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_observed = raw["Q_observed"]
        q_expected = raw["Q_expected"]
        q_gamma = raw["Q_gamma"]
        return q_observed, -q_expected, q_gamma

    def reward_metrics_from_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        groups = self.actions_to_groups(actions, N=self.N)
        return self._terminal_reward_components(groups, device=actions.device)

    def actions_to_groups(self, actions: torch.Tensor, N: int | None = None) -> list[list[list[int]]]:
        if self._reward_static_td is None:
            raise RuntimeError("actions_to_groups called before env.reset")
        B, T = actions.shape
        td = self._reward_static_td
        valid_part_mask = td.get("valid_part_mask", torch.ones((B, self.N), dtype=torch.bool, device=actions.device))
        out = []

        for b in range(B):
            group_id = torch.full((self.N,), -1, dtype=torch.long, device=actions.device)
            for node in range(1, self.N):
                if bool(valid_part_mask[b, node].item()):
                    group_id[node] = node - 1

            for t in range(T):
                a = int(actions[b, t].item())
                if a == 0:
                    break
                if 1 <= a <= len(self.group_pair_list):
                    ga, gb = self.group_pair_list[a - 1]
                    if bool((group_id == ga).any().item()) and bool((group_id == gb).any().item()):
                        group_id[group_id == gb] = ga

            groups_map: dict[int, list[int]] = {}
            for node in range(1, self.N):
                gid = int(group_id[node].item())
                if gid >= 0:
                    groups_map.setdefault(gid, []).append(node)
            groups_b = [sorted(group) for group in groups_map.values()]
            out.append(groups_b)

        return out

    def _terminal_reward_components(self, groups: list[list[list[int]]], device: torch.device) -> dict[str, torch.Tensor]:
        if self._reward_static_td is None:
            raise RuntimeError("reward_from_actions called before env.reset")

        td = self._reward_static_td
        B = len(groups)
        feasible = torch.zeros((B,), dtype=torch.float32, device=device)
        infeasible_solution = torch.zeros((B,), dtype=torch.float32, device=device)
        infeasible_groups = torch.zeros((B,), dtype=torch.float32, device=device)
        num_groups = torch.tensor([len(g) for g in groups], dtype=torch.float32, device=device)
        total_internal_strength = torch.zeros((B,), dtype=torch.float32, device=device)
        feasible_pair_count = torch.zeros((B,), dtype=torch.float32, device=device)

        compat = td["compat"]
        size = td["size"]
        build_limit = td["build_limit"]
        isstandard = td["isstandard"]
        mat_var = td["mat_var"]
        maint_diff = td["maint_diff"]
        rel_motion = td["rel_motion"]

        for b, groups_b in enumerate(groups):
            infeasible = False
            for group in groups_b:
                total_internal_strength[b] += self._group_internal_strength(group, td["W"][b])
                feasible_pair_count[b] += self._group_feasible_pair_count(group, compat[b])
                if not self._group_feasible(
                    group,
                    size[b],
                    build_limit[b],
                    isstandard[b],
                    mat_var[b],
                    maint_diff[b],
                    rel_motion[b],
                    td["assembly_adj"][b],
                ):
                    infeasible = True
                    infeasible_groups[b] += 1.0
            infeasible_solution[b] = float(infeasible)
            feasible[b] = float(not infeasible)

        normalized_internal_strength = total_internal_strength / torch.clamp(feasible_pair_count, min=1.0)
        q_gamma, q_observed, q_expected = self._group_modularity(groups, td["W"].to(device), device)

        return {
            "feasible": feasible,
            "infeasible_solution": infeasible_solution,
            "infeasible_groups": infeasible_groups,
            "num_groups": num_groups,
            "total_internal_strength": total_internal_strength,
            "feasible_pair_count": feasible_pair_count,
            "normalized_internal_strength": normalized_internal_strength,
            "Q_gamma": q_gamma,
            "Q_observed": q_observed,
            "Q_expected": q_expected,
        }

    def _group_modularity(
        self,
        groups: list[list[list[int]]],
        w: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_gamma = torch.zeros((len(groups),), dtype=torch.float32, device=device)
        q_observed = torch.zeros_like(q_gamma)
        q_expected = torch.zeros_like(q_gamma)

        for b, groups_b in enumerate(groups):
            wb = w[b].float()
            strengths = wb.sum(dim=-1)
            two_m = strengths.sum().clamp_min(self._reward_eps)

            observed = torch.tensor(0.0, dtype=torch.float32, device=device)
            expected = torch.tensor(0.0, dtype=torch.float32, device=device)
            for group in groups_b:
                if not group:
                    continue
                idx = torch.tensor(group, dtype=torch.long, device=device)
                sub_w = wb.index_select(0, idx).index_select(1, idx)
                observed = observed + sub_w.sum()
                group_strength = strengths.index_select(0, idx).sum()
                expected = expected + (group_strength * group_strength) / two_m

            q_observed[b] = observed / two_m
            q_expected[b] = self._modularity_gamma * expected / two_m
            q_gamma[b] = q_observed[b] - q_expected[b]

        return q_gamma, q_observed, q_expected

    def _group_internal_strength(self, group: list[int], w: torch.Tensor) -> torch.Tensor:
        total = torch.tensor(0.0, device=w.device)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total = total + w[group[i], group[j]]
        return total

    def _group_feasible_pair_count(self, group: list[int], compat: torch.Tensor) -> torch.Tensor:
        count = torch.tensor(0.0, device=compat.device)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                count = count + float(bool(compat[group[i], group[j]].item()))
        return count

    def _compute_dead_end(
        self,
        *args,
    ) -> torch.Tensor:
        action_mask = args[-1]
        has_valid_action = action_mask.any(dim=-1, keepdim=True)
        return ~has_valid_action

    def _group_feasible(
        self,
        group: list[int],
        size: torch.Tensor,
        build_limit: torch.Tensor,
        isstandard: torch.Tensor,
        mat_var: torch.Tensor,
        maint_diff: torch.Tensor,
        rel_motion: torch.Tensor,
        assembly_adj: torch.Tensor,
    ) -> bool:
        if not group:
            return True
        if len(group) >= 2 and isstandard[group].bool().any():
            return False
        if not torch.all(size[group].sum(dim=0) <= build_limit):
            return False
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if bool(mat_var[a, b].item()) or bool(maint_diff[a, b].item()) or bool(rel_motion[a, b].item()):
                    return False
        visited = {group[0]}
        stack = [group[0]]
        while stack:
            cur = stack.pop()
            for nxt in group:
                if bool(assembly_adj[cur, nxt].item()) and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return len(visited) == len(group)
