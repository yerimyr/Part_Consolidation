from __future__ import annotations

import torch
from tensordict import TensorDict

from rl4co.envs.pc.evaluator import DEFAULT_MODULARITY_GAMMA
from rl4co.envs.pc.generator import FPIGenerator


class PartConsolidationEnv:
    """
    General-graph Part Consolidation environment.

    Action space:
        0     : SEP (close current open group)
        1..N  : choose one real part and add it to the current group

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
        self.F = self.generator.node_feat_dim
        self._reward_static_td: TensorDict | None = None
        self._reward_eps = 1e-8
        self._modularity_gamma = DEFAULT_MODULARITY_GAMMA

    def reset(self, batch_size: int) -> TensorDict:
        td = self.generator(batch_size=batch_size, device=self.device)
        B = batch_size
        valid_part_mask = td.get("valid_part_mask", torch.ones((B, self.N), dtype=torch.bool, device=self.device))

        assigned = torch.zeros((B, self.N), dtype=torch.bool, device=self.device)
        assigned[:, 0] = True
        assigned = assigned | (~valid_part_mask)

        open_group = torch.zeros((B, self.N), dtype=torch.bool, device=self.device)
        open_group_size = torch.zeros((B, 3), dtype=torch.float32, device=self.device)
        closed_group_count = torch.zeros((B,), dtype=torch.long, device=self.device)

        td_out = TensorDict(
            {
                **td,
                "assigned": assigned,
                "open_group": open_group,
                "open_group_size": open_group_size,
                "closed_group_count": closed_group_count,
                "fallback_part_mask": torch.zeros((B, self.N), dtype=torch.bool, device=self.device),
                "dead_end": torch.zeros((B, 1), dtype=torch.bool, device=self.device),
                "done": torch.zeros((B, 1), dtype=torch.bool, device=self.device),
                "action_mask": torch.ones((B, self.N), dtype=torch.bool, device=self.device),
            },
            batch_size=[B],
        )

        td_out["action_mask"] = self.get_action_mask(td_out)
        td_out["dead_end"] = self._compute_dead_end(td_out["assigned"], td_out["open_group"], td_out["action_mask"])
        td_out["done"] = (td_out["done"] | td_out["dead_end"]).clone()
        self._reward_static_td = td_out.clone()
        return td_out

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        assigned = td["assigned"]
        open_group = td["open_group"]
        size = td["size"]
        build_limit = td["build_limit"]
        assembly_adj = td["assembly_adj"]
        isstandard = td["isstandard"]
        mat_var = td["mat_var"]
        maint_diff = td["maint_diff"]
        rel_motion = td["rel_motion"]
        valid_part_mask = td.get("valid_part_mask", torch.ones_like(assigned))

        B, N = assigned.shape
        mask = torch.zeros((B, N), dtype=torch.bool, device=assigned.device)
        fallback_part_mask = torch.zeros((B, N), dtype=torch.bool, device=assigned.device)

        open_any = open_group.any(dim=-1)
        real_valid = valid_part_mask[:, 1:]
        all_assigned = (assigned[:, 1:] | (~real_valid)).all(dim=-1)

        # --------------------------
        # Case 1: open group exists
        # --------------------------
        if open_any.any():
            feasible_part = torch.zeros((B, N), dtype=torch.bool, device=assigned.device)
            feasible_part[:, 0] = False

            for b in range(B):
                if not bool(open_any[b].item()):
                    continue

                current_group = torch.where(open_group[b])[0].tolist()
                for node in range(1, N):
                    if not bool(valid_part_mask[b, node].item()):
                        continue
                    if bool(assigned[b, node].item()):
                        continue

                    candidate = current_group + [node]
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
                        feasible_part[b, node] = True

                if self.allow_fallback and not bool(feasible_part[b, 1:].any().item()):
                    for node in range(1, N):
                        if not bool(valid_part_mask[b, node].item()):
                            continue
                        if not bool(assigned[b, node].item()):
                            fallback_part_mask[b, node] = True
                    feasible_part[b] = fallback_part_mask[b]

            mask = feasible_part

            group_card = open_group[:, 1:].sum(dim=-1)
            sep_allowed = open_any & (group_card >= self.min_group_size_before_sep)
            sep_allowed = sep_allowed | (open_any & (~mask[:, 1:].any(dim=-1)))
            mask[:, 0] = sep_allowed

        # --------------------------
        # Case 2: no open group
        # --------------------------
        no_open_rows = ~open_any
        if no_open_rows.any():
            rows = torch.where(no_open_rows)[0]
            for b in rows.tolist():
                for node in range(1, N):
                    if not bool(valid_part_mask[b, node].item()):
                        continue
                    if bool(assigned[b, node].item()):
                        continue
                    if self._group_feasible(
                        [node],
                        size[b],
                        build_limit[b],
                        isstandard[b],
                        mat_var[b],
                        maint_diff[b],
                        rel_motion[b],
                        assembly_adj[b],
                    ):
                        mask[b, node] = True
                mask[b, 0] = False

        # --------------------------
        # finished cases
        # --------------------------
        finished_but_open = all_assigned & open_any
        if finished_but_open.any():
            rows = torch.where(finished_but_open)[0]
            mask[rows] = False
            mask[rows, 0] = True

        finished_and_closed = all_assigned & (~open_any)
        if finished_and_closed.any():
            rows = torch.where(finished_and_closed)[0]
            mask[rows] = False
            mask[rows, 0] = True

        td["fallback_part_mask"] = fallback_part_mask
        return mask

    def step(self, td: TensorDict, action: torch.Tensor) -> TensorDict:
        B = td.batch_size[0]
        action = action.long().view(B)

        assigned = td["assigned"].clone()
        open_group = td["open_group"].clone()
        open_group_size = td["open_group_size"].clone()
        closed_group_count = td["closed_group_count"].clone()
        size = td["size"]
        valid_part_mask = td.get("valid_part_mask", torch.ones_like(assigned))

        is_sep = action.eq(0)
        is_part = ~is_sep

        # SEP
        if is_sep.any():
            rows = torch.where(is_sep)[0]
            had_open = open_group[rows].any(dim=-1)
            closed_group_count[rows] += had_open.long()
            open_group[rows] = False
            open_group_size[rows] = 0.0

        # PART
        if is_part.any():
            rows = torch.where(is_part)[0]
            idx = action[is_part]
            assigned[rows, idx] = True
            open_group[rows, idx] = True
            open_group_size[rows] += size[rows, idx, :]

        all_assigned = (assigned[:, 1:] | (~valid_part_mask[:, 1:])).all(dim=-1)
        open_empty = ~open_group.any(dim=-1)
        done = (all_assigned & open_empty).view(B, 1)

        td2 = td.clone()
        td2.update(
            {
                "assigned": assigned,
                "open_group": open_group,
                "open_group_size": open_group_size,
                "closed_group_count": closed_group_count,
            }
        )

        td2["action_mask"] = self.get_action_mask(td2)
        dead_end = self._compute_dead_end(assigned, open_group, td2["action_mask"])
        td2["dead_end"] = dead_end
        td2["done"] = done | dead_end
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

    @staticmethod
    def actions_to_groups(actions: torch.Tensor, N: int) -> list[list[list[int]]]:
        B, T = actions.shape
        out = []

        for b in range(B):
            groups_b = []
            cur = []
            used = set()

            for t in range(T):
                a = int(actions[b, t].item())
                if a == 0:
                    if cur:
                        groups_b.append(cur)
                        cur = []
                else:
                    if 0 < a < N and a not in used:
                        cur.append(a)
                        used.add(a)

            if cur:
                groups_b.append(cur)
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
        assigned: torch.Tensor,
        open_group: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        all_assigned = assigned[:, 1:].all(dim=-1, keepdim=True)
        has_open = open_group.any(dim=-1, keepdim=True)
        has_valid_action = action_mask.any(dim=-1, keepdim=True)
        return (~all_assigned) & (~has_open) & (~has_valid_action)

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
