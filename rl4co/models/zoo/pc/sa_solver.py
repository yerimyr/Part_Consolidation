from __future__ import annotations

import math
import random
import time

import numpy as np

from rl4co.envs.pc.evaluator import evaluate_groups
from rl4co.envs.pc.evaluator import score_metric_rows


class SASolver:
    """
    Simulated annealing baseline for part consolidation.

    Internally, a solution is encoded as an edge-based chromosome Z over the
    physical edge set. z_l = 1 keeps edge e_l, and z_l = 0 cuts it.
    """

    def __init__(
        self,
        iterations: int = 10000,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.9985,
        min_temperature: float = 1e-4,
        init_new_group_bias: float = 0.60,
        enable_post_merge_repair: bool = False,
        seed: int | None = None,
    ):
        self.iterations = int(iterations)
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.min_temperature = float(min_temperature)
        self.init_new_group_bias = float(init_new_group_bias)
        self.enable_post_merge_repair = bool(enable_post_merge_repair)
        self.rng = random.Random(seed)
        self.score_weights = None

        self.last_best_score: float | None = None
        self.last_current_scores: list[float] = []
        self.last_best_scores: list[float] = []
        self.last_temperatures: list[float] = []
        self.last_acceptance_flags: list[int] = []
        self._edge_list: list[tuple[int, int]] = []
        self._num_parts: int = 0

    def solve(self, inst):
        start = time.time()
        self._num_parts = int(inst["num_parts"])
        self._edge_list = self._build_edge_list(inst)
        current = self._initial_solution(inst)
        current_score = self._fitness(current, inst)
        best = current.copy()
        best_score = current_score

        temperature = max(self.initial_temperature, self.min_temperature)
        self.last_current_scores = [current_score]
        self.last_best_scores = [best_score]
        self.last_temperatures = [temperature]
        self.last_acceptance_flags = [1]

        for _ in range(max(self.iterations, 0)):
            candidate = self._neighbor(current, inst)
            candidate_score = self._fitness(candidate, inst)
            delta = candidate_score - current_score

            accepted = delta >= 0.0 or self.rng.random() < math.exp(delta / max(temperature, 1e-12))
            if accepted:
                current = candidate
                current_score = candidate_score

            if current_score > best_score:
                best = current.copy()
                best_score = current_score

            self.last_current_scores.append(current_score)
            self.last_best_scores.append(best_score)
            self.last_temperatures.append(temperature)
            self.last_acceptance_flags.append(int(accepted))
            temperature = max(self.min_temperature, temperature * self.cooling_rate)

        self.last_best_score = best_score
        return self._decode(best), time.time() - start

    def _build_edge_list(self, inst) -> list[tuple[int, int]]:
        adj = np.asarray(inst["assembly_adj"]).astype(bool)
        n = int(inst["num_parts"])
        return [(i, j) for i in range(n) for j in range(i + 1, n) if bool(adj[i, j])]

    def _initial_solution(self, inst) -> np.ndarray:
        if not self._edge_list:
            return np.zeros((0,), dtype=int)
        sol = np.asarray(
            [1 if self.rng.random() < 0.50 else 0 for _ in self._edge_list],
            dtype=int,
        )
        repaired = self._repair(self._canonicalize(sol), inst)
        if self._solution_feasible(repaired, inst):
            return self._canonicalize(repaired)
        return np.zeros((len(self._edge_list),), dtype=int)

    def _neighbor(self, sol: np.ndarray, inst) -> np.ndarray:
        child = self._canonicalize(sol.copy())
        op = self.rng.choice(["split", "merge", "swap", "relocation"])
        if op == "split":
            child = self._neighbor_split_edge(child, inst)
        elif op == "merge":
            child = self._neighbor_merge_edge(child, inst)
        elif op == "swap":
            child = self._neighbor_swap(child)
        else:
            child = self._neighbor_relocation(child)

        repaired = self._repair(self._canonicalize(child), inst)
        if self._solution_feasible(repaired, inst):
            return self._canonicalize(repaired)
        return self._canonicalize(sol)

    def _neighbor_split_edge(self, sol: np.ndarray, inst) -> np.ndarray:
        base = self._canonicalize(sol)
        base_group_count = len(self._decode(base))
        groups = [group for group in self._decode(base) if len(group) >= 2]
        if not groups:
            return base

        selected_group = set(self.rng.choice(groups))
        candidates = []
        for edge_idx, (u, v) in enumerate(self._edge_list):
            if int(base[edge_idx]) == 1 and u in selected_group and v in selected_group:
                candidates.append(edge_idx)
        self.rng.shuffle(candidates)

        child = base.copy()
        for edge_idx in candidates:
            child[edge_idx] = 0
            canonical = self._canonicalize(child)
            if len(self._decode(canonical)) > base_group_count:
                return canonical
        return base

    def _neighbor_merge_edge(self, sol: np.ndarray, inst) -> np.ndarray:
        base = self._canonicalize(sol)
        base_group_count = len(self._decode(base))
        if base_group_count <= 1:
            return base

        labels = self._group_labels(self._decode(base))
        candidates = []
        for edge_idx, (u, v) in enumerate(self._edge_list):
            if int(base[edge_idx]) == 0 and labels[u] != labels[v]:
                candidates.append(edge_idx)
        self.rng.shuffle(candidates)

        child = base.copy()
        for edge_idx in candidates:
            child[edge_idx] = 1
            canonical = self._canonicalize(child)
            if len(self._decode(canonical)) < base_group_count:
                return canonical
        return base

    def _neighbor_swap(self, sol: np.ndarray) -> np.ndarray:
        groups = self._decode(self._canonicalize(sol))
        labels = self._group_labels(groups)
        candidates = [node for node, gid in enumerate(labels) if gid >= 0]
        if len(candidates) < 2:
            return sol
        a, b = self.rng.sample(candidates, 2)
        labels[a], labels[b] = labels[b], labels[a]
        return self._canonicalize(self._encode(self._labels_to_groups(labels)))

    def _neighbor_relocation(self, sol: np.ndarray) -> np.ndarray:
        groups = self._decode(self._canonicalize(sol))
        labels = self._group_labels(groups)
        movable = [node for node, gid in enumerate(labels) if gid >= 0]
        active_gids = sorted({gid for gid in labels if gid >= 0})
        if not movable:
            return sol

        node = self.rng.choice(movable)
        current_gid = labels[node]
        target_gids = [gid for gid in active_gids if gid != current_gid]
        if self.rng.random() < 0.25 or not target_gids:
            target_gid = max(active_gids, default=-1) + 1
        else:
            target_gid = self.rng.choice(target_gids)
        labels[node] = target_gid
        return self._canonicalize(self._encode(self._labels_to_groups(labels)))

    def _repair(self, sol: np.ndarray, inst) -> np.ndarray:
        groups = self._decode(self._canonicalize(sol))
        repaired: list[list[int]] = []

        for group in sorted(groups, key=len, reverse=True):
            pending = list(group)
            while pending:
                candidate = [pending[0]]
                for node in list(pending[1:]):
                    trial = sorted(candidate + [node])
                    if self._group_feasible(trial, inst):
                        candidate = trial
                for node in candidate:
                    if node in pending:
                        pending.remove(node)
                repaired.append(candidate)

        if self.enable_post_merge_repair:
            repaired = self._post_merge_repair(repaired, inst)
        return self._canonicalize(self._encode(repaired))

    def _post_merge_repair(self, groups: list[list[int]], inst) -> list[list[int]]:
        repaired = [group[:] for group in groups]
        improved = True
        while improved:
            improved = False
            best_pair = None
            best_gain = float("-inf")
            for i in range(len(repaired)):
                for j in range(i + 1, len(repaired)):
                    merged = sorted(repaired[i] + repaired[j])
                    if not self._group_feasible(merged, inst):
                        continue
                    gain = self._internal_weight(merged, np.asarray(inst["W"], dtype=float))
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, j)
            if best_pair is not None:
                i, j = best_pair
                repaired[i] = sorted(repaired[i] + repaired[j])
                repaired.pop(j)
                improved = True
        return repaired

    def _fitness(self, sol: np.ndarray, inst) -> float:
        metrics = evaluate_groups(self._decode(self._canonicalize(sol)), inst)
        return float(score_metric_rows([metrics], weights=self.score_weights)[0]["score"])

    def _decode(self, sol: np.ndarray) -> list[list[int]]:
        parent = list(range(self._num_parts))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        z = self._as_array(sol)
        for bit, (u, v) in zip(z, self._edge_list):
            if int(bit) == 1:
                union(u, v)

        groups: dict[int, list[int]] = {}
        for node in range(self._num_parts):
            groups.setdefault(find(node), []).append(node)
        return [sorted(group) for group in groups.values()]

    def _encode(self, groups: list[list[int]]) -> np.ndarray:
        group_id: dict[int, int] = {}
        for gid, group in enumerate(groups):
            for node in group:
                group_id[int(node)] = gid
        return np.asarray(
            [1 if group_id.get(u) == group_id.get(v) else 0 for u, v in self._edge_list],
            dtype=int,
        )

    def _canonicalize(self, sol: np.ndarray) -> np.ndarray:
        return self._encode(self._decode(self._as_array(sol)))

    def _as_array(self, sol) -> np.ndarray:
        if isinstance(sol, np.ndarray):
            return sol.astype(int, copy=True)
        return np.asarray(list(sol), dtype=int)

    def _group_labels(self, groups: list[list[int]]) -> list[int]:
        labels = [-1] * self._num_parts
        for gid, group in enumerate(groups):
            for node in group:
                labels[int(node)] = gid
        return labels

    def _labels_to_groups(self, labels: list[int]) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for node, gid in enumerate(labels):
            if gid >= 0:
                groups.setdefault(int(gid), []).append(node)
        return [sorted(group) for group in groups.values()]

    def _solution_feasible(self, sol: np.ndarray, inst) -> bool:
        groups = self._decode(self._canonicalize(sol))
        if not all(self._group_feasible(group, inst) for group in groups):
            return False
        return self._check_r3(groups, inst) is None

    def _group_feasible(self, group: list[int], inst) -> bool:
        if any(not self._node_feasible(node, inst) for node in group):
            return False
        if len(group) >= 2 and "isstandard" in inst and np.asarray(inst["isstandard"])[group].any():
            return False
        if not self._group_size_ok(group, inst):
            return False
        if not self._no_pairwise_conflict(group, inst):
            return False
        return self._connected(group, inst)

    def _node_feasible(self, node: int, inst) -> bool:
        if "material_available" in inst and not np.asarray(inst["material_available"])[node]:
            return False
        size = np.asarray(inst["size"])
        build_limit = np.asarray(inst["build_limit"])
        if size.ndim == 1:
            return bool(size[node] <= build_limit)
        return bool(np.all(size[node] <= build_limit))

    def _group_size_ok(self, group: list[int], inst) -> bool:
        size = np.asarray(inst["size"])
        build_limit = np.asarray(inst["build_limit"])
        if size.ndim == 1:
            return bool(np.sum(size[group]) <= build_limit)
        return bool(np.all(np.sum(size[group], axis=0) <= build_limit))

    def _no_pairwise_conflict(self, group: list[int], inst) -> bool:
        mat_var = np.asarray(inst.get("mat_var", np.zeros_like(inst["assembly_adj"])))
        maint_diff = np.asarray(inst.get("maint_diff", np.zeros_like(inst["assembly_adj"])))
        rel_motion = np.asarray(inst.get("rel_motion", np.zeros_like(inst["assembly_adj"])))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if mat_var[a, b] or maint_diff[a, b] or rel_motion[a, b]:
                    return False
        return True

    def _connected(self, group: list[int], inst) -> bool:
        if not group:
            return True
        adj = np.asarray(inst["assembly_adj"])
        visited = {group[0]}
        stack = [group[0]]
        while stack:
            cur = stack.pop()
            for nxt in group:
                if adj[cur, nxt] and nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        return len(visited) == len(group)

    def _check_r3(self, groups: list[list[int]], inst):
        checker = inst.get("assembly_access_checker")
        if checker is None:
            return None
        for group in groups:
            ok, detail = checker(group, groups, inst)
            if not ok:
                return detail
        return None

    def _internal_weight(self, group: list[int], w: np.ndarray) -> float:
        total = 0.0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total += float(w[group[i], group[j]])
        return total

    def plot_history(
        self,
        save_path: str = "sa_fitness_history.png",
        show: bool = False,
        ylim: tuple[float, float] | None = None,
    ) -> str:
        if not self.last_best_scores:
            raise RuntimeError("No SA history available. Run solve(...) first.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = list(range(len(self.last_best_scores)))
        fig, ax1 = plt.subplots(1, 1, figsize=(7.5, 4.5))
        ax1.plot(steps, self.last_current_scores, label="Current Score", linewidth=1.3)
        ax1.plot(steps, self.last_best_scores, label="Best Score", linewidth=2.0)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Score")
        if ylim is not None:
            ax1.set_ylim(*ylim)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")

        ax2 = ax1.twinx()
        ax2.plot(steps, self.last_temperatures, label="Temperature", color="tab:red", alpha=0.4)
        ax2.set_ylabel("Temperature")

        fig.tight_layout()
        fig.savefig(save_path, dpi=200)
        if show:
            plt.show()
        plt.close(fig)
        return save_path


__all__ = ["SASolver"]
