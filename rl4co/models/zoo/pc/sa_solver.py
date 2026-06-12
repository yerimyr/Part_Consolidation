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

    A solution is encoded as a length-N chromosome:
    [0, 0, 1, 2] means parts 0 and 1 are in group 0, part 2 is in
    group 1, and part 3 is in group 2.
    """

    def __init__(
        self,
        iterations: int = 3000,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.995,
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

    def solve(self, inst):
        start = time.time()
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

    def _initial_solution(self, inst) -> np.ndarray:
        n = int(inst["num_parts"])
        groups: list[list[int]] = []
        nodes = list(range(n))
        self.rng.shuffle(nodes)

        for node in nodes:
            if not groups or self.rng.random() < 0.55:
                groups.append([node])
                continue
            target_idx = self.rng.randrange(len(groups))
            groups[target_idx].append(node)

        repaired = self._repair(self._encode(groups, n), inst)
        if self._solution_feasible(repaired, inst):
            return self._canonicalize(repaired)
        return np.arange(n, dtype=int)

    def _neighbor(self, sol: np.ndarray, inst) -> np.ndarray:
        child = self._canonicalize(sol.copy())
        groups = self._decode(child)
        n = len(child)
        op = self.rng.choice(["swap", "relocation", "merge", "split"])

        if op == "swap" and n >= 2:
            i, j = self.rng.sample(range(n), 2)
            child[i], child[j] = child[j], child[i]

        elif op == "relocation" and n >= 1:
            node = self.rng.randrange(n)
            current_gid = int(child[node])
            target_gids = [gid for gid in range(len(groups)) if gid != current_gid]
            if self.rng.random() < 0.25 or not target_gids:
                target_gid = int(child.max()) + 1
            else:
                target_gid = self.rng.choice(target_gids)
            child[node] = target_gid

        elif op == "merge" and len(groups) >= 2:
            gid_a, gid_b = sorted(self.rng.sample(range(len(groups)), 2))
            for node in groups[gid_b]:
                child[node] = gid_a

        elif op == "split":
            splittable = [group for group in groups if len(group) >= 2]
            if splittable:
                group = list(self.rng.choice(splittable))
                self.rng.shuffle(group)
                cut = self.rng.randrange(1, len(group))
                new_gid = int(child.max()) + 1
                for node in group[cut:]:
                    child[node] = new_gid

        repaired = self._repair(self._canonicalize(child), inst)
        if self._solution_feasible(repaired, inst):
            return self._canonicalize(repaired)
        return self._canonicalize(sol)

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
        return self._canonicalize(self._encode(repaired, len(sol)))

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
        groups = {}
        for i, gid in enumerate(sol):
            groups.setdefault(int(gid), []).append(i)
        return [sorted(group) for group in groups.values()]

    def _encode(self, groups: list[list[int]], n: int) -> np.ndarray:
        sol = np.empty(n, dtype=int)
        for gid, group in enumerate(groups):
            for node in group:
                sol[node] = gid
        return sol

    def _canonicalize(self, sol: np.ndarray) -> np.ndarray:
        mapping = {}
        next_gid = 0
        out = np.empty_like(sol)
        for i, gid in enumerate(sol):
            gid = int(gid)
            if gid not in mapping:
                mapping[gid] = next_gid
                next_gid += 1
            out[i] = mapping[gid]
        return out

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

    def plot_history(self, save_path: str = "sa_fitness_history.png", show: bool = False) -> str:
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
