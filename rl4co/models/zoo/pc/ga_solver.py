from __future__ import annotations

import csv
import copy
import random
import time

import numpy as np
from deap import base
from deap import creator
from deap import tools

from rl4co.envs.pc.evaluator import evaluate_groups
from rl4co.envs.pc.evaluator import score_metric_rows

class GASolver:
    """
    Constraint-aware GA baseline for part consolidation.

    Compared to the original toy implementation, this version:
    - scores solutions with the actual grouping constraints
    - keeps the global best solution with elitism
    - uses group-aware crossover
    - mutates with move / merge / split operators
    """

    def __init__(
        self,
        pop_size: int = 120,
        generations: int = 300,
        elite_size: int = 0,
        tournament_size: int = 2,
        mutation_rate: float = 0.05,
        init_new_group_bias: float = 0.60,
        enable_post_merge_repair: bool = False,
        exploratory_crossover_prob: float = 0.25,
        exploratory_mutation_prob: float = 0.0,
        init_diverse_fraction: float = 0.70,
        seed: int | None = None,
    ):
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.elite_size = int(elite_size)
        self.tournament_size = int(tournament_size)
        self.mutation_rate = float(mutation_rate)
        self.init_new_group_bias = float(init_new_group_bias)
        self.enable_post_merge_repair = bool(enable_post_merge_repair)
        self.exploratory_crossover_prob = float(exploratory_crossover_prob)
        self.exploratory_mutation_prob = float(exploratory_mutation_prob)
        self.init_diverse_fraction = float(init_diverse_fraction)
        self.rng = random.Random(seed)
        self.last_best_score: float | None = None
        self.last_generation_best_scores: list[float] = []
        self.last_generation_mean_scores: list[float] = []
        self.last_generation_best_raw_scores: list[float] = []
        self.last_generation_mean_raw_scores: list[float] = []
        self.last_generation_unique_raw_score_counts: list[int] = []
        self.last_generation_unique_grouping_counts: list[int] = []
        self.last_generation_best_grouping_changed: list[int] = []
        self.last_generation_parent_child_similarity: list[float] = []
        self.last_generation_parent_child_mean_similarity: list[float] = []
        self.last_generation_parent_child_examples: list[list[dict]] = []
        self.score_weights = None

    @staticmethod
    def _pop_size_for_num_parts(n: int) -> int:
        table = {
            4: 20,
            5: 40,
            6: 60,
            7: 80,
            8: 100,
            9: 120,
            10: 140,
        }
        if n in table:
            return table[n]
        if n < 4:
            return 20
        return 140

    @staticmethod
    def _elite_size_for_pop_size(pop_size: int) -> int:
        return max(0, int(round(pop_size * 0.10)))

    def solve(self, inst):
        start = time.time()
        n = int(inst["num_parts"])
        effective_pop_size = max(1, self.pop_size)
        effective_elite_size = min(max(0, self.elite_size), effective_pop_size)
        toolbox = self._build_toolbox(inst, n)
        pop = toolbox.population(n=effective_pop_size)
        self._evaluate_invalid(pop, toolbox)

        scores = self._population_scores([self._as_array(ind) for ind in pop], inst)
        raw_scores = [float(ind.fitness.values[0]) for ind in pop]

        best_idx = int(np.argmax(scores))
        best_sol = self._as_array(pop[best_idx]).copy()
        best_score = float(scores[best_idx])
        best_grouping_key = self._solution_key(best_sol)
        self.last_generation_best_scores = [best_score]
        self.last_generation_mean_scores = [float(np.mean(scores))]
        self.last_generation_best_raw_scores = [float(np.max(raw_scores))]
        self.last_generation_mean_raw_scores = [float(np.mean(raw_scores))]
        self.last_generation_unique_raw_score_counts = [self._count_unique_raw_scores(raw_scores)]
        self.last_generation_unique_grouping_counts = [self._count_unique_groupings(pop)]
        self.last_generation_best_grouping_changed = [0]
        self.last_generation_parent_child_similarity = [float("nan")]
        self.last_generation_parent_child_mean_similarity = [float("nan")]
        self.last_generation_parent_child_examples = [[]]

        cxpb = 0.9
        for generation in range(1, self.generations + 1):
            elites = [toolbox.clone(ind) for ind in tools.selBest(pop, effective_elite_size)]
            offspring = tools.selTournament(pop, effective_pop_size - effective_elite_size, tournsize=self.tournament_size)
            offspring = [toolbox.clone(ind) for ind in offspring]
            parent_child_best_similarities: list[float] = []
            parent_child_mean_similarities: list[float] = []
            parent_child_examples: list[dict] = []

            for i in range(0, len(offspring) - 1, 2):
                if self.rng.random() < cxpb:
                    parent1 = self._as_array(offspring[i])
                    parent2 = self._as_array(offspring[i + 1])
                    toolbox.mate(offspring[i], offspring[i + 1])
                    child1 = self._as_array(offspring[i])
                    child2 = self._as_array(offspring[i + 1])
                    self._record_parent_child_similarity(
                        generation,
                        i // 2,
                        parent1,
                        parent2,
                        child1,
                        child2,
                        parent_child_best_similarities,
                        parent_child_mean_similarities,
                        parent_child_examples,
                    )
                    if offspring[i].fitness.valid:
                        del offspring[i].fitness.values
                    if offspring[i + 1].fitness.valid:
                        del offspring[i + 1].fitness.values

            for ind in offspring:
                if self.rng.random() < self.mutation_rate:
                    toolbox.mutate(ind)
                    if ind.fitness.valid:
                        del ind.fitness.values

            self._evaluate_invalid(offspring, toolbox)
            pop = elites + offspring

            scores = self._population_scores([self._as_array(ind) for ind in pop], inst)
            raw_scores = [float(ind.fitness.values[0]) for ind in pop]

            gen_best_idx = int(np.argmax(scores))
            gen_best_score = float(scores[gen_best_idx])
            gen_best_sol = self._as_array(pop[gen_best_idx]).copy()
            gen_best_grouping_key = self._solution_key(gen_best_sol)
            self.last_generation_best_scores.append(gen_best_score)
            self.last_generation_mean_scores.append(float(np.mean(scores)))
            self.last_generation_best_raw_scores.append(float(np.max(raw_scores)))
            self.last_generation_mean_raw_scores.append(float(np.mean(raw_scores)))
            self.last_generation_unique_raw_score_counts.append(self._count_unique_raw_scores(raw_scores))
            self.last_generation_unique_grouping_counts.append(self._count_unique_groupings(pop))
            self.last_generation_best_grouping_changed.append(int(gen_best_grouping_key != best_grouping_key))
            self.last_generation_parent_child_similarity.append(
                float(np.mean(parent_child_best_similarities)) if parent_child_best_similarities else float("nan")
            )
            self.last_generation_parent_child_mean_similarity.append(
                float(np.mean(parent_child_mean_similarities)) if parent_child_mean_similarities else float("nan")
            )
            self.last_generation_parent_child_examples.append(parent_child_examples)
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_sol = gen_best_sol
            best_grouping_key = gen_best_grouping_key

        self.last_best_score = best_score
        end = time.time()
        return self._decode(best_sol), end - start

    def _build_toolbox(self, inst, n: int) -> base.Toolbox:
        if not hasattr(creator, "PCFitnessMax"):
            creator.create("PCFitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "PCIndividual"):
            creator.create("PCIndividual", list, fitness=creator.PCFitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("individual", self._make_individual, inst, n)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate_individual, inst=inst)
        toolbox.register("mate", self._mate_individuals, n=n, inst=inst)
        toolbox.register("mutate", self._mutate_individual, inst=inst)
        toolbox.register("clone", copy.deepcopy)
        return toolbox

    def _make_individual(self, inst, n: int):
        if self.rng.random() < self.init_diverse_fraction:
            sol = self._random_solution_diverse(inst)
        else:
            sol = self._random_solution(inst)
        return creator.PCIndividual(sol.tolist())

    def _evaluate_invalid(self, pop, toolbox) -> None:
        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

    def _evaluate_individual(self, ind, inst):
        return (self._fitness(self._as_array(ind), inst),)

    def _mate_individuals(self, ind1, ind2, n: int, inst):
        child1 = self._stabilize_child(
            self._crossover(self._as_array(ind1), self._as_array(ind2), n, inst),
            self._as_array(ind1),
            inst,
        )
        child2 = self._stabilize_child(
            self._crossover(self._as_array(ind2), self._as_array(ind1), n, inst),
            self._as_array(ind2),
            inst,
        )
        ind1[:] = child1.tolist()
        ind2[:] = child2.tolist()
        return ind1, ind2

    def _mutate_individual(self, ind, inst):
        child = self._stabilize_child(self._mutate(self._as_array(ind), inst), self._as_array(ind), inst)
        ind[:] = child.tolist()
        return (ind,)

    def _as_array(self, sol) -> np.ndarray:
        if isinstance(sol, np.ndarray):
            return sol.astype(int, copy=True)
        return np.asarray(list(sol), dtype=int)

    def plot_fitness_history(self, save_path: str = "ga_fitness_history.png", show: bool = False) -> str:
        if not self.last_generation_best_scores:
            raise RuntimeError("No GA fitness history available. Run solve(...) first.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generations = list(range(len(self.last_generation_best_scores)))

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))

        ax.plot(generations, self.last_generation_best_raw_scores, label="Best Raw Fitness", linewidth=2)
        ax.plot(generations, self.last_generation_mean_raw_scores, label="Mean Raw Fitness", linewidth=1.8)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Raw Fitness")
        ax.set_title("GA Raw Fitness by Generation")
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        diagnostics_path = save_path.replace(".png", "_diagnostics.png")
        self.plot_diagnostics_history(diagnostics_path, show=show)
        examples_path = save_path.replace(".png", "_parent_child_examples.csv")
        self.save_parent_child_similarity_examples(examples_path)
        return save_path

    def plot_diagnostics_history(self, save_path: str = "ga_fitness_diagnostics.png", show: bool = False) -> str:
        if not self.last_generation_best_scores:
            raise RuntimeError("No GA fitness history available. Run solve(...) first.")

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generations = list(range(len(self.last_generation_best_scores)))
        fig, axes = plt.subplots(4, 1, figsize=(7.5, 10.0), sharex=True)

        axes[0].plot(generations, self.last_generation_unique_raw_score_counts, linewidth=1.8)
        axes[0].set_ylabel("Unique Raw")
        axes[0].set_title("GA Diagnostics by Generation")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(generations, self.last_generation_unique_grouping_counts, linewidth=1.8)
        axes[1].set_ylabel("Unique Groupings")
        axes[1].grid(True, alpha=0.3)

        axes[2].step(generations, self.last_generation_best_grouping_changed, where="mid", linewidth=1.8)
        axes[2].set_ylabel("Best Changed")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(
            generations,
            self.last_generation_parent_child_similarity,
            label="Best Parent Similarity",
            linewidth=1.8,
        )
        axes[3].plot(
            generations,
            self.last_generation_parent_child_mean_similarity,
            label="Mean Parent Similarity",
            linewidth=1.4,
            alpha=0.8,
        )
        axes[3].set_xlabel("Generation")
        axes[3].set_ylabel("Parent-Child Sim.")
        axes[3].set_ylim(0.0, 1.05)
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="best")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return save_path

    def _stabilize_child(self, child: np.ndarray, fallback_parent: np.ndarray, inst) -> np.ndarray:
        candidate = self._repair(self._canonicalize(child), inst)
        if self._solution_feasible(candidate, inst):
            return candidate

        for _ in range(3):
            exploratory = self._repair(self._mutate_relaxed(candidate), inst)
            if self._solution_feasible(exploratory, inst):
                return exploratory
            candidate = exploratory

        return self._as_array(fallback_parent)

    def _random_solution(self, inst) -> np.ndarray:
        n = int(inst["num_parts"])
        order = list(range(n))
        self.rng.shuffle(order)

        groups: list[list[int]] = []
        for node in order:
            feasible_targets = []
            for idx in range(len(groups)):
                trial = sorted(groups[idx] + [node])
                if self._group_feasible(trial, inst):
                    feasible_targets.append(idx)

            create_new_group = (
                not feasible_targets
                or self.rng.random() < self.init_new_group_bias
            )
            if create_new_group:
                groups.append([node])
                continue

            target_idx = self.rng.choice(feasible_targets)
            groups[target_idx].append(node)

        return self._encode(groups, n)

    def _random_solution_diverse(self, inst) -> np.ndarray:
        n = int(inst["num_parts"])
        groups: list[list[int]] = []
        order = list(range(n))
        self.rng.shuffle(order)

        for node in order:
            if not groups or self.rng.random() < 0.55:
                groups.append([node])
                continue

            target_idx = self.rng.randrange(len(groups))
            groups[target_idx].append(node)

        sol = self._encode(groups, n)
        repaired = self._repair(self._canonicalize(sol), inst)
        if self._solution_feasible(repaired, inst):
            return repaired
        return self._random_solution(inst)

    def _fitness(self, sol, inst) -> float:
        groups = self._decode(sol)
        metrics = evaluate_groups(groups, inst)
        return float(score_metric_rows([metrics], weights=self.score_weights)[0]["score"])

    def _population_scores(self, pop, inst) -> list[float]:
        rows = []
        for idx, sol in enumerate(pop):
            groups = self._decode(sol)
            metrics = evaluate_groups(groups, inst)
            metrics["idx"] = idx
            rows.append(metrics)

        scored = score_metric_rows(rows, weights=self.score_weights)
        scored.sort(key=lambda x: x["idx"])
        return [float(row["score"]) for row in scored]

    def _count_unique_raw_scores(self, raw_scores: list[float]) -> int:
        return len({round(float(score), 12) for score in raw_scores})

    def _solution_key(self, sol: np.ndarray) -> tuple[int, ...]:
        return tuple(self._canonicalize(self._as_array(sol)).tolist())

    def _count_unique_groupings(self, pop) -> int:
        return len({self._solution_key(self._as_array(ind)) for ind in pop})

    def _record_parent_child_similarity(
        self,
        generation: int,
        pair_index: int,
        parent1: np.ndarray,
        parent2: np.ndarray,
        child1: np.ndarray,
        child2: np.ndarray,
        best_similarities: list[float],
        mean_similarities: list[float],
        examples: list[dict],
    ) -> None:
        for child_index, child in enumerate((child1, child2), start=1):
            sim_p1 = self._grouping_pairwise_similarity(child, parent1)
            sim_p2 = self._grouping_pairwise_similarity(child, parent2)
            best_similarities.append(max(sim_p1, sim_p2))
            mean_similarities.append((sim_p1 + sim_p2) / 2.0)

            if len(examples) < 5:
                examples.append(
                    {
                        "generation": generation,
                        "pair_index": pair_index,
                        "child_index": child_index,
                        "similarity_to_parent1": sim_p1,
                        "similarity_to_parent2": sim_p2,
                        "similarity_to_best_parent": max(sim_p1, sim_p2),
                        "parent1_chromosome": self._canonicalize(parent1).tolist(),
                        "parent2_chromosome": self._canonicalize(parent2).tolist(),
                        "child_chromosome": self._canonicalize(child).tolist(),
                        "parent1_groups": self._decode(self._canonicalize(parent1)),
                        "parent2_groups": self._decode(self._canonicalize(parent2)),
                        "child_groups": self._decode(self._canonicalize(child)),
                    }
                )

    def _grouping_pairwise_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = self._canonicalize(self._as_array(a))
        b = self._canonicalize(self._as_array(b))
        n = len(a)
        if n <= 1:
            return 1.0

        same = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                same_a = int(a[i]) == int(a[j])
                same_b = int(b[i]) == int(b[j])
                same += int(same_a == same_b)
                total += 1
        return same / max(total, 1)

    def save_parent_child_similarity_examples(
        self,
        save_path: str = "ga_parent_child_similarity_examples.csv",
    ) -> str:
        rows = [
            example
            for generation_examples in self.last_generation_parent_child_examples
            for example in generation_examples
        ]
        fieldnames = [
            "generation",
            "pair_index",
            "child_index",
            "similarity_to_parent1",
            "similarity_to_parent2",
            "similarity_to_best_parent",
            "parent1_chromosome",
            "parent2_chromosome",
            "child_chromosome",
            "parent1_groups",
            "parent2_groups",
            "child_groups",
        ]
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return save_path

    def _group_penalty(self, group: list[int], inst) -> float:
        penalty = 0.0

        if not self._group_size_ok(group, inst):
            penalty += 50.0

        for node in group:
            if not self._node_feasible(node, inst):
                penalty += 25.0

        if not self._no_pairwise_conflict(group, inst):
            penalty += 60.0
        if not self._connected(group, inst):
            penalty += 60.0

        return penalty

    def _decode(self, sol: np.ndarray) -> list[list[int]]:
        groups = {}
        for i, g in enumerate(sol):
            groups.setdefault(int(g), []).append(i)
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

    def _crossover(self, p1: np.ndarray, p2: np.ndarray, n: int, inst) -> np.ndarray:
        if self.rng.random() < self.exploratory_crossover_prob:
            return self._crossover_exploratory(p1, p2, n)
        return self._crossover_feasible_aware(p1, p2, n, inst)

    def _crossover_exploratory(self, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
        child = p1.copy()
        parent2_groups = self._decode(p2)
        self.rng.shuffle(parent2_groups)

        # Inherit a few entire groups from parent 2.
        take_k = max(1, len(parent2_groups) // 2)
        selected = parent2_groups[:take_k]
        for group in selected:
            new_gid = int(child.max()) + 1 if child.size > 0 else 0
            for node in group:
                child[node] = new_gid

        # Also align some nodes to parent 2 directly.
        for node in range(n):
            if self.rng.random() < 0.2:
                target_group = p2[node]
                members = np.where(p2 == target_group)[0]
                new_gid = int(child.max()) + 1
                child[members] = new_gid

        return self._canonicalize(child)

    def _crossover_feasible_aware(self, p1: np.ndarray, p2: np.ndarray, n: int, inst) -> np.ndarray:
        parent1_groups = self._decode(p1)
        parent2_groups = self._decode(p2)
        pool = [group[:] for group in parent1_groups] + [group[:] for group in parent2_groups]
        self.rng.shuffle(pool)

        child_groups: list[list[int]] = []
        assigned: set[int] = set()

        for group in pool:
            candidate = [node for node in group if node not in assigned]
            if not candidate:
                continue
            if len(candidate) == 1:
                child_groups.append(candidate)
                assigned.update(candidate)
                continue

            # Try the whole inherited group first.
            if self._group_feasible(candidate, inst):
                child_groups.append(sorted(candidate))
                assigned.update(candidate)
                continue

            # If the full group is infeasible, greedily keep a feasible subset.
            feasible_subset: list[int] = []
            for node in candidate:
                trial = sorted(feasible_subset + [node])
                if self._group_feasible(trial, inst):
                    feasible_subset = trial
            if feasible_subset:
                child_groups.append(feasible_subset)
                assigned.update(feasible_subset)

        remaining = [node for node in range(n) if node not in assigned]
        self.rng.shuffle(remaining)
        for node in remaining:
            feasible_targets = []
            for idx, group in enumerate(child_groups):
                trial = sorted(group + [node])
                if self._group_feasible(trial, inst):
                    feasible_targets.append(idx)
            if feasible_targets and self.rng.random() >= self.init_new_group_bias:
                target_idx = self.rng.choice(feasible_targets)
                child_groups[target_idx].append(node)
                child_groups[target_idx].sort()
            else:
                child_groups.append([node])

        return self._canonicalize(self._encode(child_groups, n))

    def _mutate(self, sol: np.ndarray, inst) -> np.ndarray:
        child = sol.copy()

        if self.rng.random() < self.exploratory_mutation_prob:
            return self._mutate_relaxed(child)

        feasible_candidates = self._collect_feasible_mutation_candidates(child, inst)
        valid_ops = [op for op, candidates in feasible_candidates.items() if candidates]
        if not valid_ops:
            return self._mutate_relaxed(child)

        op = self.rng.choice(valid_ops)
        return self.rng.choice(feasible_candidates[op])

    def _mutate_relaxed(self, sol: np.ndarray) -> np.ndarray:
        child = self._canonicalize(sol.copy())
        groups = self._decode(child)
        n = len(child)
        op = self.rng.choice(["move", "merge", "split"])

        if op == "move" and len(groups) >= 2:
            node = self.rng.randrange(n)
            current_gid = int(child[node])
            candidate_gids = [gid for gid in range(len(groups)) if gid != current_gid]
            if candidate_gids:
                child[node] = self.rng.choice(candidate_gids)
                return self._canonicalize(child)

        if op == "merge" and len(groups) >= 2:
            gid_a, gid_b = sorted(self.rng.sample(range(len(groups)), 2))
            for node in groups[gid_b]:
                child[node] = gid_a
            return self._canonicalize(child)

        large_groups = [group for group in groups if len(group) >= 2]
        if large_groups:
            group = self.rng.choice(large_groups)
            cut = self.rng.randrange(1, len(group))
            new_gid = int(child.max()) + 1
            for node in group[cut:]:
                child[node] = new_gid
            return self._canonicalize(child)

        return child

    def _collect_feasible_mutation_candidates(self, sol: np.ndarray, inst) -> dict[str, list[np.ndarray]]:
        child = self._canonicalize(sol.copy())
        groups = self._decode(child)
        n = len(child)

        candidates: dict[str, list[np.ndarray]] = {
            "move": [],
            "merge": [],
            "split": [],
        }

        for node in range(n):
            current_gid = int(child[node])
            for gid in range(len(groups)):
                if gid == current_gid:
                    continue
                trial = child.copy()
                trial[node] = gid
                if self._solution_feasible(trial, inst):
                    candidates["move"].append(self._canonicalize(trial))

        if len(groups) >= 2:
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    merged = sorted(groups[i] + groups[j])
                    if not self._group_feasible(merged, inst):
                        continue
                    trial = child.copy()
                    for node in groups[j]:
                        trial[node] = i
                    if self._solution_feasible(trial, inst):
                        candidates["merge"].append(self._canonicalize(trial))

        for group in groups:
            if len(group) < 3:
                continue
            for cut in range(1, len(group)):
                left = sorted(group[:cut])
                right = sorted(group[cut:])
                if not (self._group_feasible(left, inst) and self._group_feasible(right, inst)):
                    continue
                trial = child.copy()
                new_gid = int(child.max()) + 1
                for node in right:
                    trial[node] = new_gid
                if self._solution_feasible(trial, inst):
                    candidates["split"].append(self._canonicalize(trial))

        for op, sols in candidates.items():
            unique = {}
            for candidate in sols:
                unique[tuple(candidate.tolist())] = candidate
            candidates[op] = list(unique.values())
        return candidates

    def _repair(self, sol: np.ndarray, inst) -> np.ndarray:
        groups = self._decode(self._canonicalize(sol))
        repaired: list[list[int]] = []

        for group in sorted(groups, key=len, reverse=True):
            pending = list(group)
            while pending:
                placed_any = False
                for idx, node in enumerate(list(pending)):
                    candidate = [node]
                    for existing in list(pending):
                        if existing == node:
                            continue
                        trial = sorted(candidate + [existing])
                        if self._group_feasible(trial, inst):
                            candidate = trial
                    for node_in_candidate in candidate:
                        if node_in_candidate in pending:
                            pending.remove(node_in_candidate)
                    repaired.append(candidate)
                    placed_any = True
                    break
                if not placed_any:
                    repaired.extend([[node] for node in pending])
                    pending.clear()

        # Keep repair focused on feasibility recovery. Optional post-merge can be
        # enabled explicitly, but defaults off to preserve population diversity.
        if self.enable_post_merge_repair:
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

        return self._encode(repaired, len(sol))

    def _lightweight_valid_move(self, sol: np.ndarray, inst) -> bool:
        return self._solution_feasible(sol, inst)

    def _internal_weight(self, group: list[int], w: np.ndarray) -> float:
        total = 0.0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                total += float(w[group[i], group[j]])
        return total

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

    def _solution_feasible(self, sol: np.ndarray, inst) -> bool:
        groups = self._decode(self._canonicalize(sol))
        if not all(self._group_feasible(group, inst) for group in groups):
            return False
        return self._check_r3(groups, inst) is None

    def _check_r3(self, groups: list[list[int]], inst):
        checker = inst.get("assembly_access_checker")
        if checker is None:
            return None
        for group in groups:
            ok, detail = checker(group, groups, inst)
            if not ok:
                return detail
        return None
