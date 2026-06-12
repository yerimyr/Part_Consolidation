from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class ConflictRecord:
    code: str
    subject: tuple[int, ...]
    detail: str


class CPCCDSolver:
    """
    Community-based PCCD solver following the paper's high-level flow:
    1) detect or accept communities
    2) split the graph into intra-/inter-community subgraphs
    3) verify rules on each part separately
    4) re-assemble surviving links
    5) find a minimum grouping solution while prioritizing intra-community links
    6) optionally re-check assembly access (R3) and retry

    Notes:
    - The paper assumes richer CAD-derived attributes and manual R3 checks.
      This implementation supports those fields when present and falls back to
      the simpler benchmark fields already used in this repo.
    - The public API remains compatible: solve(inst) -> (groups, elapsed_sec)
    """

    def __init__(self, alpha: float = 0.3, max_r3_retries: int = 10):
        self.alpha = float(alpha)
        self.max_r3_retries = int(max_r3_retries)
        self.last_conflicts: list[ConflictRecord] = []
        self.last_communities: list[list[int]] = []
        self.last_modularity: float = 0.0

    def solve(self, inst):
        start = time.perf_counter()
        self.last_conflicts = []

        communities, modularity = self._get_communities(inst)
        self.last_communities = [sorted(c) for c in communities]
        self.last_modularity = modularity

        if modularity <= self.alpha or len(communities) <= 1:
            groups = self._solve_strength_only(inst)
            return groups, time.perf_counter() - start

        blocked_pairs: set[tuple[int, int]] = set()
        groups = [[i] for i in range(inst["num_parts"])]

        for _ in range(self.max_r3_retries):
            groups = self._solve_cpccd_once(inst, communities, blocked_pairs)
            violation = self._check_rule_r3(groups, inst)
            if violation is None:
                break

            pair, detail = violation
            blocked_pairs.add(pair)
            self._add_conflict("CF_AssemblyX", pair, detail)
        else:
            groups = self._solve_cpccd_once(inst, communities, blocked_pairs)

        return groups, time.perf_counter() - start

    def _get_communities(self, inst) -> tuple[list[list[int]], float]:
        if "communities" in inst and inst["communities"]:
            communities = [sorted(set(map(int, c))) for c in inst["communities"] if len(c) > 0]
            modularity = float(inst.get("community_modularity", 1.0))
            return communities, modularity

        graph = self._build_weighted_graph(inst)
        if graph.number_of_nodes() == 0:
            return [], 0.0
        if graph.number_of_edges() == 0:
            return [[n] for n in graph.nodes()], 0.0

        communities = [sorted(c) for c in nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")]
        modularity = float(nx.algorithms.community.modularity(graph, communities, weight="weight"))
        return communities, modularity

    def _build_weighted_graph(self, inst) -> nx.Graph:
        n = int(inst["num_parts"])
        adj = np.asarray(inst["assembly_adj"])
        w = np.asarray(inst["W"], dtype=float)

        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    graph.add_edge(i, j, weight=float(w[i, j]))
        return graph

    def _solve_cpccd_once(
        self,
        inst,
        communities: list[list[int]],
        blocked_pairs: set[tuple[int, int]],
    ) -> list[list[int]]:
        intra_edges, inter_edges = self._partition_edges(inst, communities, blocked_pairs)

        intra_survivors = self._filter_intra_edges(inst, intra_edges)
        inter_survivors = self._filter_inter_edges(inst, inter_edges)

        prioritized_edges = []
        for edge in intra_survivors:
            prioritized_edges.append((1, edge[2], edge[0], edge[1]))
        for edge in inter_survivors:
            prioritized_edges.append((0, edge[2], edge[0], edge[1]))

        prioritized_edges.sort(key=lambda x: (x[0], x[1]), reverse=True)

        groups = [[i] for i in range(inst["num_parts"])]
        for _, _, u, v in prioritized_edges:
            groups = self._try_merge_pair(groups, u, v, inst)

        return groups

    def _partition_edges(
        self,
        inst,
        communities: list[list[int]],
        blocked_pairs: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
        adj = np.asarray(inst["assembly_adj"])
        w = np.asarray(inst["W"], dtype=float)
        community_of = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                community_of[int(node)] = idx

        intra_edges = []
        inter_edges = []

        for i in range(inst["num_parts"]):
            for j in range(i + 1, inst["num_parts"]):
                pair = (i, j)
                if pair in blocked_pairs or not adj[i, j]:
                    continue

                weight = float(w[i, j])
                if community_of.get(i) == community_of.get(j):
                    intra_edges.append((i, j, weight))
                else:
                    inter_edges.append((i, j, weight))

        return intra_edges, inter_edges

    def _filter_intra_edges(self, inst, edges: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        survivors = []
        for u, v, weight in edges:
            if self._pair_feasible(u, v, inst, include_size=True):
                survivors.append((u, v, weight))
        return survivors

    def _filter_inter_edges(self, inst, edges: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        survivors = []
        for u, v, weight in edges:
            if self._pair_feasible(u, v, inst, include_size=False):
                survivors.append((u, v, weight))
        return survivors

    def _pair_feasible(self, u: int, v: int, inst, include_size: bool) -> bool:
        pair = tuple(sorted((u, v)))

        if not self._node_feasible(u, inst):
            return False
        if not self._node_feasible(v, inst):
            return False

        compat = np.asarray(inst.get("compat", np.ones_like(inst["assembly_adj"])))
        if compat[u, v] == 0:
            self._add_conflict("CF_PairRule", pair, "pair compatibility failed")
            return False

        if include_size and not self._group_size_ok([u, v], inst):
            self._add_conflict("CF_SizeLimit", pair, "pair size limit failed")
            return False

        if not self._connected([u, v], inst):
            self._add_conflict("CF_Disconnected", pair, "pair connectivity failed")
            return False

        return True

    def _node_feasible(self, node: int, inst) -> bool:
        if "isstandard" in inst and np.asarray(inst["isstandard"])[node]:
            self._add_conflict("CF_StandardD", (node,), "standard/electronic device")
            return False

        if "material_available" in inst and not np.asarray(inst["material_available"])[node]:
            self._add_conflict("CF_Material0", (node,), "material unavailable for AM")
            return False

        size = np.asarray(inst["size"])
        build_limit = inst["build_limit"]
        if np.ndim(size) == 2:
            if np.any(size[node] > build_limit):
                self._add_conflict("CF_SizeLimit", (node,), "single node exceeds build limit")
                return False
        else:
            if size[node] > build_limit:
                self._add_conflict("CF_SizeLimit", (node,), "single node exceeds build limit")
                return False

        return True

    def _group_size_ok(self, group: list[int], inst) -> bool:
        size = np.asarray(inst["size"])
        build_limit = np.asarray(inst["build_limit"])
        if size.ndim == 1:
            total = np.sum(size[group])
            return bool(total <= build_limit)
        total = np.sum(size[group], axis=0)
        return bool(np.all(total <= build_limit))

    def _connected(self, group: list[int], inst) -> bool:
        adj = np.asarray(inst["assembly_adj"])
        if not group:
            return True

        visited = {group[0]}
        stack = [group[0]]

        while stack:
            cur = stack.pop()
            for j in group:
                if adj[cur, j] and j not in visited:
                    visited.add(j)
                    stack.append(j)

        return len(visited) == len(group)

    def _try_merge_pair(self, groups: list[list[int]], u: int, v: int, inst) -> list[list[int]]:
        idx_u = idx_v = None
        for idx, group in enumerate(groups):
            if u in group:
                idx_u = idx
            if v in group:
                idx_v = idx

        if idx_u is None or idx_v is None or idx_u == idx_v:
            return groups

        merged = sorted(groups[idx_u] + groups[idx_v])
        if not self._group_feasible(merged, inst):
            return groups

        keep = min(idx_u, idx_v)
        drop = max(idx_u, idx_v)
        groups[keep] = merged
        groups.pop(drop)
        return groups

    def _group_feasible(self, group: list[int], inst) -> bool:
        compat = np.asarray(inst.get("compat", np.ones_like(inst["assembly_adj"])))
        for node in group:
            if not self._node_feasible(node, inst):
                return False

        if not self._group_size_ok(group, inst):
            self._add_conflict("CF_SizeLimit", tuple(group), "group size limit failed")
            return False

        for i in group:
            for j in group:
                if compat[i, j] == 0:
                    self._add_conflict("CF_PairRule", tuple(group), "group compatibility failed")
                    return False

        if not self._connected(group, inst):
            self._add_conflict("CF_Disconnected", tuple(group), "group connectivity failed")
            return False

        return True

    def _check_rule_r3(self, groups: list[list[int]], inst):
        checker = inst.get("assembly_access_checker")
        if checker is None:
            return None

        for group in groups:
            ok, detail = checker(group, groups, inst)
            if not ok:
                group_sorted = sorted(group)
                if len(group_sorted) >= 2:
                    pair = (group_sorted[-2], group_sorted[-1])
                else:
                    pair = (group_sorted[0], group_sorted[0])
                return tuple(sorted(pair)), detail
        return None

    def _solve_strength_only(self, inst) -> list[list[int]]:
        w = np.asarray(inst["W"], dtype=float)
        edges = []
        for i in range(inst["num_parts"]):
            for j in range(i + 1, inst["num_parts"]):
                if inst["assembly_adj"][i, j]:
                    edges.append((float(w[i, j]), i, j))
        edges.sort(reverse=True)

        groups = [[i] for i in range(inst["num_parts"])]
        for _, u, v in edges:
            groups = self._try_merge_pair(groups, u, v, inst)
        return groups

    def _add_conflict(self, code: str, subject: tuple[int, ...], detail: str) -> None:
        record = ConflictRecord(code=code, subject=tuple(subject), detail=detail)
        self.last_conflicts.append(record)
