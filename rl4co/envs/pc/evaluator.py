from __future__ import annotations

from collections import defaultdict

import numpy as np


SCORE_EPS = 1e-8
DEFAULT_MODULARITY_GAMMA = 1.0
DEFAULT_OBJECTIVE_SCALE = 1000.0


def group_size_ok(group: list[int], inst) -> bool:
    size = np.asarray(inst["size"])
    build_limit = np.asarray(inst["build_limit"])
    if size.ndim == 1:
        return bool(np.sum(size[group]) <= build_limit)
    return bool(np.all(np.sum(size[group], axis=0) <= build_limit))


def node_feasible(node: int, inst) -> bool:
    if "material_available" in inst and not np.asarray(inst["material_available"])[node]:
        return False

    size = np.asarray(inst["size"])
    build_limit = np.asarray(inst["build_limit"])
    if size.ndim == 1:
        return bool(size[node] <= build_limit)
    return bool(np.all(size[node] <= build_limit))


def connected(group: list[int], inst) -> bool:
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


def no_pairwise_conflict(group: list[int], inst) -> bool:
    mat_var = np.asarray(inst.get("mat_var", np.zeros_like(inst["assembly_adj"])))
    maint_diff = np.asarray(inst.get("maint_diff", np.zeros_like(inst["assembly_adj"])))
    rel_motion = np.asarray(inst.get("rel_motion", np.zeros_like(inst["assembly_adj"])))
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            if mat_var[a, b] or maint_diff[a, b] or rel_motion[a, b]:
                return False
    return True


def group_feasible(group: list[int], inst) -> bool:
    if any(not node_feasible(node, inst) for node in group):
        return False
    if len(group) >= 2 and "isstandard" in inst and np.asarray(inst["isstandard"])[group].any():
        return False
    if not group_size_ok(group, inst):
        return False
    if not no_pairwise_conflict(group, inst):
        return False
    return connected(group, inst)


def internal_strength(group: list[int], inst) -> float:
    w = np.asarray(inst["W"], dtype=float)
    total = 0.0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            total += float(w[group[i], group[j]])
    return total


def internal_edge_count(group: list[int], inst) -> int:
    adj = np.asarray(inst["assembly_adj"]).astype(bool)
    count = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if adj[group[i], group[j]]:
                count += 1
    return count


def feasible_pair_count(group: list[int], inst) -> int:
    compat = np.asarray(inst.get("compat", np.ones_like(inst["assembly_adj"])))
    count = 0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if compat[group[i], group[j]]:
                count += 1
    return count


def check_r3(groups: list[list[int]], inst) -> bool:
    checker = inst.get("assembly_access_checker")
    if checker is None:
        return True
    for group in groups:
        ok, _ = checker(group, groups, inst)
        if not ok:
            return False
    return True


def evaluate_groups(groups: list[list[int]], inst) -> dict[str, float]:
    infeasible_groups = 0
    total_internal_strength = 0.0
    total_feasible_pairs = 0

    for group in groups:
        feasible = group_feasible(group, inst)
        infeasible_groups += int(not feasible)
        total_internal_strength += internal_strength(group, inst)
        total_feasible_pairs += feasible_pair_count(group, inst)

    infeasible_solution = int(infeasible_groups > 0 or not check_r3(groups, inst))
    num_groups = float(len(groups))
    normalized_internal_strength = total_internal_strength / max(float(total_feasible_pairs), 1.0)
    q_gamma, q_observed, q_expected = modularity_objective(
        groups,
        inst,
        gamma=DEFAULT_MODULARITY_GAMMA,
    )
    return {
        "feasible": float(1 - infeasible_solution),
        "infeasible_solution": float(infeasible_solution),
        "infeasible_groups": float(infeasible_groups),
        "num_groups": num_groups,
        "total_internal_strength": float(total_internal_strength),
        "feasible_pair_count": float(total_feasible_pairs),
        "normalized_internal_strength": float(normalized_internal_strength),
        "Q_gamma": float(q_gamma * DEFAULT_OBJECTIVE_SCALE),
        "Q_observed": float(q_observed * DEFAULT_OBJECTIVE_SCALE),
        "Q_expected": float(q_expected * DEFAULT_OBJECTIVE_SCALE),
    }


def modularity_objective(
    groups: list[list[int]],
    inst,
    gamma: float = DEFAULT_MODULARITY_GAMMA,
) -> tuple[float, float, float]:
    w = np.asarray(inst["W"], dtype=float)
    strengths = np.sum(w, axis=1)
    two_m = float(np.sum(strengths))
    if two_m <= SCORE_EPS:
        return 0.0, 0.0, 0.0

    observed = 0.0
    expected = 0.0
    for group in groups:
        if not group:
            continue
        idx = np.asarray(group, dtype=int)
        observed += float(np.sum(w[np.ix_(idx, idx)]))
        group_strength = float(np.sum(strengths[idx]))
        expected += group_strength * group_strength / two_m

    q_observed = observed / two_m
    q_expected = float(gamma) * expected / two_m
    q_gamma = q_observed - q_expected
    return q_gamma, q_observed, q_expected


def _augment_reward_metrics(row: dict) -> dict:
    out = dict(row)

    num_groups = float(out.get("num_groups", out.get("groups", 0.0)))
    total_internal_strength = float(out.get("total_internal_strength", 0.0))
    feasible_pair_count_value = float(out.get("feasible_pair_count", 0.0))

    out["num_groups"] = num_groups
    out["normalized_internal_strength"] = total_internal_strength / max(feasible_pair_count_value, 1.0)
    if "Q_gamma" not in out:
        out["Q_gamma"] = float(out.get("Q_gamma", 0.0))
    if "Q_observed" not in out:
        out["Q_observed"] = float(out.get("Q_observed", 0.0))
    if "Q_expected" not in out:
        out["Q_expected"] = float(out.get("Q_expected", 0.0))
    return out


def dynamic_signed_score(row: dict) -> float:
    return float(row["Q_gamma"])


def score_metric_rows(rows: list[dict], weights: dict[str, float] | None = None) -> list[dict]:
    scored = []
    for row in rows:
        enriched = _augment_reward_metrics(row)
        if weights is None:
            score = dynamic_signed_score(enriched)
        else:
            score = 0.0
            for field, weight in weights.items():
                score += weight * float(enriched[field])
        out = dict(enriched)
        out["score"] = score
        scored.append(out)
    return scored


def score_metric_rows_by_group(
    rows: list[dict],
    group_fields: list[str],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    scored = []
    for _, items in grouped.items():
        scored.extend(score_metric_rows(items, weights=weights))
    return scored
