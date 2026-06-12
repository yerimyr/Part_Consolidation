from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np

from rl4co.envs.pc.evaluator import evaluate_groups
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.cpccd_solver import CPCCDSolver


def td_to_inst(td, num_parts: int) -> dict:
    inst = {
        "num_parts": int(num_parts),
        "material": td["material"][0, 1:].cpu().numpy(),
        "size": td["size"][0, 1:].cpu().numpy(),
        "maintfreq": td["maintfreq"][0, 1:].cpu().numpy(),
        "isstandard": td["isstandard"][0, 1:].cpu().numpy(),
        "material_available": (td["material"][0, 1:] >= 0).cpu().numpy(),
        "build_limit": td["build_limit"][0].cpu().numpy(),
        "assembly_adj": td["assembly_adj"][0, 1:, 1:].cpu().numpy(),
        "mat_var": td["mat_var"][0, 1:, 1:].cpu().numpy(),
        "stack_size": td["stack_size"][0, 1:, 1:, :].cpu().numpy(),
        "maint_diff": td["maint_diff"][0, 1:, 1:].cpu().numpy(),
        "rel_motion": td["rel_motion"][0, 1:, 1:].cpu().numpy(),
        "compat": td["compat"][0, 1:, 1:].cpu().numpy(),
        "W": td["W"][0, 1:, 1:].cpu().numpy(),
        "relation_valid": td["relation_valid"][0, 1:, 1:].cpu().numpy(),
        "relation_consistent": bool(td["relation_consistent"][0].item()),
    }
    if "topology_id" in td.keys():
        inst["topology_id"] = int(td["topology_id"][0].item())
    return inst


def fmt_bool(flag: bool) -> str:
    return "PASS" if flag else "FAIL"


def fmt_vec(vec) -> str:
    arr = np.asarray(vec, dtype=float)
    return "[" + ", ".join(f"{x:.1f}" for x in arr.tolist()) + "]"


def neighbors_of(node: int, adj: np.ndarray) -> list[int]:
    return [int(j) for j in np.where(adj[node])[0]]


def connected(group: list[int], adj: np.ndarray) -> bool:
    if not group:
        return True
    visited = {group[0]}
    stack = [group[0]]
    while stack:
        cur = stack.pop()
        for nxt in group:
            if adj[cur, nxt] and nxt not in visited:
                visited.add(nxt)
                stack.append(nxt)
    return len(visited) == len(group)


def node_feasible(node: int, inst: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True

    if "material_available" in inst and not bool(np.asarray(inst["material_available"])[node]):
        ok = False
        reasons.append("material unavailable")

    size = np.asarray(inst["size"])
    build_limit = np.asarray(inst["build_limit"])
    if not np.all(size[node] <= build_limit):
        ok = False
        reasons.append(f"single size {fmt_vec(size[node])} exceeds build_limit {fmt_vec(build_limit)}")

    if ok:
        reasons.append("node-level constraints passed")
    return ok, reasons


def pair_rule_report(i: int, j: int, inst: dict) -> dict:
    material = np.asarray(inst["material"])
    maintfreq = np.asarray(inst["maintfreq"])
    size = np.asarray(inst["size"])
    build_limit = np.asarray(inst["build_limit"])
    isstandard = np.asarray(inst["isstandard"]).astype(bool)
    assembly_adj = np.asarray(inst["assembly_adj"]).astype(bool)
    mat_var = np.asarray(inst["mat_var"]).astype(bool)
    maint_diff = np.asarray(inst["maint_diff"]).astype(bool)
    rel_motion = np.asarray(inst["rel_motion"]).astype(bool)
    stack_size = np.asarray(inst["stack_size"])
    compat = np.asarray(inst["compat"]).astype(bool)
    w = np.asarray(inst["W"], dtype=float)

    pair_ok = True
    fail_reasons: list[str] = []

    checks = {
        "adjacent": bool(assembly_adj[i, j]),
        "same_material": not bool(mat_var[i, j]),
        "same_maintfreq": not bool(maint_diff[i, j]),
        "no_relative_motion": not bool(rel_motion[i, j]),
        "pair_stack_within_limit": bool(np.all(stack_size[i, j] <= build_limit)),
        "not_standard_pair": not bool(isstandard[i] or isstandard[j]),
        "compat_matrix_true": bool(compat[i, j]),
    }

    if not checks["adjacent"]:
        pair_ok = False
        fail_reasons.append("not physically adjacent")
    if not checks["same_material"]:
        pair_ok = False
        fail_reasons.append(f"material variance ({material[i]} vs {material[j]})")
    if not checks["same_maintfreq"]:
        pair_ok = False
        fail_reasons.append(f"maintfreq difference ({maintfreq[i]} vs {maintfreq[j]})")
    if not checks["no_relative_motion"]:
        pair_ok = False
        fail_reasons.append("relative motion exists")
    if not checks["pair_stack_within_limit"]:
        pair_ok = False
        fail_reasons.append(
            f"stack size {fmt_vec(stack_size[i, j])} exceeds build_limit {fmt_vec(build_limit)}"
        )
    if not checks["not_standard_pair"]:
        pair_ok = False
        fail_reasons.append("one of the parts is standard")
    if not checks["compat_matrix_true"]:
        pair_ok = False
        fail_reasons.append("compatibility matrix blocks this pair")

    return {
        "pair": (i, j),
        "ok": pair_ok,
        "checks": checks,
        "fail_reasons": fail_reasons,
        "W": float(w[i, j]),
        "stack_size": stack_size[i, j].copy(),
        "size_i": size[i].copy(),
        "size_j": size[j].copy(),
    }


def group_report(group: list[int], inst: dict) -> dict:
    adj = np.asarray(inst["assembly_adj"]).astype(bool)
    size = np.asarray(inst["size"], dtype=float)
    build_limit = np.asarray(inst["build_limit"], dtype=float)

    total_size = size[group].sum(axis=0) if group else np.zeros_like(build_limit)
    node_reports = {node: node_feasible(node, inst) for node in group}

    pair_reports = []
    compat_all = True
    standard_ok = not (len(group) >= 2 and np.asarray(inst["isstandard"]).astype(bool)[group].any())
    for idx_i in range(len(group)):
        for idx_j in range(idx_i + 1, len(group)):
            pair = pair_rule_report(group[idx_i], group[idx_j], inst)
            pair_reports.append(pair)
            compat_all = compat_all and pair["ok"]

    node_all_ok = all(flag for flag, _ in node_reports.values())
    size_ok = bool(np.all(total_size <= build_limit))
    connected_ok = connected(group, adj)
    group_ok = node_all_ok and standard_ok and size_ok and compat_all and connected_ok

    group_fail_reasons: list[str] = []
    if not node_all_ok:
        bad_nodes = [str(node) for node, (ok, _) in node_reports.items() if not ok]
        group_fail_reasons.append("node-level failure: " + ", ".join(bad_nodes))
    if not standard_ok:
        group_fail_reasons.append("standard part can only appear as a singleton group")
    if not size_ok:
        group_fail_reasons.append(
            f"group size {fmt_vec(total_size)} exceeds build_limit {fmt_vec(build_limit)}"
        )
    if not compat_all:
        group_fail_reasons.append("at least one pair violates compatibility-related rules")
    if not connected_ok:
        group_fail_reasons.append("group is not connected in assembly_adj")

    return {
        "group": list(group),
        "ok": group_ok,
        "total_size": total_size,
        "node_reports": node_reports,
        "pair_reports": pair_reports,
        "size_ok": size_ok,
        "connected_ok": connected_ok,
        "group_fail_reasons": group_fail_reasons,
    }


def instance_consistency_checks(inst: dict) -> list[str]:
    issues: list[str] = []
    adj = np.asarray(inst["assembly_adj"]).astype(bool)
    mat_var = np.asarray(inst["mat_var"]).astype(bool)
    maint_diff = np.asarray(inst["maint_diff"]).astype(bool)
    rel_motion = np.asarray(inst["rel_motion"]).astype(bool)
    stack_size = np.asarray(inst["stack_size"], dtype=float)
    w = np.asarray(inst["W"], dtype=float)

    non_adj = ~adj
    if np.any(mat_var[non_adj]):
        issues.append("mat_var has non-zero entries on non-adjacent pairs")
    if np.any(maint_diff[non_adj]):
        issues.append("maint_diff has non-zero entries on non-adjacent pairs")
    if np.any(rel_motion[non_adj]):
        issues.append("rel_motion has non-zero entries on non-adjacent pairs")
    if np.any(stack_size[non_adj] > 0):
        issues.append("stack_size has non-zero entries on non-adjacent pairs")
    if np.any(w[non_adj] > 0):
        issues.append("W has positive entries on non-adjacent pairs")
    if not bool(inst.get("relation_consistent", True)):
        issues.append("relation_consistent flag is False")
    return issues


def render_instance_header(idx: int, inst: dict) -> list[str]:
    adj = np.asarray(inst["assembly_adj"]).astype(bool)
    topology_names = [
        "chain",
        "star",
        "tree",
        "two_module_bridge",
        "dense_clustered",
        "sparse_random",
    ]
    topo_text = "unknown"
    if "topology_id" in inst:
        topo_id = int(inst["topology_id"])
        if 0 <= topo_id < len(topology_names):
            topo_text = topology_names[topo_id]
    lines = [
        "",
        "=" * 90,
        f"INSTANCE {idx}",
        "=" * 90,
        f"num_parts: {inst['num_parts']}",
        f"topology: {topo_text}",
        f"build_limit [L, W, H]: {fmt_vec(inst['build_limit'])}",
        "part summary:",
    ]
    for node in range(int(inst["num_parts"])):
        lines.append(
            "  "
            f"part {node}: material={int(inst['material'][node])}, "
            f"maintfreq={int(inst['maintfreq'][node])}, "
            f"isstandard={int(inst['isstandard'][node])}, "
            f"size={fmt_vec(inst['size'][node])}, "
            f"neighbors={neighbors_of(node, adj)}"
        )
    return lines


def render_consistency_report(inst: dict) -> list[str]:
    issues = instance_consistency_checks(inst)
    lines = ["", "[1] GENERATOR / INSTANCE CONSISTENCY CHECK"]
    if not issues:
        lines.append("  PASS: relation tensors are consistent with the physical adjacency matrix.")
    else:
        lines.append("  FAIL:")
        for issue in issues:
            lines.append(f"    - {issue}")
    return lines


def render_structure_report(inst: dict) -> list[str]:
    adj = np.asarray(inst["assembly_adj"]).astype(int)
    lines = [
        "",
        "[2] PHYSICAL STRUCTURE (assembly_adj)",
        "  adjacency matrix:",
    ]
    for row in adj:
        lines.append("    " + " ".join(str(int(x)) for x in row))
    return lines


def render_solver_report(groups: list[list[int]], metrics: dict[str, float], elapsed: float, solver_name: str) -> list[str]:
    return [
        "",
        f"[3] FINAL GROUPING RESULT ({solver_name})",
        f"  groups: {groups}",
        f"  runtime (sec): {elapsed:.6f}",
        f"  feasible: {metrics['feasible']}",
        f"  infeasible_solution: {metrics['infeasible_solution']}",
        f"  infeasible_groups: {metrics['infeasible_groups']}",
        f"  num_groups: {metrics['num_groups']}",
        f"  total_internal_strength: {metrics['total_internal_strength']:.4f}",
        f"  feasible_pair_count: {metrics['feasible_pair_count']}",
    ]


def render_group_explanations(groups: list[list[int]], inst: dict) -> list[str]:
    lines = ["", "[4] GROUP-BY-GROUP VALIDATION"]
    for gid, group in enumerate(groups):
        report = group_report(group, inst)
        lines.append(f"  Group {gid}: {group} -> {fmt_bool(report['ok'])}")
        lines.append(
            "    "
            f"group_size={fmt_vec(report['total_size'])}, "
            f"build_limit={fmt_vec(inst['build_limit'])}, "
            f"size_ok={report['size_ok']}, connected_ok={report['connected_ok']}"
        )

        for node, (ok, reasons) in report["node_reports"].items():
            lines.append(
                "    "
                f"node {node}: {fmt_bool(ok)} | "
                + "; ".join(reasons)
            )

        if report["pair_reports"]:
            lines.append("    pair checks:")
        for pair_report in report["pair_reports"]:
            i, j = pair_report["pair"]
            checks = pair_report["checks"]
            summary = (
                f"pair ({i}, {j}) -> {fmt_bool(pair_report['ok'])} | "
                f"adjacent={checks['adjacent']}, "
                f"same_material={checks['same_material']}, "
                f"same_maintfreq={checks['same_maintfreq']}, "
                f"no_relative_motion={checks['no_relative_motion']}, "
                f"stack_ok={checks['pair_stack_within_limit']}, "
                f"not_standard_pair={checks['not_standard_pair']}, "
                f"compat={checks['compat_matrix_true']}, "
                f"W={pair_report['W']:.3f}"
            )
            lines.append("      " + summary)
            if pair_report["fail_reasons"]:
                lines.append("        fail reasons: " + "; ".join(pair_report["fail_reasons"]))
            lines.append(
                "        "
                f"stack_size={fmt_vec(pair_report['stack_size'])}, "
                f"size_i={fmt_vec(pair_report['size_i'])}, "
                f"size_j={fmt_vec(pair_report['size_j'])}"
            )

        if report["group_fail_reasons"]:
            lines.append("    group fail reasons: " + "; ".join(report["group_fail_reasons"]))
        else:
            lines.append("    explanation: all node-level, pair-level, size, and connectivity checks passed.")
    return lines


def validate_random_instances(
    num_parts: int,
    num_instances: int,
    seed: int | None,
    output_path: str | None,
) -> str:
    if seed is not None:
        np.random.seed(seed)

    generator = FPIGenerator(num_parts=num_parts)
    solver = CPCCDSolver()

    lines = [
        "PART CONSOLIDATION AUTOMATED VALIDATION REPORT",
        f"generator: FPIGenerator(num_parts={num_parts})",
        f"num_instances: {num_instances}",
        f"solver_for_grouping: CPCCDSolver",
    ]

    for idx in range(num_instances):
        td = generator(batch_size=1, device="cpu")
        inst = td_to_inst(td, num_parts=num_parts)
        groups, elapsed = solver.solve(inst)
        metrics = evaluate_groups(groups, inst)

        lines.extend(render_instance_header(idx, inst))
        lines.extend(render_consistency_report(inst))
        lines.extend(render_structure_report(inst))
        lines.extend(render_solver_report(groups, metrics, elapsed, "CPCCD"))
        lines.extend(render_group_explanations(groups, inst))

    report = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")

    return report


def main():
    parser = argparse.ArgumentParser(description="Automated validation for random part-consolidation instances.")
    parser.add_argument("--num-parts", type=int, default=6, help="Number of real parts in the random instance.")
    parser.add_argument("--num-instances", type=int, default=3, help="Number of random instances to validate.")
    parser.add_argument("--seed", type=int, default=None, help="Optional numpy seed for reproducible runs.")
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.txt",
        help="Optional text report output path.",
    )
    args = parser.parse_args()

    report = validate_random_instances(
        num_parts=args.num_parts,
        num_instances=args.num_instances,
        seed=args.seed,
        output_path=args.output,
    )
    print(report)
    print(f"\nvalidation report saved to: {args.output}")


if __name__ == "__main__":
    main()
