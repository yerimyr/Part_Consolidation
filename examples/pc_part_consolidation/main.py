import os
import argparse
from pathlib import Path
import csv
from collections import defaultdict
from itertools import combinations
import random
import sys
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import time

import torch
from tensordict import TensorDict

from rl4co.envs.pc.env import PartConsolidationEnv
from rl4co.envs.pc.evaluator import evaluate_groups
from rl4co.envs.pc.evaluator import score_metric_rows_by_group
from rl4co.envs.pc.fixed_instance_benchmark import create_fixed_instance
from rl4co.envs.pc.generator import FPIGenerator
from rl4co.models.zoo.pc.cpccd_solver import CPCCDSolver
from rl4co.models.zoo.pc.ga_solver import GASolver
from rl4co.models.zoo.pc.policy import PCPolicy
from rl4co.models.zoo.pc.sa_solver import SASolver


def _topology_name(topology_id):
    names = [
        "chain",
        "star",
        "tree",
        "two_module_bridge",
        "dense_clustered",
        "sparse_random",
    ]
    if topology_id is None:
        return None
    topology_id = int(topology_id)
    if 0 <= topology_id < len(names):
        return names[topology_id]
    return f"topology_{topology_id}"


def visualize_grouping_solution(inst, groups, method, output_path, metrics=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(inst["num_parts"])
    adj = inst["assembly_adj"]
    w = inst["W"]
    isstandard = inst["isstandard"]

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if bool(adj[i, j]):
                graph.add_edge(i, j, weight=float(w[i, j]))

    node_to_group = {}
    for gid, group in enumerate(groups):
        for node in group:
            node_to_group[int(node)] = gid

    palette = plt.get_cmap("tab10", max(len(groups), 1) + 1)
    node_colors = []
    border_colors = []
    labels = {}
    for node in range(n):
        gid = node_to_group.get(node, -1)
        node_colors.append(palette(gid % palette.N) if gid >= 0 else (0.75, 0.75, 0.75, 1.0))
        border_colors.append("crimson" if bool(isstandard[node]) else "black")
        labels[node] = f"{node}" + ("S" if bool(isstandard[node]) else "")

    intra_edges = []
    inter_edges = []
    for u, v in graph.edges():
        same_group = node_to_group.get(u, -1) >= 0 and node_to_group.get(u, -1) == node_to_group.get(v, -2)
        if same_group:
            intra_edges.append((u, v))
        else:
            inter_edges.append((u, v))

    if graph.number_of_edges() > 0:
        pos = nx.spring_layout(graph, seed=7)
    else:
        pos = {
            node: (np.cos(2 * np.pi * node / max(n, 1)), np.sin(2 * np.pi * node / max(n, 1)))
            for node in range(n)
        }

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_edges(graph, pos, edgelist=inter_edges, edge_color="#bdbdbd", width=1.3, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(graph, pos, edgelist=intra_edges, edge_color="#1b7f5f", width=2.7, alpha=0.95, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        edgecolors=border_colors,
        linewidths=2.0,
        node_size=900,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={(u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges()},
        font_size=8,
        ax=ax,
    )

    title_lines = [f"{method} | num_parts={n}"]
    topo = _topology_name(inst.get("topology_id"))
    if topo is not None:
        title_lines[0] += f" | topology={topo}"
    title_lines.append(f"groups={len(groups)} | grouping={groups}")
    if metrics is not None:
        title_lines.append(
            " | ".join(
                [
                    f"feasible={int(metrics['feasible'])}",
                    f"infeasible_solution={int(metrics['infeasible_solution'])}",
                    f"infeasible_groups={int(metrics['infeasible_groups'])}",
                    f"strength={metrics['total_internal_strength']:.2f}",
                    f"pairs={metrics['feasible_pair_count']:.2f}",
                ]
            )
        )
    ax.set_title("\n".join(title_lines), fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def actions_to_groups(actions):
    groups = []
    current = []
    used = set()

    for a in actions:
        a = int(a)
        if a == 0:
            if current:
                groups.append(current)
                current = []
        else:
            part_idx = a - 1
            if part_idx not in used:
                current.append(part_idx)
                used.add(part_idx)

    if current:
        groups.append(current)

    return groups


def run_nco(env, policy, td, max_steps=None):
    start = time.time()
    td = td.clone().to(env.device)
    actions = []
    max_steps = max_steps or (env.generator.num_parts * 2 + 4)

    policy.eval()
    with torch.no_grad():
        for _ in range(max_steps):
            if td["done"].all():
                break
            action, _, _, _ = policy.act(td, sample=False, epsilon=0.0)
            actions.append(int(action[0].item()))
            td = env.step(td, action)

    groups = actions_to_groups(actions)
    return groups, time.time() - start


def evaluate(groups):
    return len(groups)


def td_to_inst(td, num_parts):
    actual_num_parts = int(td.get("num_parts", torch.tensor([num_parts]))[0].item())
    return {
        "num_parts": actual_num_parts,
        "topology_id": int(td.get("topology_id", torch.tensor([-1]))[0].item()),
        "material": td["material"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "size": td["size"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "maintfreq": td["maintfreq"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "isstandard": td["isstandard"][0, 1 : actual_num_parts + 1].cpu().numpy(),
        "material_available": (td["material"][0, 1 : actual_num_parts + 1] >= 0).cpu().numpy(),
        "build_limit": td["build_limit"][0].cpu().numpy(),
        "assembly_adj": td["assembly_adj"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "mat_var": td["mat_var"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "stack_size": td["stack_size"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1, :].cpu().numpy(),
        "maint_diff": td["maint_diff"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "rel_motion": td["rel_motion"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "compat": td["compat"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "W": td["W"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "relation_valid": td["relation_valid"][0, 1 : actual_num_parts + 1, 1 : actual_num_parts + 1].cpu().numpy(),
        "relation_consistent": bool(td["relation_consistent"][0].item()),
    }


def inst_to_td(inst, env):
    device = env.device
    generator = env.generator
    num_parts = int(inst["num_parts"])
    num_nodes = num_parts + 1
    material_types = generator.p.material_types

    material = torch.as_tensor(inst["material"], dtype=torch.long, device=device)
    size = torch.as_tensor(inst["size"], dtype=torch.float32, device=device)
    maintfreq = torch.as_tensor(inst["maintfreq"], dtype=torch.long, device=device)
    isstandard = torch.as_tensor(inst["isstandard"], dtype=torch.long, device=device)
    build_limit = torch.as_tensor(inst["build_limit"], dtype=torch.float32, device=device)
    assembly_adj = torch.as_tensor(inst["assembly_adj"], dtype=torch.bool, device=device)
    mat_var = torch.as_tensor(inst["mat_var"], dtype=torch.float32, device=device)
    stack_size = torch.as_tensor(inst["stack_size"], dtype=torch.float32, device=device)
    maint_diff = torch.as_tensor(inst["maint_diff"], dtype=torch.float32, device=device)
    rel_motion = torch.as_tensor(inst["rel_motion"], dtype=torch.float32, device=device)
    compat = torch.as_tensor(inst["compat"], dtype=torch.bool, device=device)
    W = torch.as_tensor(inst["W"], dtype=torch.float32, device=device)
    relation_valid = torch.as_tensor(inst.get("relation_valid", inst["assembly_adj"]), dtype=torch.bool, device=device)

    degree = assembly_adj.sum(dim=-1).float()
    max_degree = degree.max().clamp_min(1.0)
    pos1d = (degree / max_degree).unsqueeze(-1)
    material_one_hot = torch.nn.functional.one_hot(material, num_classes=material_types).float()

    part_node_features = torch.cat(
        [
            material_one_hot,
            size,
            maintfreq.float().unsqueeze(-1),
            isstandard.float().unsqueeze(-1),
            pos1d,
        ],
        dim=-1,
    )
    part_edge_features = torch.cat(
        [
            assembly_adj.float().unsqueeze(-1),
            mat_var.unsqueeze(-1),
            stack_size,
            maint_diff.unsqueeze(-1),
            rel_motion.unsqueeze(-1),
        ],
        dim=-1,
    )

    node_features = torch.zeros((1, num_nodes, generator.node_feat_dim), dtype=torch.float32, device=device)
    node_features[:, 1:, :] = part_node_features.unsqueeze(0)

    edge_features = torch.zeros((1, num_nodes, num_nodes, generator.edge_feat_dim), dtype=torch.float32, device=device)
    edge_features[:, 1:, 1:, :] = part_edge_features.unsqueeze(0)

    material_all = torch.full((1, num_nodes), -1, dtype=torch.long, device=device)
    maint_all = torch.full((1, num_nodes), -1, dtype=torch.long, device=device)
    std_all = torch.full((1, num_nodes), -1, dtype=torch.long, device=device)
    size_all = torch.zeros((1, num_nodes, 3), dtype=torch.float32, device=device)
    pos_all = torch.zeros((1, num_nodes, 1), dtype=torch.float32, device=device)
    W_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    assembly_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.bool, device=device)
    mat_var_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    maint_diff_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    rel_motion_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.float32, device=device)
    stack_all = torch.zeros((1, num_nodes, num_nodes, 3), dtype=torch.float32, device=device)
    compat_all = torch.ones((1, num_nodes, num_nodes), dtype=torch.bool, device=device)
    relation_valid_all = torch.zeros((1, num_nodes, num_nodes), dtype=torch.bool, device=device)

    material_all[:, 1:] = material.unsqueeze(0)
    maint_all[:, 1:] = maintfreq.unsqueeze(0)
    std_all[:, 1:] = isstandard.unsqueeze(0)
    size_all[:, 1:, :] = size.unsqueeze(0)
    pos_all[:, 1:, :] = pos1d.unsqueeze(0)
    W_all[:, 1:, 1:] = W.unsqueeze(0)
    assembly_all[:, 1:, 1:] = assembly_adj.unsqueeze(0)
    mat_var_all[:, 1:, 1:] = mat_var.unsqueeze(0)
    maint_diff_all[:, 1:, 1:] = maint_diff.unsqueeze(0)
    rel_motion_all[:, 1:, 1:] = rel_motion.unsqueeze(0)
    stack_all[:, 1:, 1:, :] = stack_size.unsqueeze(0)
    compat_all[:, 1:, 1:] = compat.unsqueeze(0)
    relation_valid_all[:, 1:, 1:] = relation_valid.unsqueeze(0)

    assigned = torch.zeros((1, num_nodes), dtype=torch.bool, device=device)
    assigned[:, 0] = True
    open_group = torch.zeros((1, num_nodes), dtype=torch.bool, device=device)
    open_group_size = torch.zeros((1, 3), dtype=torch.float32, device=device)
    closed_group_count = torch.zeros((1,), dtype=torch.long, device=device)

    td = TensorDict(
        {
            "node_features": node_features,
            "edge_features": edge_features,
            "material": material_all,
            "size": size_all,
            "maintfreq": maint_all,
            "isstandard": std_all,
            "pos1d": pos_all,
            "W": W_all,
            "assembly_adj": assembly_all,
            "mat_var": mat_var_all,
            "stack_size": stack_all,
            "maint_diff": maint_diff_all,
            "rel_motion": rel_motion_all,
            "compat": compat_all,
            "relation_valid": relation_valid_all,
            "relation_consistent": torch.tensor([bool(inst.get("relation_consistent", True))], dtype=torch.bool, device=device),
            "build_limit": build_limit.unsqueeze(0),
            "assigned": assigned,
            "open_group": open_group,
            "open_group_size": open_group_size,
            "closed_group_count": closed_group_count,
            "fallback_part_mask": torch.zeros((1, num_nodes), dtype=torch.bool, device=device),
            "dead_end": torch.zeros((1, 1), dtype=torch.bool, device=device),
            "done": torch.zeros((1, 1), dtype=torch.bool, device=device),
            "action_mask": torch.ones((1, num_nodes), dtype=torch.bool, device=device),
        },
        batch_size=[1],
    )
    td["action_mask"] = env.get_action_mask(td)
    td["dead_end"] = env._compute_dead_end(td["assigned"], td["open_group"], td["action_mask"])
    td["done"] = td["done"] | td["dead_end"]
    return td


def print_instance(inst, title):
    print(f"\n===== {title} =====")
    print("num_parts:")
    print(inst["num_parts"])
    print("build_limit [L, W, H]:")
    print(inst["build_limit"])
    print("material:")
    print(inst["material"])
    print("maintfreq:")
    print(inst["maintfreq"])
    print("isstandard:")
    print(inst["isstandard"])
    print("size [L, W, H] per part:")
    print(inst["size"])
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


def np_round(arr, decimals=3):
    import numpy as np

    return np.round(arr, decimals=decimals)


def compute_search_space_proxy(inst):
    n = int(inst["num_parts"])
    assembly_adj = np.asarray(inst["assembly_adj"]).astype(bool)
    mat_var = np.asarray(inst["mat_var"]).astype(bool)
    maint_diff = np.asarray(inst["maint_diff"]).astype(bool)
    rel_motion = np.asarray(inst["rel_motion"]).astype(bool)

    def has_no_pairwise_conflict(nodes):
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if mat_var[a, b] or maint_diff[a, b] or rel_motion[a, b]:
                    return False
        return True

    def is_connected_subset(nodes):
        if not nodes:
            return True
        visited = {nodes[0]}
        stack = [nodes[0]]
        while stack:
            cur = stack.pop()
            for nxt in nodes:
                if nxt not in visited and assembly_adj[cur, nxt]:
                    visited.add(nxt)
                    stack.append(nxt)
        return len(visited) == len(nodes)

    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if assembly_adj[i, j] and not (mat_var[i, j] or maint_diff[i, j] or rel_motion[i, j]):
                pair_count += 1

    total_pairs = n * (n - 1) / 2
    compat_density = float(pair_count / total_pairs) if total_pairs > 0 else 0.0

    triple_count = 0
    quad_count = 0
    max_pairwise_clique_size = 1 if n > 0 else 0

    for r in range(2, min(n, 4) + 1):
        for nodes in combinations(range(n), r):
            node_list = list(nodes)
            if has_no_pairwise_conflict(node_list) and is_connected_subset(node_list):
                if r == 3:
                    triple_count += 1
                elif r == 4:
                    quad_count += 1
                if r > max_pairwise_clique_size:
                    max_pairwise_clique_size = r

    if max_pairwise_clique_size < 2 and pair_count > 0:
        max_pairwise_clique_size = 2

    return {
        "compat_density": compat_density,
        "feasible_pair_proxy": int(pair_count),
        "feasible_triple_proxy": int(triple_count),
        "feasible_quad_proxy": int(quad_count),
        "max_pairwise_clique_size": int(max_pairwise_clique_size),
    }


def instance_to_lines(inst, title):
    proxy = compute_search_space_proxy(inst)
    return [
        f"===== {title} =====",
        "num_parts:",
        str(inst["num_parts"]),
        "search_space_proxy:",
        (
            f"compat_density={proxy['compat_density']:.3f}, "
            f"pairs={proxy['feasible_pair_proxy']}, "
            f"triples={proxy['feasible_triple_proxy']}, "
            f"quads={proxy['feasible_quad_proxy']}, "
            f"max_clique={proxy['max_pairwise_clique_size']}"
        ),
        "build_limit [L, W, H]:",
        str(np_round(inst["build_limit"])),
        "material:",
        str(inst["material"]),
        "maintfreq:",
        str(inst["maintfreq"]),
        "isstandard:",
        str(inst["isstandard"]),
        "size [L, W, H] per part:",
        str(np_round(inst["size"])),
        "assembly_adj:",
        str(inst["assembly_adj"].astype(int)),
        "mat_var:",
        str(inst["mat_var"].astype(int)),
        "maint_diff:",
        str(inst["maint_diff"].astype(int)),
        "rel_motion:",
        str(inst["rel_motion"].astype(int)),
        "compat:",
        str(inst["compat"].astype(int)),
        "W:",
        str(np_round(inst["W"])),
    ]


def save_instance_info_png(inst, title, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = instance_to_lines(inst, title)
    num_lines = max(len(lines), 1)
    fig_height = max(8, 0.33 * num_lines)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    ax.text(
        0.01,
        0.99,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def result_row(instance_type, instance_id, method, groups, elapsed, metrics, search_space_proxy=None):
    group_count = evaluate(groups)
    return {
        "instance_type": instance_type,
        "instance_id": instance_id,
        "num_parts": int(metrics.get("num_parts", 0)),
        "method": method,
        "groups": group_count,
        "num_groups": group_count,
        "time": float(elapsed),
        "grouping": str(groups),
        **({} if search_space_proxy is None else search_space_proxy),
        **metrics,
    }


def summarize_result_rows(rows, label):
    prepared = []
    for row in rows:
        new_row = dict(row)
        if "num_groups" not in new_row and "groups" in new_row:
            new_row["num_groups"] = new_row["groups"]
        prepared.append(new_row)
    scored_rows = score_metric_rows_by_group(prepared, group_fields=["instance_type"])
    summary = summarize_results(scored_rows)
    print(f"\n===== {label} SUMMARY =====")
    for row in summary:
        print(row)
    return scored_rows, summary


def run_fixed(env, policy):
    inst = create_fixed_instance(
        num_parts=env.generator.min_num_parts,
        material_types=env.generator.p.material_types,
    )
    inst["num_parts"] = int(env.generator.min_num_parts)
    print_instance(inst, "FIXED INSTANCE USED IN THE EXPERIMENT")
    proxy = compute_search_space_proxy(inst)
    td = inst_to_td(inst, env)

    cpccd = CPCCDSolver()
    ga = GASolver()
    sa = SASolver()

    g1, t1 = cpccd.solve(inst)
    g2, t2 = ga.solve(inst)
    g_sa, t_sa = sa.solve(inst)
    ga.plot_fitness_history("ga_fitness_fixed.png")
    sa.plot_history("sa_fitness_fixed.png")
    m1 = evaluate_groups(g1, inst)
    m2 = evaluate_groups(g2, inst)
    m_sa = evaluate_groups(g_sa, inst)
    m1["num_parts"] = inst["num_parts"]
    m2["num_parts"] = inst["num_parts"]
    m_sa["num_parts"] = inst["num_parts"]

    visualize_grouping_solution(inst, g1, "CPCCD", Path("visualizations") / "fixed" / "cpccd.png", m1)
    visualize_grouping_solution(inst, g2, "GA", Path("visualizations") / "fixed" / "ga.png", m2)
    visualize_grouping_solution(inst, g_sa, "SA", Path("visualizations") / "fixed" / "sa.png", m_sa)
    rows = [
        result_row("fixed", 0, "CPCCD", g1, t1, m1, proxy),
        result_row("fixed", 0, "GA", g2, t2, m2, proxy),
        result_row("fixed", 0, "SA", g_sa, t_sa, m_sa, proxy),
    ]
    g3, t3 = run_nco(env, policy, td)
    m3 = evaluate_groups(g3, inst)
    m3["num_parts"] = inst["num_parts"]
    visualize_grouping_solution(inst, g3, "NCO", Path("visualizations") / "fixed" / "nco.png", m3)
    rows.append(result_row("fixed", 0, "NCO", g3, t3, m3, proxy))
    return inst, rows


def _clone_env_with_num_parts(env, num_parts: int):
    generator_params = vars(env.generator.p).copy()
    generator_params["num_parts"] = int(num_parts)
    generator_params["max_num_parts"] = int(num_parts)
    generator = FPIGenerator(**generator_params)
    return PartConsolidationEnv(
        generator=generator,
        min_group_size_before_sep=env.min_group_size_before_sep,
        allow_fallback=env.allow_fallback,
        device=str(env.device),
    )


def run_generalization(env, policy, num_instances=30, min_parts=4, max_parts=10, seed=123):
    results = []
    cpccd = CPCCDSolver()
    ga = GASolver()
    sa = SASolver()
    plot_dir = Path("ga_fitness_generalization")
    plot_dir.mkdir(parents=True, exist_ok=True)
    sa_plot_dir = Path("sa_fitness_generalization")
    sa_plot_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    for i in range(num_instances):
        sampled_num_parts = rng.randint(min_parts, max_parts)
        env_i = _clone_env_with_num_parts(env, sampled_num_parts)
        td = env_i.reset(batch_size=1)
        inst = td_to_inst(td, env_i.generator.num_parts)
        proxy = compute_search_space_proxy(inst)
        vis_dir = Path("visualizations") / "generalization" / f"instance_{i:03d}"
        save_instance_info_png(inst, f"GENERALIZATION RANDOM INSTANCE {i}", vis_dir / "instance_info.png")

        g1, t1 = cpccd.solve(inst)
        g2, t2 = ga.solve(inst)
        g_sa, t_sa = sa.solve(inst)
        ga.plot_fitness_history(str(plot_dir / f"ga_fitness_instance_{i}.png"))
        sa.plot_history(str(sa_plot_dir / f"sa_fitness_instance_{i}.png"))
        m1 = evaluate_groups(g1, inst)
        m1["num_parts"] = inst["num_parts"]
        m2 = evaluate_groups(g2, inst)
        m2["num_parts"] = inst["num_parts"]
        m_sa = evaluate_groups(g_sa, inst)
        m_sa["num_parts"] = inst["num_parts"]

        visualize_grouping_solution(inst, g1, "CPCCD", vis_dir / "cpccd.png", m1)
        visualize_grouping_solution(inst, g2, "GA", vis_dir / "ga.png", m2)
        visualize_grouping_solution(inst, g_sa, "SA", vis_dir / "sa.png", m_sa)

        results.append(result_row("generalization", i, "CPCCD", g1, t1, m1, proxy))
        results.append(result_row("generalization", i, "GA", g2, t2, m2, proxy))
        results.append(result_row("generalization", i, "SA", g_sa, t_sa, m_sa, proxy))
        g3, t3 = run_nco(env_i, policy, td)
        m3 = evaluate_groups(g3, inst)
        m3["num_parts"] = inst["num_parts"]
        visualize_grouping_solution(inst, g3, "NCO", vis_dir / "nco.png", m3)
        results.append(result_row("generalization", i, "NCO", g3, t3, m3, proxy))

        print(f"[generalization {i + 1}/{num_instances}] done | num_parts={sampled_num_parts}")

    return results


def save_results(results, filename):
    prepared = []
    for row in results:
        new_row = dict(row)
        if "num_groups" not in new_row:
            new_row["num_groups"] = new_row["groups"]
        prepared.append(new_row)
    scored_results = score_metric_rows_by_group(prepared, group_fields=["instance_type"])
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "instance_type",
            "instance_id",
            "num_parts",
            "compat_density",
            "feasible_pair_proxy",
            "feasible_triple_proxy",
            "feasible_quad_proxy",
            "max_pairwise_clique_size",
            "method",
            "grouping",
            "groups",
            "time",
            "feasible",
            "infeasible_solution",
            "infeasible_groups",
            "Q_observed",
            "Q_expected",
            "Q_gamma",
            "total_internal_strength",
            "feasible_pair_count",
            "score",
        ])
        for row in scored_results:
            writer.writerow([
                row["instance_type"],
                row["instance_id"],
                row.get("num_parts", ""),
                row.get("compat_density", ""),
                row.get("feasible_pair_proxy", ""),
                row.get("feasible_triple_proxy", ""),
                row.get("feasible_quad_proxy", ""),
                row.get("max_pairwise_clique_size", ""),
                row["method"],
                row["grouping"],
                row["groups"],
                row["time"],
                row["feasible"],
                row["infeasible_solution"],
                row["infeasible_groups"],
                row.get("Q_observed", ""),
                row.get("Q_expected", ""),
                row.get("Q_gamma", ""),
                row["total_internal_strength"],
                row["feasible_pair_count"],
                row["score"],
            ])

    summary = summarize_results(scored_results)
    with open(filename.replace(".csv", "_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_type",
                "method",
                "groups",
                "time",
                "total_time",
                "feasible",
                "infeasible_solution",
                "infeasible_groups",
                "Q_observed",
                "Q_expected",
                "Q_gamma",
                "total_internal_strength",
                "feasible_pair_count",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    return scored_results, summary


def summarize_results(results):
    agg = defaultdict(
        lambda: {
            "groups": 0.0,
            "time": 0.0,
            "total_time": 0.0,
                "feasible": 0.0,
                "infeasible_solution": 0.0,
                "infeasible_groups": 0.0,
                "Q_observed": 0.0,
                "Q_expected": 0.0,
                "Q_gamma": 0.0,
                "total_internal_strength": 0.0,
                "feasible_pair_count": 0.0,
                "score": 0.0,
            "count": 0,
        }
    )
    for row in results:
        instance_type = row["instance_type"]
        method = row["method"]
        key = (instance_type, method)
        groups = row["groups"]
        elapsed = row["time"]
        agg[key]["groups"] += float(groups)
        agg[key]["time"] += float(elapsed)
        agg[key]["total_time"] += float(elapsed)
        agg[key]["feasible"] += float(row["feasible"])
        agg[key]["infeasible_solution"] += float(row["infeasible_solution"])
        agg[key]["infeasible_groups"] += float(row["infeasible_groups"])
        agg[key]["Q_observed"] += float(row.get("Q_observed", 0.0))
        agg[key]["Q_expected"] += float(row.get("Q_expected", 0.0))
        agg[key]["Q_gamma"] += float(row.get("Q_gamma", 0.0))
        agg[key]["total_internal_strength"] += float(row["total_internal_strength"])
        agg[key]["feasible_pair_count"] += float(row["feasible_pair_count"])
        agg[key]["score"] += float(row["score"])
        agg[key]["count"] += 1

    summary = []
    for (instance_type, method), vals in sorted(agg.items()):
        count = max(vals["count"], 1)
        summary.append(
            {
                "instance_type": instance_type,
                "method": method,
                "groups": vals["groups"] / count,
                "time": vals["time"] / count,
                "total_time": vals["total_time"],
                "feasible": vals["feasible"] / count,
                "infeasible_solution": vals["infeasible_solution"] / count,
                "infeasible_groups": vals["infeasible_groups"] / count,
                "Q_observed": vals["Q_observed"] / count,
                "Q_expected": vals["Q_expected"] / count,
                "Q_gamma": vals["Q_gamma"] / count,
                "total_internal_strength": vals["total_internal_strength"] / count,
                "feasible_pair_count": vals["feasible_pair_count"] / count,
                "score": vals["score"] / count,
            }
        )
    return summary


def plot_results(results, title, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"skip plot '{title}': {exc}")
        return

    grouped_groups = defaultdict(list)
    grouped_time = defaultdict(list)
    grouped_score = defaultdict(list)
    for row in results:
        grouped_groups[row["method"]].append(row["groups"])
        grouped_time[row["method"]].append(row["time"])
        grouped_score[row["method"]].append(row.get("score", 0.0))

    methods = sorted(grouped_groups.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    def _draw_violin(ax, data, ylabel, subplot_title):
        positions = np.arange(1, len(methods) + 1)
        parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor("#9ecae1")
            body.set_edgecolor("#4a4a4a")
            body.set_alpha(0.8)

        means = [float(np.mean(vals)) for vals in data]
        cis = []
        for vals in data:
            arr = np.asarray(vals, dtype=float)
            if len(arr) <= 1:
                cis.append(0.0)
            else:
                cis.append(1.96 * arr.std(ddof=1) / np.sqrt(len(arr)))

        ax.errorbar(
            positions,
            means,
            yerr=cis,
            fmt="o",
            color="#d62728",
            ecolor="#d62728",
            elinewidth=1.4,
            capsize=4,
            label="Mean ± 95% CI",
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(methods)
        ax.set_xlabel("Method")
        ax.set_ylabel(ylabel)
        ax.set_title(subplot_title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()

    _draw_violin(axes[0], [grouped_groups[m] for m in methods], "Groups", "Group Count")
    _draw_violin(axes[1], [grouped_time[m] for m in methods], "Seconds", "Runtime")
    _draw_violin(axes[2], [grouped_score[m] for m in methods], "Score", "Score")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_difference_heatmap(results, title, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        print(f"skip difference heatmap '{title}': {exc}")
        return

    def _representative_rows(rows):
        best = {}
        for row in rows:
            key = (row["instance_type"], row["instance_id"], row["method"])
            current = best.get(key)
            if current is None or float(row.get("score", 0.0)) > float(current.get("score", 0.0)):
                best[key] = row
        return best

    rep = _representative_rows(results)
    instance_keys = sorted({(row["instance_type"], row["instance_id"]) for row in results})
    pair_specs = [
        ("CPCCD", "GA"),
        ("CPCCD", "NCO"),
        ("GA", "NCO"),
    ]

    group_diff = np.full((len(instance_keys), len(pair_specs)), np.nan, dtype=float)
    score_diff = np.full((len(instance_keys), len(pair_specs)), np.nan, dtype=float)

    for r, inst_key in enumerate(instance_keys):
        for c, (m1, m2) in enumerate(pair_specs):
            row1 = rep.get((inst_key[0], inst_key[1], m1))
            row2 = rep.get((inst_key[0], inst_key[1], m2))
            if row1 is None or row2 is None:
                continue
            group_diff[r, c] = float(str(row1["grouping"]) != str(row2["grouping"]))
            score_diff[r, c] = float(row1.get("score", 0.0)) - float(row2.get("score", 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, len(instance_keys) * 0.18)))

    im0 = axes[0].imshow(group_diff, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0].set_title("Grouping Difference (1=different)")
    axes[0].set_xticks(range(len(pair_specs)))
    axes[0].set_xticklabels([f"{a}\nvs\n{b}" for a, b in pair_specs])
    axes[0].set_yticks(range(len(instance_keys)))
    axes[0].set_yticklabels([f"{it}:{idx}" for it, idx in instance_keys], fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax = np.nanmax(np.abs(score_diff)) if np.isfinite(score_diff).any() else 1.0
    vmax = max(vmax, 1e-6)
    im1 = axes[1].imshow(score_diff, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Score Difference (left - right)")
    axes[1].set_xticks(range(len(pair_specs)))
    axes[1].set_xticklabels([f"{a}\n-\n{b}" for a, b in pair_specs])
    axes[1].set_yticks(range(len(instance_keys)))
    axes[1].set_yticklabels([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def load_policy(gen, device, ckpt_path):
    policy = PCPolicy(
        node_feat_dim=gen.node_feat_dim,
        edge_feat_dim=gen.edge_feat_dim,
    ).to(device)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = "cpu"

    generator_params = dict(
        num_parts=4,
        max_num_parts=20,
        material_types=2,
        p_relative_motion=0.03,
        p_extra_edge=0.80,
        L_low=20.0,
        L_high=120.0,
        W_low=10.0,
        W_high=55.0,
        H_low=2.0,
        H_high=24.0,
        build_limit_L=1000.0,
        build_limit_W=1000.0,
        build_limit_H=500.0,
        p_maint_H=0.02,
        p_standard=0.01,
    )

    gen = FPIGenerator(**generator_params)
    env = PartConsolidationEnv(generator=gen, device=device)
    ckpt = Path("checkpoints") / "best_model.pt"
    policy = load_policy(gen, device, ckpt)

    print("\n===== FIXED INSTANCE EXPERIMENT =====")
    fixed_inst, fixed_results = run_fixed(env, policy)
    df_fixed, summary_fixed = save_results(fixed_results, "fixed_results.csv")
    for row in summary_fixed:
        print(row)

    print("\n===== GENERALIZATION EXPERIMENT =====")
    gen_results = run_generalization(
        env,
        policy,
        num_instances=100,
        min_parts=generator_params["num_parts"],
        max_parts=generator_params["max_num_parts"],
    )
    df_gen, summary_gen = save_results(gen_results, "generalization_results.csv")
    for row in summary_gen:
        print(row)

    all_results = fixed_results + gen_results
    df_all, summary_all = save_results(all_results, "all_results.csv")
    summarize_result_rows(all_results, "ALL")

    plot_results(df_fixed, "Fixed_Instance_Result", "Fixed_Instance_Result.png")
    plot_results(df_gen, "Generalization_Result", "Generalization_Result.png")
    plot_results(df_all, "All_Results", "All_Results.png")
    plot_difference_heatmap(df_gen, "Generalization_Differences", "Generalization_Differences.png")
    plot_difference_heatmap(df_all, "All_Results_Differences", "All_Results_Differences.png")

    return {
        "fixed_instance": fixed_inst,
        "fixed_results": df_fixed,
        "fixed_summary": summary_fixed,
        "generalization_results": df_gen,
        "generalization_summary": summary_gen,
        "all_summary": summary_all,
    }


if __name__ == "__main__":
    main()
