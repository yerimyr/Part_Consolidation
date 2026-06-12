import numpy as np


TOPOLOGY_NAMES = {
    "chain",
    "star",
    "tree",
    "two_module_bridge",
}


def _add_edge(adj, i, j):
    if i == j:
        return
    adj[i, j] = True
    adj[j, i] = True


def _connect_sequence(adj, nodes):
    for idx in range(len(nodes) - 1):
        _add_edge(adj, nodes[idx], nodes[idx + 1])


def _build_chain_adjacency(num_parts: int) -> np.ndarray:
    adj = np.zeros((num_parts, num_parts), dtype=bool)
    _connect_sequence(adj, list(range(num_parts)))
    return adj


def _build_star_adjacency(num_parts: int) -> np.ndarray:
    adj = np.zeros((num_parts, num_parts), dtype=bool)
    center = 0
    for node in range(1, num_parts):
        _add_edge(adj, center, node)
    return adj


def _build_tree_adjacency(num_parts: int) -> np.ndarray:
    adj = np.zeros((num_parts, num_parts), dtype=bool)
    if num_parts <= 1:
        return adj

    # Deterministic branching tree:
    # parent(i) = floor((i - 1) / 2)
    for node in range(1, num_parts):
        parent = (node - 1) // 2
        _add_edge(adj, parent, node)
    return adj


def _build_two_module_bridge_adjacency(num_parts: int) -> np.ndarray:
    adj = np.zeros((num_parts, num_parts), dtype=bool)
    split = max(1, num_parts // 2)
    if split >= num_parts:
        split = num_parts - 1

    left = list(range(split))
    right = list(range(split, num_parts))

    _connect_sequence(adj, left)
    _connect_sequence(adj, right)

    for group in (left, right):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                _add_edge(adj, group[i], group[j])

    _add_edge(adj, left[-1], right[0])
    return adj


def _build_topology_adjacency(num_parts: int, topology: str) -> np.ndarray:
    if topology == "chain":
        return _build_chain_adjacency(num_parts)
    if topology == "star":
        return _build_star_adjacency(num_parts)
    if topology == "tree":
        return _build_tree_adjacency(num_parts)
    if topology == "two_module_bridge":
        return _build_two_module_bridge_adjacency(num_parts)
    raise ValueError(f"Unknown fixed topology: {topology}")


def _sample_embeddedness_weights(
    assembly_adj: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n = int(assembly_adj.shape[0])
    weights = np.zeros((n, n), dtype=float)
    if n <= 1:
        return weights

    denom = float(max(n - 1, 1))  # Emb_max + 1, where Emb_max = n - 2.
    adj_float = assembly_adj.astype(float)
    common_neighbors = adj_float @ adj_float

    for i in range(n):
        for j in range(i + 1, n):
            if not assembly_adj[i, j]:
                continue
            embeddedness = common_neighbors[i, j]
            sampled = rng.uniform(
                embeddedness / denom,
                (embeddedness + 1.0) / denom,
            )
            weights[i, j] = sampled
            weights[j, i] = sampled
    return weights


def create_fixed_instance(
    num_parts=4,
    build_limit=(260.0, 120.0, 80.0),
    seed=42,
    topology="chain",
    material_types=3,
):
    if topology not in TOPOLOGY_NAMES:
        raise ValueError(f"topology must be one of {sorted(TOPOLOGY_NAMES)}")

    rng = np.random.default_rng(seed)

    # Match generator.py semantics:
    # - per-node size is a 3D vector [L, W, H]
    # - build_limit is also a 3D vector
    # - pairwise stack_size keeps the same [L, W, H] shape
    build_limit = np.asarray(build_limit, dtype=float)
    if build_limit.shape != (3,):
        raise ValueError("build_limit must be a length-3 iterable: (L_limit, W_limit, H_limit)")

    if material_types < 1:
        raise ValueError("material_types must be >= 1")

    material = np.array([(i // 2) % material_types for i in range(num_parts)], dtype=int)
    maintfreq = np.array([(i % 3) == 0 for i in range(num_parts)], dtype=int)
    isstandard = np.zeros(num_parts, dtype=int)
    material_available = np.ones(num_parts, dtype=bool)

    # Deterministic-but-nontrivial node sizes.
    L = rng.uniform(5.0, 160.0, size=num_parts)
    W = rng.uniform(5.0, 70.0, size=num_parts)
    H = rng.uniform(5.0, 30.0, size=num_parts)
    size = np.stack([L, W, H], axis=-1).astype(float)

    assembly_adj = _build_topology_adjacency(num_parts, topology)
    eye = np.eye(num_parts, dtype=bool)

    mat_var = (material[:, None] != material[None, :]) & assembly_adj
    maint_diff = (maintfreq[:, None] != maintfreq[None, :]) & assembly_adj

    rel_motion = np.zeros((num_parts, num_parts), dtype=bool)
    if topology == "chain" and num_parts >= 4:
        rel_motion[2, 3] = True
        rel_motion[3, 2] = True
    elif topology == "star" and num_parts >= 3:
        rel_motion[0, 1] = True
        rel_motion[1, 0] = True
    elif topology == "tree" and num_parts >= 5:
        rel_motion[1, 3] = True
        rel_motion[3, 1] = True
    elif topology == "two_module_bridge" and num_parts >= 4:
        split = max(1, num_parts // 2)
        bridge_left = split - 1
        bridge_right = split
        rel_motion[bridge_left, bridge_right] = True
        rel_motion[bridge_right, bridge_left] = True
    rel_motion &= assembly_adj

    stack_size_full = size[:, None, :] + size[None, :, :]
    stack_size = stack_size_full * assembly_adj[:, :, None].astype(float)
    stack_ok = (stack_size_full <= build_limit.reshape(1, 1, 3)).all(axis=-1)

    standard_pair_block = isstandard[:, None].astype(bool) | isstandard[None, :].astype(bool)
    compat = (
        assembly_adj
        & (~mat_var)
        & (~maint_diff)
        & (~rel_motion)
        & stack_ok
        & (~standard_pair_block)
    )
    compat |= eye

    W_matrix = _sample_embeddedness_weights(assembly_adj, rng)

    return {
        "num_parts": num_parts,
        "topology": topology,
        "material": material,
        "size": size,
        "maintfreq": maintfreq,
        "isstandard": isstandard,
        "material_available": material_available,
        "build_limit": build_limit,
        "assembly_adj": assembly_adj,
        "mat_var": mat_var,
        "stack_size": stack_size,
        "maint_diff": maint_diff,
        "rel_motion": rel_motion,
        "compat": compat,
        "W": W_matrix,
        "relation_valid": assembly_adj.copy(),
        "relation_consistent": True,
    }
