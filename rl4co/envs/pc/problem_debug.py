from generator import FPIGenerator


def print_instance(td):
    td = td.clone()

    N = td["material"].shape[1] - 1

    print("=" * 60)
    print("🔹 BASIC INFO")
    print("=" * 60)
    print(f"num_parts: {N}")
    print("device: cpu")
    print()

    # ============================================================
    # 🔹 NODE FEATURES
    # ============================================================
    print("=" * 60)
    print("🔹 NODE FEATURES")
    print("=" * 60)

    for i in range(N + 1):
        mat = td["material"][0, i].item()
        size = td["size"][0, i]
        maint = td["maintfreq"][0, i].item()
        std = td["isstandard"][0, i].item()

        print(
            f"[Part {i}] "
            f"material={mat}, "
            f"size=({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}), "
            f"maint={maint:.3f}, "
            f"is_standard={std}"
        )

    print()

    # ============================================================
    # 🔥 EDGE FEATURES (ALL PAIRS 출력)
    # ============================================================
    print("=" * 60)
    print("🔹 EDGE FEATURES (ALL PAIRS)")
    print("=" * 60)

    W = td["W"][0]
    adj = td["assembly_adj"][0]

    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):

            print(
                f"[{i}-{j}] "
                f"adj={int(adj[i,j].item())}, "
                f"W={W[i,j]:.3f}, "
                f"rel_motion={td['rel_motion'][0,i,j]:.3f}, "
                f"compat={int(td['compat'][0,i,j].item())}"
            )

    print()

    # ============================================================
    # 🔹 ADJ MATRIX
    # ============================================================
    print("=" * 60)
    print("🔹 POSITION / ADJACENCY MATRIX")
    print("=" * 60)

    for i in range(N + 1):
        row = " ".join(str(int(x.item())) for x in adj[i])
        print(f"{i}: {row}")

    print()

    # ============================================================
    # 🔹 VALIDITY
    # ============================================================
    print("=" * 60)
    print("🔹 VALIDITY CHECK (CRITICAL)")
    print("=" * 60)

    invalid = (
        (td["rel_motion"] == 1) &
        (td["compat"] == 1)
    ).sum()

    if invalid == 0:
        print("✅ VALID: No invalid relations")
    else:
        print("❌ INVALID")

    print()

    # ============================================================
    # 🔥 GLOBAL STATS (두 개로 분리)
    # ============================================================
    print("=" * 60)
    print("🔹 GLOBAL STATS")
    print("=" * 60)

    valid_W = W[adj == 1]
    all_W = W[1:, 1:]

    print("▶ Connected edges only:")
    if valid_W.numel() > 0:
        print(f"W mean: {valid_W.mean().item():.4f}")
        print(f"W std : {valid_W.std().item():.4f}")
    else:
        print("No connected edges")

    print()

    print("▶ All pairs:")
    print(f"W mean: {all_W.mean().item():.4f}")
    print(f"W std : {all_W.std().item():.4f}")

    print("=" * 60)


def debug_one_instance():
    gen = FPIGenerator(num_parts=20)
    td = gen(batch_size=1)
    print_instance(td)


if __name__ == "__main__":
    debug_one_instance()