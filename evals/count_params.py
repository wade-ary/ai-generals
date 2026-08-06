import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import equinox as eqx
from networks.transformer import HistoryTransformer

def count_params(depth, embed_dim, n_head=4, ff_factor=2, patch_size=4, pad_to=24,
                 num_bins=128, value_loss="ce"):
    key = jax.random.PRNGKey(0)
    net = HistoryTransformer(
        grid_size=pad_to,
        pad_to=pad_to,
        history_size=7,
        patch_size=patch_size,
        depth=depth,
        embed_dim=embed_dim,
        n_head=n_head,
        ff_factor=ff_factor,
        use_bf16=True,
        value_loss=value_loss,
        num_bins=num_bins,
        v_min=-1.0,
        v_max=1.0,
        key=key,
    )
    params = jax.tree.leaves(eqx.filter(net, eqx.is_array))
    total = sum(p.size for p in params)
    return total

# If called with args, just count for those
if len(sys.argv) > 1:
    depth = int(sys.argv[1])
    embed_dim = int(sys.argv[2])
    n_head = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ff_factor = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    patch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    pad_to = int(sys.argv[6]) if len(sys.argv) > 6 else 24
    total = count_params(depth, embed_dim, n_head, ff_factor, patch_size, pad_to)
    print(f"depth={depth} embed_dim={embed_dim} n_head={n_head} ff={ff_factor} ps={patch_size} pad={pad_to}: {total:,} params ({total/1e6:.2f}M)")
    sys.exit(0)

# Sweep
print("=" * 80)
print("Parameter count sweep for HistoryTransformer (pad_to=24, patch_size=4, ff=2, CE)")
print("=" * 80)
print(f"{'depth':>6} {'embed':>6} {'heads':>6} {'params':>12} {'M':>8}")
print("-" * 44)

configs = []
for depth in [4, 6, 8, 10, 12]:
    for embed_dim in [192, 256, 320, 384, 448, 512]:
        n_head = 4 if embed_dim <= 320 else (8 if embed_dim <= 512 else 8)
        total = count_params(depth, embed_dim, n_head)
        m = total / 1e6
        configs.append((depth, embed_dim, n_head, total, m))
        marker = ""
        if 4.5 <= m <= 5.5: marker = " <-- ~5M"
        elif 7.0 <= m <= 8.0: marker = " <-- ~7.5M"
        elif 9.5 <= m <= 10.5: marker = " <-- ~10M"
        elif 12.0 <= m <= 13.0: marker = " <-- ~12.5M"
        elif 14.5 <= m <= 15.5: marker = " <-- ~15M"
        print(f"{depth:>6} {embed_dim:>6} {n_head:>6} {total:>12,} {m:>8.2f}{marker}")

print()
print("Current runs for reference:")
for name, d, e, nh, pt in [("pop_S", 8, 320, 4, 16), ("pop", 10, 384, 4, 24)]:
    total = count_params(d, e, nh, pad_to=pt)
    print(f"  {name}: depth={d} embed={e} pad={pt}: {total:,} ({total/1e6:.2f}M)")
