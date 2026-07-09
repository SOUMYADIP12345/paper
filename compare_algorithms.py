"""
================================================================================
  MODERN BASELINE COMPARISON for the Solar-Aware GA Multi-Sink WSN Protocol
================================================================================

  WHY THIS FILE EXISTS
  --------------------
  The paper (see Solar_Aware_GA_Multi_Sink_WSN_Paper.md, Section 8) admits that
  comparing SGA-MS *only* against LEACH is a weak evaluation: LEACH (2000) picks
  cluster heads AT RANDOM and is neither energy-aware nor a metaheuristic, so
  beating it proves very little. This module adds stronger, MODERN baselines so
  the GA can be judged against the algorithms actually used in 2023-2025 WSN
  clustering papers.

  THE THREE ADDED BASELINES
  -------------------------
    1. HEED   - energy-aware, deterministic clustering (Younis & Fahmy, 2004).
                The classic "fair" non-random baseline: CHs are the high-energy,
                well-separated nodes. No metaheuristic.
    2. PSO-MS - Particle Swarm Optimisation selects the CH set. PSO is the most
                common metaheuristic competitor to GA in the WSN literature
                (LEACH-PSO / PSO-C / DPFCP families).
    3. GWO-MS - Grey Wolf Optimiser selects the CH set. GWO (and EECHIGWO) is the
                single most-cited "modern" metaheuristic for CH selection in
                2023-2025 papers.

  THE KEY FAIRNESS DECISION
  -------------------------
  For PSO-MS and GWO-MS we swap ONLY THE OPTIMISER. The energy model, the solar
  model, the per-node panel efficiency, the EXACT SAME multi-objective fitness
  function (via solar_ga_wsn._evaluate_population), the path decision, and the
  full solar-aware multi-sink tier are all identical to SGA-MS. This isolates a
  single scientific question:

        "Given my fitness function and my 3-tier architecture, is the GENETIC
         ALGORITHM actually the best optimiser, or would PSO / GWO do better?"

  That is a far more defensible contribution than "GA beats random LEACH".

  HEED and LEACH use their own native CH-selection logic (energy-aware and
  random, respectively) but STILL run through the same multi-sink tier and
  energy/solar models, so every protocol is evaluated under identical physics.

  HOW TO RUN  (needs numpy + matplotlib; run in Colab or locally, NOT required
  to be interactive):

        python compare_algorithms.py                 # default config, all protocols
        SOLAR_GA_PROTOCOLS="GA,PSO,GWO,HEED,LEACH" python compare_algorithms.py

  Outputs:
        comparison_all_algorithms.png   - overlaid lifetime/energy/deaths/balance
        comparison_summary_bars.png     - bar chart of key metrics
        printed summary table
================================================================================
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Callable

import numpy as np
import matplotlib.pyplot as plt

# Reuse EVERYTHING from the main protocol so the comparison is provably fair:
# same Node, same energy/solar physics, same fitness function, same multi-sink
# tier. We only add NEW cluster-head selectors (PSO, GWO, HEED).
import solar_ga_wsn as base
from solar_ga_wsn import (
    Node, World, Chromosome,
    default_config,
    get_num_chs, get_num_ms,
    solar_rate_for_round,
    _evaluate_population,
    run_ga_ch_election,
    decide_ch_paths, elect_ms_chs, _solar_aware_score,
    vectorized_assign,
    make_stats, _record_stats,
    create_nodes,
    simulate_round_leach,
    MS_REELECT_THR, COMM_RANGE_PCT,
)


# ==============================================================================
# SECTION 1 - SHARED DECODE + SCORING HELPERS (for PSO / GWO)
# ==============================================================================
#
# CH selection is a COMBINATORIAL problem: choose K distinct node IDs. PSO and
# GWO are continuous optimisers, so each "agent" is a real vector of length K
# whose values live in [0, S) (S = number of alive nodes). We decode a vector
# into K DISTINCT node IDs, then score it with the project's own fitness
# function so the objective is byte-for-byte identical to the GA's.

def _decode_positions(positions: np.ndarray,
                      alive_idx: np.ndarray,
                      num_chs: int) -> List[List[int]]:
    """
    Decode a (P, K) matrix of continuous positions into P gene-lists, each a
    list of K DISTINCT alive-node IDs. Duplicate indices are repaired by
    linear probing to the next unused slot (a standard permutation-repair).
    """
    S = int(alive_idx.shape[0])
    genes_list: List[List[int]] = []
    for p in range(positions.shape[0]):
        used = set()
        genes: List[int] = []
        for k in range(num_chs):
            idx = int(round(float(positions[p, k]))) % S
            start = idx
            while idx in used:
                idx = (idx + 1) % S
                if idx == start:      # every slot used (K == S) - accept dup
                    break
            used.add(idx)
            genes.append(int(alive_idx[idx]))
        genes_list.append(genes)
    return genes_list


def _score_positions(positions: np.ndarray,
                     world: World,
                     cfg: dict,
                     solar_now: float,
                     num_chs: int):
    """
    Decode every agent's position to a CH set and evaluate ALL of them with the
    project's shared batched fitness function. Returns (fitness_array, genes).
    Fresh Chromosome objects (fitness = -1) force a real evaluation each call.
    """
    genes_list = _decode_positions(positions, world.alive_idx, num_chs)
    chromos = [Chromosome(g) for g in genes_list]
    _evaluate_population(chromos, world, cfg, solar_now, num_chs)
    fits = np.fromiter((c.fitness for c in chromos), dtype=np.float64,
                       count=len(chromos))
    return fits, genes_list


# ==============================================================================
# SECTION 2 - PSO CLUSTER-HEAD SELECTOR  (drop-in replacement for the GA)
# ==============================================================================

def run_pso_ch_election(nodes: List[Node],
                        world: World,
                        cfg: dict,
                        round_num: int,
                        num_chs: int) -> Optional[Chromosome]:
    """
    Particle Swarm Optimisation for CH selection.

    Same population size (GA_POP -> swarm size) and same iteration budget
    (GA_GEN) as the GA, so compute budgets match. Same fitness function.
    Standard global-best PSO with inertia w and cognitive/social coefficients.
    """
    S = int(world.alive_idx.size)
    if S < num_chs or num_chs <= 0:
        return None

    solar_now = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])
    P     = int(cfg["GA_POP"])
    iters = int(cfg["GA_GEN"])
    hi    = S - 1e-6

    X = np.random.uniform(0.0, hi, size=(P, num_chs))
    V = np.random.uniform(-1.0, 1.0, size=(P, num_chs)) * (S * 0.1)

    fit, genes = _score_positions(X, world, cfg, solar_now, num_chs)
    pbest      = X.copy()
    pbest_fit  = fit.copy()
    pbest_gen  = list(genes)

    g0        = int(fit.argmax())
    gbest     = X[g0].copy()
    gbest_fit = float(fit[g0])
    gbest_gen = list(genes[g0])

    w, c1, c2 = 0.7, 1.5, 1.5           # standard PSO coefficients
    vmax      = S * 0.5
    patience  = max(5, iters // 8)
    stale     = 0

    for _ in range(iters):
        r1 = np.random.random((P, num_chs))
        r2 = np.random.random((P, num_chs))
        V  = (w * V
              + c1 * r1 * (pbest - X)
              + c2 * r2 * (gbest[None, :] - X))
        V  = np.clip(V, -vmax, vmax)
        X  = np.clip(X + V, 0.0, hi)

        fit, genes = _score_positions(X, world, cfg, solar_now, num_chs)

        improved = fit > pbest_fit
        if improved.any():
            pbest[improved]     = X[improved]
            pbest_fit[improved] = fit[improved]
            for i in np.nonzero(improved)[0]:
                pbest_gen[int(i)] = genes[int(i)]

        gi = int(fit.argmax())
        if fit[gi] > gbest_fit + 1e-9:
            gbest     = X[gi].copy()
            gbest_fit = float(fit[gi])
            gbest_gen = list(genes[gi])
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    best = Chromosome(gbest_gen)
    best.fitness = gbest_fit
    return best


# ==============================================================================
# SECTION 3 - GWO CLUSTER-HEAD SELECTOR  (drop-in replacement for the GA)
# ==============================================================================

def run_gwo_ch_election(nodes: List[Node],
                        world: World,
                        cfg: dict,
                        round_num: int,
                        num_chs: int) -> Optional[Chromosome]:
    """
    Grey Wolf Optimiser for CH selection - the most-cited "modern" metaheuristic
    for this task in 2023-2025 WSN papers.

    Same swarm size (GA_POP) and iteration budget (GA_GEN) as the GA. The three
    best wolves (alpha, beta, delta) steer the pack; the exploration coefficient
    `a` decays linearly from 2 to 0. Same fitness function as the GA.
    """
    S = int(world.alive_idx.size)
    if S < num_chs or num_chs <= 0:
        return None

    solar_now = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])
    P     = int(cfg["GA_POP"])
    iters = int(cfg["GA_GEN"])
    hi    = S - 1e-6

    X = np.random.uniform(0.0, hi, size=(P, num_chs))

    best_gen: Optional[List[int]] = None
    best_fit = -1.0

    for it in range(iters):
        fit, genes = _score_positions(X, world, cfg, solar_now, num_chs)
        order = np.argsort(fit)[::-1]

        a_idx = int(order[0])
        b_idx = int(order[1]) if P > 1 else a_idx
        d_idx = int(order[2]) if P > 2 else b_idx
        alpha = X[a_idx].copy()
        beta  = X[b_idx].copy()
        delta = X[d_idx].copy()

        if fit[a_idx] > best_fit:
            best_fit = float(fit[a_idx])
            best_gen = list(genes[a_idx])

        a = 2.0 - 2.0 * it / max(iters - 1, 1)     # 2 -> 0

        def _pull(leader: np.ndarray) -> np.ndarray:
            A = 2.0 * a * np.random.random((P, num_chs)) - a
            C = 2.0 * np.random.random((P, num_chs))
            D = np.abs(C * leader[None, :] - X)
            return leader[None, :] - A * D

        X = (_pull(alpha) + _pull(beta) + _pull(delta)) / 3.0
        X = np.clip(X, 0.0, hi)

    # Safety: if the pack never improved on the initial draw (degenerate),
    # fall back to a fresh evaluation of the final positions.
    if best_gen is None:
        fit, genes = _score_positions(X, world, cfg, solar_now, num_chs)
        gi = int(fit.argmax())
        best_gen = list(genes[gi])
        best_fit = float(fit[gi])

    best = Chromosome(best_gen)
    best.fitness = best_fit
    return best


# ==============================================================================
# SECTION 4 - HEED CLUSTER-HEAD SELECTOR  (energy-aware, non-metaheuristic)
# ==============================================================================

def run_heed_ch_election(nodes: List[Node],
                         world: World,
                         cfg: dict,
                         round_num: int,
                         num_chs: int) -> Optional[Chromosome]:
    """
    HEED-style cluster-head selection (Younis & Fahmy, 2004), simplified for
    this simulator.

    HEED's essence, and what distinguishes it from random LEACH, is:
      (a) a node's chance of becoming a CH grows with its RESIDUAL ENERGY, and
      (b) CHs are kept SPATIALLY SEPARATED (a tentative CH yields to a
          higher-energy CH already claiming its neighbourhood), which avoids the
          clumping that wastes coverage.

    We implement this deterministically: walk nodes from highest to lowest
    residual energy, accept a node as CH only if it is at least `min_sep` away
    from every already-accepted CH, and back-fill by pure energy order if the
    separation constraint leaves us short of num_chs. This gives an energy-aware,
    well-spread head set - a genuinely fair (non-random) baseline - while
    remaining a clear, documented simplification of the full iterated-cost HEED.
    """
    S = int(world.alive_idx.size)
    if S < num_chs or num_chs <= 0:
        return None

    ids = world.alive_idx
    e   = world.alive_e
    xy  = world.alive_xy

    # Separation ~ half the intra-cluster communication radius.
    min_sep = cfg["FIELD"] * COMM_RANGE_PCT * 0.5

    order = np.argsort(e)[::-1]         # highest residual energy first
    chosen: List[int] = []
    chosen_xy: List[np.ndarray] = []

    for oi in order:
        if len(chosen) >= num_chs:
            break
        p = xy[oi]
        if chosen_xy:
            d = np.sqrt(((np.asarray(chosen_xy) - p) ** 2).sum(axis=1)).min()
            if d < min_sep:            # too close to a higher-energy CH -> skip
                continue
        chosen.append(int(ids[oi]))
        chosen_xy.append(p)

    if len(chosen) < num_chs:          # separation left us short - back-fill
        cset = set(chosen)
        for oi in order:
            nid = int(ids[oi])
            if nid not in cset:
                chosen.append(nid)
                cset.add(nid)
                if len(chosen) >= num_chs:
                    break

    best = Chromosome(chosen)
    best.fitness = 0.0                 # HEED has no GA fitness; sentinel only
    return best


# ==============================================================================
# SECTION 5 - GENERIC PROTOCOL ROUND
# ==============================================================================
#
# This mirrors solar_ga_wsn.simulate_round_ga EXACTLY, except the cluster-head
# set is produced by a pluggable `selector`. Everything downstream - path
# decision, multi-sink election, sensor assignment, aggregation, mid-round
# re-election, stats - is identical, so the ONLY variable across protocols is
# how the CHs are chosen.

Selector = Callable[[List[Node], World, dict, int, int], Optional[Chromosome]]

_SELECTORS: Dict[str, Selector] = {
    "GA":   run_ga_ch_election,
    "PSO":  run_pso_ch_election,
    "GWO":  run_gwo_ch_election,
    "HEED": run_heed_ch_election,
}


def simulate_round_generic(nodes: List[Node],
                           world: World,
                           round_num: int,
                           cfg: dict,
                           stats: dict,
                           selector: Selector) -> bool:
    """One protocol round with a pluggable CH selector (multi-sink tier kept)."""
    solar_now = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])

    # Step 1 - solar harvest
    if solar_now > 0:
        for n in nodes:
            if n.alive:
                n.harvest_solar(solar_now)

    # Step 2 - reset roles
    for n in nodes:
        if n.alive:
            n.reset_role()

    # Step 3 - refresh world, dynamic CH count
    world.refresh()
    alive_count = int(world.alive_idx.size)
    num_chs = get_num_chs(alive_count, cfg)

    if alive_count < num_chs + 1:
        _record_stats(nodes, stats)
        stats["ch_counts"].append(0)
        stats["ms_counts"].append(0)
        return False

    # Step 4 - elect CHs with the chosen optimiser
    solution = selector(nodes, world, cfg, round_num, num_chs)
    if solution is None:
        _record_stats(nodes, stats)
        stats["ch_counts"].append(0)
        stats["ms_counts"].append(0)
        return False

    ch_nodes = [nodes[g] for g in solution.genes
                if g < len(nodes) and nodes[g].alive]
    for ch in ch_nodes:
        ch.role = "CH"

    # Step 5 - path decision (before MS-CH election)
    direct_chs, relay_chs = decide_ch_paths(ch_nodes, cfg)

    # Step 6 - MS-CH election (only if needed)
    num_ms = get_num_ms(len(relay_chs), cfg)
    ms_chs = elect_ms_chs(relay_chs, cfg, solar_now, num_ms) if num_ms else []

    # Step 7 - sensors -> nearest CH (vectorized)
    if ch_nodes:
        ch_xy     = np.array([(c.x, c.y) for c in ch_nodes], dtype=np.float64)
        ch_id_arr = np.array([c.id for c in ch_nodes], dtype=np.int64)
        ch_id_set = {c.id for c in ch_nodes}
        sensor_mask = np.array([nid not in ch_id_set
                                for nid in world.alive_idx], dtype=bool)
        sensor_xy  = world.alive_xy[sensor_mask]
        sensor_ids = world.alive_idx[sensor_mask]
        if sensor_xy.shape[0] > 0:
            assign_idx = vectorized_assign(sensor_xy, ch_xy)
            for sid, ai in zip(sensor_ids.tolist(), assign_idx.tolist()):
                nodes[sid].assigned_ch = int(ch_id_arr[ai])

    # Step 8 - sensors transmit
    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch is not None:
            ch = nodes[n.assigned_ch]
            if ch.alive:
                n.transmit(ch.x, ch.y)
                ch.receive()

    # Step 9 - CHs aggregate + forward
    member_count: Dict[int, int] = {c.id: 0 for c in ch_nodes}
    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch in member_count:
            member_count[n.assigned_ch] += 1

    ms_inbound: Dict[int, int] = {m.id: 0 for m in ms_chs}

    for ch in ch_nodes:
        if not ch.alive or ch.role != "CH":
            continue
        ch.aggregate(member_count[ch.id])
        if not ch.alive:
            continue
        if ch.goes_direct:
            ch.transmit(cfg["BS_X"], cfg["BS_Y"])
        elif ch.assigned_ms is not None and ch.assigned_ms in ms_inbound:
            ms = nodes[ch.assigned_ms]
            if ms.alive:
                ch.transmit(ms.x, ms.y)
                ms.receive()
                ms_inbound[ms.id] += 1

    # Step 10 - re-elect critically depleted MS-CHs
    for ms in list(ms_chs):
        if ms.energy < MS_REELECT_THR * cfg["E_INITIAL"]:
            cluster = [c for c in relay_chs
                       if c.alive and c.id != ms.id and c.assigned_ms == ms.id]
            if cluster:
                ms.role = "CH"
                new_ms = max(cluster, key=lambda c:
                             _solar_aware_score(c, cluster, cfg, solar_now))
                new_ms.role = "MS-CH"
                for r in cluster:
                    if r.id != new_ms.id:
                        r.assigned_ms = new_ms.id
                ms_inbound[new_ms.id] = ms_inbound.pop(ms.id, 0)
                ms_chs[ms_chs.index(ms)] = new_ms
                stats["reelections"] += 1

    # Step 11 - MS-CHs transmit to BS
    for ms in ms_chs:
        if ms.alive and ms.role == "MS-CH":
            own_members = member_count.get(ms.id, 0)
            inbound     = ms_inbound.get(ms.id, 0)
            ms.aggregate(own_members + inbound)
            if ms.alive:
                ms.transmit(cfg["BS_X"], cfg["BS_Y"])
                stats["packets_to_bs"] += 1

    for ch in direct_chs:
        if ch.alive:
            stats["packets_to_bs"] += 1

    # Step 12 - record
    _record_stats(nodes, stats)
    stats["ch_counts"].append(len(ch_nodes))
    stats["ms_counts"].append(len(ms_chs))
    return any(n.alive for n in nodes)


# ==============================================================================
# SECTION 6 - PER-PROTOCOL SIMULATION RUNNER
# ==============================================================================

def run_protocol(cfg: dict, protocol: str):
    """
    Run one protocol end-to-end. Nodes are re-created with the SAME fixed seed
    for every protocol (create_nodes seeds 42 internally), so all protocols see
    an identical field, identical starting batteries, and identical per-node
    solar efficiencies. Returns (stats, first_dead, network_dead).
    """
    protocol = protocol.upper()
    nodes = create_nodes(cfg)          # reseeds random + numpy to 42
    world = World(nodes=nodes)
    stats = make_stats()

    selector = _SELECTORS.get(protocol)   # None for LEACH (native round)

    print(f"\n{'=' * 58}")
    print(f"  Protocol  : {protocol}")
    print(f"  Nodes     : {cfg['NUM_NODES']}   Field : "
          f"{cfg['FIELD']}x{cfg['FIELD']}m   Rounds: {cfg['NUM_ROUNDS']}")
    print(f"{'=' * 58}")

    first_dead   = None
    network_dead = cfg["NUM_ROUNDS"]

    for r in range(cfg["NUM_ROUNDS"]):
        if protocol == "LEACH":
            alive = simulate_round_leach(nodes, world, r, cfg, stats)
        else:
            alive = simulate_round_generic(nodes, world, r, cfg, stats, selector)

        dead_now = cfg["NUM_NODES"] - sum(1 for n in nodes if n.alive)
        if first_dead is None and dead_now >= 1:
            first_dead = r
            print(f"  *  First node died : round {r}")
        if not alive:
            network_dead = r
            print(f"  X  Network dead    : round {r}")
            break

    if network_dead == cfg["NUM_ROUNDS"]:
        print(f"  v  Survived all {cfg['NUM_ROUNDS']} rounds")
    print(f"  Packets to BS      : {stats['packets_to_bs']}")
    return stats, first_dead, network_dead


# ==============================================================================
# SECTION 7 - MULTI-PROTOCOL PLOTS + SUMMARY
# ==============================================================================

_COLORS = {
    "GA":    "#1A5FAD",
    "PSO":   "#8E44AD",
    "GWO":   "#16A085",
    "HEED":  "#E67E22",
    "LEACH": "#C0392B",
}
_STYLES = {
    "GA": "-", "PSO": "-", "GWO": "-", "HEED": "--", "LEACH": ":",
}


def plot_multi_results(results: Dict[str, dict], cfg: dict) -> None:
    """Overlay all protocols on four lifetime/energy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "SGA-MS (GA) vs Modern Baselines (PSO, GWO, HEED) vs LEACH\n"
        f"(Nodes={cfg['NUM_NODES']}, Field={cfg['FIELD']}m, "
        f"Rounds={cfg['NUM_ROUNDS']}, CH%={cfg['CH_PERCENT'] * 100:.0f}% dynamic)",
        fontsize=13, fontweight="bold")

    panels = [
        ("alive_nodes",   "Alive nodes",              "Network Lifetime"),
        ("total_energy",  "Total residual energy (J)", "Total Residual Energy"),
        ("dead_nodes",    "Cumulative dead nodes",    "Node Deaths Over Time"),
        ("energy_stddev", "Std dev of energy (J)",    "Energy Balance (lower=better)"),
    ]
    for ax, (key, ylabel, title) in zip(axes.flat, panels):
        for proto, res in results.items():
            series = res["stats"][key]
            ax.plot(range(len(series)), series,
                    color=_COLORS.get(proto, "#333"),
                    ls=_STYLES.get(proto, "-"), lw=2, label=proto)
        ax.set(xlabel="Round", ylabel=ylabel, title=title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if key == "alive_nodes":
            ax.set_ylim(0, cfg["NUM_NODES"] + 2)

    plt.tight_layout()
    plt.savefig("comparison_all_algorithms.png", dpi=150, bbox_inches="tight")
    print("  Plot saved -> comparison_all_algorithms.png")
    plt.close()


def plot_summary_bars(results: Dict[str, dict], cfg: dict) -> None:
    """Grouped bar chart of the headline metrics."""
    protocols = list(results.keys())
    lifetime  = [results[p]["network_dead"] for p in protocols]
    first_d   = [results[p]["first_dead"] if results[p]["first_dead"] is not None
                 else cfg["NUM_ROUNDS"] for p in protocols]
    packets   = [results[p]["stats"]["packets_to_bs"] for p in protocols]
    residual  = [results[p]["stats"]["total_energy"][-1]
                 if results[p]["stats"]["total_energy"] else 0.0
                 for p in protocols]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Headline Metrics by Protocol (higher lifetime / first-death /"
                 " packets / residual is better)", fontsize=12, fontweight="bold")
    colors = [_COLORS.get(p, "#333") for p in protocols]

    for ax, vals, title in [
        (axes[0, 0], lifetime, "Network lifetime (rounds)"),
        (axes[0, 1], first_d,  "First node death (round)"),
        (axes[1, 0], packets,  "Packets delivered to BS"),
        (axes[1, 1], residual, "Final residual energy (J)"),
    ]:
        bars = ax.bar(protocols, vals, color=colors)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2f}" if isinstance(v, float) else f"{v}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("comparison_summary_bars.png", dpi=150, bbox_inches="tight")
    print("  Plot saved -> comparison_summary_bars.png")
    plt.close()


def print_summary_table(results: Dict[str, dict], cfg: dict) -> None:
    protocols = list(results.keys())
    print("\n" + "=" * (26 + 12 * len(protocols)))
    print("  MULTI-PROTOCOL RESULTS SUMMARY")
    print("=" * (26 + 12 * len(protocols)))

    header = f"  {'Metric':<26}" + "".join(f"{p:>12}" for p in protocols)
    print(header)
    print("  " + "-" * (24 + 12 * len(protocols)))

    def row(label, fn):
        line = f"  {label:<26}"
        for p in protocols:
            line += f"{fn(results[p]):>12}"
        print(line)

    row("First node death",
        lambda r: str(r["first_dead"]) if r["first_dead"] is not None
        else f">{cfg['NUM_ROUNDS']}")
    row("Network lifetime",   lambda r: str(r["network_dead"]))
    row("Packets to BS",      lambda r: str(r["stats"]["packets_to_bs"]))
    row("Final residual (J)",
        lambda r: f"{r['stats']['total_energy'][-1]:.3f}"
        if r["stats"]["total_energy"] else "0")
    row("Final energy std (J)",
        lambda r: f"{r['stats']['energy_stddev'][-1]:.4f}"
        if r["stats"]["energy_stddev"] else "0")
    row("Avg CHs / round",
        lambda r: f"{np.mean(r['stats']['ch_counts']):.1f}"
        if r["stats"]["ch_counts"] else "0")
    row("Avg MS-CHs / round",
        lambda r: f"{np.mean(r['stats']['ms_counts']):.2f}"
        if r["stats"]["ms_counts"] else "0")
    print("=" * (26 + 12 * len(protocols)))

    # Relative gains vs each baseline, using GA as the reference protocol.
    if "GA" in results:
        ga_life = results["GA"]["network_dead"]
        print("\n  Network-lifetime gain of GA vs each baseline:")
        for p in protocols:
            if p == "GA":
                continue
            base_life = results[p]["network_dead"]
            if base_life > 0:
                pct = (ga_life - base_life) / base_life * 100
                print(f"    GA vs {p:<6}: {pct:+.1f}%")
    print()


# ==============================================================================
# MAIN
# ==============================================================================

def main(cfg: Optional[dict] = None,
         protocols: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    Run every requested protocol under identical conditions and emit the
    comparison plots + summary table.

    Choose protocols via the SOLAR_GA_PROTOCOLS env var (comma-separated) or the
    `protocols` argument. Default: GA, PSO, GWO, HEED, LEACH.
    """
    if cfg is None:
        cfg = default_config()

    if protocols is None:
        env = os.environ.get("SOLAR_GA_PROTOCOLS", "GA,PSO,GWO,HEED,LEACH")
        protocols = [p.strip().upper() for p in env.split(",") if p.strip()]

    print("\n" + "#" * 64)
    print("  MODERN-BASELINE COMPARISON  -  Solar-Aware GA Multi-Sink WSN")
    print("  Protocols: " + ", ".join(protocols))
    print("#" * 64)

    results: Dict[str, dict] = {}
    for proto in protocols:
        stats, fd, nd = run_protocol(cfg, proto)
        results[proto] = {"stats": stats, "first_dead": fd, "network_dead": nd}

    print_summary_table(results, cfg)
    print("  Generating plots...")
    plot_multi_results(results, cfg)
    plot_summary_bars(results, cfg)
    print("\n  Done.\n")
    return results


if __name__ == "__main__":
    main()
