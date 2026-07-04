"""
================================================================================
   Solar-Aware Genetic Algorithm Based Multi-Sink Data Aggregation Protocol
              for Wireless Sensor Assisted IoT Networks
                      ---  GA-ONLY  RUNNER  ---
================================================================================
  This file is the GA-only spin-off of `solar_ga_wsn`. LEACH has been removed
  so this script outputs only the GA-protocol results.

  It shares the same core formulas, GA logic, MS-CH election, and snapshot
  output as the GA branch of the combined script, with ONE intentional
  addition: the GA fitness function here includes a 5th "BS-proximity" term
  (bs_score) that rewards CH sets with members close enough to reach the base
  station directly (PATH A). The combined-script fitness uses 4 terms.

  Architecture (3-Tier):
    Tier 1 -> Sensor nodes : sense data, send to nearest CH
    Tier 2 -> Cluster Heads: aggregate sensors data, then either:
               PATH A -> direct to BS   (close + healthy battery)
               PATH B -> via MS-CH      (far or low battery)
    Tier 3 -> MS-CH node(s): collect from relay CHs, aggregate, send to BS
================================================================================
"""

import math
import os
import random
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Headless-friendly: never block on a GUI, even if matplotlib is configured for one.
plt.switch_backend("Agg")

SNAPSHOT_DIR = "topology_snapshots"


# ==============================================================================
# SECTION 1 - USER INPUT
# ==============================================================================

def get_user_input() -> dict:
    """
    Collect simulation parameters. Press ENTER to accept the default.
    """
    print("\n" + "=" * 65)
    print("  Solar-Aware GA Multi-Sink WSN Protocol - Configuration")
    print("=" * 65)
    print("  Press ENTER to use default value shown in [brackets]\n")

    def ask(prompt, default, cast=float, vmin=None, vmax=None):
        while True:
            try:
                raw = input(f"  {prompt} [{default}]: ").strip()
                val = cast(raw) if raw else default
                if vmin is not None and val < vmin:
                    print(f"    !  Must be >= {vmin}")
                    continue
                if vmax is not None and val > vmax:
                    print(f"    !  Must be <= {vmax}")
                    continue
                return val
            except ValueError:
                print(f"    !  Invalid - expected {cast.__name__}")

    print("  -- Network Setup --")
    field    = ask("Field size in metres (square)",        100, int,   50, 2000)
    n_nodes  = ask("Number of sensor nodes",                50, int,   10, 5000)
    bs_x     = ask("Base station X position (metres)",  field // 2, float, 0)
    bs_y     = ask("Base station Y position (metres)",  field + 20, float, 0)
    n_rounds = ask("Number of simulation rounds",           300, int,   50, 5000)

    print("\n  -- CH / MS-CH Percentages (both scale with alive nodes) --")
    ch_pct   = ask("CH percentage of alive nodes (e.g. 10 = 10%)",
                   10, float, 2, 30)
    relay_per_ms = ask("Relay CHs handled by ONE MS-CH",
                       4, int, 1, 50)

    print("\n  -- Energy Settings --")
    e_init   = ask("Initial node energy (Joules)",          0.5, float, 0.01, 10.0)
    e_solar  = ask("Peak solar harvest rate (J/round)",   0.002, float, 0.0,   0.1)
    pkt      = ask("Packet size (bits)",                   4000, int,   100, 100000)

    print("\n  -- GA Settings --")
    ga_pop   = ask("GA population size",                     30, int,   10,  300)
    ga_gen   = ask("GA generations per round",               50, int,   10,  500)
    ga_mut   = ask("GA mutation rate  (0.0-1.0)",           0.1, float, 0.0, 1.0)
    ga_cx    = ask("GA crossover rate (0.0-1.0)",           0.8, float, 0.0, 1.0)

    print("\n  -- Routing Thresholds --")
    d_dist   = ask("Max distance (m) for CH->BS direct",
                   round(field * 0.75), float, 10)
    d_nrg    = ask("Min battery fraction for CH->BS direct (0.0-1.0)",
                   0.4, float, 0.0, 1.0)

    print("\n  -- Topology Snapshots --")
    snap_every = ask("Save topology image every N rounds (0 = off)",
                     50, int, 0, 10000)

    print("\n  -- Validation --")
    init_chs = max(1, round(n_nodes * ch_pct / 100))
    print(f"  v  Initial CH count     : {init_chs}  "
          f"(= {n_nodes} nodes x {ch_pct}%)")
    print(f"  v  CH and MS-CH counts auto-scale every round")
    if bs_y > field * 2:
        print("  !  Warning: BS is very far - high energy cost")
    print("  v  Configuration accepted\n")

    return {
        "FIELD"        : field,
        "NUM_NODES"    : n_nodes,
        "BS_X"         : bs_x,
        "BS_Y"         : bs_y,
        "NUM_ROUNDS"   : n_rounds,
        "CH_PERCENT"   : ch_pct / 100.0,
        "RELAYS_PER_MS": relay_per_ms,
        "E_INITIAL"    : e_init,
        "MAX_HARVEST"  : e_solar,
        "PACKET_SIZE"  : pkt,
        "GA_POP"       : ga_pop,
        "GA_GEN"       : ga_gen,
        "GA_MUT"       : ga_mut,
        "GA_CX"        : ga_cx,
        "DIRECT_DIST"  : d_dist,
        "DIRECT_NRG"   : d_nrg,
        "SNAPSHOT_EVERY": snap_every,
    }


def get_num_chs(alive_count: int, cfg: dict) -> int:
    """Dynamic CH count - recalculated every round."""
    if alive_count <= 0:
        return 0
    return max(1, min(alive_count, round(alive_count * cfg["CH_PERCENT"])))


def get_num_ms(num_relay_chs: int, cfg: dict) -> int:
    """
    Dynamic MS-CH count.  One MS-CH per RELAYS_PER_MS relay CHs.
    Returns 0 when no CH needs a relay (MS-CH stage gets skipped wholesale).
    """
    if num_relay_chs <= 0:
        return 0
    return max(1, min(num_relay_chs,
                      math.ceil(num_relay_chs / cfg["RELAYS_PER_MS"])))


# ==============================================================================
# SECTION 2 - RADIO & ENERGY CONSTANTS
# ==============================================================================

E_ELEC         = 50e-9
E_AMP          = 100e-12
E_MP           = 0.0013e-12
E_DA           = 5e-9
BATTERY_MAX    = 2.0
COMM_RANGE_PCT = 0.4
D0             = math.sqrt(E_AMP / E_MP)
MS_REELECT_THR = 0.15
# Per-node solar harvesting efficiency range. Each node draws a fixed value
# in [SOLAR_EFF_MIN, SOLAR_EFF_MAX] at creation, modelling differences in
# panel orientation, shading and dust. This spatial heterogeneity is what
# makes the "solar-aware" score actually differentiate one node from another.
SOLAR_EFF_MIN  = 0.6
SOLAR_EFF_MAX  = 1.0


# ==============================================================================
# SECTION 3 - NODE
# ==============================================================================

class Node:
    __slots__ = ("id", "x", "y", "energy", "alive", "role",
                 "assigned_ch", "assigned_ms", "goes_direct",
                 "packets_sent", "_cfg", "_dist_bs", "solar_eff")

    def __init__(self, node_id: int, x: float, y: float, cfg: dict,
                 solar_eff: float = 1.0):
        self.id           = node_id
        self.x            = x
        self.y            = y
        self.energy       = cfg["E_INITIAL"]
        self.alive        = True
        self.role         = "sensor"
        self.assigned_ch  = None
        self.assigned_ms  = None
        self.goes_direct  = False
        self.packets_sent = 0
        self._cfg         = cfg
        self._dist_bs     = math.hypot(cfg["BS_X"] - x, cfg["BS_Y"] - y)
        # Fixed per-node harvesting efficiency (panel orientation / shading).
        self.solar_eff    = solar_eff

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    @property
    def distance_to_bs(self) -> float:
        return self._dist_bs

    def _tx_cost(self, bits: int, d: float) -> float:
        if d <= D0:
            return E_ELEC * bits + E_AMP * bits * d * d
        d2 = d * d
        return E_ELEC * bits + E_MP * bits * d2 * d2

    def _rx_cost(self, bits: int) -> float:
        return E_ELEC * bits

    def _agg_cost(self, bits: int, n: int) -> float:
        return E_DA * bits * n

    def transmit(self, to_x: float, to_y: float) -> bool:
        cost = self._tx_cost(self._cfg["PACKET_SIZE"],
                             math.hypot(self.x - to_x, self.y - to_y))
        self.energy       -= cost
        self.packets_sent += 1
        if self.energy <= 0:
            self.energy = 0
            self.alive  = False
        return self.alive

    def receive(self) -> bool:
        self.energy -= self._rx_cost(self._cfg["PACKET_SIZE"])
        if self.energy <= 0:
            self.energy = 0
            self.alive  = False
        return self.alive

    def aggregate(self, n_packets: int) -> bool:
        self.energy -= self._agg_cost(self._cfg["PACKET_SIZE"], n_packets)
        if self.energy <= 0:
            self.energy = 0
            self.alive  = False
        return self.alive

    def harvest_solar(self, solar_rate_now: float) -> None:
        # solar_rate_now is the network-wide daylight rate; scale it by this
        # node's own panel efficiency so shaded / poorly-oriented nodes
        # genuinely harvest less than well-placed ones.
        if solar_rate_now <= 0:
            return
        rate    = solar_rate_now * self.solar_eff
        harvest = max(0.0, rate + random.gauss(0, rate * 0.05))
        self.energy = min(self.energy + harvest, BATTERY_MAX)

    def reset_role(self) -> None:
        self.role        = "sensor"
        self.assigned_ch = None
        self.assigned_ms = None
        self.goes_direct = False

    @property
    def energy_fraction(self) -> float:
        # Fraction of INITIAL energy remaining. With solar harvesting this can
        # exceed 1.0 (battery caps at BATTERY_MAX = 4x E_INITIAL); callers that
        # need a [0, 1] value clamp with min(..., 1.0).
        # Uses E_INITIAL to stay consistent with the GA fitness energy term
        # (which also normalises by E_INITIAL) and with the documented design
        # in SOLAR_GA_WSN_TECHNICAL_DEEP_DIVE.md.
        return self.energy / self._cfg["E_INITIAL"]

    def __repr__(self) -> str:
        return (f"Node({self.id}, {self.role}, "
                f"{self.energy:.4f}J, alive={self.alive})")

def solar_rate_for_round(round_num: int, max_harvest: float) -> float:
    """
    24-hour cycle:
        00:00 -> 0
        06:00 -> sunrise
        12:00 -> peak solar
        18:00 -> sunset
        18:00-24:00 -> 0
    """
    hour = round_num % 24
    return max_harvest * max(
        0.0,
        math.sin(math.pi * (hour - 6) / 12)
    )

# ==============================================================================
# SECTION 4 - WORLD STATE
# ==============================================================================

@dataclass
class World:
    nodes: List[Node]
    alive_idx:  np.ndarray = dc_field(default_factory=lambda: np.empty(0, dtype=np.int32))
    alive_xy:   np.ndarray = dc_field(default_factory=lambda: np.empty((0, 2)))
    alive_e:    np.ndarray = dc_field(default_factory=lambda: np.empty(0))
    alive_dbs:  np.ndarray = dc_field(default_factory=lambda: np.empty(0))
    alive_seff: np.ndarray = dc_field(default_factory=lambda: np.empty(0))
    id_to_idx:  Dict[int, int] = dc_field(default_factory=dict)

    def refresh(self) -> None:
        alive = [n for n in self.nodes if n.alive]
        if not alive:
            self.alive_idx  = np.empty(0, dtype=np.int32)
            self.alive_xy   = np.empty((0, 2))
            self.alive_e    = np.empty(0)
            self.alive_dbs  = np.empty(0)
            self.alive_seff = np.empty(0)
            self.id_to_idx  = {}
            return
        self.alive_idx = np.fromiter((n.id for n in alive), dtype=np.int32,
                                     count=len(alive))
        self.alive_xy  = np.array([(n.x, n.y) for n in alive], dtype=np.float64)
        self.alive_e   = np.fromiter((n.energy for n in alive),
                                     dtype=np.float64, count=len(alive))
        self.alive_dbs = np.fromiter((n._dist_bs for n in alive),
                                     dtype=np.float64, count=len(alive))
        self.alive_seff = np.fromiter((n.solar_eff for n in alive),
                                      dtype=np.float64, count=len(alive))
        self.id_to_idx = {int(nid): i for i, nid in enumerate(self.alive_idx)}


def vectorized_assign(sensor_xy: np.ndarray, ch_xy: np.ndarray) -> np.ndarray:
    if ch_xy.shape[0] == 0 or sensor_xy.shape[0] == 0:
        return np.empty(sensor_xy.shape[0], dtype=np.int64)
    diff = sensor_xy[:, None, :] - ch_xy[None, :, :]
    d2   = np.einsum("ijk,ijk->ij", diff, diff)
    return d2.argmin(axis=1)


# ==============================================================================
# SECTION 5 - GENETIC ALGORITHM
# ==============================================================================

class Chromosome:
    __slots__ = ("genes", "fitness")

    def __init__(self, genes):
        self.genes   = list(genes)
        self.fitness = -1.0

    def copy(self) -> "Chromosome":
        c = Chromosome(self.genes)
        c.fitness = self.fitness
        return c


def _evaluate_population(pop: List["Chromosome"],
                        world: World,
                        cfg: dict,
                        solar_now: float,
                        num_chs: int) -> None:
    pending = [c for c in pop if c.fitness < 0]
    if not pending:
        return

    if (world.alive_idx.size == 0 or num_chs == 0
            or world.alive_xy.shape[0] < num_chs):
        for c in pending:
            c.fitness = 0.0
        return

    alive_xy   = world.alive_xy
    alive_e    = world.alive_e
    alive_seff = world.alive_seff
    S          = alive_xy.shape[0]
    id_map     = world.id_to_idx
    K        = num_chs
    P        = len(pending)

    gene_idx = np.empty((P, K), dtype=np.int64)
    valid    = np.ones(P, dtype=bool)
    for p, c in enumerate(pending):
        try:
            gene_idx[p] = [id_map[g] for g in c.genes]
        except (KeyError, ValueError):
            valid[p] = False
            gene_idx[p] = 0

    ch_xy  = alive_xy[gene_idx]
    ch_e   = alive_e[gene_idx]

    e_score = np.minimum(ch_e.sum(axis=1) / (K * cfg["E_INITIAL"]), 1.0)

    if cfg["MAX_HARVEST"] > 0:
        daylight = solar_now / cfg["MAX_HARVEST"]        # 0.0 night, 1.0 noon

        # NODE-SPECIFIC solar: scale the shared daylight level by the average
        # panel efficiency of THIS chromosome's CHs. Because solar_eff varies
        # per node, this term now differs between candidate CH sets and can
        # actually change which chromosome wins (previously it was a constant
        # offset that had no effect on ranking).
        ch_seff        = alive_seff[gene_idx]            # (P, K)
        solar_fraction = daylight * ch_seff.mean(axis=1)  # per-chromosome

        energy_fraction = np.minimum(
            ch_e.mean(axis=1) / cfg["E_INITIAL"],
            1.0
        )

        s_score = (
            0.5 * solar_fraction +
            0.5 * energy_fraction
        )
    else:
        s_score = np.full(P, 0.5)

    comm_range2 = (cfg["FIELD"] * COMM_RANGE_PCT) ** 2
    big_alloc   = P * S * K
    if big_alloc <= 4_000_000:
        diff   = alive_xy[None, :, None, :] - ch_xy[:, None, :, :]
        d2     = (diff * diff).sum(axis=-1)
        d2_min = d2.min(axis=2)
        is_ch = np.zeros((P, S), dtype=bool)
        rows  = np.arange(P)[:, None]
        is_ch[rows, gene_idx] = True
        sensor_mask = ~is_ch
        within = (d2_min <= comm_range2) & sensor_mask
        denom  = np.maximum(sensor_mask.sum(axis=1), 1)
        c_score = within.sum(axis=1) / denom
    else:
        c_score = np.zeros(P)
        for p in range(P):
            if not valid[p]:
                continue
            mask = np.ones(S, dtype=bool)
            mask[gene_idx[p]] = False
            sxy = alive_xy[mask]
            if sxy.shape[0] == 0:
                c_score[p] = 1.0
                continue
            diff = sxy[:, None, :] - ch_xy[p][None, :, :]
            d2   = (diff * diff).sum(axis=-1)
            covered = int((d2.min(axis=1) <= comm_range2).sum())
            c_score[p] = covered / sxy.shape[0]

    if K > 1:
        centroid = ch_xy.mean(axis=1, keepdims=True)
        spread   = np.sqrt(((ch_xy - centroid) ** 2).sum(axis=2)).mean(axis=1)
        sp_score = np.minimum(spread / cfg["FIELD"], 1.0)
    else:
        sp_score = np.full(P, 0.5)

    # BS-proximity score: reward chromosome sets where at least some CHs
    # are close to the BS (enabling PATH A direct routing).
    bs_xy    = np.array([[cfg["BS_X"], cfg["BS_Y"]]], dtype=np.float64)
    max_dist = math.hypot(cfg["FIELD"], cfg["BS_Y"])
    ch_bs_d  = np.sqrt(((ch_xy - bs_xy[:, None, :]) ** 2).sum(axis=2))   # (P, K)
    # Score = fraction of CHs that are within DIRECT_DIST of BS
    bs_score = (ch_bs_d <= cfg["DIRECT_DIST"]).sum(axis=1) / max(K, 1)
    bs_score = bs_score.astype(np.float64)

    fitness = (0.20 * e_score + 0.20 * s_score
               + 0.30 * c_score + 0.15 * sp_score + 0.15 * bs_score)
    fitness = np.where(valid, fitness, 0.0)

    for p, c in enumerate(pending):
        c.fitness = float(fitness[p])


def _smart_initial_population(alive_ids: List[int],
                              alive_energies: List[float],
                              num_chs: int,
                              pop_size: int) -> List[Chromosome]:
    pop: List[Chromosome] = []
    if not alive_ids or num_chs > len(alive_ids):
        return pop
    energies = np.asarray(alive_energies, dtype=np.float64)
    weights = energies + 1e-9
    weights = weights / weights.sum()

    n_smart = pop_size // 2
    for _ in range(n_smart):
        chosen = np.random.choice(alive_ids, size=num_chs,
                                  replace=False, p=weights)
        pop.append(Chromosome(chosen.tolist()))
    while len(pop) < pop_size:
        pop.append(Chromosome(random.sample(alive_ids, num_chs)))
    return pop


def _tournament(pop: List[Chromosome], k: int = 3) -> Chromosome:
    if len(pop) <= k:
        return max(pop, key=lambda c: c.fitness)
    return max(random.sample(pop, k), key=lambda c: c.fitness)


def _crossover(p1: Chromosome, p2: Chromosome,
               cfg: dict, num_chs: int) -> Chromosome:
    if random.random() > cfg["GA_CX"] or num_chs < 2:
        return p1.copy()
    pt    = random.randint(1, num_chs - 1)
    seen  = set(p1.genes[:pt])
    genes = list(p1.genes[:pt])
    for g in p2.genes:
        if g not in seen:
            genes.append(g); seen.add(g)
            if len(genes) == num_chs:
                break
    if len(genes) < num_chs:
        for g in p1.genes:
            if g not in seen:
                genes.append(g); seen.add(g)
                if len(genes) == num_chs:
                    break
    return Chromosome(genes[:num_chs])


def _mutate(chromo: Chromosome, alive_ids: List[int],
            mut_rate: float) -> Chromosome:
    if random.random() >= mut_rate or not chromo.genes:
        return chromo
    genes = chromo.genes[:]
    gset  = set(genes)
    candidates = [i for i in alive_ids if i not in gset]
    if not candidates:
        return Chromosome(genes)
    idx = random.randint(0, len(genes) - 1)
    genes[idx] = random.choice(candidates)
    return Chromosome(genes)


def run_ga_ch_election(nodes: List[Node],
                       world: World,
                       cfg: dict,
                       round_num: int,
                       num_chs: int) -> Optional[Chromosome]:
    if world.alive_idx.size < num_chs:
        return None

    alive_ids = world.alive_idx.tolist()
    alive_e   = world.alive_e.tolist()
    solar_now = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])

    pop = _smart_initial_population(alive_ids, alive_e, num_chs, cfg["GA_POP"])
    if not pop:
        return None

    PATIENCE = max(5, cfg["GA_GEN"] // 8)
    best_overall: Optional[Chromosome] = None
    best_score = -1.0
    stale = 0
    cur_mut = cfg["GA_MUT"]

    for gen in range(cfg["GA_GEN"]):
        _evaluate_population(pop, world, cfg, solar_now, num_chs)
        pop.sort(key=lambda c: c.fitness, reverse=True)

        if pop[0].fitness > best_score + 1e-9:
            best_score   = pop[0].fitness
            best_overall = pop[0].copy()
            stale = 0
            cur_mut = cfg["GA_MUT"]
        else:
            stale += 1
            cur_mut = min(0.5, cfg["GA_MUT"] * (1 + stale * 0.25))

        if stale >= PATIENCE:
            break

        new_pop = [pop[0].copy()]
        if len(pop) > 1:
            new_pop.append(pop[1].copy())

        while len(new_pop) < cfg["GA_POP"]:
            child = _crossover(_tournament(pop), _tournament(pop),
                               cfg, num_chs)
            new_pop.append(_mutate(child, alive_ids, cur_mut))
        pop = new_pop

    return best_overall


# ==============================================================================
# SECTION 6 - PATH DECISION
# ==============================================================================

def decide_ch_paths(ch_nodes: List[Node],
                    cfg: dict) -> Tuple[List[Node], List[Node]]:
    direct_chs: List[Node] = []
    relay_chs:  List[Node] = []
    for ch in ch_nodes:
        if not ch.alive:
            continue
        close   = ch.distance_to_bs <= cfg["DIRECT_DIST"]
        healthy = ch.energy_fraction >= cfg["DIRECT_NRG"]
        if close and healthy:
            ch.goes_direct = True
            ch.assigned_ms = None
            direct_chs.append(ch)
        else:
            ch.goes_direct = False
            ch.assigned_ms = None
            relay_chs.append(ch)
    return direct_chs, relay_chs


# ==============================================================================
# SECTION 7 - MS-CH ELECTION (THE SOLAR-AWARE NOVELTY)
# ==============================================================================

def _solar_aware_score(ch: Node, peers: List[Node], cfg: dict,
                       solar_now: float) -> float:
    bat = min(ch.energy_fraction, 1.0)
    if cfg["MAX_HARVEST"] > 0:
        # Node-specific: shared daylight level scaled by THIS candidate's own
        # panel efficiency, so the solar term genuinely differentiates the
        # relay CHs competing to become the MS-CH.
        solar = (solar_now / cfg["MAX_HARVEST"]) * ch.solar_eff
    else:
        solar = 0.5

    if len(peers) > 1:
        d_sum = sum(ch.distance_to(p.x, p.y) for p in peers if p.id != ch.id)
        avg_d = d_sum / max(len(peers) - 1, 1)
        cent  = 1.0 - min(avg_d / (cfg["FIELD"] * math.sqrt(2)), 1.0)
    else:
        cent = 1.0

    max_d = math.sqrt(cfg["FIELD"] ** 2 + cfg["BS_Y"] ** 2)
    bs_s  = 1.0 - min(ch.distance_to_bs / max_d, 1.0)

    return 0.35 * bat + 0.30 * solar + 0.20 * cent + 0.15 * bs_s


def _kmedoids_split(relay_chs: List[Node], k: int,
                    iters: int = 10) -> List[List[Node]]:
    if k <= 1 or len(relay_chs) <= k:
        if k <= 1:
            return [relay_chs]
        return [[ch] for ch in relay_chs]

    xy = np.array([(c.x, c.y) for c in relay_chs], dtype=np.float64)
    n  = len(relay_chs)
    medoids = [random.randrange(n)]
    for _ in range(k - 1):
        d2 = np.min(((xy[:, None, :] - xy[medoids][None, :, :]) ** 2)
                    .sum(axis=2), axis=1)
        medoids.append(int(d2.argmax()))

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        diff = xy[:, None, :] - xy[medoids][None, :, :]
        d2   = np.einsum("ijk,ijk->ij", diff, diff)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for ki in range(k):
            members = np.where(labels == ki)[0]
            if members.size == 0:
                continue
            centroid = xy[members].mean(axis=0)
            d_to_c   = ((xy[members] - centroid) ** 2).sum(axis=1)
            medoids[ki] = int(members[int(d_to_c.argmin())])

    clusters: List[List[Node]] = [[] for _ in range(k)]
    for i, lab in enumerate(labels):
        clusters[lab].append(relay_chs[i])
    return [c for c in clusters if c]


def elect_ms_chs(relay_chs: List[Node], cfg: dict,
                 solar_now: float, num_ms: int) -> List[Node]:
    if num_ms <= 0 or not relay_chs:
        return []

    if num_ms == 1 or len(relay_chs) <= 2:
        best = max(relay_chs,
                   key=lambda c: _solar_aware_score(c, relay_chs, cfg, solar_now))
        best.role = "MS-CH"
        for r in relay_chs:
            if r.id != best.id:
                r.assigned_ms = best.id
        return [best]

    clusters = _kmedoids_split(relay_chs, num_ms)
    elected: List[Node] = []
    for cluster in clusters:
        best = max(cluster,
                   key=lambda c: _solar_aware_score(c, cluster, cfg, solar_now))
        best.role = "MS-CH"
        elected.append(best)
        for r in cluster:
            if r.id != best.id:
                r.assigned_ms = best.id

    return elected


# ==============================================================================
# SECTION 8 - GA PROTOCOL ROUND
# ==============================================================================

def simulate_round_ga(nodes: List[Node],
                      world: World,
                      round_num: int,
                      cfg: dict,
                      stats: dict) -> bool:
    solar_now = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])

    if solar_now > 0:
        for n in nodes:
            if n.alive:
                n.harvest_solar(solar_now)

    for n in nodes:
        if n.alive:
            n.reset_role()

    world.refresh()
    alive_count = world.alive_idx.size
    num_chs = get_num_chs(alive_count, cfg)

    if alive_count < num_chs + 1:
        _record_stats(nodes, stats)
        stats["ch_counts"].append(0)
        stats["ms_counts"].append(0)
        return False

    solution = run_ga_ch_election(nodes, world, cfg, round_num, num_chs)
    if solution is None:
        _record_stats(nodes, stats)
        stats["ch_counts"].append(0)
        stats["ms_counts"].append(0)
        return False

    ch_nodes = [nodes[g] for g in solution.genes
                if g < len(nodes) and nodes[g].alive]
    for ch in ch_nodes:
        ch.role = "CH"

    direct_chs, relay_chs = decide_ch_paths(ch_nodes, cfg)

    num_ms = get_num_ms(len(relay_chs), cfg)
    ms_chs = elect_ms_chs(relay_chs, cfg, solar_now, num_ms) if num_ms else []

    if ch_nodes:
        ch_xy = np.array([(c.x, c.y) for c in ch_nodes], dtype=np.float64)
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

    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch is not None:
            ch = nodes[n.assigned_ch]
            if ch.alive:
                n.transmit(ch.x, ch.y)
                ch.receive()

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

    _record_stats(nodes, stats)
    stats["ch_counts"].append(len(ch_nodes))
    stats["ms_counts"].append(len(ms_chs))

    snap = cfg.get("SNAPSHOT_EVERY", 0)
    if snap and round_num % snap == 0:
        try:
            plot_topology_snapshot(nodes, ch_nodes, direct_chs, relay_chs,
                                   ms_chs, round_num, cfg, protocol="GA")
        except Exception as e:
            print(f"  !  Snapshot failed at round {round_num}: {e}")
    return any(n.alive for n in nodes)


# ==============================================================================
# SECTION 9 - STATS
# ==============================================================================

def make_stats() -> dict:
    return {
        "alive_nodes"   : [],
        "dead_nodes"    : [],
        "total_energy"  : [],
        "energy_stddev" : [],
        "ch_counts"     : [],
        "ms_counts"     : [],
        "packets_to_bs" : 0,
        "reelections"   : 0,
    }


def _record_stats(nodes: List[Node], stats: dict) -> None:
    energies = [n.energy for n in nodes if n.alive]
    alive = len(energies)
    stats["alive_nodes"].append(alive)
    stats["dead_nodes"].append(len(nodes) - alive)
    stats["total_energy"].append(sum(energies))
    stats["energy_stddev"].append(float(np.std(energies)) if energies else 0.0)


# ==============================================================================
# SECTION 10 - SIMULATION RUNNER
# ==============================================================================
def create_nodes(cfg: dict, seed=None) -> List[Node]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nodes = [Node(
        i,
        random.uniform(0, cfg["FIELD"]),
        random.uniform(0, cfg["FIELD"]),
        cfg
    ) for i in range(cfg["NUM_NODES"])]
    # Assign each node a fixed solar-harvesting efficiency. Drawn AFTER all
    # positions so the field topology is byte-for-byte identical to before;
    # this heterogeneity is what makes the solar-aware selection meaningful.
    for n in nodes:
        n.solar_eff = random.uniform(SOLAR_EFF_MIN, SOLAR_EFF_MAX)
    return nodes


def run_simulation(cfg: dict):
    nodes = create_nodes(cfg, seed=42)
    world = World(nodes=nodes)
    stats = make_stats()
    init_chs = max(1, round(cfg["NUM_NODES"] * cfg["CH_PERCENT"]))
    print(f"\n{'=' * 58}")
    print(f"  Protocol  : GA (Solar-Aware Multi-Sink)")
    print(f"  Nodes     : {cfg['NUM_NODES']}   "
          f"Field : {cfg['FIELD']}x{cfg['FIELD']}m")
    print(f"  Rounds    : {cfg['NUM_ROUNDS']}  "
          f"Initial CHs: {init_chs} "
          f"({cfg['CH_PERCENT'] * 100:.0f}% - dynamic)")
    print(f"  MS-CH     : 1 per {cfg['RELAYS_PER_MS']} relay CHs (dynamic, "
          f"0 if no relays)")
    if cfg.get("SNAPSHOT_EVERY", 0):
        print(f"  Snapshots : every {cfg['SNAPSHOT_EVERY']} rounds -> "
              f"./{SNAPSHOT_DIR}/")
    print(f"{'=' * 58}")

    first_dead   = None
    network_dead = cfg["NUM_ROUNDS"]

    for r in range(cfg["NUM_ROUNDS"]):
        alive = simulate_round_ga(nodes, world, r, cfg, stats)

        dead_now = cfg["NUM_NODES"] - sum(1 for n in nodes if n.alive)
        if first_dead is None and dead_now >= 1:
            first_dead = r
            print(f"  *  First node died   : Round {r}")

        if not alive:
            network_dead = r
            print(f"  X  Network dead      : Round {r}")
            break

        step = max(cfg["NUM_ROUNDS"] // 5, 1)
        if r % step == 0:
            a = stats["alive_nodes"][-1]
            e = stats["total_energy"][-1]
            nchs = stats["ch_counts"][-1] if stats["ch_counts"] else "?"
            nms  = stats["ms_counts"][-1] if stats["ms_counts"] else "?"
            print(f"  Round {r:4d} | Alive: {a:3d}/{cfg['NUM_NODES']} "
                  f"| CHs: {nchs} | MS: {nms} | Energy: {e:.4f} J")

    if network_dead == cfg["NUM_ROUNDS"]:
        print(f"  v  Network survived all {cfg['NUM_ROUNDS']} rounds!")
    print(f"  Packets to BS     : {stats['packets_to_bs']}")
    print(f"  MS-CH re-elections: {stats['reelections']}")
    print(f"{'=' * 58}")

    return stats, first_dead, network_dead, nodes


# ==============================================================================
# SECTION 11 - PLOTS
# ==============================================================================

def plot_results(ga_stats, ga_fd, cfg) -> None:
    """6 graphs: lifetime, energy, deaths, balance, dynamic CH/MS, solar."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        "Solar-Aware GA Multi-Sink Protocol\n"
        f"(Nodes={cfg['NUM_NODES']}, Field={cfg['FIELD']}m, "
        f"Rounds={cfg['NUM_ROUNDS']}, "
        f"CH%={cfg['CH_PERCENT'] * 100:.0f}% dynamic, "
        f"1 MS-CH per {cfg['RELAYS_PER_MS']} relay CHs)",
        fontsize=12, fontweight="bold")

    C_GA, C_MS = "#1A5FAD", "#27AE60"
    rg = range(len(ga_stats["alive_nodes"]))

    # 1. Alive
    ax = axes[0, 0]
    ax.plot(rg, ga_stats["alive_nodes"], color=C_GA, lw=2, label="GA")

    if ga_fd is not None:
        ax.axvline(ga_fd, color=C_GA, ls=":", alpha=0.6,
                   label=f"GA 1st death r{ga_fd}")
    ax.set(xlabel="Round", ylabel="Alive nodes",
           title="Network Lifetime",
           ylim=(0, cfg["NUM_NODES"] + 2))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Total residual energy
    ax = axes[0, 1]
    ax.plot(rg, ga_stats["total_energy"], color=C_GA, lw=2, label="GA")
    ax.set(xlabel="Round", ylabel="Total residual energy (J)",
           title="Total Residual Energy")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Dead nodes
    ax = axes[0, 2]
    ax.plot(rg, ga_stats["dead_nodes"], color=C_GA, lw=2, label="GA")
    ax.set(xlabel="Round", ylabel="Cumulative dead nodes",
           title="Node Deaths Over Time")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Energy balance
    ax = axes[1, 0]
    ax.plot(rg, ga_stats["energy_stddev"], color=C_GA, lw=2, label="GA")
    ax.set(xlabel="Round", ylabel="Std deviation of energy (J)",
           title="Energy Balance\n(lower = more balanced)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Dynamic CH + MS counts
    ax = axes[1, 1]
    if ga_stats["ch_counts"]:
        ax.plot(range(len(ga_stats["ch_counts"])),
                ga_stats["ch_counts"], color=C_GA, lw=2, label="GA CHs")
    if ga_stats["ms_counts"]:
        ax.plot(range(len(ga_stats["ms_counts"])),
                ga_stats["ms_counts"], color=C_MS, lw=2, label="GA MS-CHs")
    ax.set(xlabel="Round", ylabel="Count",
           title=f"Dynamic CH & MS-CH Counts\n"
                 f"(both scale with alive nodes)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Solar harvest cycle
    ax = axes[1, 2]
    hours   = np.linspace(0, 48, 500)
    # Reuse the SAME formula as solar_rate_for_round() so the plotted curve
    # can never drift from the model actually used in the simulation.
    # (Bug fix: the "-6" phase shift was previously missing here, which made
    #  the plot peak at 06:00 instead of noon.)
    harvest = [cfg["MAX_HARVEST"] * max(0.0,
                                        math.sin(math.pi * ((h % 24) - 6) / 12))
               for h in hours]
    ax.fill_between(hours, harvest, alpha=0.25, color="orange")
    ax.plot(hours, harvest, color="darkorange", lw=2)
    ax.set_xticks(range(0, 49, 6))
    ax.set_xticklabels([f"{h % 24:02d}:00" for h in range(0, 49, 6)])
    ax.set(xlabel="Hour of day", ylabel="Harvest rate (J/round)",
           title="Solar Harvest Model\n(30% weight in MS-CH election)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ga_results.png", dpi=150, bbox_inches="tight")
    print("  Plot saved -> ga_results.png")
    plt.close()


def plot_topology_snapshot(nodes: List["Node"],
                           ch_nodes: List["Node"],
                           direct_chs: List["Node"],
                           relay_chs: List["Node"],
                           ms_chs: List["Node"],
                           round_num: int,
                           cfg: dict,
                           protocol: str = "GA") -> None:
    F, BSX, BSY = cfg["FIELD"], cfg["BS_X"], cfg["BS_Y"]
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    alive_sensors_x: List[float] = []
    alive_sensors_y: List[float] = []
    dead_x: List[float] = []
    dead_y: List[float] = []
    sensor_to_ch: Dict[int, int] = {}
    ms_id_set = {m.id for m in ms_chs}
    for n in nodes:
        if not n.alive:
            dead_x.append(n.x)
            dead_y.append(n.y)
            continue
        if n.role == "sensor":
            alive_sensors_x.append(n.x)
            alive_sensors_y.append(n.y)
            if n.assigned_ch is not None:
                sensor_to_ch[n.id] = n.assigned_ch

    n_alive  = sum(1 for n in nodes if n.alive)
    n_direct = len(direct_chs)
    n_relay  = len(relay_chs) - len(ms_chs)
    n_ms     = len(ms_chs)
    ms_used  = n_ms > 0 and any(c.assigned_ms is not None for c in relay_chs)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_facecolor("#f4f6f9")
    ax.set_xlim(-8, F + 8)
    ax.set_ylim(-8, max(BSY + 20, F + 20))

    comm_range = F * COMM_RANGE_PCT
    for c in ch_nodes:
        ax.add_patch(plt.Circle((c.x, c.y), comm_range,
                                color="gray", fill=False,
                                alpha=0.08, lw=0.7, ls="--"))

    for sid, cid in sensor_to_ch.items():
        s, c = nodes[sid], nodes[cid]
        if c.alive:
            ax.plot([s.x, c.x], [s.y, c.y],
                    color="#AABCD4", lw=0.45, alpha=0.40, zorder=1)

    for c in direct_chs:
        if c.alive:
            ax.annotate("", xy=(BSX, BSY), xytext=(c.x, c.y),
                        arrowprops=dict(arrowstyle="->", color="#1A5FAD",
                                        lw=1.6, linestyle="dashed",
                                        alpha=0.85),
                        zorder=4)

    for c in relay_chs:
        if c.alive and c.assigned_ms is not None and c.id not in ms_id_set:
            ms = nodes[c.assigned_ms]
            if ms.alive:
                ax.annotate("", xy=(ms.x, ms.y), xytext=(c.x, c.y),
                            arrowprops=dict(arrowstyle="->", color="#27AE60",
                                            lw=1.6, alpha=0.9),
                            zorder=4)

    for m in ms_chs:
        if m.alive:
            ax.annotate("", xy=(BSX, BSY), xytext=(m.x, m.y),
                        arrowprops=dict(arrowstyle="->", color="#C0392B",
                                        lw=2.4, alpha=0.95),
                        zorder=5)

    if alive_sensors_x:
        ax.scatter(alive_sensors_x, alive_sensors_y,
                   c="#7FB3D3", s=32, zorder=3, alpha=0.85,
                   edgecolors="white", lw=0.5)
    if dead_x:
        ax.scatter(dead_x, dead_y, c="#888", s=22, marker="x",
                   zorder=3, alpha=0.6, lw=1.0)

    for c in direct_chs:
        if c.alive:
            ax.scatter(c.x, c.y, c="#1A5FAD", s=170, marker="^",
                       zorder=5, edgecolors="white", lw=1.2)
    for c in relay_chs:
        if c.alive and c.id not in ms_id_set:
            ax.scatter(c.x, c.y, c="#E67E22", s=170, marker="^",
                       zorder=5, edgecolors="white", lw=1.2)
    for m in ms_chs:
        if m.alive:
            ax.scatter(m.x, m.y, c="#27AE60", s=360, marker="*",
                       zorder=6, edgecolors="white", lw=1.4)

    ax.scatter(BSX, BSY, c="#C0392B", s=420, marker="s",
               zorder=7, edgecolors="darkred", lw=2)
    ax.text(BSX + 7, BSY + 3, "BS",
            fontsize=10, fontweight="bold", color="darkred")

    ax.add_patch(plt.Rectangle((0, 0), F, F, fill=False,
                               edgecolor="#999", lw=1.2, ls="--", alpha=0.5))

    if ms_used:
        mode_label = f"MS-USED  ({n_ms} MS-CH{'s' if n_ms != 1 else ''})"
        title_color = "#27AE60"
    else:
        mode_label = "MS-SKIPPED  (all CHs reach BS directly)"
        title_color = "#1A5FAD"

    ax.set_title(
        f"GA  Round {round_num:4d}   |   {mode_label}\n"
        f"Alive: {n_alive}/{cfg['NUM_NODES']}   "
        f"CHs: {len(ch_nodes)}   "
        f"Direct: {n_direct}   Relay: {n_relay}   MS-CH: {n_ms}",
        fontsize=12, fontweight="bold", color=title_color, pad=12)

    handles = [
        mpatches.Patch(color="#7FB3D3", label="Sensor (alive)"),
        mpatches.Patch(color="#888",    label="Sensor (dead)"),
        mpatches.Patch(color="#1A5FAD", label="CH -> BS direct  (PATH A)"),
        mpatches.Patch(color="#E67E22", label="CH -> MS-CH relay (PATH B)"),
        mpatches.Patch(color="#27AE60", label="MS-CH -> BS       (PATH C)"),
        mpatches.Patch(color="#C0392B", label="Base Station"),
    ]
    ax.legend(handles=handles, loc="lower right",
              fontsize=8, framealpha=0.93, edgecolor="gray")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.18)

    fname = os.path.join(SNAPSHOT_DIR,
                         f"ga_round_{round_num:04d}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_topology_both_paths(cfg: dict) -> None:
    """Deterministic illustrative topology with multiple MS-CHs."""
    F, BSX, BSY = cfg["FIELD"], cfg["BS_X"], cfg["BS_Y"]

    def d(x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    sensors = [
        (0,  F * .80, F * .85), (1,  F * .90, F * .70), (2,  F * .75, F * .72),
        (3,  F * .15, F * .88), (4,  F * .25, F * .75), (5,  F * .10, F * .70),
        (6,  F * .10, F * .35), (7,  F * .20, F * .22), (8,  F * .08, F * .15),
        (9,  F * .80, F * .30), (10, F * .90, F * .18), (11, F * .70, F * .12),
        (12, F * .40, F * .45), (13, F * .60, F * .42), (14, F * .35, F * .30),
        (15, F * .65, F * .28),
    ]

    chs = {
        20: (F * .85, F * .92, "CH-direct", "CH-1\n(direct)"),
        21: (F * .45, F * .88, "CH-direct", "CH-2\n(direct)"),
        22: (F * .15, F * .20, "CH-relay",  "CH-3\n(relay W)"),
        23: (F * .82, F * .15, "CH-relay",  "CH-4\n(relay E)"),
        24: (F * .25, F * .50, "MS-CH",     "MS-CH 1\n(west zone)"),
        25: (F * .75, F * .55, "MS-CH",     "MS-CH 2\n(east zone)"),
    }

    sensor_ch = {0: 20, 1: 20, 2: 20, 3: 21, 4: 21, 5: 21,
                 6: 22, 7: 22, 8: 22, 9: 23, 10: 23, 11: 23,
                 12: 24, 13: 24, 14: 25, 15: 25}
    relay_to_ms = {22: 24, 23: 25}

    fig, ax = plt.subplots(figsize=(11, 12))
    ax.set_facecolor("#f4f6f9")
    ax.set_xlim(-8, F + 8); ax.set_ylim(-8, BSY + 20)

    for cid, (cx, cy, role, _) in chs.items():
        ax.add_patch(plt.Circle((cx, cy), F * COMM_RANGE_PCT,
                                color="gray", fill=False,
                                alpha=0.10, lw=0.8, ls="--"))

    for sid, (_, sx, sy) in enumerate(sensors):
        chx, chy = chs[sensor_ch[sid]][:2]
        ax.plot([sx, chx], [sy, chy],
                color="#AABCD4", lw=0.6, alpha=0.45, zorder=1)

    for cid, rad, lbl in [(20, 0.20, "CH-1"), (21, -0.15, "CH-2")]:
        cx, cy = chs[cid][:2]
        ax.annotate("", xy=(BSX, BSY), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#1A5FAD",
                                    lw=2.2, linestyle="dashed",
                                    connectionstyle=f"arc3,rad={rad}"),
                    zorder=4)
        ax.text((cx + BSX) / 2 + (8 if rad > 0 else -8), (cy + BSY) / 2,
                f"PATH A\n{lbl}->BS\n({d(cx, cy, BSX, BSY):.0f}m)",
                fontsize=7.5, color="#1A5FAD", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#1A5FAD", alpha=0.80, lw=0.7))

    for r_id, ms_id in relay_to_ms.items():
        cx, cy = chs[r_id][:2]
        mx, my = chs[ms_id][:2]
        ax.annotate("", xy=(mx, my), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color="#27AE60",
                                    lw=2.2,
                                    connectionstyle="arc3,rad=0.15"),
                    zorder=4)
        ax.text((cx + mx) / 2 + 6, (cy + my) / 2 + 4,
                f"PATH B\nCH->MS-CH\n({d(cx, cy, mx, my):.0f}m)",
                fontsize=7.5, color="#27AE60", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#27AE60", alpha=0.80, lw=0.7))

    for ms_id in (24, 25):
        mx, my = chs[ms_id][:2]
        ax.annotate("", xy=(BSX, BSY), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->", color="#C0392B",
                                    lw=3.0,
                                    connectionstyle="arc3,rad=0.0"),
                    zorder=5)
        ax.text((mx + BSX) / 2 + 10, (my + BSY) / 2,
                f"PATH C\nMS->BS\n({d(mx, my, BSX, BSY):.0f}m)",
                fontsize=8, color="#C0392B", fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#C0392B", alpha=0.88, lw=0.9))

    sx = [s[1] for s in sensors]; sy = [s[2] for s in sensors]
    ax.scatter(sx, sy, c="#7FB3D3", s=45, zorder=3, alpha=0.85,
               edgecolors="white", lw=0.6)

    for cid, (cx, cy, role, lbl) in chs.items():
        if role == "CH-direct":
            ax.scatter(cx, cy, c="#1A5FAD", s=210, marker="^", zorder=5,
                       edgecolors="white", lw=1.3)
        elif role == "CH-relay":
            ax.scatter(cx, cy, c="#E67E22", s=210, marker="^", zorder=5,
                       edgecolors="white", lw=1.3)
        else:
            ax.scatter(cx, cy, c="#27AE60", s=420, marker="*", zorder=6,
                       edgecolors="white", lw=1.5)
        ax.annotate(lbl, (cx, cy), textcoords="offset points",
                    xytext=(9, 7), fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.75, ec="gray", lw=0.5))

    ax.scatter(BSX, BSY, c="#C0392B", s=520, marker="s", zorder=7,
               edgecolors="darkred", lw=2)
    ax.text(BSX + 9, BSY + 3, "Base\nStation",
            fontsize=10, fontweight="bold", color="darkred")

    ax.text(0.01, 0.99,
            "Data Flow Summary\n"
            "------------------------\n"
            "Sensor -> CH  (always)\n"
            "PATH A: CH -> BS direct\n"
            "  (close + healthy batt)\n"
            "PATH B: CH -> MS-CH\n"
            "  (far or low battery)\n"
            "PATH C: MS-CH -> BS\n"
            "  (one long-range TX)\n"
            "Multiple MS-CHs scale\n"
            "with relay CH count.",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec="#555", alpha=0.90, lw=0.8))

    ax.add_patch(plt.Rectangle((0, 0), F, F, fill=False,
                               edgecolor="#999", lw=1.5,
                               ls="--", alpha=0.5))
    ax.text(3, F - 5, f"{F}x{F}m sensor field",
            fontsize=8.5, color="#888")

    handles = [
        mpatches.Patch(color="#7FB3D3", label="Sensor node"),
        mpatches.Patch(color="#1A5FAD", label="CH -> BS direct (close + healthy)"),
        mpatches.Patch(color="#E67E22", label="CH -> MS-CH relay (far or low batt)"),
        mpatches.Patch(color="#27AE60", label="MS-CH (solar-aware, multiple zones)"),
        mpatches.Patch(color="#C0392B", label="Base Station (BS)"),
        mlines.Line2D([], [], color="#AABCD4", lw=1.2, label="Sensor -> CH"),
        mlines.Line2D([], [], color="#1A5FAD", lw=2.2, ls="--",
                      label="PATH A: CH -> BS directly"),
        mlines.Line2D([], [], color="#27AE60", lw=2.2,
                      label="PATH B: CH -> MS-CH (relay)"),
        mlines.Line2D([], [], color="#C0392B", lw=3.0,
                      label="PATH C: MS-CH -> BS (final hop)"),
    ]
    ax.legend(handles=handles, loc="lower right",
              fontsize=8.2, framealpha=0.93, edgecolor="gray")
    ax.set_xlabel("X position (metres)", fontsize=11)
    ax.set_ylabel("Y position (metres)", fontsize=11)
    ax.set_title(
        "WSN Network Topology - Solar-Aware GA Multi-Sink Protocol\n"
        "PATH A: CH->BS direct  |  PATH B: CH->MS-CH->BS  |  "
        "PATH C: MS-CH->BS",
        fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.18)
    plt.tight_layout()
    plt.savefig("ga_topology.png", dpi=160, bbox_inches="tight")
    print("  Plot saved -> ga_topology.png")
    plt.close()


# ==============================================================================
# SECTION 12 - SUMMARY
# ==============================================================================

def print_summary(ga_s, ga_fd, ga_nd, cfg) -> None:
    print("\n" + "=" * 64)
    print("  GA SIMULATION RESULTS SUMMARY")
    print("=" * 64)
    print(f"  {'Metric':<40} {'GA':>10}")
    print(f"  {'-' * 50}")

    def fmt(v):
        return str(v) if v is not None else f">{cfg['NUM_ROUNDS']}"

    rows = [
        ("First node death (round)",      fmt(ga_fd)),
        ("Network lifetime (rounds)",     str(ga_nd)),
        ("Packets delivered to BS",       str(ga_s["packets_to_bs"])),
        ("Final residual energy (J)",
         f"{ga_s['total_energy'][-1]:.4f}" if ga_s["total_energy"] else "0"),
        ("Final energy std dev (J)",
         f"{ga_s['energy_stddev'][-1]:.4f}" if ga_s["energy_stddev"] else "0"),
        ("MS-CH re-elections",            str(ga_s["reelections"])),
        ("Avg CHs / round",
         f"{np.mean(ga_s['ch_counts']):.1f}"  if ga_s['ch_counts']  else "0"),
        ("Avg MS-CHs / round",
         f"{np.mean(ga_s['ms_counts']):.2f}"  if ga_s['ms_counts']  else "0"),
        ("CH% used", f"{cfg['CH_PERCENT'] * 100:.0f}% dynamic"),
    ]
    for label, ga_val in rows:
        print(f"  {label:<40} {ga_val:>10}")
    print("=" * 64)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "#" * 65)
    print("  Solar-Aware GA Multi-Sink Data Aggregation Protocol")
    print("  for Wireless Sensor Assisted IoT  --  GA-ONLY RUNNER")
    print("#" * 65)

    cfg = get_user_input()

    ga_stats, ga_fd, ga_nd, _ = run_simulation(cfg)

    print_summary(ga_stats, ga_fd, ga_nd, cfg)

    print("\n  Generating plots...")
    plot_results(ga_stats, ga_fd, cfg)
    plot_topology_both_paths(cfg)

    print("\n  All outputs saved.  Done.\n")


if __name__ == "__main__":
    main()
