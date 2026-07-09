"""
================================================================================
   Solar-Aware Genetic Algorithm Based Multi-Sink Data Aggregation Protocol
              for Wireless Sensor Assisted IoT Networks
================================================================================
  Architecture (3-Tier):
    Tier 1 -> Sensor nodes : sense data, send to nearest CH
    Tier 2 -> Cluster Heads: aggregate sensors data, then either:
               PATH A -> direct to BS   (close + healthy battery)
               PATH B -> via MS-CH      (far or low battery)
    Tier 3 -> MS-CH node(s): collect from relay CHs, aggregate, send to BS

  GA runs every round - 3 jobs:
    Job 1 -> Elect CHs  (energy + solar + coverage + spread)
    Job 2 -> Decide each CH path  (direct vs relay)
    Job 3 -> Elect MS-CH(s) ONLY among relay CHs

  ----------------------------------------------------------------------------
  USER REQUIREMENTS - POINT-BY-POINT COMPLIANCE
  ----------------------------------------------------------------------------
  (1) MS-CH count scales with relay CH count
        ->  get_num_ms()           : one MS-CH per RELAYS_PER_MS relay CHs
        ->  elect_ms_chs()         : k-medoids over relay CHs picks the
                                     best one per zone (multi-MS placement)
  (2) CH count scales with alive node count
        ->  get_num_chs()          : recomputed EVERY round, capped at
                                     alive_count
  (3) MS-CH stage runs ONLY when actually needed
        ->  get_num_ms() returns 0 when no CH needs a relay; the round's
            MS-CH election is skipped wholesale
  (4) A CH that can reach BS directly NEVER goes through an MS-CH
        ->  decide_ch_paths() runs BEFORE elect_ms_chs(), so direct CHs
            are filtered out before the relay-pool is built
  (5) Big-network optimisation
        ->  World caches alive arrays + id->index map; sensor->CH
            assignment, coverage, and clustering are vectorized in NumPy
  (6) GA optimisation
        ->  smart energy-weighted seeding, elitism (top-2),
            BATCHED whole-population fitness in a single NumPy pass,
            adaptive mutation, early-stop on plateau, fitness caching
  Plus: per-round topology snapshot every SNAPSHOT_EVERY rounds, with the
        title flagged "MS-USED" or "MS-SKIPPED" so the project demo shows
        when the MS-CH tier is engaged vs bypassed.
  ----------------------------------------------------------------------------

  References:
    [1] Muruganantham & El-Ocla (2020) - Routing using GA in WSN
    [2] Wu et al. (2013) IET - GA for energy harvesting WSN
    [3] GA-based energy efficient routing (ACSIJ 2014)
    [4] Routing optimisation in IoT using GA (2023)
    [5] Heinzelman et al. - LEACH protocol baseline
    [6] First-order radio model E(k,d)=E_elec*k + E_amp*k*d^2
================================================================================
"""

import math
import os
import random
import time                       # v2: per-round CH-election timing
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# only for Google Colab 
def _in_colab() -> bool:
    """Detect whether we are running inside Google Colab."""
    try:
        import google.colab  # noqa: F401  (presence check only)
        return True
    except ImportError:
        return False

# and this is for all code editor like vs code , codex etc except Google Colab
def _in_notebook() -> bool:
    """Detect a Jupyter/IPython kernel (Colab counts too)."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        return ip.__class__.__name__ != "TerminalInteractiveShell"
    except ImportError:
        return False


# Headless-friendly outside notebooks: never block on a GUI.
# Inside Colab / Jupyter we keep the inline backend so plots render in cells.
if not (_in_colab() or _in_notebook()):
    plt.switch_backend("Agg")

SNAPSHOT_DIR = "topology_snapshots"

# v2: optional convergence recorder. When set to a Python list, each CH-election
# optimiser (GA / PSO / GWO) appends its best-fitness-so-far after every
# generation, letting us plot convergence curves and directly compare how fast
# each optimiser improves. Left as None during normal runs => zero overhead.
_CONV_SINK: Optional[list] = None


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
                   round(field * 0.55), float, 10)
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


def default_config(**overrides) -> dict:
    """
    Return a sensible default configuration dictionary - no prompts.

    Useful in Google Colab and other non-interactive environments. Pass
    keyword overrides to tweak any field, e.g.:

        cfg = default_config(NUM_NODES=200, NUM_ROUNDS=500)
    """
    cfg = {
        "FIELD"         : 100,
        "NUM_NODES"     : 50,
        "BS_X"          : 50.0,
        "BS_Y"          : 120.0,
        "NUM_ROUNDS"    : 300,
        "CH_PERCENT"    : 0.10,
        "RELAYS_PER_MS" : 4,
        "E_INITIAL"     : 0.5,
        "MAX_HARVEST"   : 0.002,
        "PACKET_SIZE"   : 4000,
        "GA_POP"        : 30,
        "GA_GEN"        : 50,
        "GA_MUT"        : 0.1,
        "GA_CX"         : 0.8,
        "DIRECT_DIST"   : 55.0,
        "DIRECT_NRG"    : 0.4,
        "SNAPSHOT_EVERY": 50,

        # ---------------------------------------------------------------------
        # v2 ADDITIONS - each one removes a drawback of the original model.
        # ---------------------------------------------------------------------
        # (Drawback 3) Realistic radio / MAC layer -----------------------------
        #   PACKET_LOSS  : base probability a single transmission is dropped
        #                  (collision / fading). Scaled up with distance.
        #   MAX_RETX     : max automatic retransmissions on a dropped packet
        #                  (each retry costs TX energy again -> ARQ realism).
        #   IDLE_ENERGY  : per-round idle-listening drain for EVERY alive node
        #                  (radios are not free when silent).
        #   SENSING_ENERGY: per-round sensing/CPU drain for every alive node.
        "PACKET_LOSS"    : 0.05,
        "MAX_RETX"       : 2,
        "IDLE_ENERGY"    : 5.0e-5,
        "SENSING_ENERGY" : 2.0e-5,

        # (Drawback 4) Realistic solar harvesting ------------------------------
        #   SOLAR_MODEL  : "clearsky" (old smooth half-sine) or "realistic"
        #                  (adds seasonal drift + stochastic clouds + noise).
        #   CLOUD_PROB   : chance a given round is (partly) clouded.
        #   CLOUD_MIN    : worst-case fraction of sun that gets through a cloud.
        #   SEASON_AMP   : +/- amplitude of the yearly seasonal envelope.
        #   SOLAR_TRACE  : optional path to a CSV of measured harvest rates;
        #                  when set it overrides the analytic model entirely.
        "SOLAR_MODEL"    : "realistic",
        "CLOUD_PROB"     : 0.30,
        "CLOUD_MIN"      : 0.15,
        "SEASON_AMP"     : 0.30,
        "SOLAR_TRACE"    : None,

        # (Drawback 5) Solar-awareness ablation --------------------------------
        #   USE_SOLAR_TERM: when False the solar weighting is removed from BOTH
        #                   the GA fitness AND the MS-CH score, so you can
        #                   measure exactly how much the "solar-aware" idea
        #                   actually contributes.
        "USE_SOLAR_TERM" : True,

        # (Drawback 1) Monte-Carlo repetition ----------------------------------
        #   MC_SEEDS      : how many independent random seeds to average over.
        "MC_SEEDS"       : 15,
    }
    cfg.update(overrides)
    return cfg


def get_num_chs(alive_count: int, cfg: dict) -> int:
    """
    Dynamic CH count - recalculated EVERY round (Point 2).
        alive=50, CH_PERCENT=0.10  -> num_chs = 5
        alive=8 , CH_PERCENT=0.10  -> num_chs = 1  (minimum enforced)
        alive=0                    -> num_chs = 0  (network dead)
    Capped at `alive_count` so we can never elect more CHs than nodes alive.
    """
    if alive_count <= 0:
        return 0
    return max(1, min(alive_count, round(alive_count * cfg["CH_PERCENT"])))


def get_num_ms(num_relay_chs: int, cfg: dict) -> int:
    """
    Dynamic MS-CH count (Point 1).

    One MS-CH is elected per RELAYS_PER_MS relay CHs.
    Returns 0 when no CH needs a relay (Points 3 + 4):
      * Point 3: MS-CH is only applied if it is actually needed.
      * Point 4: a CH that can transmit directly to BS never reaches the
                 MS-CH stage, so if every CH chose PATH A,
                 num_relay_chs == 0  -> num_ms == 0  -> MS step skipped.

        relay=0   -> 0   (skip MS-CH entirely)
        relay=3   -> 1
        relay=8, RELAYS_PER_MS=4  -> 2
    Capped at `num_relay_chs` so an MS-CH always has someone to serve.
    """
    if num_relay_chs <= 0:
        return 0
    return max(1, min(num_relay_chs,
                      math.ceil(num_relay_chs / cfg["RELAYS_PER_MS"])))


# ==============================================================================
# SECTION 2 - RADIO & ENERGY CONSTANTS  (Reference [6])
# ==============================================================================

E_ELEC         = 50e-9        # J/bit
E_AMP          = 100e-12      # J/bit/m^2
E_MP           = 0.0013e-12   # J/bit/m^4
E_DA           = 5e-9         # J/bit
BATTERY_MAX    = 2.0          # J - hard battery cap
COMM_RANGE_PCT = 0.4
D0             = math.sqrt(E_AMP / E_MP)   # ~87 m crossover distance
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
    """
    One sensor node. Roles: "sensor" | "CH" | "MS-CH"
    All nodes are identical hardware - role flips each round.
    """
    __slots__ = ("id", "x", "y", "energy", "alive", "role",
                 "assigned_ch", "assigned_ms", "goes_direct",
                 "packets_sent", "_cfg", "_dist_bs", "solar_eff",
                 # v2 realism counters
                 "delivered", "packets_delivered")

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
        # Cache distance to BS - it never changes
        self._dist_bs     = math.hypot(cfg["BS_X"] - x, cfg["BS_Y"] - y)
        # Fixed per-node harvesting efficiency (panel orientation / shading).
        self.solar_eff    = solar_eff
        # v2: outcome of the most recent transmit() (was the packet delivered?)
        # and a lifetime counter of successfully delivered packets. Used to
        # compute the Packet Delivery Ratio (PDR) - impossible in the old
        # loss-free model where every packet always arrived.
        self.delivered         = True
        self.packets_delivered = 0

    # ---- Geometry ------------------------------------------------------------
    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    @property
    def distance_to_bs(self) -> float:
        return self._dist_bs

    # ---- Radio energy model (Reference [6]) ----------------------------------
    def _tx_cost(self, bits: int, d: float) -> float:
        if d <= D0:
            return E_ELEC * bits + E_AMP * bits * d * d
        d2 = d * d
        return E_ELEC * bits + E_MP * bits * d2 * d2

    def _rx_cost(self, bits: int) -> float:
        return E_ELEC * bits

    def _agg_cost(self, bits: int, n: int) -> float:
        return E_DA * bits * n

    # ---- Energy operations ---------------------------------------------------
    def transmit(self, to_x: float, to_y: float) -> bool:
        """
        v2 realistic transmit with an ARQ (automatic-repeat-request) layer.

        In the old model a transmission always succeeded and cost energy once.
        Real radios drop packets (collisions, fading) and retransmit, burning
        extra energy. Here:
          * p_drop grows with distance (longer links are lossier),
          * on a drop we retry up to MAX_RETX times,
          * EVERY attempt (including retries) costs full TX energy,
          * self.delivered records whether the packet finally got through.
        With PACKET_LOSS == 0 this reduces exactly to the original behaviour.
        """
        bits = self._cfg["PACKET_SIZE"]
        d    = math.hypot(self.x - to_x, self.y - to_y)
        loss = float(self._cfg.get("PACKET_LOSS", 0.0))

        if loss <= 0.0:
            # ---- original loss-free path (backward compatible) --------------
            self.energy       -= self._tx_cost(bits, d)
            self.packets_sent += 1
            self.delivered     = True
            if self.energy <= 0:
                self.energy = 0
                self.alive  = False
            else:
                self.packets_delivered += 1
            return self.alive

        # ---- realistic lossy path with retransmissions ----------------------
        d0     = max(float(self._cfg.get("DIRECT_DIST", 55.0)), 1e-9)
        p_drop = min(0.95, loss * (1.0 + 0.5 * d / d0))   # distance-scaled
        max_retx = int(self._cfg.get("MAX_RETX", 0))
        self.packets_sent += 1
        self.delivered = False
        for _attempt in range(max_retx + 1):
            self.energy -= self._tx_cost(bits, d)
            if self.energy <= 0:
                self.energy = 0
                self.alive  = False
                break
            if random.random() >= p_drop:      # this attempt got through
                self.delivered = True
                break
        if self.delivered:
            self.packets_delivered += 1
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

    # ---- Solar harvesting (Reference [2]) ------------------------------------
    def harvest_solar(self, solar_rate_now: float) -> None:
        """Caller passes the network-wide daylight rate for this round; the
        node's own panel efficiency (solar_eff) scales it, so a shaded or
        badly-oriented node genuinely harvests less than a well-placed one."""
        if solar_rate_now <= 0:
            return
        rate    = solar_rate_now * self.solar_eff
        harvest = max(0.0, rate + random.gauss(0, rate * 0.05))
        self.energy = min(self.energy + harvest, BATTERY_MAX)

    # ---- Role reset ----------------------------------------------------------
    def reset_role(self) -> None:
        self.role        = "sensor"
        self.assigned_ch = None
        self.assigned_ms = None
        self.goes_direct = False

    # ---- v2: always-on energy drain (idle listening + sensing) ---------------
    def idle_sense_drain(self) -> None:
        """
        Every alive node spends energy each round just being ON - the radio
        idle-listens and the sensor/CPU takes a reading - regardless of whether
        it transmits. The original model ignored this, which flattered lifetime.
        This is a small but constant tax that also makes solar harvesting
        genuinely matter (a node must out-harvest its idle cost to survive).
        """
        drain = (float(self._cfg.get("IDLE_ENERGY", 0.0))
                 + float(self._cfg.get("SENSING_ENERGY", 0.0)))
        if drain <= 0.0:
            return
        self.energy -= drain
        if self.energy <= 0:
            self.energy = 0
            self.alive  = False

    @property
    def energy_fraction(self) -> float:
        return self.energy / self._cfg["E_INITIAL"]

    def __repr__(self) -> str:
        return (f"Node({self.id}, {self.role}, "
                f"{self.energy:.4f}J, alive={self.alive})")


def solar_rate_for_round(round_num: int, max_harvest: float) -> float:
    """
    Pre-computable once per round (same for every node).

    24-hour day with daylight from 06:00 to 18:00 and PEAK AT NOON (12:00):
        hour  0  -> 0.000   midnight  - no sun
        hour  6  -> 0.000   sunrise   - just starting
        hour 12  -> 1.000   noon      - peak harvest
        hour 18  -> 0.000   sunset    - done
        hour 18-24 clamped to 0 (negative sine -> max(0, ...))
    """
    hour = round_num % 24
    return max_harvest * max(0.0, math.sin(math.pi * (hour - 6) / 12))


def load_solar_trace(path: str) -> List[float]:
    """
    Load a measured solar-harvest trace from a CSV/text file: one number per
    line (or a single comma-separated line), interpreted as the harvest rate
    (J/round) per round. Lets you drive the simulation from REAL irradiance
    data instead of any analytic model. Returns [] on failure.
    """
    vals: List[float] = []
    try:
        with open(path, "r") as fh:
            for line in fh:
                for tok in line.replace(",", " ").split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
    except OSError:
        return []
    return vals


def actual_solar_rate(round_num: int, cfg: dict) -> float:
    """
    v2 REALISTIC solar model - what a panel ACTUALLY harvests this round.

    The original model was a perfectly smooth, repeating half-sine: every day
    identical, never a cloud. Real harvesting is messier, so on top of the
    clear-sky curve we layer:

      1. Seasonal envelope  - a slow yearly sine (SEASON_AMP) so summer days
                              out-harvest winter days.
      2. Stochastic clouds  - with probability CLOUD_PROB the round is (partly)
                              clouded, attenuating output down to as little as
                              CLOUD_MIN of clear-sky.
      3. Measurement noise  - small Gaussian jitter.

    If SOLAR_TRACE points at a data file, that measured series is used verbatim
    (cycled) and the analytic model is bypassed entirely.

    IMPORTANT design choice: this "actual" value is used for HARVESTING only.
    The GA / PSO / GWO still PLAN against the clear-sky forecast
    (solar_rate_for_round), exactly like a real controller that has a forecast
    but not tomorrow's weather. Modelling that forecast-vs-reality gap is itself
    part of the added realism.
    """
    # Cached trace path -> list (loaded once).
    trace = cfg.get("_solar_trace_cache")
    if trace is None and cfg.get("SOLAR_TRACE"):
        trace = load_solar_trace(cfg["SOLAR_TRACE"])
        cfg["_solar_trace_cache"] = trace if trace else []
        trace = cfg["_solar_trace_cache"]
    if trace:
        return max(0.0, float(trace[round_num % len(trace)]))

    base = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])
    if base <= 0.0 or cfg.get("SOLAR_MODEL", "clearsky") != "realistic":
        return base

    day    = round_num // 24
    season = 1.0 + cfg.get("SEASON_AMP", 0.0) * math.sin(2 * math.pi * day / 365.0)

    cloud = 1.0
    if random.random() < cfg.get("CLOUD_PROB", 0.0):
        cloud = random.uniform(cfg.get("CLOUD_MIN", 0.15), 0.85)

    noise = random.gauss(1.0, 0.05)
    return max(0.0, base * season * cloud * noise)


# ==============================================================================
# SECTION 4 - WORLD STATE  (vectorized helpers for big networks)
# ==============================================================================

@dataclass
class World:
    """
    Pre-computed numpy views over alive nodes for the current round.
    Rebuilt at the start of each round - O(N) once, then everything else
    is vectorized.
    """
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
        # id -> position in alive_xy / alive_e / alive_dbs (O(1) lookup,
        # used by the batched GA fitness evaluator)
        self.id_to_idx = {int(nid): i for i, nid in enumerate(self.alive_idx)}


def vectorized_assign(sensor_xy: np.ndarray,
                      ch_xy: np.ndarray) -> np.ndarray:
    """
    Vectorized nearest-CH lookup. O(S * C) but in pure NumPy.
    Returns array of CH indices (into ch_xy) for each sensor.
    """
    if ch_xy.shape[0] == 0 or sensor_xy.shape[0] == 0:
        return np.empty(sensor_xy.shape[0], dtype=np.int64)
    # Broadcasted squared distance
    diff = sensor_xy[:, None, :] - ch_xy[None, :, :]
    d2   = np.einsum("ijk,ijk->ij", diff, diff)
    return d2.argmin(axis=1)


# ==============================================================================
# SECTION 5 - GENETIC ALGORITHM  (Reference [1][3][4])
# ==============================================================================

class Chromosome:
    __slots__ = ("genes", "fitness")

    def __init__(self, genes):
        self.genes   = list(genes)
        self.fitness = -1.0    # -1 sentinel = not yet computed

    def copy(self) -> "Chromosome":
        c = Chromosome(self.genes)
        c.fitness = self.fitness
        return c


def _evaluate_population(pop: List["Chromosome"],
                        world: World,
                        cfg: dict,
                        solar_now: float,
                        num_chs: int) -> None:
    """
    Batched fitness evaluation - evaluates ALL chromosomes in the
    population in one vectorized NumPy pass.

    For population P, alive S, CHs K:
        per-call cost   ~  O(P * S * K)  in C-level NumPy
        per-call memory ~  P * S * K * 8 bytes (worst case)

    On big networks (P=30, S=2000, K=50) that's 24 MB and runs in ~0.05 s.
    For very large jobs (S * P * K > 4M elements) we automatically fall
    back to per-chromosome evaluation to keep memory bounded.

    Skips chromosomes whose fitness has already been cached (>= 0).

    Weights:  25% energy + 25% solar + 30% coverage + 20% spread.
    """
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

    # Build (P, K) array of indices into alive_xy / alive_e for each chromo.
    # If any gene refers to a now-dead node (shouldn't happen but be safe),
    # mark that chromosome invalid.
    gene_idx = np.empty((P, K), dtype=np.int64)
    valid    = np.ones(P, dtype=bool)
    for p, c in enumerate(pending):
        try:
            gene_idx[p] = [id_map[g] for g in c.genes]
        except (KeyError, ValueError):
            valid[p] = False
            gene_idx[p] = 0   # placeholder, fitness forced to 0 below

    ch_xy  = alive_xy[gene_idx]                       # (P, K, 2)
    ch_e   = alive_e[gene_idx]                        # (P, K)

    # 1. residual energy ------------------------------------------------------
    e_score = np.minimum(ch_e.sum(axis=1) / (K * cfg["E_INITIAL"]), 1.0)

    # 2. solar score - the TRUE solar-aware score (Bug 3 fix) ---------------
    #    A high score means: this CH has good battery AND is currently
    #    harvesting sun. At night solar_fraction == 0 so the score reduces to
    #    half the energy fraction; at noon a charged-and-charging CH scores
    #    near 1.0. This is what differentiates 9am from 9pm even when the
    #    battery is identical.
    if cfg["MAX_HARVEST"] > 0 and cfg.get("USE_SOLAR_TERM", True):
        daylight = solar_now / cfg["MAX_HARVEST"]            # 0.0 night, 1.0 noon
        # NODE-SPECIFIC solar: scale the shared daylight level by the average
        # panel efficiency of THIS chromosome's CHs. Because solar_eff varies
        # per node, this term now differs between candidate CH sets and can
        # actually change which chromosome wins (previously it was a constant
        # offset that had no effect on ranking).
        ch_seff         = alive_seff[gene_idx]               # (P, K)
        solar_fraction  = daylight * ch_seff.mean(axis=1)    # per-chromosome
        energy_fraction = np.minimum(ch_e.mean(axis=1) / cfg["E_INITIAL"], 1.0)
        s_score = 0.5 * solar_fraction + 0.5 * energy_fraction
    else:
        s_score = np.full(P, 0.5)

    # 3. coverage -------------------------------------------------------------
    comm_range2 = (cfg["FIELD"] * COMM_RANGE_PCT) ** 2
    big_alloc   = P * S * K
    if big_alloc <= 4_000_000:
        # diff: (P, S, K, 2) -> reduce to (P, S, K) -> min over K -> (P, S)
        diff   = alive_xy[None, :, None, :] - ch_xy[:, None, :, :]
        d2     = (diff * diff).sum(axis=-1)
        d2_min = d2.min(axis=2)
        # mask out CHs themselves so they don't count as "covered sensors"
        is_ch = np.zeros((P, S), dtype=bool)
        rows  = np.arange(P)[:, None]
        is_ch[rows, gene_idx] = True
        sensor_mask = ~is_ch
        within = (d2_min <= comm_range2) & sensor_mask
        denom  = np.maximum(sensor_mask.sum(axis=1), 1)
        c_score = within.sum(axis=1) / denom
    else:
        # Fallback per-chromosome (memory-safe for very big networks)
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

    # 4. spatial spread -------------------------------------------------------
    if K > 1:
        centroid = ch_xy.mean(axis=1, keepdims=True)
        spread   = np.sqrt(((ch_xy - centroid) ** 2).sum(axis=2)).mean(axis=1)
        sp_score = np.minimum(spread / cfg["FIELD"], 1.0)
    else:
        sp_score = np.full(P, 0.5)

    fitness = (0.25 * e_score + 0.25 * s_score
               + 0.30 * c_score + 0.20 * sp_score)
    fitness = np.where(valid, fitness, 0.0)

    for p, c in enumerate(pending):
        c.fitness = float(fitness[p])


def _smart_initial_population(alive_ids: List[int],
                              alive_energies: List[float],
                              num_chs: int,
                              pop_size: int) -> List[Chromosome]:
    """
    Half of the population: random.
    Half: energy-weighted sampling (higher-energy nodes more likely picked).
    Faster GA convergence on big networks.
    """
    pop: List[Chromosome] = []
    if not alive_ids or num_chs > len(alive_ids):
        return pop
    energies = np.asarray(alive_energies, dtype=np.float64)
    # avoid all-zero
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
    """
    GA Job 1 - elect CHs.
    Optimizations vs original:
      * smart energy-weighted initialization
      * elitism (top-2 carried over)
      * vectorized fitness
      * adaptive mutation: doubles when fitness plateaus
      * early-stop when best unchanged for ELITE_PATIENCE generations
    """
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
        # Batched fitness eval - whole population in one numpy pass.
        # Cached chromosomes (fitness >= 0) are skipped automatically.
        _evaluate_population(pop, world, cfg, solar_now, num_chs)
        pop.sort(key=lambda c: c.fitness, reverse=True)

        if pop[0].fitness > best_score + 1e-9:
            best_score   = pop[0].fitness
            best_overall = pop[0].copy()
            stale = 0
            cur_mut = cfg["GA_MUT"]
        else:
            stale += 1
            # Adaptive: bump mutation when stuck
            cur_mut = min(0.5, cfg["GA_MUT"] * (1 + stale * 0.25))

        if _CONV_SINK is not None:            # v2 convergence recording
            _CONV_SINK.append(best_score)

        if stale >= PATIENCE:
            break

        # Elitism: top-2 directly into next pop
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
# SECTION 6 - PATH DECISION  (runs BEFORE MS-CH election)
# ==============================================================================

def decide_ch_paths(ch_nodes: List[Node],
                    cfg: dict) -> Tuple[List[Node], List[Node]]:
    """
    Each CH independently picks PATH A or PATH B.
    PATH A direct  : distance to BS <= DIRECT_DIST AND energy >= DIRECT_NRG
    PATH B relay   : otherwise

    NOTE: This runs BEFORE MS-CH election. CHs that CAN go direct will
    NEVER reach the MS-CH election stage (Point 4).
    """
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
            ch.assigned_ms = None     # filled in after MS-CH election
            relay_chs.append(ch)
    return direct_chs, relay_chs


# ==============================================================================
# SECTION 7 - MS-CH ELECTION  (THE SOLAR-AWARE NOVELTY)
# Reference [2][4]
# ==============================================================================

def _solar_aware_score(ch: Node,
                       peers: List[Node],
                       cfg: dict,
                       solar_now: float) -> float:
    """
    Per-CH MS-CH suitability score. Higher = better.
        35% battery level
        30% solar rate right now
        20% centrality among peers (relay CHs)
        15% closeness to BS
    """
    bat = min(ch.energy_fraction, 1.0)
    if cfg["MAX_HARVEST"] > 0 and cfg.get("USE_SOLAR_TERM", True):
        # Node-specific: shared daylight level scaled by THIS candidate's own
        # panel efficiency, so the solar term genuinely differentiates the
        # relay CHs competing to become the MS-CH.
        # v2: when USE_SOLAR_TERM is False this drops to a neutral constant,
        # ablating solar-awareness from MS-CH election too.
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
    """
    Lightweight k-medoids on relay CHs. Returns k clusters.
    Used to place MS-CHs across spatially-distinct relay zones (Point 1).
    """
    if k <= 1 or len(relay_chs) <= k:
        if k <= 1:
            return [relay_chs]
        return [[ch] for ch in relay_chs]

    xy = np.array([(c.x, c.y) for c in relay_chs], dtype=np.float64)
    n  = len(relay_chs)
    # Seed: farthest-first traversal (BUILD step of PAM, simplified)
    medoids = [random.randrange(n)]
    for _ in range(k - 1):
        d2 = np.min(((xy[:, None, :] - xy[medoids][None, :, :]) ** 2)
                    .sum(axis=2), axis=1)
        medoids.append(int(d2.argmax()))

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        # assign
        diff = xy[:, None, :] - xy[medoids][None, :, :]
        d2   = np.einsum("ijk,ijk->ij", diff, diff)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        # update medoid per cluster (point closest to cluster centroid)
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
    return [c for c in clusters if c]   # drop empty clusters


def elect_ms_chs(relay_chs: List[Node],
                 cfg: dict,
                 solar_now: float,
                 num_ms: int) -> List[Node]:
    """
    GA Job 3 - elect num_ms MS-CHs from the relay CH pool.

    Steps:
      1. If num_ms == 0  -> return [] (Point 3)
      2. If num_ms == 1  -> pick best by solar-aware score
      3. Else            -> k-medoids partition, best per cluster

    Each chosen MS-CH then collects only from the relay CHs in its cluster.
    """
    if num_ms <= 0 or not relay_chs:
        return []

    if num_ms == 1 or len(relay_chs) <= 2:
        best = max(relay_chs,
                   key=lambda c: _solar_aware_score(c, relay_chs, cfg, solar_now))
        best.role = "MS-CH"
        # all relay CHs report to this one
        for r in relay_chs:
            if r.id != best.id:
                r.assigned_ms = best.id
        return [best]

    # multiple MS-CHs - cluster relay CHs spatially first
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
    """
    Step 1  Solar harvest (vectorized rate)
    Step 2  Reset all roles
    Step 3  Refresh world state, compute dynamic num_chs
    Step 4  GA Job 1 -> elect CHs
    Step 5  GA Job 2 -> path decision (BEFORE MS-CH)
    Step 6  Compute dynamic num_ms, GA Job 3 -> elect MS-CHs (only if needed)
    Step 7  Assign sensors to nearest CH (vectorized)
    Step 8  Sensors transmit
    Step 9  CHs aggregate + forward
    Step 10 Re-elect MS-CH(s) if any battery critical
    Step 11 MS-CH(s) transmit to BS
    Step 12 Record stats
    """
    solar_now    = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])  # forecast
    solar_actual = actual_solar_rate(round_num, cfg)                    # realised

    # Step 1 - realistic harvest + always-on idle/sensing drain (v2)
    for n in nodes:
        if n.alive:
            if solar_actual > 0:
                n.harvest_solar(solar_actual)
            n.idle_sense_drain()

    # Step 2
    for n in nodes:
        if n.alive:
            n.reset_role()

    # Step 3
    world.refresh()
    alive_count = world.alive_idx.size
    num_chs = get_num_chs(alive_count, cfg)

    if alive_count < num_chs + 1:
        _record_stats(nodes, stats)
        stats["ch_counts"].append(0)
        stats["ms_counts"].append(0)
        return False

    # Step 4
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

    # Step 5 - path decision FIRST (Point 4)
    direct_chs, relay_chs = decide_ch_paths(ch_nodes, cfg)

    # Step 6 - MS-CH election only if needed (Points 1, 3)
    num_ms = get_num_ms(len(relay_chs), cfg)
    ms_chs = elect_ms_chs(relay_chs, cfg, solar_now, num_ms) if num_ms else []

    # Step 7 - sensor -> nearest CH (vectorized)
    if ch_nodes:
        ch_xy = np.array([(c.x, c.y) for c in ch_nodes], dtype=np.float64)
        ch_id_arr = np.array([c.id for c in ch_nodes], dtype=np.int64)
        # Sensors = alive non-CH
        ch_id_set = {c.id for c in ch_nodes}
        sensor_mask = np.array([nid not in ch_id_set
                                for nid in world.alive_idx], dtype=bool)
        sensor_xy  = world.alive_xy[sensor_mask]
        sensor_ids = world.alive_idx[sensor_mask]
        if sensor_xy.shape[0] > 0:
            assign_idx = vectorized_assign(sensor_xy, ch_xy)
            for sid, ai in zip(sensor_ids.tolist(), assign_idx.tolist()):
                nodes[sid].assigned_ch = int(ch_id_arr[ai])

    # Step 8 - sensors transmit (receiver only spends RX if the packet arrived)
    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch is not None:
            ch = nodes[n.assigned_ch]
            if ch.alive:
                n.transmit(ch.x, ch.y)
                if n.delivered:
                    ch.receive()

    # Step 9 - CHs aggregate + forward
    # Build member counts
    member_count: Dict[int, int] = {c.id: 0 for c in ch_nodes}
    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch in member_count:
            member_count[n.assigned_ch] += 1

    # MS-CH inbound count (relay CHs that survive the aggregate)
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
                if ch.delivered:            # relay hop only counts if it arrived
                    ms.receive()
                    ms_inbound[ms.id] += 1

    # Step 10 - re-elect any critically depleted MS-CH
    for ms in list(ms_chs):
        if ms.energy < MS_REELECT_THR * cfg["E_INITIAL"]:
            cluster = [c for c in relay_chs
                       if c.alive and c.id != ms.id and c.assigned_ms == ms.id]
            if cluster:
                ms.role = "CH"
                new_ms = max(cluster, key=lambda c:
                             _solar_aware_score(c, cluster, cfg, solar_now))
                new_ms.role = "MS-CH"
                # reroute cluster
                for r in cluster:
                    if r.id != new_ms.id:
                        r.assigned_ms = new_ms.id
                ms_inbound[new_ms.id] = ms_inbound.pop(ms.id, 0)
                ms_chs[ms_chs.index(ms)] = new_ms
                stats["reelections"] += 1

    # Step 11 - MS-CHs transmit to BS
    # An MS-CH was originally a relay CH, so it ALSO collected packets from
    # its own sensor members (step 8). Aggregate both streams before TX.
    for ms in ms_chs:
        if ms.alive and ms.role == "MS-CH":
            own_members = member_count.get(ms.id, 0)
            inbound     = ms_inbound.get(ms.id, 0)
            ms.aggregate(own_members + inbound)
            if ms.alive:
                ms.transmit(cfg["BS_X"], cfg["BS_Y"])
                if ms.delivered:                 # count only successful delivery
                    stats["packets_to_bs"] += 1

    for ch in direct_chs:
        if ch.alive and ch.delivered:            # delivered set during Step 9 TX
            stats["packets_to_bs"] += 1

    # Step 12
    _record_stats(nodes, stats)
    stats["ch_counts"].append(len(ch_nodes))
    stats["ms_counts"].append(len(ms_chs))

    # Topology snapshot (every SNAPSHOT_EVERY rounds) - illustrates the
    # protocol live: title says MS-USED or MS-SKIPPED so the user can
    # immediately see when the MS-CH stage was bypassed (Points 3 + 4).
    snap = cfg.get("SNAPSHOT_EVERY", 0)
    if snap and round_num % snap == 0:
        try:
            plot_topology_snapshot(nodes, ch_nodes, direct_chs, relay_chs,
                                   ms_chs, round_num, cfg, protocol="GA")
        except Exception as e:
            print(f"  !  Snapshot failed at round {round_num}: {e}")
    return any(n.alive for n in nodes)


# ==============================================================================
# SECTION 9 - LEACH BASELINE  (Reference [5])
# ==============================================================================

def simulate_round_leach(nodes: List[Node],
                         world: World,
                         round_num: int,
                         cfg: dict,
                         stats: dict) -> bool:
    """
    LEACH - same solar model and dynamic CH count for fair comparison.
    No MS-CH; every CH ships direct to BS.
    """
    solar_now    = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])  # forecast
    solar_actual = actual_solar_rate(round_num, cfg)                    # realised
    # v2: realistic harvest + always-on idle/sensing drain (same physics as GA
    # so the LEACH comparison stays fair).
    for n in nodes:
        if n.alive:
            if solar_actual > 0:
                n.harvest_solar(solar_actual)
            n.idle_sense_drain()
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

    elected_ids = random.sample(world.alive_idx.tolist(),
                                min(num_chs, alive_count))
    elected = [nodes[i] for i in elected_ids]
    for n in elected:
        n.role = "CH"

    # Vectorized sensor assignment
    ch_xy = np.array([(c.x, c.y) for c in elected], dtype=np.float64)
    ch_id_arr = np.array([c.id for c in elected], dtype=np.int64)
    ch_id_set = set(elected_ids)
    sensor_mask = np.array([nid not in ch_id_set
                            for nid in world.alive_idx], dtype=bool)
    sensor_xy  = world.alive_xy[sensor_mask]
    sensor_ids = world.alive_idx[sensor_mask]
    if sensor_xy.shape[0] > 0:
        assign_idx = vectorized_assign(sensor_xy, ch_xy)
        for sid, ai in zip(sensor_ids.tolist(), assign_idx.tolist()):
            nodes[sid].assigned_ch = int(ch_id_arr[ai])

    # Sensors transmit (receiver spends RX only if the packet arrived)
    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch is not None:
            ch = nodes[n.assigned_ch]
            if ch.alive:
                n.transmit(ch.x, ch.y)
                if n.delivered:
                    ch.receive()

    # CHs aggregate + send direct
    member_count = {c.id: 0 for c in elected}
    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch in member_count:
            member_count[n.assigned_ch] += 1

    for ch in elected:
        if not ch.alive:
            continue
        ch.aggregate(member_count[ch.id])
        if ch.alive:
            ch.transmit(cfg["BS_X"], cfg["BS_Y"])
            if ch.delivered:                     # count only successful delivery
                stats["packets_to_bs"] += 1

    _record_stats(nodes, stats)
    stats["ch_counts"].append(num_chs)
    stats["ms_counts"].append(0)

    # LEACH snapshot - never has MS-CH (single-tier baseline)
    snap = cfg.get("SNAPSHOT_EVERY", 0)
    if snap and round_num % snap == 0:
        try:
            plot_topology_snapshot(nodes, elected, elected, [], [],
                                   round_num, cfg, protocol="LEACH")
        except Exception as e:
            print(f"  !  Snapshot failed at round {round_num}: {e}")
    return any(n.alive for n in nodes)


# ==============================================================================
# SECTION 10 - STATS
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
# SECTION 11 - SIMULATION RUNNER
# ==============================================================================

def create_nodes(cfg: dict, seed: int = 42) -> List[Node]:
    # v2: the seed is now a parameter (was hard-wired to 42). This is what
    # makes multi-seed Monte-Carlo runs possible - each repetition lays out a
    # different-but-reproducible field so results can be averaged with error
    # bars instead of resting on a single lucky/unlucky layout.
    random.seed(seed)
    np.random.seed(seed)
    nodes = [Node(i,
                  random.uniform(0, cfg["FIELD"]),
                  random.uniform(0, cfg["FIELD"]),
                  cfg)
             for i in range(cfg["NUM_NODES"])]
    # Assign each node a fixed solar-harvesting efficiency. Drawn AFTER all
    # positions so the field topology is byte-for-byte identical to before;
    # this heterogeneity is what makes the solar-aware selection meaningful.
    for n in nodes:
        n.solar_eff = random.uniform(SOLAR_EFF_MIN, SOLAR_EFF_MAX)
    return nodes


def run_simulation(cfg: dict, protocol: str = "GA"):
    nodes = create_nodes(cfg)
    world = World(nodes=nodes)
    stats = make_stats()
    init_chs = max(1, round(cfg["NUM_NODES"] * cfg["CH_PERCENT"]))
    print(f"\n{'=' * 58}")
    print(f"  Protocol  : {protocol}")
    print(f"  Nodes     : {cfg['NUM_NODES']}   "
          f"Field : {cfg['FIELD']}x{cfg['FIELD']}m")
    print(f"  Rounds    : {cfg['NUM_ROUNDS']}  "
          f"Initial CHs: {init_chs} "
          f"({cfg['CH_PERCENT'] * 100:.0f}% - dynamic)")
    if protocol == "GA":
        print(f"  MS-CH     : 1 per {cfg['RELAYS_PER_MS']} relay CHs (dynamic, "
              f"0 if no relays)")
    if cfg.get("SNAPSHOT_EVERY", 0):
        print(f"  Snapshots : every {cfg['SNAPSHOT_EVERY']} rounds -> "
              f"./{SNAPSHOT_DIR}/")
    print(f"{'=' * 58}")

    first_dead   = None
    network_dead = cfg["NUM_ROUNDS"]

    for r in range(cfg["NUM_ROUNDS"]):
        if protocol == "GA":
            alive = simulate_round_ga(nodes, world, r, cfg, stats)
        else:
            alive = simulate_round_leach(nodes, world, r, cfg, stats)

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
            extra = f"| MS: {nms}" if protocol == "GA" else ""
            print(f"  Round {r:4d} | Alive: {a:3d}/{cfg['NUM_NODES']} "
                  f"| CHs: {nchs} {extra} | Energy: {e:.4f} J")

    if network_dead == cfg["NUM_ROUNDS"]:
        print(f"  v  Network survived all {cfg['NUM_ROUNDS']} rounds!")
    print(f"  Packets to BS     : {stats['packets_to_bs']}")
    if protocol == "GA":
        print(f"  MS-CH re-elections: {stats['reelections']}")
    print(f"{'=' * 58}")

    return stats, first_dead, network_dead, nodes


# ==============================================================================
# SECTION 12 - PLOTS
# ==============================================================================

def plot_results(ga_stats, leach_stats, ga_fd, leach_fd, cfg) -> None:
    """6 graphs: lifetime, energy, deaths, balance, dynamic CH/MS, solar."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        "Solar-Aware GA Multi-Sink Protocol  vs  LEACH\n"
        f"(Nodes={cfg['NUM_NODES']}, Field={cfg['FIELD']}m, "
        f"Rounds={cfg['NUM_ROUNDS']}, "
        f"CH%={cfg['CH_PERCENT'] * 100:.0f}% dynamic, "
        f"1 MS-CH per {cfg['RELAYS_PER_MS']} relay CHs)",
        fontsize=12, fontweight="bold")

    C_GA, C_LEACH, C_MS = "#1A5FAD", "#C0392B", "#27AE60"
    rg = range(len(ga_stats["alive_nodes"]))
    rl = range(len(leach_stats["alive_nodes"]))

    # 1. Alive
    ax = axes[0, 0]
    ax.plot(rg, ga_stats["alive_nodes"],    color=C_GA,    lw=2, label="GA")
    ax.plot(rl, leach_stats["alive_nodes"], color=C_LEACH, lw=2,
            ls="--", label="LEACH")
    if ga_fd is not None:
        ax.axvline(ga_fd, color=C_GA, ls=":", alpha=0.6,
                   label=f"GA 1st death r{ga_fd}")
    if leach_fd is not None:
        ax.axvline(leach_fd, color=C_LEACH, ls=":", alpha=0.6,
                   label=f"LEACH 1st death r{leach_fd}")
    ax.set(xlabel="Round", ylabel="Alive nodes",
           title="Network Lifetime",
           ylim=(0, cfg["NUM_NODES"] + 2))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Total residual energy
    ax = axes[0, 1]
    ax.plot(rg, ga_stats["total_energy"],    color=C_GA,    lw=2, label="GA")
    ax.plot(rl, leach_stats["total_energy"], color=C_LEACH, lw=2,
            ls="--", label="LEACH")
    ax.set(xlabel="Round", ylabel="Total residual energy (J)",
           title="Total Residual Energy")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Dead nodes
    ax = axes[0, 2]
    ax.plot(rg, ga_stats["dead_nodes"],    color=C_GA,    lw=2, label="GA")
    ax.plot(rl, leach_stats["dead_nodes"], color=C_LEACH, lw=2,
            ls="--", label="LEACH")
    ax.set(xlabel="Round", ylabel="Cumulative dead nodes",
           title="Node Deaths Over Time")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Energy balance
    ax = axes[1, 0]
    ax.plot(rg, ga_stats["energy_stddev"],    color=C_GA,    lw=2, label="GA")
    ax.plot(rl, leach_stats["energy_stddev"], color=C_LEACH, lw=2,
            ls="--", label="LEACH")
    ax.set(xlabel="Round", ylabel="Std deviation of energy (J)",
           title="Energy Balance\n(lower = more balanced)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Dynamic CH + MS counts (the new picture)
    ax = axes[1, 1]
    if ga_stats["ch_counts"]:
        ax.plot(range(len(ga_stats["ch_counts"])),
                ga_stats["ch_counts"], color=C_GA, lw=2, label="GA CHs")
    if ga_stats["ms_counts"]:
        ax.plot(range(len(ga_stats["ms_counts"])),
                ga_stats["ms_counts"], color=C_MS, lw=2, label="GA MS-CHs")
    if leach_stats["ch_counts"]:
        ax.plot(range(len(leach_stats["ch_counts"])),
                leach_stats["ch_counts"], color=C_LEACH, lw=1.5,
                ls="--", label="LEACH CHs")
    ax.set(xlabel="Round", ylabel="Count",
           title=f"Dynamic CH & MS-CH Counts\n"
                 f"(both scale with alive nodes)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Solar harvest cycle
    ax = axes[1, 2]
    hours   = np.linspace(0, 48, 500)
    # Reuse the model so the plot can never drift from solar_rate_for_round.
    # round_num is integer in the model, but linspace gives floats - fine,
    # the formula is continuous in `hour`.
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
    plt.savefig("results_comparison.png", dpi=150, bbox_inches="tight")
    print("  Plot saved -> results_comparison.png")
    plt.close()


def plot_topology_snapshot(nodes: List["Node"],
                           ch_nodes: List["Node"],
                           direct_chs: List["Node"],
                           relay_chs: List["Node"],
                           ms_chs: List["Node"],
                           round_num: int,
                           cfg: dict,
                           protocol: str = "GA") -> None:
    """
    Live-state snapshot of the WSN at round `round_num`.

    Saved as `topology_snapshots/{protocol}_round_XXXX.png`.

    Title makes the routing mode obvious so the user can flip through
    snapshots and immediately see when MS-CH is in use vs skipped:
        * "MS-USED  (n MS-CHs)"     - at least one CH is relayed
        * "MS-SKIPPED (all direct)" - every CH could reach BS directly,
                                      MS-CH stage was bypassed entirely
                                      (Points 3 + 4 of the spec)

    Drawn:
        - sensor positions (alive vs dead)
        - sensor -> CH membership (faint lines)
        - PATH A: direct CH -> BS (blue dashed arrow)
        - PATH B: relay CH -> MS-CH (green arrow)
        - PATH C: MS-CH -> BS (red bold arrow)
        - communication-range halos around each CH
    """
    F, BSX, BSY = cfg["FIELD"], cfg["BS_X"], cfg["BS_Y"]
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # Classify nodes -----------------------------------------------------------
    alive_sensors_x: List[float] = []
    alive_sensors_y: List[float] = []
    dead_x: List[float] = []
    dead_y: List[float] = []
    sensor_to_ch: Dict[int, int] = {}
    ms_id_set = {m.id for m in ms_chs}
    ch_id_set = {c.id for c in ch_nodes}
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
    n_relay  = len(relay_chs) - len(ms_chs)   # relay CHs that REPORT TO an MS-CH
    n_ms     = len(ms_chs)
    ms_used  = n_ms > 0 and any(c.assigned_ms is not None for c in relay_chs)

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_facecolor("#f4f6f9")
    ax.set_xlim(-8, F + 8)
    ax.set_ylim(-8, max(BSY + 20, F + 20))

    # CH range halos
    comm_range = F * COMM_RANGE_PCT
    for c in ch_nodes:
        ax.add_patch(plt.Circle((c.x, c.y), comm_range,
                                color="gray", fill=False,
                                alpha=0.08, lw=0.7, ls="--"))

    # Sensor -> CH lines (faint)
    for sid, cid in sensor_to_ch.items():
        s, c = nodes[sid], nodes[cid]
        if c.alive:
            ax.plot([s.x, c.x], [s.y, c.y],
                    color="#AABCD4", lw=0.45, alpha=0.40, zorder=1)

    # PATH A: every direct CH -> BS (blue dashed)
    for c in direct_chs:
        if c.alive:
            ax.annotate("", xy=(BSX, BSY), xytext=(c.x, c.y),
                        arrowprops=dict(arrowstyle="->", color="#1A5FAD",
                                        lw=1.6, linestyle="dashed",
                                        alpha=0.85),
                        zorder=4)

    # PATH B: relay CH -> its MS-CH (green)
    for c in relay_chs:
        if c.alive and c.assigned_ms is not None and c.id not in ms_id_set:
            ms = nodes[c.assigned_ms]
            if ms.alive:
                ax.annotate("", xy=(ms.x, ms.y), xytext=(c.x, c.y),
                            arrowprops=dict(arrowstyle="->", color="#27AE60",
                                            lw=1.6, alpha=0.9),
                            zorder=4)

    # PATH C: MS-CH -> BS (red, bold)
    for m in ms_chs:
        if m.alive:
            ax.annotate("", xy=(BSX, BSY), xytext=(m.x, m.y),
                        arrowprops=dict(arrowstyle="->", color="#C0392B",
                                        lw=2.4, alpha=0.95),
                        zorder=5)

    # In LEACH (no MS-CH stage), every CH ships direct - draw all CH->BS
    if protocol != "GA":
        for c in ch_nodes:
            if c.alive and c not in direct_chs:
                ax.annotate("", xy=(BSX, BSY), xytext=(c.x, c.y),
                            arrowprops=dict(arrowstyle="->", color="#1A5FAD",
                                            lw=1.4, linestyle="dashed",
                                            alpha=0.7),
                            zorder=4)

    # Sensors (alive & dead)
    if alive_sensors_x:
        ax.scatter(alive_sensors_x, alive_sensors_y,
                   c="#7FB3D3", s=32, zorder=3, alpha=0.85,
                   edgecolors="white", lw=0.5)
    if dead_x:
        ax.scatter(dead_x, dead_y, c="#888", s=22, marker="x",
                   zorder=3, alpha=0.6, lw=1.0)

    # CHs
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

    # Base station
    ax.scatter(BSX, BSY, c="#C0392B", s=420, marker="s",
               zorder=7, edgecolors="darkred", lw=2)
    ax.text(BSX + 7, BSY + 3, "BS",
            fontsize=10, fontweight="bold", color="darkred")

    # Field outline
    ax.add_patch(plt.Rectangle((0, 0), F, F, fill=False,
                               edgecolor="#999", lw=1.2, ls="--", alpha=0.5))

    # Title with the killer indicator
    if protocol == "GA":
        if ms_used:
            mode_label = f"MS-USED  ({n_ms} MS-CH{'s' if n_ms != 1 else ''})"
            title_color = "#27AE60"
        else:
            mode_label = "MS-SKIPPED  (all CHs reach BS directly)"
            title_color = "#1A5FAD"
    else:
        mode_label = "LEACH baseline (no MS-CH)"
        title_color = "#C0392B"

    ax.set_title(
        f"{protocol}  Round {round_num:4d}   |   {mode_label}\n"
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
                         f"{protocol.lower()}_round_{round_num:04d}.png")
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

    # PATH A: direct CH -> BS
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

    # PATH B: relay CH -> matching MS-CH
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

    # PATH C: each MS-CH -> BS
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
    plt.savefig("topology.png", dpi=160, bbox_inches="tight")
    print("  Plot saved -> topology.png")
    plt.close()


# ==============================================================================
# SECTION 13 - SUMMARY
# ==============================================================================

def print_summary(ga_s, leach_s, ga_fd, leach_fd, ga_nd, leach_nd, cfg) -> None:
    print("\n" + "=" * 64)
    print("  SIMULATION RESULTS SUMMARY")
    print("=" * 64)
    print(f"  {'Metric':<40} {'GA':>10} {'LEACH':>10}")
    print(f"  {'-' * 60}")

    def fmt(v):
        # Use "is not None" so a first death at round 0 is reported as "0"
        # rather than being mistaken for "never died".
        return str(v) if v is not None else f">{cfg['NUM_ROUNDS']}"

    rows = [
        ("First node death (round)",      fmt(ga_fd),       fmt(leach_fd)),
        ("Network lifetime (rounds)",     str(ga_nd),       str(leach_nd)),
        ("Packets delivered to BS",       str(ga_s["packets_to_bs"]),
                                          str(leach_s["packets_to_bs"])),
        ("Final residual energy (J)",
         f"{ga_s['total_energy'][-1]:.4f}"    if ga_s["total_energy"]    else "0",
         f"{leach_s['total_energy'][-1]:.4f}" if leach_s["total_energy"] else "0"),
        ("Final energy std dev (J)",
         f"{ga_s['energy_stddev'][-1]:.4f}"    if ga_s["energy_stddev"]    else "0",
         f"{leach_s['energy_stddev'][-1]:.4f}" if leach_s["energy_stddev"] else "0"),
        ("MS-CH re-elections",            str(ga_s["reelections"]), "N/A"),
        ("Avg CHs / round",
         f"{np.mean(ga_s['ch_counts']):.1f}"    if ga_s['ch_counts']    else "0",
         f"{np.mean(leach_s['ch_counts']):.1f}" if leach_s['ch_counts'] else "0"),
        ("Avg MS-CHs / round (GA)",
         f"{np.mean(ga_s['ms_counts']):.2f}"    if ga_s['ms_counts']    else "0",
         "N/A"),
        ("CH% used",
         f"{cfg['CH_PERCENT'] * 100:.0f}% dynamic",
         f"{cfg['CH_PERCENT'] * 100:.0f}% dynamic"),
    ]
    for label, ga_val, leach_val in rows:
        print(f"  {label:<40} {ga_val:>10} {leach_val:>10}")

    print()
    if leach_nd > 0 and ga_nd != leach_nd:
        print(f"  v GA extends network lifetime by "
              f"{(ga_nd - leach_nd) / leach_nd * 100:+.1f}%")
    if leach_s["total_energy"] and ga_s["total_energy"]:
        ga_fe, leach_fe = (ga_s["total_energy"][-1],
                           leach_s["total_energy"][-1])
        if leach_fe > 0:
            print(f"  v GA saves {(ga_fe - leach_fe) / leach_fe * 100:+.1f}% "
                  f"more residual energy")
    print("=" * 64)


# ==============================================================================
# MAIN
# ==============================================================================

def run_ga_vs_leach(cfg: Optional[dict] = None, use_defaults: bool = False) -> dict:
    """
    Original top-level driver: GA vs LEACH only.

    Args:
        cfg          : optional pre-built configuration dict (skips prompts).
        use_defaults : when True, use default_config() instead of prompting.

    Default behaviour: prompts the user for every field (works in terminals
    and in Colab/Jupyter, where Colab pops up an input box per prompt).
    Press ENTER at any prompt to accept the default shown in [brackets].

    Pass use_defaults=True (or supply your own cfg) for non-interactive runs.

    Returns:
        dict with keys: cfg, ga_stats, leach_stats, ga_first_dead,
                        leach_first_dead, ga_network_dead, leach_network_dead.
    """
    print("\n" + "#" * 65)
    print("  Solar-Aware GA Multi-Sink Data Aggregation Protocol")
    print("  for Wireless Sensor Assisted IoT")
    print("#" * 65)

    if cfg is None:
        if use_defaults:
            cfg = default_config()
            print("  (running with default configuration - no prompts)")
        else:
            cfg = get_user_input()

    ga_stats,    ga_fd,    ga_nd,    _ = run_simulation(cfg, "GA")
    leach_stats, leach_fd, leach_nd, _ = run_simulation(cfg, "LEACH")

    print_summary(ga_stats, leach_stats,
                  ga_fd, leach_fd, ga_nd, leach_nd, cfg)

    print("\n  Generating plots...")
    plot_results(ga_stats, leach_stats, ga_fd, leach_fd, cfg)
    plot_topology_both_paths(cfg)

    print("\n  All outputs saved.  Done.\n")

    return {
        "cfg"               : cfg,
        "ga_stats"          : ga_stats,
        "leach_stats"       : leach_stats,
        "ga_first_dead"     : ga_fd,
        "leach_first_dead"  : leach_fd,
        "ga_network_dead"   : ga_nd,
        "leach_network_dead": leach_nd,
    }


# NOTE: the __main__ entry point lives at the very bottom of this file,
# after the modern-baseline comparison code that follows.



# ##############################################################################
# ##############################################################################
# ##                                                                          ##
# ##   MODERN BASELINE COMPARISON  (PSO, GWO, HEED)  -  SAME FILE             ##
# ##                                                                          ##
# ##   Everything below reuses the Node / World / energy model / solar model ##
# ##   / GA fitness function / multi-sink tier defined ABOVE in this same     ##
# ##   file. Only the cluster-head OPTIMISER changes, so the comparison is    ##
# ##   provably fair: PSO-MS and GWO-MS run the EXACT same fitness function   ##
# ##   and multi-sink tier as the GA; only the search strategy differs.       ##
# ##                                                                          ##
# ##   Baselines added:                                                        ##
# ##     * HEED  - energy-aware, spatially-separated CHs (deterministic).     ##
# ##     * PSO   - Particle Swarm Optimisation picks the CH set.              ##
# ##     * GWO   - Grey Wolf Optimiser picks the CH set.                      ##
# ##   LEACH (random) is kept for continuity.                                  ##
# ##                                                                          ##
# ##   RUN THIS FILE DIRECTLY:  python solar_ga_wsn_with_baselines.py         ##
# ##############################################################################
# ##############################################################################

from typing import Callable as _Callable


# ==============================================================================
# CMP-1  -  SHARED DECODE + SCORING HELPERS (for PSO / GWO)
# ==============================================================================
#
# CH selection is COMBINATORIAL: choose K distinct node IDs. PSO and GWO are
# continuous optimisers, so each "agent" is a real vector of length K whose
# values live in [0, S) (S = number of alive nodes). We decode a vector into K
# DISTINCT node IDs, then score it with THIS project's own fitness function
# (_evaluate_population, defined above) so the objective is byte-for-byte
# identical to the GA's.

def _decode_positions(positions: np.ndarray,
                      alive_idx: np.ndarray,
                      num_chs: int) -> List[List[int]]:
    """Decode a (P, K) matrix of continuous positions into P gene-lists, each a
    list of K DISTINCT alive-node IDs. Duplicate indices are repaired by linear
    probing to the next unused slot (standard permutation repair)."""
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
                if idx == start:          # every slot used (K == S) - accept
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
    """Decode every agent's position to a CH set and evaluate ALL of them with
    the project's shared batched fitness function. Returns (fitness, genes).
    Fresh Chromosome objects (fitness = -1) force a real evaluation each call."""
    genes_list = _decode_positions(positions, world.alive_idx, num_chs)
    chromos = [Chromosome(g) for g in genes_list]
    _evaluate_population(chromos, world, cfg, solar_now, num_chs)
    fits = np.fromiter((c.fitness for c in chromos), dtype=np.float64,
                       count=len(chromos))
    return fits, genes_list


# ==============================================================================
# CMP-2  -  PSO CLUSTER-HEAD SELECTOR  (drop-in replacement for the GA)
# ==============================================================================

def run_pso_ch_election(nodes: List[Node],
                        world: World,
                        cfg: dict,
                        round_num: int,
                        num_chs: int) -> Optional[Chromosome]:
    """Particle Swarm Optimisation for CH selection. Same population size
    (GA_POP -> swarm size) and iteration budget (GA_GEN) as the GA, so compute
    budgets match. Same fitness function. Standard global-best PSO."""
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

    w, c1, c2 = 0.7, 1.5, 1.5          # standard PSO coefficients
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

        if _CONV_SINK is not None:            # v2 convergence recording
            _CONV_SINK.append(gbest_fit)

        if stale >= patience:
            break

    best = Chromosome(gbest_gen)
    best.fitness = gbest_fit
    return best


# ==============================================================================
# CMP-3  -  GWO CLUSTER-HEAD SELECTOR  (drop-in replacement for the GA)
# ==============================================================================

def run_gwo_ch_election(nodes: List[Node],
                        world: World,
                        cfg: dict,
                        round_num: int,
                        num_chs: int) -> Optional[Chromosome]:
    """Grey Wolf Optimiser for CH selection - the most-cited "modern"
    metaheuristic for this task in 2023-2025 WSN papers. Same swarm size
    (GA_POP) and iteration budget (GA_GEN) as the GA. Three best wolves
    (alpha, beta, delta) steer the pack; exploration coefficient `a` decays
    from 2 to 0. Same fitness function as the GA."""
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

        if _CONV_SINK is not None:            # v2 convergence recording
            _CONV_SINK.append(best_fit)

        a = 2.0 - 2.0 * it / max(iters - 1, 1)      # 2 -> 0

        def _pull(leader: np.ndarray) -> np.ndarray:
            A = 2.0 * a * np.random.random((P, num_chs)) - a
            C = 2.0 * np.random.random((P, num_chs))
            D = np.abs(C * leader[None, :] - X)
            return leader[None, :] - A * D

        X = (_pull(alpha) + _pull(beta) + _pull(delta)) / 3.0
        X = np.clip(X, 0.0, hi)

    if best_gen is None:                # degenerate safety fallback
        fit, genes = _score_positions(X, world, cfg, solar_now, num_chs)
        gi = int(fit.argmax())
        best_gen = list(genes[gi])
        best_fit = float(fit[gi])

    best = Chromosome(best_gen)
    best.fitness = best_fit
    return best


# ==============================================================================
# CMP-4  -  HEED CLUSTER-HEAD SELECTOR  (energy-aware, non-metaheuristic)
# ==============================================================================

def run_heed_ch_election(nodes: List[Node],
                         world: World,
                         cfg: dict,
                         round_num: int,
                         num_chs: int) -> Optional[Chromosome]:
    """HEED-style CH selection (Younis & Fahmy, 2004), simplified.

    HEED's essence, and what distinguishes it from random LEACH:
      (a) a node's chance of becoming CH grows with its RESIDUAL ENERGY, and
      (b) CHs are kept SPATIALLY SEPARATED (a tentative CH yields to a
          higher-energy CH already claiming its neighbourhood).

    Implemented deterministically: walk nodes from highest to lowest residual
    energy, accept a node as CH only if it is at least `min_sep` from every
    already-accepted CH, back-filling by pure energy order if separation leaves
    us short. Energy-aware + well-spread = a genuinely fair (non-random)
    baseline, while remaining a documented simplification of full HEED."""
    S = int(world.alive_idx.size)
    if S < num_chs or num_chs <= 0:
        return None

    ids = world.alive_idx
    e   = world.alive_e
    xy  = world.alive_xy

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
            if d < min_sep:            # too close to a higher-energy CH
                continue
        chosen.append(int(ids[oi]))
        chosen_xy.append(p)

    if len(chosen) < num_chs:          # back-fill by energy order
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
# CMP-5  -  GENERIC PROTOCOL ROUND  (pluggable CH selector, same multi-sink tier)
# ==============================================================================

_Selector = _Callable[[List[Node], World, dict, int, int], Optional[Chromosome]]

_SELECTORS: Dict[str, _Selector] = {
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
                           selector: _Selector) -> bool:
    """One protocol round with a pluggable CH selector. Mirrors
    simulate_round_ga EXACTLY except for the CH-selection call, so the ONLY
    variable across protocols is how the cluster heads are chosen."""
    solar_now    = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])  # forecast
    solar_actual = actual_solar_rate(round_num, cfg)                    # realised

    # Step 1 - REALISTIC solar harvest (clouds / season / noise) followed by
    # the v2 always-on idle-listen + sensing drain every alive node pays each
    # round whether or not it transmits.
    for n in nodes:
        if n.alive:
            if solar_actual > 0:
                n.harvest_solar(solar_actual)
            n.idle_sense_drain()

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

    # Step 8 - sensors transmit (receiver only spends RX if the packet arrived)
    for nid in world.alive_idx.tolist():
        n = nodes[nid]
        if n.role == "sensor" and n.assigned_ch is not None:
            ch = nodes[n.assigned_ch]
            if ch.alive:
                n.transmit(ch.x, ch.y)
                if n.delivered:
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
                if ch.delivered:            # relay hop only counts if it arrived
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
                if ms.delivered:                 # count only successful delivery
                    stats["packets_to_bs"] += 1

    for ch in direct_chs:
        if ch.alive and ch.delivered:            # delivered set during Step 9 TX
            stats["packets_to_bs"] += 1

    # Step 12 - record
    _record_stats(nodes, stats)
    stats["ch_counts"].append(len(ch_nodes))
    stats["ms_counts"].append(len(ms_chs))
    return any(n.alive for n in nodes)


# ==============================================================================
# CMP-6  -  PER-PROTOCOL SIMULATION RUNNER
# ==============================================================================

def run_protocol(cfg: dict, protocol: str, seed: int = 42, quiet: bool = False):
    """Run one protocol end-to-end for a GIVEN seed.

    v2 additions vs the old runner:
      * `seed` parameter -> enables multi-seed Monte-Carlo repetition.
      * per-round CH-election time is measured (addresses the "GA is expensive"
        critique with real numbers instead of hand-waving).
      * Packet Delivery Ratio (PDR) is computed from the realistic lossy radio.

    Returns (stats, first_dead, network_dead, extra) where `extra` is a dict
    with keys: pdr, avg_elect_ms, tx_attempts, tx_delivered."""
    protocol = protocol.upper()
    nodes = create_nodes(cfg, seed)          # reproducible per-seed field
    world = World(nodes=nodes)
    stats = make_stats()

    base_selector = _SELECTORS.get(protocol)   # None for LEACH (native round)

    # Wrap the selector so we can time ONLY the cluster-head election, per round.
    elect = {"total": 0.0, "calls": 0}
    selector = None
    if base_selector is not None:
        def selector(nn, ww, cc, rr, kk, _bs=base_selector):
            t0 = time.perf_counter()
            res = _bs(nn, ww, cc, rr, kk)
            elect["total"] += time.perf_counter() - t0
            elect["calls"] += 1
            return res

    if not quiet:
        print(f"\n{'=' * 58}")
        print(f"  Protocol  : {protocol}   (seed {seed})")
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
            if not quiet:
                print(f"  *  First node died : round {r}")
        if not alive:
            network_dead = r
            if not quiet:
                print(f"  X  Network dead    : round {r}")
            break

    # PDR from the realistic radio: delivered / attempted (retries counted).
    tx_attempts   = sum(n.packets_sent for n in nodes)
    tx_delivered  = sum(n.packets_delivered for n in nodes)
    pdr           = (tx_delivered / tx_attempts) if tx_attempts else 0.0
    avg_elect_ms  = (elect["total"] / elect["calls"] * 1000.0
                     if elect["calls"] else 0.0)
    extra = {"pdr": pdr, "avg_elect_ms": avg_elect_ms,
             "tx_attempts": tx_attempts, "tx_delivered": tx_delivered}

    if not quiet:
        if network_dead == cfg["NUM_ROUNDS"]:
            print(f"  v  Survived all {cfg['NUM_ROUNDS']} rounds")
        print(f"  Packets to BS      : {stats['packets_to_bs']}")
        print(f"  PDR                : {pdr * 100:.1f}%   "
              f"(delivered {tx_delivered}/{tx_attempts} attempts)")
        print(f"  Avg CH-election    : {avg_elect_ms:.2f} ms/round")
    return stats, first_dead, network_dead, extra


# ==============================================================================
# CMP-7  -  MULTI-PROTOCOL PLOTS + SUMMARY
# ==============================================================================

_CMP_COLORS = {
    "GA": "#1A5FAD", "PSO": "#8E44AD", "GWO": "#16A085",
    "HEED": "#E67E22", "LEACH": "#C0392B",
}
_CMP_STYLES = {
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
        ("alive_nodes",   "Alive nodes",               "Network Lifetime"),
        ("total_energy",  "Total residual energy (J)", "Total Residual Energy"),
        ("dead_nodes",    "Cumulative dead nodes",     "Node Deaths Over Time"),
        ("energy_stddev", "Std dev of energy (J)",     "Energy Balance (lower=better)"),
    ]
    for ax, (key, ylabel, title) in zip(axes.flat, panels):
        for proto, res in results.items():
            series = res["stats"][key]
            ax.plot(range(len(series)), series,
                    color=_CMP_COLORS.get(proto, "#333"),
                    ls=_CMP_STYLES.get(proto, "-"), lw=2, label=proto)
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
    fig.suptitle("Headline Metrics by Protocol (higher is better)",
                 fontsize=12, fontweight="bold")
    colors = [_CMP_COLORS.get(p, "#333") for p in protocols]

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


def print_comparison_table(results: Dict[str, dict], cfg: dict) -> None:
    protocols = list(results.keys())
    width = 26 + 12 * len(protocols)
    print("\n" + "=" * width)
    print("  MULTI-PROTOCOL RESULTS SUMMARY")
    print("=" * width)

    print(f"  {'Metric':<26}" + "".join(f"{p:>12}" for p in protocols))
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
    row("PDR (%)",
        lambda r: f"{r.get('extra', {}).get('pdr', 0.0) * 100:.1f}"
        if "extra" in r else "n/a")
    row("CH-elect (ms/round)",
        lambda r: f"{r.get('extra', {}).get('avg_elect_ms', 0.0):.2f}"
        if "extra" in r else "n/a")
    print("=" * width)

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
# CMP-8  -  COMPARISON DRIVER
# ==============================================================================

def run_comparison(cfg: Optional[dict] = None,
                   protocols: Optional[List[str]] = None) -> Dict[str, dict]:
    """Run every requested protocol under identical conditions and emit the
    comparison plots + summary table.

    Choose protocols via the SOLAR_GA_PROTOCOLS env var (comma-separated) or the
    `protocols` argument. Default: GA, PSO, GWO, HEED, LEACH."""
    if cfg is None:
        cfg = default_config()

    if protocols is None:
        env = os.environ.get("SOLAR_GA_PROTOCOLS", "GA,PSO,GWO,HEED,LEACH")
        protocols = [p.strip().upper() for p in env.split(",") if p.strip()]

    print("\n" + "#" * 64)
    print("  MODERN-BASELINE COMPARISON  -  Solar-Aware GA Multi-Sink WSN")
    print("  Protocols: " + ", ".join(protocols))
    print("#" * 64)

    seed = int(cfg.get("_single_seed", 42))
    results: Dict[str, dict] = {}
    for proto in protocols:
        stats, fd, nd, extra = run_protocol(cfg, proto, seed=seed)
        results[proto] = {"stats": stats, "first_dead": fd,
                          "network_dead": nd, "extra": extra}

    print_comparison_table(results, cfg)
    print("  Generating plots...")
    plot_multi_results(results, cfg)
    plot_summary_bars(results, cfg)
    print("\n  Done.\n")
    return results


# ==============================================================================
# CMP-9  -  MONTE-CARLO (MULTI-SEED) EVALUATION + STATISTICAL SIGNIFICANCE
# ==============================================================================
#
# This is the single most important v2 addition. The original paper reported
# ONE run of seed 42; metaheuristic comparisons are only credible when averaged
# over many independent seeds with error bars and a significance test.

_MC_METRICS = ("lifetime", "first_death", "packets", "residual", "pdr", "elect_ms")


def _one_run_metrics(cfg: dict, protocol: str, seed: int) -> Dict[str, float]:
    stats, fd, nd, extra = run_protocol(cfg, protocol, seed=seed, quiet=True)
    return {
        "lifetime"   : float(nd),
        "first_death": float(fd if fd is not None else cfg["NUM_ROUNDS"]),
        "packets"    : float(stats["packets_to_bs"]),
        "residual"   : float(stats["total_energy"][-1] if stats["total_energy"] else 0.0),
        "pdr"        : float(extra["pdr"]),
        "elect_ms"   : float(extra["avg_elect_ms"]),
    }


def _significance(a: List[float], b: List[float]) -> Dict[str, float]:
    """GA-vs-baseline significance. Uses SciPy (Welch t-test + Mann-Whitney U)
    when available; otherwise falls back to a manual Welch t-statistic plus
    Cohen's d and honestly reports that no p-value is available."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        from scipy import stats as _sp   # type: ignore
        t, p_t = _sp.ttest_ind(a, b, equal_var=False)
        try:
            u, p_u = _sp.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:               # e.g. all values identical
            u, p_u = float("nan"), float("nan")
        return {"backend": "scipy", "t": float(t), "p_ttest": float(p_t),
                "u": float(u), "p_mwu": float(p_u)}
    except Exception:
        na, nb = len(a), len(b)
        am, bm = float(a.mean()), float(b.mean())
        av = float(a.var(ddof=1)) if na > 1 else 0.0
        bv = float(b.var(ddof=1)) if nb > 1 else 0.0
        se = math.sqrt(av / na + bv / nb) if na and nb else 0.0
        t  = (am - bm) / se if se > 0 else float("nan")
        psd = (math.sqrt(((av * (na - 1)) + (bv * (nb - 1))) / (na + nb - 2))
               if (na + nb - 2) > 0 else 0.0)
        d = (am - bm) / psd if psd > 0 else float("nan")
        return {"backend": "manual", "t": float(t), "p_ttest": float("nan"),
                "cohens_d": float(d)}


def run_monte_carlo(cfg: Optional[dict] = None,
                    protocols: Optional[List[str]] = None,
                    seeds: Optional[List[int]] = None) -> Dict[str, dict]:
    """Run every protocol over MANY seeds, report mean +/- std for each metric,
    and test GA vs each baseline for statistical significance."""
    if cfg is None:
        cfg = default_config()
    if protocols is None:
        env = os.environ.get("SOLAR_GA_PROTOCOLS", "GA,PSO,GWO,HEED,LEACH")
        protocols = [p.strip().upper() for p in env.split(",") if p.strip()]
    if seeds is None:
        n_seeds = int(cfg.get("MC_SEEDS", 15))
        seeds = list(range(1, n_seeds + 1))

    print("\n" + "#" * 66)
    print("  MONTE-CARLO EVALUATION  -  Solar-Aware GA Multi-Sink WSN (v2)")
    print(f"  Protocols : {', '.join(protocols)}")
    print(f"  Seeds     : {len(seeds)}  ({seeds[0]}..{seeds[-1]})")
    print(f"  Radio     : loss={cfg['PACKET_LOSS']}, retx={cfg['MAX_RETX']} | "
          f"Solar={cfg['SOLAR_MODEL']} | SolarTerm={cfg['USE_SOLAR_TERM']}")
    print("#" * 66)

    # raw[proto][metric] = list over seeds
    raw: Dict[str, Dict[str, List[float]]] = {
        p: {m: [] for m in _MC_METRICS} for p in protocols}

    for si, seed in enumerate(seeds, 1):
        print(f"  Seed {si}/{len(seeds)} (={seed}) ...", end="", flush=True)
        for proto in protocols:
            m = _one_run_metrics(cfg, proto, seed)
            for k in _MC_METRICS:
                raw[proto][k].append(m[k])
        print(" done")

    # Aggregate mean/std.
    agg: Dict[str, dict] = {}
    for proto in protocols:
        agg[proto] = {"raw": raw[proto]}
        for m in _MC_METRICS:
            arr = np.asarray(raw[proto][m], dtype=float)
            agg[proto][m] = (float(arr.mean()), float(arr.std(ddof=1))
                             if arr.size > 1 else 0.0)

    _print_mc_table(agg, protocols, cfg)
    _print_significance(raw, protocols)
    print("  Generating Monte-Carlo plot ...")
    plot_monte_carlo(agg, protocols, cfg)
    print("\n  Monte-Carlo evaluation done.\n")
    return agg


def _print_mc_table(agg: Dict[str, dict], protocols: List[str], cfg: dict) -> None:
    width = 26 + 16 * len(protocols)
    print("\n" + "=" * width)
    print("  MEAN +/- STD OVER SEEDS")
    print("=" * width)
    print(f"  {'Metric':<26}" + "".join(f"{p:>16}" for p in protocols))
    print("  " + "-" * (24 + 16 * len(protocols)))

    labels = [
        ("lifetime",    "Network lifetime"),
        ("first_death", "First node death"),
        ("packets",     "Packets to BS"),
        ("residual",    "Final residual (J)"),
        ("pdr",         "PDR (fraction)"),
        ("elect_ms",    "CH-elect (ms/round)"),
    ]
    for key, label in labels:
        line = f"  {label:<26}"
        for p in protocols:
            mean, std = agg[p][key]
            if key in ("residual", "pdr"):
                line += f"{mean:>9.3f}+/-{std:<4.2f}"
            elif key == "elect_ms":
                line += f"{mean:>9.2f}+/-{std:<4.2f}"
            else:
                line += f"{mean:>9.1f}+/-{std:<4.1f}"
        print(line)
    print("=" * width)


def _print_significance(raw: Dict[str, Dict[str, List[float]]],
                        protocols: List[str]) -> None:
    if "GA" not in protocols:
        return
    print("\n  STATISTICAL SIGNIFICANCE - GA vs each baseline (network lifetime):")
    ga = raw["GA"]["lifetime"]
    ga_mean = float(np.mean(ga))
    for p in protocols:
        if p == "GA":
            continue
        base = raw[p]["lifetime"]
        s = _significance(ga, base)
        gain = ((ga_mean - float(np.mean(base))) / float(np.mean(base)) * 100
                if np.mean(base) else float("nan"))
        if s["backend"] == "scipy":
            sig = "SIGNIFICANT" if (s["p_ttest"] == s["p_ttest"]
                                    and s["p_ttest"] < 0.05) else "not sig."
            print(f"    GA vs {p:<6}: lifetime {gain:+.1f}% | "
                  f"Welch t={s['t']:+.2f}, p={s['p_ttest']:.4g} | "
                  f"Mann-Whitney p={s['p_mwu']:.4g}  -> {sig} (a=0.05)")
        else:
            print(f"    GA vs {p:<6}: lifetime {gain:+.1f}% | "
                  f"t={s['t']:+.2f}, Cohen's d={s.get('cohens_d', float('nan')):+.2f} "
                  f"| p-value needs SciPy (pip install scipy)")
    print()


def plot_monte_carlo(agg: Dict[str, dict], protocols: List[str],
                     cfg: dict) -> None:
    """Bar charts with error bars (mean +/- std) for the headline metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Monte-Carlo results over {cfg.get('MC_SEEDS', '?')} seeds "
        f"(mean +/- std)\nRealistic radio (loss/retx) + realistic solar",
        fontsize=12, fontweight="bold")
    colors = [_CMP_COLORS.get(p, "#333") for p in protocols]

    panels = [
        ("lifetime", "Network lifetime (rounds)", axes[0, 0]),
        ("pdr",      "Packet delivery ratio",     axes[0, 1]),
        ("packets",  "Packets delivered to BS",   axes[1, 0]),
        ("elect_ms", "CH-election time (ms/round)", axes[1, 1]),
    ]
    for key, title, ax in panels:
        means = [agg[p][key][0] for p in protocols]
        stds  = [agg[p][key][1] for p in protocols]
        bars = ax.bar(protocols, means, yerr=stds, capsize=5, color=colors)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, mv in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{mv:.2f}" if mv < 100 else f"{mv:.0f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("comparison_monte_carlo.png", dpi=150, bbox_inches="tight")
    print("  Plot saved -> comparison_monte_carlo.png")
    plt.close()


# ==============================================================================
# CMP-10  -  CONVERGENCE CURVES  (GA vs PSO vs GWO, same budget)
# ==============================================================================

def plot_convergence(cfg: Optional[dict] = None, seed: int = 42) -> None:
    """Run ONE cluster-head election with each metaheuristic on an identical
    field and capture best-fitness per generation, so the reader can see how
    fast each optimiser converges under the same population/iteration budget."""
    global _CONV_SINK
    if cfg is None:
        cfg = default_config()

    curves: Dict[str, list] = {}
    for proto, fn in (("GA", run_ga_ch_election),
                      ("PSO", run_pso_ch_election),
                      ("GWO", run_gwo_ch_election)):
        nodes = create_nodes(cfg, seed)          # identical field each time
        world = World(nodes=nodes)
        world.refresh()
        num_chs = get_num_chs(int(world.alive_idx.size), cfg)
        _CONV_SINK = []
        try:
            fn(nodes, world, cfg, 12, num_chs)   # round 12 ~ midday sun
        finally:
            curves[proto] = list(_CONV_SINK)
            _CONV_SINK = None

    plt.figure(figsize=(9, 6))
    for proto in ("GA", "PSO", "GWO"):
        c = curves.get(proto, [])
        if c:
            plt.plot(range(1, len(c) + 1), c, lw=2,
                     color=_CMP_COLORS.get(proto, "#333"), label=proto)
    plt.xlabel("Generation / iteration")
    plt.ylabel("Best fitness so far")
    plt.title("Optimiser convergence (same population & iteration budget)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison_convergence.png", dpi=150, bbox_inches="tight")
    print("  Plot saved -> comparison_convergence.png")
    plt.close()


# ==============================================================================
# CMP-11  -  SOLAR-AWARENESS ABLATION  (does the solar term actually help?)
# ==============================================================================

def run_solar_ablation(cfg: Optional[dict] = None,
                       seeds: Optional[List[int]] = None) -> Dict[str, dict]:
    """Run GA WITH and WITHOUT the solar term over multiple seeds and quantify
    the difference. If the gap is negligible, the 'solar-aware' claim must be
    softened - this makes the paper honest and defensible."""
    if cfg is None:
        cfg = default_config()
    if seeds is None:
        seeds = list(range(1, int(cfg.get("MC_SEEDS", 15)) + 1))

    print("\n" + "#" * 66)
    print("  SOLAR-AWARENESS ABLATION  -  GA with vs without the solar term")
    print("#" * 66)

    out: Dict[str, dict] = {}
    for label, use_solar in (("solar-aware", True), ("solar-OFF", False)):
        c = dict(cfg)
        c["USE_SOLAR_TERM"] = use_solar
        lifetimes, pdrs = [], []
        for seed in seeds:
            _, _, nd, extra = run_protocol(c, "GA", seed=seed, quiet=True)
            lifetimes.append(float(nd))
            pdrs.append(float(extra["pdr"]))
        out[label] = {"lifetime": lifetimes, "pdr": pdrs}
        print(f"  {label:<12}: lifetime {np.mean(lifetimes):.1f}"
              f"+/-{np.std(lifetimes, ddof=1) if len(lifetimes) > 1 else 0:.1f}"
              f" | PDR {np.mean(pdrs) * 100:.1f}%")

    on  = np.mean(out["solar-aware"]["lifetime"])
    off = np.mean(out["solar-OFF"]["lifetime"])
    if off > 0:
        print(f"\n  Solar term changes GA network lifetime by "
              f"{(on - off) / off * 100:+.1f}% (mean over seeds).")
        s = _significance(out["solar-aware"]["lifetime"],
                          out["solar-OFF"]["lifetime"])
        if s["backend"] == "scipy":
            print(f"  Significance: Welch t={s['t']:+.2f}, p={s['p_ttest']:.4g}")
        else:
            print(f"  Effect size: Cohen's d="
                  f"{s.get('cohens_d', float('nan')):+.2f} (install SciPy for p)")
    print()
    return out


# ==============================================================================
# MAIN ENTRY POINT  (v2 - one file, all drawbacks addressed)
# ==============================================================================

def main_v2(cfg: Optional[dict] = None) -> None:
    """
    Full v2 pipeline, each stage removing one of the previously-listed drawbacks:
      1. Monte-Carlo over many seeds + significance test   (single-seed fix)
      2. CH-election timing (inside the MC table/plot)      (GA-cost fix)
      3. Convergence curves GA vs PSO vs GWO                (GA-cost fix)
      4. Solar-awareness ablation                           (solar-edge fix)
    Realistic radio (loss/retx/idle) and realistic solar (clouds/season) are ON
    by default via default_config(), so every run already reflects those fixes.
    """
    if cfg is None:
        cfg = default_config()

    stage = os.environ.get("SOLAR_GA_STAGE", "all").lower()

    if stage in ("all", "mc", "montecarlo"):
        run_monte_carlo(cfg)
    if stage in ("all", "convergence", "conv"):
        print("  Generating convergence curves ...")
        plot_convergence(cfg)
    if stage in ("all", "ablation"):
        run_solar_ablation(cfg)
    if stage in ("single",):
        # Single-seed detailed run with lifetime/energy curves (fast smoke test).
        run_comparison(cfg)


if __name__ == "__main__":
    # v2 default: Monte-Carlo + convergence + solar ablation, all with the
    # realistic radio and solar models enabled. Non-interactive; Colab/CI ready.
    #
    #   python solar_ga_wsn_v2.py
    #   SOLAR_GA_PROTOCOLS="GA,PSO,GWO,HEED,LEACH" python solar_ga_wsn_v2.py
    #   SOLAR_GA_STAGE=single   python solar_ga_wsn_v2.py   # quick 1-seed run
    #   SOLAR_GA_STAGE=mc       python solar_ga_wsn_v2.py   # only Monte-Carlo
    #
    # For fewer seeds / faster runs: default_config(MC_SEEDS=5) in your own call,
    # or edit MC_SEEDS. The original GA-vs-LEACH driver is still run_ga_vs_leach().
    main_v2()
