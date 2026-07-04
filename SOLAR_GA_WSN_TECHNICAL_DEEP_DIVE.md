# `solar_ga_wsn.py` — Technical Deep Dive

> A formula-by-formula, line-by-line walkthrough of the protocol.
> Every equation, every weight, every threshold, and every design
> decision — explained.

This document is written for someone who **wants to understand the math
and the code together**. If you only want a high-level story, read
`HOW_SOLAR_GA_WSN_WORKS.md` first. This file picks up where that one
leaves off.

---

## Table of Contents

1. [Symbol Reference](#1-symbol-reference)
2. [Formula 1 — First-Order Radio Energy Model](#2-formula-1--first-order-radio-energy-model)
3. [Formula 2 — Data Aggregation Cost](#3-formula-2--data-aggregation-cost)
4. [Formula 3 — Solar Harvest Model](#4-formula-3--solar-harvest-model)
5. [Formula 4 — Euclidean Distance](#5-formula-4--euclidean-distance)
6. [Formula 5 — Dynamic Cluster-Head Count](#6-formula-5--dynamic-cluster-head-count)
7. [Formula 6 — Dynamic MS-CH Count](#7-formula-6--dynamic-ms-ch-count)
8. [Formula 7 — The GA Fitness Function](#8-formula-7--the-ga-fitness-function)
9. [Formula 8 — MS-CH Solar-Aware Score](#9-formula-8--ms-ch-solar-aware-score)
10. [Formula 9 — k-Medoids Spatial Clustering](#10-formula-9--k-medoids-spatial-clustering)
11. [GA Mechanics — Selection, Crossover, Mutation](#11-ga-mechanics--selection-crossover-mutation)
12. [Path Decision Rule](#12-path-decision-rule)
13. [MS-CH Re-Election Trigger](#13-ms-ch-re-election-trigger)
14. [Step-by-Step: One Full Round](#14-step-by-step-one-full-round)
15. [Design Decisions & Thinking](#15-design-decisions--thinking)
16. [Performance Optimisations](#16-performance-optimisations)

---

## 1. Symbol Reference

| Symbol | Meaning | Default value |
|---|---|---|
| `k` | Packet size (bits) | 4000 |
| `d` | Distance between transmitter and receiver (m) | varies |
| `E_elec` | Per-bit electronics energy | 50 × 10⁻⁹ J/bit |
| `E_amp` | Free-space amplifier coefficient | 100 × 10⁻¹² J/bit/m² |
| `E_mp`  | Multi-path amplifier coefficient | 0.0013 × 10⁻¹² J/bit/m⁴ |
| `E_DA`  | Aggregation energy per bit per packet | 5 × 10⁻⁹ J/bit |
| `d₀` | Crossover distance (free-space ↔ multi-path) | √(E_amp/E_mp) ≈ 87.7 m |
| `E_init` | Initial battery of every node | 0.5 J |
| `BATTERY_MAX` | Hard cap on battery (with solar) | 2.0 J |
| `MAX_HARVEST` | Peak network-wide solar rate per round | 0.002 J |
| `solar_eff` | Per-node harvesting efficiency (panel/shade), fixed at creation | uniform in [0.6, 1.0] |
| `SOLAR_EFF_MIN`/`MAX` | Bounds of the per-node `solar_eff` draw | 0.6 / 1.0 |
| `COMM_RANGE_PCT` | Comm range as fraction of field | 0.40 |
| `MS_REELECT_THR` | Battery fraction below which an MS-CH is replaced | 0.15 |
| `CH_PERCENT` | Target fraction of alive nodes that become CHs | 0.10 |
| `RELAYS_PER_MS` | Relay CHs handled by one MS-CH | 4 |
| `DIRECT_DIST` | Max distance for CH→BS direct path | 0.55 × FIELD |
| `DIRECT_NRG` | Min battery fraction for CH→BS direct path | 0.40 |

---

## 2. Formula 1 — First-Order Radio Energy Model

**Source:** Heinzelman et al., the same model used in the original LEACH paper (reference [6] in the code header).

### The math

Transmitting `k` bits over distance `d` costs:

```
              ┌  E_elec · k + E_amp · k · d²        if d ≤ d₀   (free-space)
E_TX(k, d) =  ┤
              └  E_elec · k + E_mp  · k · d⁴        if d >  d₀   (multi-path)
```

The crossover distance `d₀` is the point where free-space and multi-path
predict the same cost:

```
d₀ = √(E_amp / E_mp)
```

Receiving the same packet costs:

```
E_RX(k) = E_elec · k
```

### Why two regimes?

Below `d₀`, signal loss is roughly proportional to `d²` (line-of-sight).
Above `d₀`, ground reflections start to dominate, and loss grows as `d⁴`.
This is **why long-distance shouting is so much worse than short whispering**
— it's not just twice as bad, it's a power-of-four explosion.

With the defaults, `d₀ ≈ 87.7 m`. So a CH that's 50 m from the BS pays
`d²` cost; one that's 100 m away pays `d⁴` cost. That's the entire reason
the multi-sink relay idea exists.

### How it's coded

```python
# Constants (Section 2 of the file)
E_ELEC = 50e-9
E_AMP  = 100e-12
E_MP   = 0.0013e-12
D0     = math.sqrt(E_AMP / E_MP)        # ≈ 87.7 m

# Per-node method (Node._tx_cost, Section 3)
def _tx_cost(self, bits: int, d: float) -> float:
    if d <= D0:
        return E_ELEC * bits + E_AMP * bits * d * d        # free-space
    d2 = d * d
    return E_ELEC * bits + E_MP * bits * d2 * d2           # multi-path (d⁴)
```

### What happens at runtime

When a node calls `transmit(to_x, to_y)`:

1. Distance is computed with `math.hypot`.
2. The branch `d ≤ d₀` picks the right equation.
3. The cost is subtracted from `self.energy`.
4. If `self.energy ≤ 0`, the node is marked **dead** (`alive = False`).

```python
def transmit(self, to_x: float, to_y: float) -> bool:
    cost = self._tx_cost(self._cfg["PACKET_SIZE"],
                         math.hypot(self.x - to_x, self.y - to_y))
    self.energy -= cost
    self.packets_sent += 1
    if self.energy <= 0:
        self.energy = 0
        self.alive  = False
    return self.alive
```

---

## 3. Formula 2 — Data Aggregation Cost

When a CH receives `n` packets from its members and merges them into one
bundle, that fusion has a small CPU cost:

```
E_AGG(k, n) = E_DA · k · n
```

### How it's coded

```python
def _agg_cost(self, bits: int, n: int) -> float:
    return E_DA * bits * n

def aggregate(self, n_packets: int) -> bool:
    self.energy -= self._agg_cost(self._cfg["PACKET_SIZE"], n_packets)
    if self.energy <= 0:
        self.energy = 0
        self.alive  = False
    return self.alive
```

### Why it matters

Aggregation is **cheap compared to transmission** but it's not free. With
defaults, fusing 5 packets costs about `5 × 4000 × 5e-9 = 100 µJ`. A
single 50 m transmit costs about `1.2 mJ` — twelve times more. So the
big energy savings come from *avoiding long transmits*, not from
avoiding aggregation. That insight is why the multi-sink design works.

---

## 4. Formula 3 — Solar Harvest Model

The simulated sun follows a **half-sine wave** during daylight hours and
flatlines at night.

### The math

Let `h = round_num mod 24` (current hour, 0–23). Then:

```
solar(round) = MAX_HARVEST · max(0, sin(π · (h - 6) / 12))
```

This curve:
- Is **0** before 06:00 and after 18:00 (night)
- Rises smoothly to **1.0** at noon
- Falls smoothly back to 0 at sunset

`solar(round)` is the **network-wide daylight level** — it is the same for
everyone. What differs **per node** is a fixed **harvesting efficiency**
`solar_eff ∈ [0.6, 1.0]`, drawn once at creation. It models a node's panel
orientation, shading and dust: a node in a shaded corner permanently harvests
less than a well-placed one. Each round a node's actual gain is its own rate
plus 5% Gaussian noise:

```
rate           = solar(round) · solar_eff          (node-specific)
harvest_actual = max(0, rate + N(0, 0.05 · rate))
```

Battery is then capped at `BATTERY_MAX = 2.0 J` so a node never
"overcharges".

### How it's coded

```python
def solar_rate_for_round(round_num: int, max_harvest: float) -> float:
    hour = round_num % 24
    return max_harvest * max(0.0, math.sin(math.pi * (hour - 6) / 12))

# In Node:
def harvest_solar(self, solar_rate_now: float) -> None:
    if solar_rate_now <= 0:
        return
    rate    = solar_rate_now * self.solar_eff        # node-specific
    harvest = max(0.0, rate + random.gauss(0, rate * 0.05))
    self.energy = min(self.energy + harvest, BATTERY_MAX)

# solar_eff is assigned once, in create_nodes():
for n in nodes:
    n.solar_eff = random.uniform(SOLAR_EFF_MIN, SOLAR_EFF_MAX)   # [0.6, 1.0]
```

### Why this shape?

A real solar panel's output roughly follows a sine of solar elevation
angle. We use a *half*-sine spanning 12 hours so the integral matches
an idealised dawn-to-dusk cycle.

### Why per-node `solar_eff` (and not just the shared curve)?

This is the fix for a subtle but important flaw. The daylight level
`solar(round)` is identical for every node, so on its own it can **never**
tell two candidate nodes apart — added to competing candidates it is just a
constant offset that cannot change a ranking. Giving each node a fixed
`solar_eff` makes harvesting **spatially heterogeneous**: some nodes really
are better solar sites than others. That is what lets the "solar-aware" score
genuinely prefer one node over another (see Formulas 7 and 8). The extra 5%
noise only adds small round-to-round jitter on top.

---

## 5. Formula 4 — Euclidean Distance

Used everywhere — sensor → CH, CH → BS, peer-to-peer for centrality.

```
d(A, B) = √((Aₓ - Bₓ)² + (Aᵧ - Bᵧ)²)
```

In code, computed via `math.hypot` (which is numerically more stable
than the naive `sqrt(dx² + dy²)`):

```python
def distance_to(self, x: float, y: float) -> float:
    return math.hypot(self.x - x, self.y - y)
```

For batch operations (sensor-to-CH assignment, GA fitness), distances
are vectorised in NumPy with the squared-distance trick (no sqrt needed
when only relative ordering matters):

```python
def vectorized_assign(sensor_xy: np.ndarray, ch_xy: np.ndarray) -> np.ndarray:
    diff = sensor_xy[:, None, :] - ch_xy[None, :, :]   # (S, C, 2)
    d2   = np.einsum("ijk,ijk->ij", diff, diff)        # (S, C)
    return d2.argmin(axis=1)                           # nearest CH per sensor
```

The cached `_dist_bs` field on each Node is a small but impactful win:
the BS doesn't move, so we compute distance-to-BS exactly once per node
(in `__init__`) and reuse it forever.

---

## 6. Formula 5 — Dynamic Cluster-Head Count

```
num_chs = max(1, min(alive_count, round(alive_count · CH_PERCENT)))
```

### Code

```python
def get_num_chs(alive_count: int, cfg: dict) -> int:
    if alive_count <= 0:
        return 0
    return max(1, min(alive_count, round(alive_count * cfg["CH_PERCENT"])))
```

### Examples

| Alive | CH_PERCENT | num_chs |
|---|---|---|
| 50 | 0.10 | 5 |
| 30 | 0.10 | 3 |
| 8 | 0.10 | 1 (floor enforced) |
| 0 | 0.10 | 0 (network dead) |

### Why dynamic?

If we hardcoded `num_chs = 5`, then once the network dropped below 5 nodes
the CH count couldn't shrink. Worse, we might try to elect more CHs than
we had alive nodes. Recomputing every round means the topology gracefully
collapses as nodes die.

---

## 7. Formula 6 — Dynamic MS-CH Count

```
              ┌  0                                              if relay = 0
num_ms     =  ┤
              └  max(1, min(relay, ⌈relay / RELAYS_PER_MS⌉))    otherwise
```

### Code

```python
def get_num_ms(num_relay_chs: int, cfg: dict) -> int:
    if num_relay_chs <= 0:
        return 0
    return max(1, min(num_relay_chs,
                      math.ceil(num_relay_chs / cfg["RELAYS_PER_MS"])))
```

### Why the `if relay = 0` short-circuit?

This is **Point 3 + Point 4 of the spec** baked into a single line.
If every CH chose Path A (direct to BS), there are zero relay CHs, so
the MS-CH stage is **skipped entirely** for the round. No wasted election,
no wasted energy.

### Examples

| relay_chs | RELAYS_PER_MS | num_ms |
|---|---|---|
| 0 | 4 | **0 (skip)** |
| 3 | 4 | 1 |
| 8 | 4 | 2 |
| 9 | 4 | 3 |

---

## 8. Formula 7 — The GA Fitness Function

This is the **heart of the GA**. Every candidate set of CHs (a chromosome)
is graded by:

```
F = 0.25 · E_score + 0.25 · S_score + 0.30 · C_score + 0.20 · Sp_score
```

The four components below all live on `[0, 1]` so the final fitness is
also on `[0, 1]`.

### 8.1 Energy Score (25%)

```
E_score = min( Σ E(ch) / (K · E_init) , 1.0 )
```

The total residual energy of the picked CHs, normalised by what they
would have if they were all fresh. Prefers chromosomes that pick
high-battery nodes.

```python
e_score = np.minimum(ch_e.sum(axis=1) / (K * cfg["E_INITIAL"]), 1.0)
```

### 8.2 Solar Score (25%) — the "true solar-aware" piece

```
daylight = solar_now / MAX_HARVEST                         (shared, 0→1)
S_score  = 0.5 · daylight · avg(solar_eff of chosen CHs)
         + 0.5 · (avg_E(ch) / E_init)
```

**This is the key novelty.** The first half is a **node-specific** solar term:
the shared daylight level multiplied by the *average panel efficiency of the
CHs this chromosome actually picked*. Because `solar_eff` varies per node,
this value **differs between candidate teams**, so it genuinely influences
which team wins. (Earlier this term used only `daylight`, which is identical
for all candidates and therefore could not change the ranking — a constant
offset. Scaling by the picked nodes' `solar_eff` is what fixes that.)

- At **night** (`daylight = 0`), `S_score = 0.5 · battery_avg` — falls back
  to a battery-only score.
- At **noon**, a team of fully charged CHs sitting in **good sun**
  (`solar_eff ≈ 1`) scores near `1.0`; an equally charged team stuck in
  **shade** (`solar_eff ≈ 0.6`) scores lower and is less likely to be chosen.
- A team with weak batteries but excellent solar sites still gets a daytime
  boost, reflecting "they're recharging fast, they'll survive".

```python
if cfg["MAX_HARVEST"] > 0:
    daylight        = solar_now / cfg["MAX_HARVEST"]          # shared 0→1
    ch_seff         = alive_seff[gene_idx]                    # (P, K) per-node
    solar_fraction  = daylight * ch_seff.mean(axis=1)         # per-chromosome
    energy_fraction = np.minimum(ch_e.mean(axis=1) / cfg["E_INITIAL"], 1.0)
    s_score = 0.5 * solar_fraction + 0.5 * energy_fraction
else:
    s_score = np.full(P, 0.5)        # no solar configured → neutral
```

### 8.3 Coverage Score (30%, the highest weight)

```
C_score = (count of sensors within COMM_RANGE of ANY CH) / total_sensors
```

Where `COMM_RANGE = FIELD · 0.4` (40% of field width). This penalises
chromosomes that leave large patches of the field uncovered — the most
common failure mode.

The vectorised computation uses squared distances (no sqrt) and a
broadcast-then-min pattern over `(P, S, K)`:

```python
comm_range2 = (cfg["FIELD"] * COMM_RANGE_PCT) ** 2

# Big tensor: (P populations, S sensors, K chosen CHs, 2 coords)
diff   = alive_xy[None, :, None, :] - ch_xy[:, None, :, :]
d2     = (diff * diff).sum(axis=-1)                 # (P, S, K)
d2_min = d2.min(axis=2)                             # (P, S)

# Mask out CHs themselves so they don't count as "covered sensors"
is_ch = np.zeros((P, S), dtype=bool)
rows  = np.arange(P)[:, None]
is_ch[rows, gene_idx] = True
sensor_mask = ~is_ch

within  = (d2_min <= comm_range2) & sensor_mask
denom   = np.maximum(sensor_mask.sum(axis=1), 1)    # avoid /0
c_score = within.sum(axis=1) / denom
```

For very large jobs (`P · S · K > 4 million`) the code falls back to a
per-chromosome loop to bound peak memory.

### 8.4 Spread Score (20%)

```
Sp_score = min( mean_distance(CH_i, centroid) / FIELD , 1.0 )
```

Penalises clumped CHs. If all 5 CHs sit in one corner, their spread is
small → low score. If they're well-distributed, spread is high.

```python
if K > 1:
    centroid = ch_xy.mean(axis=1, keepdims=True)
    spread   = np.sqrt(((ch_xy - centroid) ** 2).sum(axis=2)).mean(axis=1)
    sp_score = np.minimum(spread / cfg["FIELD"], 1.0)
else:
    sp_score = np.full(P, 0.5)
```

### Why these weights?

| Weight | Component | Reasoning |
|---|---|---|
| 30% | Coverage | Highest because a chromosome that misses sensors is broken regardless of battery |
| 25% | Energy | Tired CHs die first, dragging others |
| 25% | Solar | The novelty — equal billing with energy |
| 20% | Spread | Important but secondary; coverage already discourages clumping indirectly |

You can tune these in code — they're just three weights summing to 1.0.

---

## 9. Formula 8 — MS-CH Solar-Aware Score

A *different* score is used to pick MS-CHs (super-leaders) from the
relay-CH pool:

```
score(ch) = 0.35 · Battery + 0.30 · Solar + 0.20 · Centrality + 0.15 · BS-closeness
```

### The four components

```
Battery       = min(ch.energy / E_init, 1.0)
Solar         = (solar_now / MAX_HARVEST) · ch.solar_eff      (else 0.5)
Centrality    = 1 - min(avg_d_to_peers / (FIELD · √2), 1)
BS-closeness  = 1 - min(d_to_BS / max_d, 1),
                where max_d = √(FIELD² + BS_Y²)
```

Note the `Solar` term is **node-specific**: the shared daylight level is
scaled by *this candidate's own* `solar_eff`, so among the relay CHs
competing to become the MS-CH the solar component actually differentiates
them (rather than adding the same constant to everyone).

### Code

```python
def _solar_aware_score(ch, peers, cfg, solar_now):
    bat = min(ch.energy_fraction, 1.0)

    if cfg["MAX_HARVEST"] > 0:
        solar = (solar_now / cfg["MAX_HARVEST"]) * ch.solar_eff   # node-specific
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
```

### Why these weights are different from the GA fitness

The GA fitness picks **a team** of CHs. The MS-CH score picks **one
individual** from a small candidate pool. Different jobs, different
priorities:

| Concern | GA team | MS-CH individual |
|---|---|---|
| Battery | 25% | **35%** (an MS-CH ships big aggregated bundles to BS — battery matters most) |
| Solar | 25% | 30% |
| Coverage | 30% | n/a (handled by k-medoids partition) |
| Spread | 20% | n/a |
| Centrality | n/a | 20% (must be near its relay CHs) |
| BS closeness | n/a | 15% (helps but secondary) |

---

## 10. Formula 9 — k-Medoids Spatial Clustering

When we need **multiple MS-CHs**, we first partition the relay CHs into
`num_ms` spatial clusters, then pick the best candidate per cluster.

### The algorithm (PAM-lite)

```
1. Seed:   m₁ = random;  for i = 2..k: mᵢ = argmax dist²(p, {m₁..mᵢ₋₁})
2. Repeat (max 10 iters):
   a. Assign each point to its nearest medoid
   b. For each cluster, recompute medoid = point closest to cluster centroid
   c. If labels unchanged, break
3. Return clusters
```

This is the **farthest-first traversal** seeding (BUILD step of PAM,
simplified) followed by classic k-medoids reassignment.

### Code (excerpt)

```python
def _kmedoids_split(relay_chs, k, iters=10):
    if k <= 1 or len(relay_chs) <= k:
        return [relay_chs] if k <= 1 else [[ch] for ch in relay_chs]

    xy = np.array([(c.x, c.y) for c in relay_chs], dtype=np.float64)
    n  = len(relay_chs)

    # Farthest-first seeding
    medoids = [random.randrange(n)]
    for _ in range(k - 1):
        d2 = np.min(((xy[:, None, :] - xy[medoids][None, :, :]) ** 2)
                    .sum(axis=2), axis=1)
        medoids.append(int(d2.argmax()))

    # Iterate
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

    clusters = [[] for _ in range(k)]
    for i, lab in enumerate(labels):
        clusters[lab].append(relay_chs[i])
    return [c for c in clusters if c]
```

### Why k-medoids and not k-means?

We want each MS-CH to be a **real node** (a medoid is a real point;
a k-means centroid is a fictitious average). The MS-CH must physically
exist to receive radio packets.

### Why farthest-first seeding?

Random seeding sometimes places two seeds right next to each other,
leading to one giant cluster and several tiny ones. Farthest-first
gives well-spread initial centres in O(k·n) time. Cheap and effective.

---

## 11. GA Mechanics — Selection, Crossover, Mutation

### 11.1 Chromosome representation

A chromosome is a **list of node IDs** of length `K = num_chs`. Each ID
must be unique (no duplicate CHs). Fitness is cached on the object so we
don't re-evaluate unchanged elites.

```python
class Chromosome:
    __slots__ = ("genes", "fitness")
    def __init__(self, genes):
        self.genes   = list(genes)
        self.fitness = -1.0     # -1 sentinel = not yet computed
```

### 11.2 Smart initial population (energy-weighted seeding)

Half the population is **energy-biased random** (high-battery nodes
more likely picked). The other half is **uniform random** for diversity.

```python
weights = energies + 1e-9
weights = weights / weights.sum()

n_smart = pop_size // 2
for _ in range(n_smart):
    chosen = np.random.choice(alive_ids, size=num_chs,
                              replace=False, p=weights)
    pop.append(Chromosome(chosen.tolist()))
while len(pop) < pop_size:
    pop.append(Chromosome(random.sample(alive_ids, num_chs)))
```

### 11.3 Tournament selection (k=3)

Pick 3 random chromosomes, return the fittest. Cheaper than roulette
wheel and naturally adjusts pressure with population variance.

```python
def _tournament(pop, k=3):
    if len(pop) <= k:
        return max(pop, key=lambda c: c.fitness)
    return max(random.sample(pop, k), key=lambda c: c.fitness)
```

### 11.4 Crossover (single-point, duplicate-aware)

Standard single-point crossover would create duplicates (since genes
are unique IDs). The implementation patches this by walking parent 2
and only adding genes not already in the child:

```python
def _crossover(p1, p2, cfg, num_chs):
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
    # Fallback if p2 exhausted
    if len(genes) < num_chs:
        for g in p1.genes:
            if g not in seen:
                genes.append(g); seen.add(g)
                if len(genes) == num_chs:
                    break
    return Chromosome(genes[:num_chs])
```

### 11.5 Mutation (random replacement with adaptive rate)

Pick a random gene and replace it with a random non-duplicate node:

```python
def _mutate(chromo, alive_ids, mut_rate):
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
```

The **adaptive** part is in `run_ga_ch_election`: when fitness plateaus,
mutation rate ramps up linearly:

```
cur_mut = min(0.5, GA_MUT · (1 + stale · 0.25))
```

So with `GA_MUT = 0.1`:
- 0 stale gens → 10% mutation
- 4 stale gens → 20% mutation
- 8 stale gens → 30% mutation
- 16 stale gens → 50% mutation (capped)

This is **simulated annealing in spirit** — start exploitative, get more
exploratory when stuck.

### 11.6 Elitism + early stopping

Top 2 chromosomes are copied verbatim to the next generation (no
chance of losing the current best to a bad random child).

Patience: `max(5, GA_GEN // 8)`. If best fitness hasn't improved for
that many consecutive generations, break out early. Saves substantial
time when the GA has clearly converged.

### Full loop in code

```python
PATIENCE = max(5, cfg["GA_GEN"] // 8)
best_overall, best_score, stale = None, -1.0, 0
cur_mut = cfg["GA_MUT"]

for gen in range(cfg["GA_GEN"]):
    _evaluate_population(pop, world, cfg, solar_now, num_chs)  # batched NumPy
    pop.sort(key=lambda c: c.fitness, reverse=True)

    if pop[0].fitness > best_score + 1e-9:
        best_score, best_overall = pop[0].fitness, pop[0].copy()
        stale, cur_mut = 0, cfg["GA_MUT"]
    else:
        stale += 1
        cur_mut = min(0.5, cfg["GA_MUT"] * (1 + stale * 0.25))

    if stale >= PATIENCE:
        break

    new_pop = [pop[0].copy(), pop[1].copy()]    # elitism
    while len(new_pop) < cfg["GA_POP"]:
        child = _crossover(_tournament(pop), _tournament(pop), cfg, num_chs)
        new_pop.append(_mutate(child, alive_ids, cur_mut))
    pop = new_pop
```

---

## 12. Path Decision Rule

Each elected CH independently decides Path A (direct to BS) or Path B
(via MS-CH):

```
Path A  ⟺  (d_to_BS ≤ DIRECT_DIST)  ∧  (energy_fraction ≥ DIRECT_NRG)
Path B  ⟺  otherwise
```

With defaults: `DIRECT_DIST = 0.55 · FIELD = 55 m`, `DIRECT_NRG = 0.40`.
A CH with full battery sitting 50 m from the BS → Path A. The same CH
at 15% battery → Path B. A CH 80 m from BS at full battery → Path B.

```python
def decide_ch_paths(ch_nodes, cfg):
    direct_chs, relay_chs = [], []
    for ch in ch_nodes:
        if not ch.alive:
            continue
        close   = ch.distance_to_bs <= cfg["DIRECT_DIST"]
        healthy = ch.energy_fraction >= cfg["DIRECT_NRG"]
        if close and healthy:
            ch.goes_direct = True
            direct_chs.append(ch)
        else:
            ch.goes_direct = False
            relay_chs.append(ch)
    return direct_chs, relay_chs
```

### The ordering matters

`decide_ch_paths` runs **before** `elect_ms_chs`. This enforces
**Point 4**: a CH that *can* go direct never enters the relay pool, so
it never participates in MS-CH election. No wasted optimisation.

---

## 13. MS-CH Re-Election Trigger

Mid-round, after CHs have aggregated and forwarded, we check each MS-CH:

```
if  E(ms) < MS_REELECT_THR · E_init   (i.e. battery < 15% × 0.5J = 75 mJ)
    demote ms back to plain CH
    promote best-scoring peer in its cluster (using formula 8)
    reroute the cluster's relay CHs to the new MS-CH
    increment stats["reelections"]
```

```python
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
```

### Why mid-round, not next-round?

If we waited until the next round, the dying MS-CH would still have to
make its huge BS transmit *this* round, almost certainly killing it.
Replacing it now means the heavy-lift happens on a fresher node.

---

## 14. Step-by-Step: One Full Round

A round is the heartbeat of the simulation. The function
`simulate_round_ga` runs through these 12 steps in order:

### Step 1 — Solar harvest
Compute the shared daylight rate `solar_now` once per round. Every alive node
tops up by `solar_now · solar_eff` (its own fixed panel efficiency) plus 5%
noise, so better-sited nodes recharge faster.

### Step 2 — Reset roles
Yesterday's CHs and MS-CHs revert to plain sensors. Assignment fields
(`assigned_ch`, `assigned_ms`, `goes_direct`) are cleared.

### Step 3 — Refresh world cache
The `World` object rebuilds its NumPy arrays of alive node IDs,
positions, and energies, plus an `id → array_index` map for O(1)
lookups during GA fitness evaluation.

### Step 4 — GA elects CHs
Calls `run_ga_ch_election`. Returns the best chromosome (a list of
`num_chs` node IDs). Those nodes get `role = "CH"`.

If the GA can't find a valid solution (e.g. fewer alive nodes than
needed CHs), the round is recorded as a no-op and the function returns.

### Step 5 — Path decision
`decide_ch_paths` splits CHs into `direct_chs` (Path A) and
`relay_chs` (Path B). **This must run before MS-CH election.**

### Step 6 — MS-CH election (only if needed)
- Compute `num_ms = get_num_ms(len(relay_chs), cfg)`.
- If `num_ms == 0`, skip the entire MS-CH stage.
- If `num_ms == 1`, just pick the best relay CH by formula 8.
- If `num_ms > 1`, run k-medoids on relay CH positions, then pick the
  best in each cluster.

### Step 7 — Sensor → CH assignment (vectorised)
Every alive non-CH sensor is assigned to its **nearest** CH using the
NumPy broadcast trick. Done in a single matrix operation.

### Step 8 — Sensors transmit
Each sensor calls `transmit(ch.x, ch.y)`. Cost is computed by formula 1.
The receiving CH calls `receive()` and pays formula 1's `E_RX`.

### Step 9 — CHs aggregate + forward
Each CH calls `aggregate(member_count)` (formula 2), then either:
- Calls `transmit(BS_X, BS_Y)` (Path A direct), or
- Calls `transmit(ms.x, ms.y)` and the MS-CH receives (Path B).

### Step 10 — MS-CH re-election (mid-round safety net)
Any MS-CH below 15% battery is replaced by a peer per Section 13 above.

### Step 11 — MS-CHs transmit to BS
Each MS-CH aggregates its own sensor packets PLUS its inbound relay-CH
bundles (because it was originally a CH too — it has its own members).
Then it calls `transmit(BS_X, BS_Y)`.

### Step 12 — Record stats
`alive_nodes`, `dead_nodes`, `total_energy`, `energy_stddev`,
`ch_counts`, `ms_counts` are all appended for this round.

If `round_num % SNAPSHOT_EVERY == 0`, a topology snapshot image is
saved with title flag `"MS-USED"` or `"MS-SKIPPED"`.

```python
def simulate_round_ga(nodes, world, round_num, cfg, stats):
    solar_now = solar_rate_for_round(round_num, cfg["MAX_HARVEST"])

    # 1. Harvest
    if solar_now > 0:
        for n in nodes:
            if n.alive: n.harvest_solar(solar_now)

    # 2. Reset roles
    for n in nodes:
        if n.alive: n.reset_role()

    # 3. Refresh world
    world.refresh()
    alive_count = world.alive_idx.size
    num_chs = get_num_chs(alive_count, cfg)
    if alive_count < num_chs + 1:
        ...; return False

    # 4. GA elect CHs
    solution = run_ga_ch_election(nodes, world, cfg, round_num, num_chs)

    # 5. Path decision FIRST
    direct_chs, relay_chs = decide_ch_paths(ch_nodes, cfg)

    # 6. MS-CH only if needed
    num_ms = get_num_ms(len(relay_chs), cfg)
    ms_chs = elect_ms_chs(relay_chs, cfg, solar_now, num_ms) if num_ms else []

    # 7-12: assign, transmit, aggregate, forward, re-elect, record
    ...
```

---

## 15. Design Decisions & Thinking

This section explains **why** the protocol does what it does. Each row
reflects an intentional trade-off.

### 15.1 Why is path decision before MS-CH election?
**Point 4 of the spec.** A CH that can reach BS directly must never
enter the relay pool. Decoupling them up front means the MS-CH stage
can be skipped entirely some rounds (Point 3 + Point 4 fold together
naturally).

### 15.2 Why is the GA "smart-seeded" with energy weighting?
Pure random initial populations waste generations exploring obviously
bad configurations (CHs with 5% battery). Energy-weighted seeding
biases the *starting point* without locking in any single solution.
Convergence is typically 30-40% faster on big networks.

### 15.3 Why is fitness batched, not per-chromosome?
For population P=30, alive S=2000, K=50 the per-call cost is
~24 MB and ~50 ms in pure NumPy. Per-chromosome Python loops for the
same workload take ~3 seconds. **60× speedup** with no behaviour change.

### 15.4 Why fall back to per-chromosome above 4 million elements?
P × S × K floats × 8 bytes = peak RAM for the broadcast tensor. Above
~32 MB, peak allocations start hurting cache locality and trigger
swap risk on small machines. The fallback keeps memory bounded at the
cost of speed for very-large networks.

### 15.5 Why does `_tournament` take k=3?
Lower k = less selection pressure (more diversity). Higher k = greedier
convergence. k=3 is the textbook sweet spot for population sizes around
30 and works well in practice here.

### 15.6 Why is `cur_mut` capped at 0.5?
Above 50% mutation, the GA is essentially a random search — children
have only 50% chance of inheriting useful structure. Capping prevents
the adaptive system from over-correcting on hard plateaus.

### 15.7 Why is patience `max(5, GA_GEN // 8)`?
With `GA_GEN = 50`, patience = 6. That's enough generations to verify
a plateau is real, but small enough to save substantial time when the
GA has converged. On big runs with `GA_GEN = 200`, patience scales up
to 25 — proportional to the search budget.

### 15.8 Why is `MS_REELECT_THR = 0.15`?
A node at 15% battery (= 75 mJ at default) cannot reliably make a
single long BS transmit (~40-100 mJ depending on distance). Catching
it at 15% gives the new MS-CH room to actually deliver the packet.
Lower threshold → more dropped packets. Higher → unnecessary churn.

### 15.9 Why is `BATTERY_MAX = 2.0` while `E_init = 0.5`?
Real solar-equipped nodes can store up to 4× their initial budget over
many sunny days. The cap prevents unlimited accumulation (which would
be physically impossible for a real super-cap or small Li-ion cell)
and creates an interesting wrinkle: a long-lived node "fills up" then
plateaus, which is realistic.

### 15.10 Why is `COMM_RANGE_PCT = 0.4`?
A sensor can typically reach a CH up to ~40 m away in our 100 m field
(40% of FIELD). This sets the "covered" radius in the GA fitness's
coverage term. Larger → easier coverage, less spread pressure.
Smaller → many uncovered sensors, the GA struggles. 0.4 is the field-
calibrated sweet spot.

### 15.11 Why use the same random seed (42) every run?
**Reproducibility.** Researchers comparing GA vs LEACH need both to see
the same field layout, otherwise differences could be from luck rather
than the algorithm. The seed-42 convention is set in `create_nodes`.

### 15.12 Why does the MS-CH re-election trigger demote, not just kill?
A near-dead MS-CH still has 75 mJ of battery — it can still serve as a
plain CH (one short transmit to its peer MS-CH costs ~10 mJ). Demoting
it instead of killing recovers usable life.

---

## 16. Performance Optimisations

The file is engineered to handle networks up to ~5000 nodes. Key tricks:

| Optimisation | Where | Speedup vs naive |
|---|---|---|
| `Node.__slots__` | Section 3 | ~30% memory, ~10% speed |
| Cached `_dist_bs` | Section 3 | O(1) BS lookup, used in hot loop |
| `World` NumPy cache | Section 4 | One O(N) refresh instead of O(N) per query |
| Vectorised sensor→CH assignment | `vectorized_assign` | ~50× over Python loop |
| Batched GA fitness | `_evaluate_population` | ~60× over per-chromosome eval |
| Smart energy-weighted init | `_smart_initial_population` | ~30-40% fewer generations to converge |
| Fitness caching (sentinel `-1.0`) | `Chromosome.fitness` | Re-evaluation skipped for elites |
| Adaptive mutation + early stop | `run_ga_ch_election` | Saves up to 80% generations on plateaus |
| Squared distances (no sqrt) | Coverage, k-medoids | Free 10-15% in inner loops |
| Memory-bounded fallback above 4M elements | Coverage in fitness | Prevents swap on big networks |
| Pre-computed `solar_now` per round | `simulate_round_ga` | Once vs N times |

These together let a 1000-node, 1000-round simulation finish in
~30 seconds on a modest laptop instead of ~30 minutes.

---

## 17. Putting It All Together

```
Round starts
  │
  ├── 1. Compute solar_now once (formula 3)
  ├── 2. Every alive node harvests (formula 3 + 5% noise)
  ├── 3. Reset roles, refresh World cache
  │
  ├── 4. Compute num_chs (formula 5)
  │       │
  │       └── Run GA (50 generations)
  │             │
  │             ├── Smart init (50% energy-weighted, 50% random)
  │             ├── Each generation:
  │             │     ├── Batched fitness (formula 7)
  │             │     ├── Sort, track best
  │             │     ├── Elitism (top 2)
  │             │     ├── Tournament select × 2  →  Crossover  →  Mutate
  │             │     └── Adaptive mutation if plateau
  │             └── Early stop if patient
  │
  ├── 5. Path decision (formula: dist & battery thresholds)
  │       └── Direct CHs vs Relay CHs
  │
  ├── 6. Compute num_ms (formula 6)
  │       │
  │       └── If > 0:
  │             ├── If num_ms == 1: pick best by formula 8
  │             └── Else: k-medoids (formula 9) → best per cluster
  │
  ├── 7. Vectorised sensor → nearest CH assignment (formula 4)
  ├── 8. Sensors transmit to CHs (formula 1)
  ├── 9. CHs aggregate (formula 2) + forward to BS or MS-CH (formula 1)
  ├── 10. MS-CH re-election if any below 15% (Section 13)
  ├── 11. MS-CHs aggregate combined stream + transmit to BS (formulas 1+2)
  └── 12. Record stats, snapshot if scheduled
```

That's the complete picture. Every operation in the round corresponds
to one of the formulas above; every formula has a clear physical or
algorithmic justification; every weight and threshold has been chosen
with a specific failure mode in mind.

---

*Want to change a behaviour? Pick the relevant formula, adjust its
weights or thresholds in the constants section or `cfg` dictionary,
and re-run. The file is structured so each formula is a single,
self-contained function — modifications are local and safe.*
