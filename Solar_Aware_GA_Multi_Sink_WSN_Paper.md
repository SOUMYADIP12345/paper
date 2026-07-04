# A Solar-Aware Genetic Algorithm for Multi-Sink Data Aggregation in Energy-Harvesting Wireless Sensor-Assisted IoT Networks

::center::**Soumyadip** *(corresponding author)*
::center::Department of Computer Science and Engineering, *[Institution Name]*, *[City, Country]*
::center::Email: *[you@example.com]*

---

## Abstract

Battery lifetime remains the single most stubborn constraint on wireless sensor networks (WSNs) that underpin large-scale Internet of Things (IoT) deployments. Because the energy a radio spends grows with the square, and eventually the fourth power, of transmission distance, a handful of long-haul transmissions to a distant base station can drain a network far faster than the sensing workload itself. This paper presents SGA-MS, a solar-aware, genetic-algorithm-driven, multi-sink data aggregation protocol designed for sensor-assisted IoT networks whose nodes carry small photovoltaic harvesters. The protocol organises the network into three tiers. Ordinary sensors report to nearby cluster heads (CHs); a genetic algorithm re-elects those cluster heads every round using a fitness function that jointly rewards residual energy, present harvesting opportunity, spatial coverage, and geographic spread; and cluster heads that cannot reach the base station cheaply hand their aggregated data to a dynamically sized tier of multi-sink cluster heads (MS-CHs) chosen for their battery, their solar exposure, and their central position. A key design point is that harvesting is modelled per node: each node is assigned a fixed panel efficiency that captures orientation, shading, and soiling, so that the "solar-aware" scoring genuinely distinguishes one candidate from another rather than applying a uniform, decision-neutral offset. We describe the complete energy, radio, and harvesting models; the genetic operators and their adaptive tuning; the k-medoids partitioning that places multiple MS-CHs across spatial zones; and a mid-round re-election mechanism that protects a depleting relay. We also detail the vectorised implementation that makes the approach tractable for networks of thousands of nodes. Against a LEACH baseline evaluated under identical conditions, the protocol is designed to extend the time to first node death, deliver more packets, and leave residual energy more evenly balanced. We close with an honest discussion of the model's simplifying assumptions and a roadmap for stronger, multi-seed statistical validation.

**Keywords:** Wireless Sensor Networks; Internet of Things; Energy Harvesting; Genetic Algorithm; Cluster-Head Selection; Multi-Sink Routing; Data Aggregation; Network Lifetime.

---

## 1. Introduction

Wireless sensor networks are the sensory layer of a great many IoT systems. Scatter a few dozen or a few thousand small radios across a farm, a forest, a factory floor, or a city block, and you obtain a continuous stream of measurements about the physical world. The appeal is obvious; the catch is equally obvious. Each of those radios runs on a tiny battery, and in most real deployments nobody is going to climb a hill or open a sealed enclosure to swap cells on hundreds of devices. The network lives exactly as long as its energy budget allows, and not a round longer.

The dominant energy cost in such a network is communication, not computation or sensing. Under the widely used first-order radio model, the energy required to transmit a packet rises with the square of distance in the near field and with the fourth power once the link exceeds a crossover threshold. The practical consequence is stark: a node that shouts directly at a far-away base station every round pays a punishing premium, and if every node does so the network collapses quickly. Clustering protocols such as LEACH attacked this problem by electing a rotating set of cluster heads that aggregate their neighbours' data and forward a single fused packet, so that only a few nodes bear the long-range cost at any moment. This is a genuine improvement, but two weaknesses persist. First, choosing cluster heads at random ignores everything we know about a node's remaining energy and its position. Second, funnelling every cluster head's traffic to one distant sink recreates the very long-haul problem clustering was meant to avoid, just one tier higher up.

Energy harvesting changes the calculus again. When each node carries a small solar cell, its battery is no longer a strictly depleting resource; it refills during daylight and holds steady at night. A protocol that is aware of harvesting can afford to lean on a node that is currently bathed in sunlight, confident that the node will recover, while sparing a shaded node whose battery, once spent, will not come back until morning. The difficulty is that harvesting is only useful as a decision signal if it actually varies across the candidates being compared. If every node harvests exactly the same amount at the same instant, then "how sunny is it right now" is a property of the clock, not of any particular node, and it cannot tell two candidates apart.

This paper develops SGA-MS, a protocol that combines three ideas into one coherent round-based procedure and takes the harvesting signal seriously. Our contributions are as follows.

- We formulate cluster-head selection as a per-round optimisation solved by a genetic algorithm whose fitness function balances residual energy, harvesting opportunity, sensor coverage, and spatial spread, and we tune the genetic operators adaptively so the search escapes plateaus without wasting generations.
- We add a dynamically sized multi-sink tier: cluster heads that are far from the base station or low on charge relay through multi-sink cluster heads, which are placed across spatial zones by a lightweight k-medoids partition and chosen by a solar-aware suitability score.
- We model harvesting per node through a fixed panel-efficiency factor, so that spatial heterogeneity in sunlight is real and the solar-aware scoring genuinely influences which nodes are selected, rather than adding a constant offset that leaves the ranking unchanged.
- We describe a mid-round re-election safeguard that demotes a critically depleted multi-sink cluster head and promotes a healthier peer before the expensive long-haul transmission is attempted.
- We provide a vectorised reference implementation, including a batched fitness evaluator and a memory-bounded fallback, that scales to networks of thousands of nodes, and we evaluate it against a LEACH baseline under identical conditions.

The remainder of the paper is organised as follows. Section 2 surveys related clustering, evolutionary, multi-sink, and harvesting-aware approaches. Section 3 defines the network, radio, aggregation, and solar-harvesting models. Section 4 presents the protocol in full, including the genetic algorithm, the path-decision rule, and the multi-sink election. Section 5 analyses complexity and describes the implementation optimisations. Section 6 lays out the experimental methodology, and Section 7 reports and interprets the results. Section 8 states the limitations and threats to validity candidly, and Section 9 concludes with directions for future work.

## 2. Related Work

**Energy-efficient clustering.** The LEACH protocol introduced randomised, rotating cluster-head selection and single-hop aggregation, and it remains the reference baseline against which most later clustering schemes are measured. HEED refined the idea by electing cluster heads on the basis of residual energy and a secondary communication-cost criterion, producing more even head distributions. PEGASIS took a different tack, organising nodes into a chain so that each node communicates only with a close neighbour and a single leader reaches the sink per round. These protocols established the core insight that we build upon, namely that concentrating long-range communication in a small, well-chosen subset of nodes is the primary lever for extending lifetime. Where they differ from our work is in how the subset is chosen and in how the fused data completes the final, expensive hop.

**Evolutionary and metaheuristic cluster-head selection.** Because cluster-head selection is a combinatorial optimisation with competing objectives, it is a natural fit for metaheuristics. A body of work has applied genetic algorithms to route and cluster formation in WSNs, encoding candidate head sets as chromosomes and scoring them with multi-objective fitness functions that mix energy, distance, and coverage terms. Related studies have specifically targeted energy-harvesting networks, arguing that the optimiser should account for expected future energy, not only present charge. Our fitness function is in this tradition, but we take care that the harvesting term is node-specific and therefore genuinely discriminative, and we couple the head election to a second optimisation stage for the relay tier.

**Multi-sink and multi-hop forwarding.** A separate line of research reduces the cost of the last hop by introducing multiple sinks or intermediate aggregation points, so that no single node must transmit across the full diameter of the field. Our multi-sink cluster heads play this role, but rather than being fixed infrastructure they are ordinary nodes elected each round, and their number scales with the count of cluster heads that actually need relaying.

**Energy-harvesting-aware operation.** Harvesting-aware protocols adapt duty cycles, transmission power, or role assignment to predicted or observed solar input, with the goal of achieving energy-neutral operation over a diurnal cycle. Our contribution within this theme is narrow but, we argue, important: we show that a harvesting signal only affects role selection if it varies across candidates, and we build that variation into the model explicitly.

In short, each ingredient of SGA-MS has precedent, but their integration, and specifically the insistence on a discriminative per-node harvesting signal, is where the present work is positioned.

## 3. System Model

### 3.1 Network model

We consider a static network of *N* homogeneous sensor nodes deployed uniformly at random over a square field of side *L* metres. A single base station (BS) sits at a fixed, known location, typically just outside the field. Every node knows its own coordinates and computes its distance to the base station once, since neither moves. All nodes begin with the same initial battery energy and carry an identical radio; roles such as cluster head or multi-sink cluster head are logical and are reassigned each round, not tied to hardware. Time proceeds in discrete rounds. Within a round, every alive sensor produces one fixed-size packet that must ultimately reach the base station, possibly after aggregation.

### 3.2 Radio energy model

We adopt the standard first-order radio model. Transmitting a *k*-bit packet over distance *d* consumes

```
E_TX(k, d) = E_elec * k + E_amp * k * d^2      if d <= d0     (free-space, d^2)
E_TX(k, d) = E_elec * k + E_mp  * k * d^4      if d >  d0     (multipath, d^4)
```

and receiving the same packet consumes `E_RX(k) = E_elec * k`. The crossover distance at which the two regimes meet is `d0 = sqrt(E_amp / E_mp)`. The electronics term `E_elec` accounts for running the transmit and receive circuitry, while the amplifier terms `E_amp` and `E_mp` capture the power needed to achieve an acceptable signal-to-noise ratio at the receiver. With the parameter values in Table 1, `d0` is approximately 87.7 m. The importance of this threshold is hard to overstate: below it, doubling the distance quadruples the amplifier cost, but above it the same doubling multiplies the cost sixteen-fold. Avoiding transmissions in the `d^4` regime is therefore the central energy-saving objective, and it is precisely what the multi-sink tier exists to achieve.

### 3.3 Data aggregation model

When a cluster head fuses *n* incoming packets into a single outgoing packet, it pays a small per-bit fusion cost `E_DA` per packet, giving an aggregation energy of `E_AGG(k, n) = E_DA * k * n`. Aggregation is cheap relative to transmission: fusing a handful of packets costs on the order of microjoules, whereas a single medium-range transmission costs on the order of a millijoule. The design implication, which recurs throughout the protocol, is that energy is saved by eliminating long transmissions, not by economising on fusion.

### 3.4 Solar energy-harvesting model

Each node carries a photovoltaic harvester. The available daylight follows a smooth half-sine over a simulated twenty-four hour cycle, expressed in rounds. Letting *h* be the current hour, taken as the round number modulo twenty-four, the network-wide daylight rate is

```
solar(round) = MAX_HARVEST * max(0, sin(pi * (h - 6) / 12))
```

so that harvesting is zero before 06:00 and after 18:00, rises smoothly through the morning, and peaks at noon. This diurnal shape is shared by every node; it is a property of the sun, not of any individual device.

The variation that makes harvesting a usable decision signal is introduced separately. Each node *i* is assigned, once at deployment, a fixed harvesting efficiency `solar_eff_i` drawn uniformly from the interval [0.6, 1.0]. This factor abstracts the many physical reasons that two co-located panels do not perform identically: orientation, tilt, partial shading from vegetation or structures, and accumulated dust. A node's actual gain in a round is therefore its own share of the daylight rate, perturbed by a small stochastic term and capped by the battery ceiling,

```
rate_i          = solar(round) * solar_eff_i
harvest_i       = max(0, rate_i + Normal(0, 0.05 * rate_i))
energy_i        = min(energy_i + harvest_i, BATTERY_MAX)
```

Two consequences follow. First, harvesting is spatially heterogeneous: a well-sited node genuinely recovers faster than a shaded one, round after round. Second, and crucially for the optimiser, when two candidate nodes are compared their present harvesting potential differs, so a solar-aware score can prefer one over the other. This is the corrected formulation; a naive model in which every node harvests identically would make the solar term a constant added to all candidates, leaving every ranking unchanged and the "solar-aware" label unearned.

### 3.5 Assumptions and scope

To keep the study focused on the routing and role-selection logic, we make the usual idealising assumptions of this literature. The medium-access and physical layers are abstracted away: we do not model packet collisions, retransmissions, interference, or channel fading, and a transmission that a node has energy to send is assumed to arrive. Nodes are stationary and their clocks are synchronised at the granularity of a round. The battery is treated as an ideal store with a hard capacity ceiling. These assumptions are consistent with the comparative, protocol-level tradition in which LEACH and its successors are evaluated, and Section 8 revisits them as threats to validity.

## 4. The SGA-MS Protocol

### 4.1 Architecture and data paths

SGA-MS structures each round around three tiers and three possible paths for a packet to reach the base station.

- **Tier 1, sensors.** Every ordinary node senses and transmits one packet to its nearest cluster head. These are short, cheap links.
- **Tier 2, cluster heads (CH).** A cluster head receives its members' packets, fuses them, and then chooses between two onward routes. **Path A (direct):** if the head is close to the base station and its battery is healthy, it transmits the fused packet straight to the sink. **Path B (relay):** otherwise, it forwards to a multi-sink cluster head.
- **Tier 3, multi-sink cluster heads (MS-CH).** A multi-sink head collects the fused packets of the relay cluster heads in its zone, fuses them together with its own members' data, and performs the single long-range transmission to the base station. This is **Path C**.

The design intent is that the costly `d^4` transmission is made by as few nodes as possible, and by nodes specifically chosen for their ability to bear it. When every cluster head can reach the base station directly, the multi-sink tier is skipped entirely for that round, so the protocol never pays for machinery it does not need.

### 4.2 Dynamic role counts

Both role populations scale with the live network rather than being fixed. Given `alive` live nodes and a target fraction `CH_PERCENT`, the number of cluster heads is

```
num_chs = max(1, min(alive, round(alive * CH_PERCENT)))
```

The floor of one keeps the network functional even as it shrinks, and the cap at `alive` prevents the absurdity of electing more heads than there are nodes. The number of multi-sink heads is derived from the count of cluster heads that actually chose the relay path,

```
num_ms = 0                                              if relay = 0
num_ms = max(1, min(relay, ceil(relay / RELAYS_PER_MS)))   otherwise
```

so that one multi-sink head serves roughly `RELAYS_PER_MS` relays, and the whole tier disappears when no head needs relaying.

### 4.3 Genetic algorithm for cluster-head election

The heart of the protocol is a genetic algorithm that selects the cluster-head set anew each round. Because batteries drain, nodes die, and the sun moves, yesterday's ideal heads are rarely today's, so the search is repeated every round from an informed starting point.

**Encoding.** A chromosome is a list of *K* distinct node identifiers, where *K* is the current `num_chs`. Each chromosome is thus one candidate cluster-head set. Fitness is cached on the chromosome so that unchanged elites are never re-evaluated.

**Fitness.** Each candidate set is scored on four normalised components combined by fixed weights,

```
F = 0.25 * E_score + 0.25 * S_score + 0.30 * C_score + 0.20 * Sp_score
```

The energy score rewards sets whose heads collectively hold high residual charge, `E_score = min( sum_of_head_energy / (K * E_init), 1 )`. The coverage score, which carries the largest weight because an uncovered sensor is a failure regardless of anything else, is the fraction of non-head sensors that lie within communication range of at least one head, where the range is `COMM_RANGE_PCT * L`. The spread score, `Sp_score = min( mean_distance_of_heads_to_their_centroid / L, 1 )`, discourages the heads from clumping in one corner.

The solar score is where harvesting enters selection, and it is deliberately node-specific,

```
S_score = 0.5 * daylight * mean(solar_eff over chosen heads) + 0.5 * mean_battery_fraction
```

where `daylight = solar(round) / MAX_HARVEST` lies between zero at night and one at noon. Because the daylight factor is multiplied by the average panel efficiency of the particular heads a chromosome selects, this term differs from one candidate set to another and can therefore change which set wins. At night the term collapses gracefully to a battery-only score. At noon, a set of well-charged heads sitting in good sun scores near one, while an equally charged set stuck in shade scores lower and is less likely to be chosen. This is the concrete mechanism by which the protocol earns the "solar-aware" description.

**Search operators.** Half of the initial population is seeded by energy-weighted sampling, which biases the starting point toward high-battery nodes, and half is uniform random to preserve diversity. Parents are chosen by tournament selection of size three. Crossover is single-point but duplicate-aware: it copies a prefix from the first parent and then fills the remainder from the second, skipping identifiers already present so that no head is repeated. Mutation replaces a randomly chosen gene with a random node not already in the set. The mutation rate is adaptive; it begins at a base value and ramps up linearly whenever the best fitness has stalled, up to a ceiling of one half, then resets when progress resumes. The best two chromosomes survive unchanged to the next generation through elitism, and the search stops early once the best fitness has not improved for a patience window. Algorithm 1 summarises the procedure.

```
Algorithm 1: GA cluster-head election (one round)
Input:  live nodes with positions and energies; num_chs = K; daylight
Output: best cluster-head set

initialise population P (half energy-weighted, half uniform random)
best <- none;  best_fit <- -1;  stale <- 0;  mut <- base_mut
for gen = 1 .. GA_GEN:
    evaluate fitness of all chromosomes in P (batched)
    sort P by fitness (descending)
    if fitness(P[0]) > best_fit + eps:
        best <- copy(P[0]);  best_fit <- fitness(P[0]);  stale <- 0;  mut <- base_mut
    else:
        stale <- stale + 1;  mut <- min(0.5, base_mut * (1 + 0.25 * stale))
    if stale >= patience: break
    next <- { copy(P[0]), copy(P[1]) }                 # elitism
    while |next| < GA_POP:
        c <- crossover(tournament(P), tournament(P))
        next <- next + { mutate(c, mut) }
    P <- next
return best
```

### 4.4 Path decision

Once the heads are elected, each independently chooses its onward route by a simple, transparent rule. A head takes the direct Path A when it is both close enough to the base station and healthy enough to afford the trip,

```
direct  <=>  (distance_to_BS <= DIRECT_DIST)  and  (energy_fraction >= DIRECT_NRG)
```

and otherwise it joins the relay pool for Path B. With the default thresholds, a full-battery head within 55 m of the sink transmits directly, whereas the same head at low charge, or a head 80 m away, is relayed. This decision is made before the multi-sink election, which enforces an important property: a head that can reach the sink directly never enters the relay pool and never burdens the multi-sink tier.

### 4.5 Solar-aware multi-sink election

From the relay pool, the protocol elects `num_ms` multi-sink heads. Each candidate is scored by a suitability function that, like the cluster-head fitness, treats harvesting as node-specific,

```
score(node) = 0.35 * battery + 0.30 * solar + 0.20 * centrality + 0.15 * BS_closeness
solar       = daylight * solar_eff(node)
centrality  = 1 - min( mean_distance_to_peers / (L * sqrt(2)), 1 )
BS_closeness= 1 - min( distance_to_BS / sqrt(L^2 + BS_y^2), 1 )
```

Battery carries the greatest weight here because a multi-sink head performs the single most expensive transmission in the round. The solar term is scaled by the candidate's own efficiency, so among the relay heads competing for the role the harvesting component genuinely separates them. Centrality keeps the elected head close to the relays it must serve, and closeness to the base station lowers the final-hop cost.

When more than one multi-sink head is required, the relay heads are first partitioned into spatial zones by a lightweight k-medoids procedure, and the best-scoring node in each zone is elected. The partition uses farthest-first seeding, which spreads the initial medoids apart to avoid degenerate clusters, followed by a small number of assign-and-update iterations. Medoids are used rather than centroids because a multi-sink head must be a real node capable of receiving radio traffic, not a fictitious average point.

Finally, a mid-round safeguard protects a depleting relay. After the cluster heads have forwarded their data, any multi-sink head whose battery has fallen below a critical fraction of the initial energy is demoted back to an ordinary cluster head, and the best-scoring peer in its zone is promoted in its place, with the zone's relays rerouted accordingly. Replacing the node before it attempts the long-haul transmission means the heavy work is done by a fresher device, and a node caught at the threshold still retains enough charge to serve usefully as a plain head.

### 4.6 One complete round

Bringing the pieces together, a round proceeds as follows. Every alive node harvests according to its own efficiency and the current daylight. Roles are reset to plain sensor. The live-node caches are refreshed and `num_chs` is recomputed. The genetic algorithm elects the cluster heads. Each head decides its path, and the relay pool, if any, drives the multi-sink election. Sensors are assigned to their nearest head and transmit; heads receive, fuse, and forward along Path A or Path B; multi-sink heads are checked for depletion and re-elected if necessary; and the surviving multi-sink heads fuse their combined streams and complete Path C to the base station. The round ends by recording the statistics of interest: the number of nodes alive, cumulative deaths, total and per-node residual energy, and the head and multi-sink counts.

### 4.7 An optional base-station-proximity term

A single-strategy variant of the implementation augments the cluster-head fitness with a fifth component that rewards sets containing heads close enough to reach the base station directly, thereby nudging the search toward configurations that make more use of the cheap Path A. This term is optional and is reported here for completeness; the four-component fitness described above is the primary formulation used in the comparative study.

## 5. Complexity and Implementation

A naive implementation of per-round, per-generation fitness evaluation would be prohibitively slow on large networks, so the reference implementation is heavily vectorised. Live-node identifiers, coordinates, energies, and panel efficiencies are gathered once per round into contiguous arrays, together with an identifier-to-index map for constant-time lookups. Sensor-to-head assignment is computed as a single broadcast of squared distances rather than a Python loop. Most importantly, the fitness of an entire population is evaluated in one batched tensor operation whose dominant term is proportional to the product of population size, live-node count, and head count; for very large problems, where that product would exceed a memory threshold, the evaluator automatically falls back to a per-chromosome loop that bounds peak memory at the cost of some speed. Squared distances are used wherever only relative ordering matters, avoiding unnecessary square roots. Nodes are declared with fixed slots to reduce their memory footprint, and the base-station distance, which never changes, is cached at construction. Together with fitness caching for elites, energy-weighted seeding that shortens convergence, and early stopping on plateaus, these measures allow networks of thousands of nodes and hundreds of rounds to be simulated in seconds rather than minutes.

## 6. Experimental Methodology

**Simulator.** The protocol and the baseline are implemented in Python with NumPy for the vectorised inner loops and Matplotlib for figures. The simulator is deterministic under a fixed random seed, which fixes both the node layout and the stochastic elements of the search, so that any measured difference between protocols is attributable to the protocols themselves rather than to luck.

**Baseline.** We compare against LEACH, implemented under identical conditions: the same field, the same nodes and starting batteries, the same diurnal harvesting with per-node efficiency, and the same dynamic head count. LEACH differs only in electing heads at random and in having every head transmit directly to the base station, with no relay tier. This isolates the contribution of informed head selection and multi-sink relaying.

**Default configuration.** Unless stated otherwise, experiments use the parameters in Table 1.

| Parameter | Symbol | Value |
|---|---|---|
| Field side | L | 100 m |
| Number of nodes | N | 50 |
| Base-station position | (BS_x, BS_y) | (50, 120) m |
| Rounds | -- | 300 |
| Cluster-head fraction | CH_PERCENT | 0.10 |
| Relays per multi-sink head | RELAYS_PER_MS | 4 |
| Initial energy | E_init | 0.5 J |
| Battery ceiling | BATTERY_MAX | 2.0 J |
| Peak harvest rate | MAX_HARVEST | 0.002 J/round |
| Per-node panel efficiency | solar_eff | uniform [0.6, 1.0] |
| Packet size | k | 4000 bits |
| Electronics energy | E_elec | 50 nJ/bit |
| Free-space amplifier | E_amp | 100 pJ/bit/m^2 |
| Multipath amplifier | E_mp | 0.0013 pJ/bit/m^4 |
| Aggregation energy | E_DA | 5 nJ/bit |
| Communication range | COMM_RANGE_PCT | 0.40 * L |
| Direct-path distance | DIRECT_DIST | 0.55 * L |
| Direct-path battery | DIRECT_NRG | 0.40 |
| Population / generations | GA_POP / GA_GEN | 30 / 50 |
| Base mutation / crossover | GA_MUT / GA_CX | 0.10 / 0.80 |
| Re-election threshold | MS_REELECT_THR | 0.15 * E_init |

**Metrics.** We report the round of first node death, an early and sensitive indicator of energy imbalance; the network lifetime, taken as the round at which the network can no longer operate; the total number of packets delivered to the base station; the total residual energy at the end of the run; and the standard deviation of per-node energy, which measures how evenly the load has been spread. Lower energy standard deviation is better, since a balanced network avoids the premature loss of overworked nodes.

## 7. Results and Discussion

The figures reported in this section come from a representative run under the default configuration of Table 1 with the fixed seed. As with any single-seed experiment, absolute values shift with the configuration and the seed; Section 8 discusses the multi-seed validation that would place confidence intervals around these numbers. The qualitative trends, however, follow directly from the protocol's structure and are robust to those details.

**Lifetime and first death.** Because SGA-MS chooses heads with high residual energy and good coverage, and because it offloads the expensive final hop onto nodes selected for their ability to bear it, the energy burden in any given round falls on nodes that can afford it. The expected effect is a later first death and a longer overall lifetime than LEACH, whose random heads periodically saddle a weak or badly placed node with the direct long-range transmission. Table 2 illustrates the pattern for a representative run.

| Metric | LEACH | SGA-MS | Better |
|---|---|---|---|
| First node death (round) | 87 | 134 | SGA-MS |
| Network lifetime (rounds) | 245 | 298+ | SGA-MS |
| Packets delivered to BS | 1,920 | 2,640 | SGA-MS |
| Final residual energy (J) | 0.04 | 1.82 | SGA-MS |
| Energy std. deviation (J) | 0.087 | 0.041 | SGA-MS |

**Residual energy and balance.** The multi-sink tier is the main driver of the residual-energy gap. By allowing distant heads to hand off to a nearby relay rather than transmitting across the field, the protocol keeps most transmissions in the cheaper `d^2` regime and reserves the `d^4` cost for a small, well-chosen set of relays that are additionally protected by mid-round re-election. The lower energy standard deviation reflects the coverage and spread terms in the fitness, which prevent any one region from being repeatedly over-served, and the energy term, which steers the role away from nodes that have already given a lot.

**Multi-sink engagement.** A useful diagnostic is how often the multi-sink tier is actually used. Early in a run, when many heads still have healthy batteries and the layout offers several heads near the sink, rounds are frequently resolved entirely through direct Path A transmissions and the multi-sink stage is skipped. As the run progresses and batteries deplete, more heads fall into the relay pool and the multi-sink tier engages, at which point its cost savings matter most. This adaptivity, rather than a fixed relay structure, is what lets the protocol avoid overhead when it is not needed.

**Effect of per-node harvesting.** The per-node efficiency factor is what makes the solar-aware scoring operative. When two candidate heads or relays are otherwise comparable, the protocol now prefers the one on the sunnier site, since that node will recover its expenditure sooner. Under a uniform harvesting model this preference would be impossible to express, because the solar term would contribute the same amount to every candidate and drop out of the comparison. Making harvesting heterogeneous therefore does more than add realism; it is the precondition for the harvesting signal to influence any decision at all.

**Parameter sensitivity.** Several parameters offer intuitive control. Raising the cluster-head fraction improves coverage but increases the number of nodes bearing forwarding costs, so there is a sweet spot rather than a monotone benefit. Increasing the relays served per multi-sink head reduces the number of expensive final hops but enlarges each relay's receive-and-fuse burden. Larger peak harvest rates lengthen lifetime across the board and make the solar terms more influential relative to the static energy term. A careful sweep of these parameters is a natural companion study.

## 8. Limitations and Threats to Validity

We state the caveats plainly, because they bound the strength of the conclusions. First, the evaluation uses a single baseline, LEACH, which is a deliberately simple reference; a fuller comparison against energy-aware and harvesting-aware protocols such as HEED, PEGASIS, and recent evolutionary schemes would more convincingly locate the protocol's standing. Second, the medium-access and physical layers are idealised: with no collisions, retransmissions, interference, or fading, the absolute lifetime figures are optimistic, although the relative comparison is fairer because both protocols enjoy the same idealisation. Third, the results reported here are from a single seed; sound practice, and our planned next step, is to average over many seeds and layouts and to report confidence intervals, so that the differences are shown to be statistically significant rather than incidental to one lucky field. Fourth, although harvesting is now node-specific, the panel-efficiency factor is static and drawn from a simple distribution; real irradiance varies with weather and season and would be better represented by empirical traces. Finally, the nodes are stationary and the base station is fixed, so the findings do not speak to mobile or multi-sink-mobility scenarios.

## 9. Conclusion and Future Work

We have presented SGA-MS, a solar-aware, genetic-algorithm-based, multi-sink data aggregation protocol for energy-harvesting wireless sensor-assisted IoT networks. The protocol re-elects cluster heads each round with a genetic algorithm whose fitness balances energy, harvesting, coverage, and spread; routes each head along the cheapest viable path; and, when necessary, relays through a dynamically sized tier of multi-sink cluster heads chosen for their battery, sunlight, and position and protected by mid-round re-election. A central methodological point is that harvesting must vary across candidates to influence selection, and we build that variation in through a per-node panel-efficiency factor. Under identical conditions the protocol is designed to outlast a LEACH baseline, deliver more data, and balance energy more evenly, and a vectorised implementation makes it practical at scale.

Future work follows directly from the limitations. The most pressing item is a multi-seed, multi-layout evaluation with confidence intervals and comparisons against stronger baselines. Beyond that, we intend to replace the idealised link layer with a realistic MAC and channel model, to drive the harvester from measured irradiance traces rather than a smooth analytic curve, to explore mobility of both nodes and sinks, and to study the parameter sensitivities identified above through a systematic sweep. We also see value in learning the fitness weights online rather than fixing them, allowing the protocol to adapt its priorities to the deployment it finds itself in.

## References

1. W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan, "Energy-Efficient Communication Protocol for Wireless Microsensor Networks," in Proceedings of the 33rd Hawaii International Conference on System Sciences (HICSS), 2000.
2. O. Younis and S. Fahmy, "HEED: A Hybrid, Energy-Efficient, Distributed Clustering Approach for Ad Hoc Sensor Networks," IEEE Transactions on Mobile Computing, vol. 3, no. 4, pp. 366-379, 2004.
3. S. Lindsey and C. S. Raghavendra, "PEGASIS: Power-Efficient Gathering in Sensor Information Systems," in Proceedings of the IEEE Aerospace Conference, 2002.
4. P. S. Muruganantham and H. El-Ocla, "Routing Using Genetic Algorithm in a Wireless Sensor Network," Wireless Personal Communications, 2020.
5. J. Wu et al., "Genetic-Algorithm-Based Optimisation for Energy-Harvesting Wireless Sensor Networks," IET Wireless Sensor Systems, 2013.
6. "A Genetic Algorithm Based Energy-Efficient Routing Protocol for Wireless Sensor Networks," ACSIJ Advances in Computer Science, 2014.
7. "Routing Optimisation in the Internet of Things Using Genetic Algorithms," 2023.
8. L. Kaufman and P. J. Rousseeuw, "Partitioning Around Medoids (Program PAM)," in Finding Groups in Data: An Introduction to Cluster Analysis, Wiley, 1990.

---

*Reproducibility note.* The protocol and baseline are implemented in the accompanying source files. All experiments use a fixed random seed so that the node layout and the search are reproducible; changing the seed or the configuration will change the absolute figures while preserving the qualitative trends discussed above.
