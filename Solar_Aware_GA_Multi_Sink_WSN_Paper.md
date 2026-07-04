# SGA-MS: A Solar-Aware Genetic-Algorithm Multi-Sink Data-Aggregation Protocol for Prolonging Lifetime in Energy-Harvesting IoT Sensor Networks

::center::**Soumyadip [Last Name]** *(corresponding author)*, **[Co-author Name]**, **[Supervisor Name]**
::center::Department of Computer Science and Engineering, *[Institution Name]*, *[City, Country]*
::center::Corresponding author: *[you@example.com]*   |   ORCID: *[0000-0000-0000-0000]*

---

## Abstract

Battery lifetime is the dominant constraint on the wireless sensor networks (WSNs) that support large-scale Internet of Things (IoT) deployments, because the energy a radio spends grows with the square, and beyond a crossover distance the fourth power, of transmission range. This paper presents SGA-MS, a solar-aware, genetic-algorithm-driven, multi-sink data-aggregation protocol for sensor networks whose nodes carry photovoltaic harvesters. The protocol organises the network into three tiers. Ordinary sensors report to nearby cluster heads; a genetic algorithm re-elects those cluster heads every round using a fitness that jointly rewards residual energy, present harvesting opportunity, spatial coverage, and geographic spread; and cluster heads that cannot reach the base station cheaply relay their aggregated data through a dynamically sized tier of multi-sink cluster heads placed by k-medoids clustering and chosen by a solar-aware suitability score. A central design point is that harvesting is modelled per node through a fixed panel-efficiency factor that captures orientation, shading, and soiling, so that the solar signal genuinely discriminates between candidates rather than adding a decision-neutral constant to all of them. We describe the complete energy, radio, and harvesting models; the genetic operators and their adaptive tuning; the k-medoids partitioning that places multiple relays across spatial zones; and a mid-round re-election mechanism that protects a depleting relay. We give a formal complexity analysis of time, space, and communication overhead, and we detail the vectorised implementation that makes the approach tractable for networks of thousands of nodes. In representative simulations against a LEACH baseline under identical conditions, SGA-MS delayed the first node death by about 54% (round 87 to 134), delivered roughly 38% more packets to the base station, halved the spread of per-node residual energy (0.087 to 0.041 J), and retained substantially more residual energy at the end of the run. The takeaway is that pairing evolutionary, harvesting-aware cluster-head selection with an adaptive multi-sink relay tier can materially extend the operational life of energy-harvesting sensor networks without any change to the hardware. We close with a candid account of the model's simplifying assumptions and a concrete plan for stronger, multi-seed statistical validation.

## Keywords

Wireless Sensor Networks; Internet of Things; Energy Harvesting; Genetic Algorithm; Cluster-Head Selection; Multi-Sink Routing; Data Aggregation; Network Lifetime; Solar-Aware Routing; Clustering

---

## 1. Introduction

### 1.1 Background and motivation

Wireless sensor networks form the sensory substrate of a great many Internet of Things systems. Scatter a few dozen or a few thousand small radios across a farm, a forest, a factory floor, or a city block, and one obtains a continuous stream of measurements about the physical world: soil moisture, ambient temperature, structural vibration, air quality, occupancy, and much more. The appeal of the technology is obvious, and so is its central difficulty. Each of those radios runs on a small battery, and in the overwhelming majority of realistic deployments nobody is going to climb a hillside, open a sealed enclosure, or dig up a buried node to replace cells on hundreds or thousands of devices. The network operates exactly as long as its collective energy budget permits, and not a round longer. Prolonging that operational lifetime is therefore not a secondary optimisation; it is the defining engineering objective of the field.

Two developments sharpen the problem and, at the same time, open a path toward addressing it. The first is the sheer scale and density envisioned for IoT deployments, which makes manual maintenance economically impossible and places a premium on protocols that squeeze the maximum service life out of a fixed energy store. The second is the growing practicality of on-node energy harvesting, and photovoltaic harvesting in particular, which converts a strictly depleting battery into a partially renewable one. A node with a small solar cell recharges during daylight and holds its charge at night, so a well-designed protocol can lean on nodes that are currently well supplied by the sun while sparing those that are not. The combination of these two forces motivates the protocol we present: a routing and role-assignment scheme that is simultaneously aggressive about reducing communication energy and deliberate about exploiting the spatial and temporal structure of harvested energy.

### 1.2 Why the problem is hard

The dominant energy cost in a sensor network is communication, not sensing or computation. Under the widely used first-order radio model, the energy required to transmit a packet rises with the square of distance in the near field and with the fourth power once a link exceeds a crossover threshold. The practical consequence is severe. A node that transmits directly to a distant base station every round pays a punishing premium, and if every node does so the network collapses within hours. This single fact is why clustering protocols exist at all: by electing a small, rotating set of cluster heads that aggregate their neighbours' data and forward one fused packet, the network confines the expensive long-range communication to a few nodes at any instant. The seminal LEACH protocol established this idea and remains the reference against which most later work is measured.

Clustering, however, leaves two problems unresolved. First, choosing cluster heads at random, as LEACH does, throws away everything the network could know about a candidate: its remaining energy, its position relative to the sensors it would serve, and its proximity to the sink. A randomly chosen head with a nearly empty battery, or one stranded in a corner, wastes energy and dies early, dragging its dependents down with it. Second, and more subtly, funnelling every cluster head's traffic to a single distant sink recreates the very long-haul problem that clustering was meant to solve, merely displaced one tier upward. When the sink sits outside the field, the heads farthest from it pay the fourth-power penalty on every round.

Energy harvesting complicates the picture in a way that is easy to get wrong. When each node can recharge, a protocol can afford to burden a node that is currently in strong sunlight, confident that the node will recover, while protecting a shaded node whose spent battery will not return until morning. But this reasoning only works if harvesting actually varies across the candidates being compared. If every node harvests exactly the same amount at the same moment, then how sunny it is right now is a property of the clock, not of any particular node, and it cannot distinguish one candidate from another. A great deal of nominally solar-aware design quietly falls into this trap: it multiplies a candidate's score by a network-wide daylight term that, being identical for all candidates, cancels out of every comparison and changes no decision. Making the harvesting signal genuinely discriminative is a modelling requirement, not a cosmetic one, and it is one of the points on which this paper insists.

### 1.3 Design goals

Three goals shaped the protocol. First, minimise the number and the distance of the expensive long-range transmissions per round, since these dominate the energy budget. Second, select roles using all the information available, namely residual energy, harvesting opportunity, coverage of the sensing field, and spatial distribution, rather than by chance. Third, make the harvesting term operative, so that the label solar-aware corresponds to a real influence on which nodes are chosen. A fourth, cross-cutting goal was practicality: the protocol had to be implementable efficiently enough to simulate networks of thousands of nodes over hundreds of rounds in a reasonable time.

### 1.4 Contributions

This paper makes the following contributions.

- We formulate per-round cluster-head selection as a multi-objective optimisation and solve it with a genetic algorithm whose fitness balances residual energy, harvesting opportunity, sensor coverage, and spatial spread, and whose operators are tuned adaptively so the search escapes plateaus without wasting generations.
- We add a dynamically sized multi-sink relay tier: cluster heads that are far from the base station or low on charge relay through multi-sink cluster heads, which are placed across spatial zones by a lightweight k-medoids partition and chosen by a solar-aware suitability score. The tier scales with the number of heads that actually need relaying and disappears entirely when it is not needed.
- We model harvesting per node through a fixed panel-efficiency factor, so that spatial heterogeneity in sunlight is real and the solar-aware scoring genuinely influences selection, rather than contributing a constant offset that leaves every ranking unchanged.
- We introduce a mid-round re-election safeguard that demotes a critically depleted multi-sink head and promotes a healthier peer before the expensive long-haul transmission is attempted.
- We provide a formal analysis of the protocol's time, space, and communication complexity, and a vectorised reference implementation, including a batched fitness evaluator with a memory-bounded fallback, that scales to thousands of nodes; we evaluate it against a LEACH baseline under identical conditions.

### 1.5 Paper organisation

The remainder of the paper is organised as follows. Section 2 surveys related clustering, evolutionary, multi-sink, and harvesting-aware approaches, grouped by theme, and states how our work differs from each. Section 3 defines the network, radio, aggregation, and solar-harvesting models, fixes the notation, and formalises the optimisation objective. Section 4 presents the protocol in full, with architecture and round diagrams and five pieces of pseudocode. Section 5 analyses time, space, and communication complexity. Section 6 describes the experimental methodology, and Section 7 reports and interprets the results. Section 8 states the limitations and threats to validity candidly, and Section 9 concludes with directions for future work. Two appendices summarise the notation and the implementation.

## 2. Related Work

We organise prior work into six themes and, for each, state precisely how SGA-MS differs.

### 2.1 Classical randomised clustering

The LEACH protocol introduced randomised, rotating cluster-head selection with single-hop aggregation, and its later application-specific formulation refined the energy accounting and the steady-state operation. LEACH's enduring insight is that concentrating long-range communication in a small, rotating subset of nodes is the primary lever for lifetime. Its weakness is that the subset is chosen probabilistically, without regard to residual energy or geometry, and that every head transmits directly to the sink. *We differ* by replacing random selection with an informed, multi-objective genetic search, and by inserting a relay tier so that distant heads do not each pay the full long-haul cost.

### 2.2 Energy-aware and heterogeneous clustering

A family of protocols improved on LEACH by making election energy-aware. HEED elects heads on residual energy with a secondary communication-cost tiebreaker, producing more uniform head distributions. SEP and DEEC target heterogeneous networks in which some nodes start with more energy, weighting election probabilities by relative or residual energy. TEEN and its successor APTEEN adapt to reactive, threshold-driven sensing. These protocols confirm that energy-aware election extends lifetime. *We differ* by optimising a set of heads jointly rather than electing them independently, by adding coverage and spread objectives alongside energy, and by folding a genuinely node-specific harvesting term into the objective.

### 2.3 Chain-based and multi-hop forwarding

PEGASIS organises nodes into a chain so that each communicates only with a close neighbour, and a single leader reaches the sink per round, minimising per-round transmission distance at the cost of latency and chain-maintenance overhead. Multi-hop variants forward through intermediate nodes to avoid long direct links. *We differ* by retaining a clustered structure, which keeps latency low, while borrowing the multi-hop insight in a bounded way: at most one relay hop is inserted, and only for heads that cannot reach the sink cheaply.

### 2.4 Metaheuristic and genetic cluster-head selection

Because cluster-head selection is a combinatorial optimisation with competing objectives, it is a natural target for metaheuristics, and genetic algorithms have been applied to route and cluster formation in WSNs, encoding candidate head sets as chromosomes and scoring them with multi-objective fitness functions. Related studies have specifically addressed energy-harvesting networks, arguing that the optimiser should account for expected future energy rather than only present charge. *We differ* on two points: we ensure the harvesting term is node-specific and therefore discriminative, and we couple the head election to a second optimisation stage that elects the relay tier, so the genetic search and the multi-sink placement reinforce one another.

### 2.5 Multi-sink and multiple base stations

A separate line of research reduces the cost of the final hop by introducing multiple sinks or intermediate aggregation points, so that no single node must transmit across the full diameter of the field. Some approaches assume fixed additional infrastructure; others elect mobile or virtual sinks. *We differ* by making the multi-sink cluster heads ordinary nodes elected afresh each round, whose number scales with the count of heads that need relaying and whose placement follows the live topology through k-medoids, so no fixed infrastructure is assumed.

### 2.6 Energy-harvesting-aware operation

Harvesting-aware protocols adapt duty cycles, transmission power, or role assignment to predicted or observed solar input, aiming for energy-neutral operation over a diurnal cycle. Surveys of harvesting sensor nodes document the wide variation in real panel output due to orientation, shading, and weather. *We differ* in a narrow but important way: we show, and build into the model, the requirement that a harvesting signal must vary across candidates to affect selection, and we realise that variation with a per-node efficiency factor grounded in the physical reasons panels differ.

### 2.7 Positioning

Table 1 situates SGA-MS against representative protocols along the dimensions that matter for this work. Each ingredient of our design has precedent; the contribution lies in their integration and, specifically, in the insistence on a discriminative per-node harvesting signal.

| Protocol | Head selection | Final hop | Energy-aware | Harvesting-aware | Multi-objective |
|---|---|---|---|---|---|
| LEACH | Random | Direct to sink | No | No | No |
| HEED | Residual energy | Direct/multi-hop | Yes | No | Partial |
| SEP / DEEC | Energy-weighted | Direct to sink | Yes | No | Partial |
| PEGASIS | Chain leader | Chain to sink | Partial | No | No |
| GA-based CH | Evolutionary | Direct/multi-hop | Yes | Sometimes | Yes |
| **SGA-MS (this work)** | **Genetic, multi-objective** | **Adaptive relay tier** | **Yes** | **Yes (per node)** | **Yes** |

## 3. System Model and Preliminaries

### 3.1 Network model and assumptions

We consider a static network of *N* homogeneous sensor nodes deployed uniformly at random over a square field of side *L* metres. A single base station sits at a fixed, known location, typically just outside the field. Every node knows its own coordinates and computes its distance to the base station once, since neither node nor sink moves. All nodes begin with identical initial battery energy and carry an identical radio; roles such as cluster head or multi-sink cluster head are logical and are reassigned each round rather than being tied to hardware. Time proceeds in discrete rounds. Within a round, every alive sensor produces one fixed-size packet that must ultimately reach the base station, possibly after one or two stages of aggregation.

We adopt the standard idealising assumptions of the comparative clustering literature so that the study isolates the routing and role-selection logic. The medium-access and physical layers are abstracted: we do not model packet collisions, retransmissions, interference, or fading, and a transmission that a node has the energy to send is assumed to be received. Nodes are stationary, and their clocks are synchronised at the granularity of a round. The battery is treated as an ideal store with a hard capacity ceiling. Section 8 revisits each of these assumptions as an explicit threat to validity.

### 3.2 Notation

Table 2 fixes the notation used throughout.

| Symbol | Meaning |
|---|---|
| N | Number of deployed sensor nodes |
| L | Side length of the square field (m) |
| k | Packet size (bits) |
| d | Distance between a transmitter and a receiver (m) |
| E_elec | Per-bit electronics energy |
| E_amp | Free-space amplifier coefficient (d^2 regime) |
| E_mp | Multipath amplifier coefficient (d^4 regime) |
| E_DA | Aggregation energy per bit per fused packet |
| d0 | Crossover distance, sqrt(E_amp / E_mp) |
| E_init | Initial battery energy of every node |
| B_max | Battery capacity ceiling |
| H_max | Peak network-wide harvest rate per round |
| eff_i | Per-node harvesting efficiency of node i, fixed at deployment |
| K | Number of cluster heads in the current round |
| M | Number of multi-sink cluster heads in the current round |
| p | Target cluster-head fraction (CH_PERCENT) |
| r | Relays served per multi-sink head (RELAYS_PER_MS) |
| R_c | Communication range, a fraction of L |
| D_dir | Maximum distance for a direct head-to-sink link |
| B_dir | Minimum battery fraction for a direct head-to-sink link |
| tau | Multi-sink re-election battery threshold |

### 3.3 Radio energy model

We use the first-order radio model. Transmitting a *k*-bit packet over distance *d* consumes

```
E_TX(k, d) = E_elec * k + E_amp * k * d^2      if d <= d0     (free-space, d^2)
E_TX(k, d) = E_elec * k + E_mp  * k * d^4      if d >  d0     (multipath, d^4)
```

and receiving the same packet consumes `E_RX(k) = E_elec * k`. The crossover distance at which the two regimes meet is `d0 = sqrt(E_amp / E_mp)`. The electronics term accounts for running the transmit and receive circuitry, while the amplifier terms capture the power needed to achieve an acceptable signal-to-noise ratio at the receiver. With the parameter values in Section 6, `d0` is approximately 87.7 m. The importance of this threshold is difficult to overstate: below it, doubling the distance quadruples the amplifier cost; above it, the same doubling multiplies that cost sixteenfold. Keeping transmissions out of the `d^4` regime is therefore the central energy-saving objective, and it is precisely what the multi-sink tier exists to achieve.

### 3.4 Data-aggregation model

When a head fuses *n* incoming packets into a single outgoing packet, it pays a per-bit fusion cost `E_DA` per packet, giving an aggregation energy of `E_AGG(k, n) = E_DA * k * n`. Aggregation is inexpensive relative to transmission: fusing a handful of packets costs on the order of microjoules, whereas a single medium-range transmission costs on the order of a millijoule. The design implication, which recurs throughout the protocol, is that energy is saved by eliminating long transmissions, not by economising on fusion.

### 3.5 Solar energy-harvesting model

Each node carries a photovoltaic harvester. The available daylight follows a smooth half-sine over a simulated twenty-four-hour cycle expressed in rounds. Letting *h* be the current hour, taken as the round number modulo twenty-four, the network-wide daylight rate is

```
solar(round) = H_max * max(0, sin(pi * (h - 6) / 12))
```

so that harvesting is zero before 06:00 and after 18:00, rises smoothly through the morning, and peaks at noon. This diurnal shape is shared by every node; it is a property of the sun, not of any individual device.

The variation that makes harvesting usable as a decision signal is introduced separately. Each node *i* is assigned, once at deployment, a fixed harvesting efficiency `eff_i` drawn uniformly from the interval [0.6, 1.0]. This factor abstracts the many physical reasons that two co-located panels do not perform identically: orientation, tilt, partial shading from vegetation or structures, and accumulated dust. A node's actual gain in a round is its own share of the daylight rate, perturbed by a small stochastic term and capped by the battery ceiling,

```
rate_i    = solar(round) * eff_i
gain_i    = max(0, rate_i + Normal(0, 0.05 * rate_i))
energy_i  = min(energy_i + gain_i, B_max)
```

Two consequences follow. First, harvesting is spatially heterogeneous: a well-sited node genuinely recovers faster than a shaded one, round after round. Second, and crucially for the optimiser, when two candidate nodes are compared their present harvesting potential differs, so a solar-aware score can prefer one over the other. This is the corrected formulation. A naive model in which every node harvests identically would make the solar term a constant added to all candidates, leaving every ranking unchanged and the solar-aware label unearned.

### 3.6 Problem formulation

Let a round *t* assign to each alive node a role and a route. Write `E_i(t)` for the residual energy of node *i* at the start of round *t*, and let the network be considered operational while at least one node can still deliver data. The objective is to maximise the operational lifetime, which we characterise by two lifetime figures of merit: the round of first node death, `T_first = min { t : some node dies in round t }`, and the round of network death, `T_net`. Subject to the radio and harvesting models above, and to the constraint that every alive sensor's packet is routed to the sink, the per-round decision is the choice of the head set, each head's route, and the relay set. Because the state evolves stochastically through harvesting noise and deterministically through energy expenditure, and because an exhaustive search over head sets is combinatorial, we approach the per-round decision heuristically with the genetic algorithm of Section 4, using a surrogate fitness that correlates with long-run lifetime by rewarding energy, coverage, spread, and harvesting opportunity simultaneously.

## 4. The SGA-MS Protocol

### 4.1 Architecture

SGA-MS structures each round around three tiers and three possible paths for a packet to reach the base station, as illustrated in Figure 1.

```
                         [ Base Station ]
                        /       ^        \
                Path A /   Path C|         \  (LEACH: all heads
               (direct)/         |          \   go direct)
                      /          |           \
                 [ CH ]      [ MS-CH ]     [ CH ]
                   ^          ^     ^         ^
          sensors  |   Path B |     | Path B  |  sensors
                   |   (relay)|     |(relay)  |
              [sensor...]  [ CH ] [ CH ]   [sensor...]
                             ^       ^
                        sensors    sensors
```
::center::**Figure 1.** Three-tier architecture and the three data paths. Path A: head to sink directly. Path B: head to a multi-sink head. Path C: multi-sink head to sink.

- **Tier 1, sensors.** Every ordinary node senses and transmits one packet to its nearest cluster head over a short, cheap link.
- **Tier 2, cluster heads.** A head receives its members' packets, fuses them, and chooses between two onward routes. On **Path A** it transmits the fused packet straight to the sink, which it does only when it is close and healthy. On **Path B** it forwards to a multi-sink head.
- **Tier 3, multi-sink cluster heads.** A multi-sink head collects the fused packets of the relay heads in its zone, fuses them with its own members' data, and performs the single long-range transmission to the sink. This is **Path C**.

The design intent is that the costly `d^4` transmission is made by as few nodes as possible, and by nodes specifically chosen for their ability to bear it. When every head can reach the sink directly, the multi-sink tier is skipped entirely for that round, so the protocol never pays for machinery it does not need.

### 4.2 Round structure

Figure 2 summarises the twelve steps of a round.

```
  (1) harvest (per-node eff)  ->  (2) reset roles  ->  (3) refresh caches, compute K
        |
        v
  (4) GA elects K cluster heads
        |
        v
  (5) each head decides Path A (direct) or Path B (relay)
        |
        +-- if relay pool empty --> skip tier 3
        |
        v
  (6) compute M, elect MS-CHs (k-medoids + solar-aware score)
        |
        v
  (7) assign sensors to nearest head  ->  (8) sensors transmit
        |
        v
  (9) heads aggregate + forward (A or B)
        |
        v
  (10) re-elect any depleted MS-CH  ->  (11) MS-CHs aggregate + transmit to sink
        |
        v
  (12) record statistics; snapshot if scheduled
```
::center::**Figure 2.** Control flow of one simulation round.

### 4.3 Dynamic role sizing

Both role populations scale with the live network rather than being fixed. Given `alive` live nodes and a target fraction *p*, the number of cluster heads is

```
K = max(1, min(alive, round(alive * p)))
```

The floor of one keeps the network functional as it shrinks, and the cap at `alive` prevents the absurdity of electing more heads than there are nodes. The number of multi-sink heads is derived from the count of heads that actually chose the relay path,

```
M = 0                                          if relay = 0
M = max(1, min(relay, ceil(relay / r)))        otherwise
```

so that one multi-sink head serves roughly *r* relays, and the whole tier disappears when no head needs relaying. This rule encodes two properties at once: the relay tier is applied only when it is needed, and a head that can reach the sink directly never enters the relay pool because path decision precedes multi-sink election.

### 4.4 Genetic algorithm for cluster-head election

The heart of the protocol is a genetic algorithm that selects the cluster-head set anew each round. Because batteries drain, nodes die, and the sun moves, yesterday's ideal heads are rarely today's, so the search is repeated every round from an informed starting point.

#### 4.4.1 Chromosome encoding

A chromosome is a list of *K* distinct node identifiers, where *K* is the current head count. Each chromosome is thus one candidate cluster-head set. Fitness is cached on the chromosome so that unchanged elites are never re-evaluated.

#### 4.4.2 Fitness function

Each candidate set is scored on four normalised components combined by fixed weights,

```
F = 0.25 * E_score + 0.25 * S_score + 0.30 * C_score + 0.20 * Sp_score
```

The **energy score** rewards sets whose heads collectively hold high residual charge,

```
E_score = min( sum_of_head_energy / (K * E_init), 1 )
```

The **coverage score**, which carries the largest weight because an uncovered sensor is a failure regardless of anything else, is the fraction of non-head sensors that lie within communication range `R_c = R_c_pct * L` of at least one head. The **spread score** discourages the heads from clumping in one corner,

```
Sp_score = min( mean_distance_of_heads_to_their_centroid / L, 1 )
```

The **solar score** is where harvesting enters selection, and it is deliberately node-specific,

```
S_score = 0.5 * daylight * mean(eff over chosen heads) + 0.5 * mean_battery_fraction
daylight = solar(round) / H_max        (0 at night, 1 at noon)
```

Because the daylight factor is multiplied by the average panel efficiency of the particular heads a chromosome selects, this term differs from one candidate set to another and can therefore change which set wins. At night it collapses gracefully to a battery-only score. At noon, a set of well-charged heads on good sites scores near one, while an equally charged set stuck in shade scores lower and is less likely to be chosen. This is the concrete mechanism by which the protocol earns the solar-aware description.

#### 4.4.3 Initialisation

Half of the initial population is seeded by energy-weighted sampling, which biases the starting point toward high-battery nodes; the other half is uniform random to preserve diversity. This hybrid seeding typically shortens convergence on large networks without prematurely narrowing the search.

#### 4.4.4 Operators

Parents are chosen by tournament selection of size three, which balances selection pressure against diversity. Crossover is single-point but duplicate-aware: it copies a prefix from the first parent and then fills the remainder from the second, skipping identifiers already present so that no head is repeated. Mutation replaces a randomly chosen gene with a random node not already in the set. The mutation rate is adaptive; it begins at a base value and ramps up linearly whenever the best fitness has stalled, up to a ceiling of one half, then resets when progress resumes, which is simulated annealing in spirit. The best two chromosomes survive unchanged through elitism, and the search stops early once the best fitness has not improved for a patience window.

#### 4.4.5 Pseudocode

```
Algorithm 1  GA cluster-head election (one round)
Input:  live nodes with positions, energies, efficiencies; head count K; daylight
Output: best cluster-head set
1  P <- init_population(GA_POP)        # half energy-weighted, half uniform
2  best <- none; best_fit <- -inf; stale <- 0; mut <- base_mut
3  for gen = 1 .. GA_GEN:
4      evaluate_fitness(P)             # Algorithm 2, batched
5      sort P by fitness descending
6      if fitness(P[0]) > best_fit + eps:
7          best <- copy(P[0]); best_fit <- fitness(P[0]); stale <- 0; mut <- base_mut
8      else:
9          stale <- stale + 1; mut <- min(0.5, base_mut * (1 + 0.25 * stale))
10     if stale >= patience: break
11     next <- { copy(P[0]), copy(P[1]) }          # elitism
12     while |next| < GA_POP:
13         c <- crossover(tournament(P), tournament(P))
14         next <- next + { mutate(c, mut) }
15     P <- next
16 return best
```

```
Algorithm 2  Batched fitness evaluation
Input:  population P of size p; live count S; head count K
1  build index matrix G[p][K] mapping each chromosome's genes to live-array rows
2  gather head coords, energies, efficiencies via G
3  E_score  <- min( rowsum(head_energy)/(K*E_init), 1 )
4  S_score  <- 0.5*daylight*rowmean(head_eff) + 0.5*min(rowmean(head_energy)/E_init,1)
5  if p*S*K <= MEM_LIMIT:
6      D2 <- squared distances (p x S x K); Dmin <- min over K
7      C_score <- fraction of non-head sensors with Dmin <= R_c^2
8  else:
9      compute C_score per chromosome (memory-bounded loop)
10 Sp_score <- min( mean distance of heads to their centroid / L, 1 )
11 F <- 0.25*E + 0.25*S + 0.30*C + 0.20*Sp   (invalid chromosomes -> 0)
```

### 4.5 Path decision

Once the heads are elected, each independently chooses its onward route by a simple, transparent rule. A head takes the direct Path A when it is both close enough to the sink and healthy enough to afford the trip, and otherwise it joins the relay pool.

```
Algorithm 3  Path decision (per head)
1  close   <- distance_to_sink <= D_dir
2  healthy <- energy_fraction  >= B_dir
3  if close and healthy:  route <- DIRECT (Path A)
4  else:                  route <- RELAY  (Path B)
```

With the default thresholds, a full-battery head within 55 m of the sink transmits directly, whereas the same head at low charge, or a head 80 m away, is relayed. Because this decision precedes the multi-sink election, a head that can reach the sink directly never enters the relay pool and never burdens the multi-sink tier.

### 4.6 Solar-aware multi-sink election

From the relay pool, the protocol elects *M* multi-sink heads. Each candidate is scored by a suitability function that, like the head fitness, treats harvesting as node-specific,

```
score(node) = 0.35 * battery + 0.30 * solar + 0.20 * centrality + 0.15 * BS_closeness
solar        = daylight * eff(node)
centrality   = 1 - min( mean_distance_to_peers / (L * sqrt(2)), 1 )
BS_closeness = 1 - min( distance_to_sink / sqrt(L^2 + BS_y^2), 1 )
```

Battery carries the greatest weight here because a multi-sink head performs the single most expensive transmission in the round. The solar term is scaled by the candidate's own efficiency, so among the relay heads competing for the role the harvesting component genuinely separates them. Centrality keeps the elected head close to the relays it must serve, and closeness to the sink lowers the final-hop cost.

When more than one multi-sink head is required, the relay heads are first partitioned into spatial zones by a lightweight k-medoids procedure, and the best-scoring node in each zone is elected. Medoids, rather than centroids, are used because a multi-sink head must be a real node capable of receiving radio traffic, not a fictitious average point.

```
Algorithm 4  Multi-sink election
Input:  relay heads; count M; daylight
1  if M <= 0 or relay heads empty: return {}
2  if M == 1 or |relay| <= 2:
3      pick argmax score(.) over relay heads; assign all others to it; return it
4  clusters <- kmedoids(relay heads, M)      # farthest-first seed + PAM-lite
5  elected  <- {}
6  for each cluster c:
7      m <- argmax score(node) over c
8      elected <- elected + { m };  assign the rest of c to m
9  return elected
```

### 4.7 Mid-round re-election

A mid-round safeguard protects a depleting relay. After the heads have forwarded their data, any multi-sink head whose battery has fallen below `tau` times the initial energy is demoted back to an ordinary head, and the best-scoring peer in its zone is promoted in its place, with the zone's relays rerouted accordingly. Replacing the node before it attempts the long-haul transmission means the heavy work is done by a fresher device, and a node caught at the threshold still retains enough charge to serve usefully as a plain head. Doing this mid-round rather than next-round matters, because a node left in place would almost certainly die completing its expensive transmission.

### 4.8 One complete round

```
Algorithm 5  simulate_round (SGA-MS)
1  daylight <- solar(round)/H_max; every alive node harvests (per-node eff)
2  reset all roles to sensor; refresh live caches; K <- role sizing
3  heads <- GA_elect(K)                       # Algorithm 1
4  (direct, relay) <- path_decision(heads)    # Algorithm 3
5  M <- role sizing(relay); ms <- elect(relay, M)   # Algorithm 4, if M>0
6  assign each sensor to nearest head (vectorised); sensors transmit
7  each head aggregates members; forwards on Path A or Path B
8  re-elect any ms below tau (Section 4.7)
9  each surviving ms aggregates its combined stream and transmits on Path C
10 record alive count, deaths, total and per-node residual energy, K and M
```

### 4.9 An optional base-station-proximity term

A single-strategy variant of the implementation augments the head fitness with a fifth component that rewards sets containing heads close enough to reach the sink directly, nudging the search toward configurations that make more use of the cheap Path A. This term is optional and is reported for completeness; the four-component fitness described above is the primary formulation used in the comparative study.

## 5. Complexity Analysis

### 5.1 Time complexity

Let *S* be the number of live nodes in a round, *K* the head count, *G* the generation budget, and *P* the population size. The dominant per-round cost is the genetic search. A single batched fitness evaluation gathers head attributes in `O(P*K)` and computes coverage over a broadcast of squared distances in `O(P*S*K)`; the energy, solar, and spread terms are `O(P*K)`. Across at most *G* generations the head election is therefore `O(G*P*S*K)` in the worst case, though early stopping and fitness caching for elites typically reduce this substantially in practice. Sensor-to-head assignment is a single nearest-neighbour pass in `O(S*K)`. The multi-sink election runs k-medoids over the relay set of size `|relay| <= K` for a fixed number of iterations, costing `O(K*M)` per iteration, which is negligible beside the head search since `K << S`. The per-round total is thus governed by `O(G*P*S*K)`.

### 5.2 Space complexity

The live caches hold coordinates, energies, and efficiencies in `O(S)`. The batched coverage computation is the peak allocator; its temporary tensor is `O(P*S*K)`. To bound this on very large networks, the implementation falls back to a per-chromosome loop once `P*S*K` exceeds a fixed element threshold, trading time for a `O(S*K)` working set. All other structures are `O(S)` or smaller.

### 5.3 Communication overhead

Communication overhead is best measured by the number and distance of transmissions per round, since these determine energy. In LEACH, each of the `S - K` non-head sensors transmits once to its head, and each of the *K* heads transmits once directly to the sink; the *K* head-to-sink transmissions include the farthest heads, which incur the `d^4` penalty. SGA-MS keeps the same `S - K` sensor transmissions, but the *K* head transmissions split into `|direct|` short head-to-sink links and `|relay|` head-to-relay links, followed by only *M* multi-sink-to-sink transmissions, where `M = ceil(|relay| / r)`. The number of genuinely long-range transmissions is therefore reduced from *K* in LEACH to roughly `|direct| + M`, and because direct heads are, by construction, close to the sink, the truly expensive `d^4` links number about *M*, which for the default of four relays per multi-sink head is at most a quarter of the relay count. This reduction in the count and distance of long-haul transmissions is the mechanism behind the residual-energy gap reported in Section 7. The relay tier does add extra receptions at the multi-sink heads and a modest number of medium-range head-to-relay links, but reception and medium-range transmission are far cheaper than long-range transmission, so the trade is favourable.

A separate form of overhead deserves honest mention: the head election in the reference implementation assumes global knowledge and centralised computation, as is common in simulation studies of this kind. A real deployment would need to disseminate the elected assignments, incurring control-message overhead not accounted for in the energy model. We treat this as a threat to validity in Section 8 and a target for future distributed variants.

### 5.4 Convergence

Elitism guarantees monotone non-decreasing best fitness across generations, so the search never regresses. Adaptive mutation raises exploration on plateaus, and the patience-based early stop halts once improvement ceases, so the effective generation count is usually well below the budget *G*. Because the fitness is bounded in [0, 1] and elitism preserves the incumbent, the algorithm converges to a local optimum of the surrogate fitness within the generation budget; global optimality is neither claimed nor required, since the surrogate is itself a heuristic proxy for long-run lifetime.

## 6. Experimental Setup

### 6.1 Simulator

The protocol and the baseline are implemented in Python, with NumPy for the vectorised inner loops and Matplotlib for figures. The simulator is deterministic under a fixed random seed, which fixes both the node layout and the stochastic elements of the search, so that any measured difference between protocols is attributable to the protocols themselves rather than to chance. Each run advances round by round until a fixed horizon or until the network can no longer operate, recording the metrics of Section 6.4 at every round and optionally emitting topology snapshots.

### 6.2 Parameters

Unless stated otherwise, experiments use the values in Table 3.

| Parameter | Symbol | Value |
|---|---|---|
| Field side | L | 100 m |
| Number of nodes | N | 50 |
| Base-station position | (BS_x, BS_y) | (50, 120) m |
| Rounds | -- | 300 |
| Cluster-head fraction | p | 0.10 |
| Relays per multi-sink head | r | 4 |
| Initial energy | E_init | 0.5 J |
| Battery ceiling | B_max | 2.0 J |
| Peak harvest rate | H_max | 0.002 J/round |
| Per-node panel efficiency | eff | uniform [0.6, 1.0] |
| Packet size | k | 4000 bits |
| Electronics energy | E_elec | 50 nJ/bit |
| Free-space amplifier | E_amp | 100 pJ/bit/m^2 |
| Multipath amplifier | E_mp | 0.0013 pJ/bit/m^4 |
| Aggregation energy | E_DA | 5 nJ/bit |
| Communication range | R_c | 0.40 * L |
| Direct-path distance | D_dir | 0.55 * L |
| Direct-path battery | B_dir | 0.40 |
| Population / generations | P / G | 30 / 50 |
| Base mutation / crossover | -- | 0.10 / 0.80 |
| Re-election threshold | tau | 0.15 * E_init |

### 6.3 Baselines

We compare against LEACH, implemented under identical conditions: the same field, the same nodes and starting batteries, the same diurnal harvesting with per-node efficiency, and the same dynamic head count. LEACH differs only in electing heads at random and in having every head transmit directly to the sink, with no relay tier. This isolates the joint contribution of informed head selection and multi-sink relaying. Comparisons against stronger baselines, notably HEED and PEGASIS, are identified as future work in Section 9.

### 6.4 Metrics

We report five metrics. The **round of first node death** is an early and sensitive indicator of energy imbalance. The **network lifetime** is the round at which the network can no longer operate. The **packets delivered to the sink** measure useful throughput. The **total residual energy** at the end of the run measures how much budget the protocol conserved. The **standard deviation of per-node energy** measures how evenly the load was spread; a lower value is better, since a balanced network avoids the premature loss of overworked nodes.

### 6.5 Seeds and reproducibility

All results in this paper use a single fixed seed so that the node layout and the search are reproducible from the accompanying source. We stress that a single seed characterises one field, and Section 6.6 describes the multi-seed protocol needed for statistically sound claims.

### 6.6 Planned evaluation protocol

For a rigorous evaluation, each configuration should be run over many independent seeds and topologies, with the metrics reported as means accompanied by 95% confidence intervals, and with differences between protocols tested for significance using a paired test such as the Wilcoxon signed-rank test. Node-count sweeps (for example 50, 100, 200, and 500 nodes) should probe scalability, and a parameter sweep over *p*, *r*, and `H_max` should map the sensitivity of the results. The present study reports a single representative run and is therefore best read as an existence demonstration of the trends, not as a statistically validated benchmark.

## 7. Results and Discussion

The figures below come from a representative run under the default configuration of Table 3 with the fixed seed. As with any single-seed experiment, absolute values shift with the configuration and the seed; the qualitative trends, however, follow directly from the protocol's structure and are robust to those details.

### 7.1 Representative comparison

Table 4 reports the headline metrics for a representative run.

| Metric | LEACH | SGA-MS | Change |
|---|---|---|---|
| First node death (round) | 87 | 134 | +54% later |
| Network lifetime (rounds) | 245 | 298+ | +21% or more |
| Packets delivered to sink | 1,920 | 2,640 | +38% |
| Final residual energy (J) | 0.04 | 1.82 | much higher |
| Energy std. deviation (J) | 0.087 | 0.041 | 53% lower |

### 7.2 Network lifetime and first death

Because SGA-MS chooses heads with high residual energy and good coverage, and because it offloads the expensive final hop onto nodes selected for their ability to bear it, the energy burden in any given round falls on nodes that can afford it. The expected effect, borne out in the representative run, is a later first death and a longer overall lifetime than LEACH, whose random heads periodically saddle a weak or badly placed node with the direct long-range transmission. Figure 3 plots alive nodes against round for both protocols and marks the first-death rounds; the SGA-MS curve stays higher for longer and its first-death marker lies well to the right of LEACH's.

::center::**Figure 3.** Alive nodes versus round for SGA-MS and LEACH, with first-death markers. *(Generated by the simulator; see the alive-nodes panel of the results figure.)*

### 7.3 Residual energy and balance

The multi-sink tier is the main driver of the residual-energy gap. By allowing distant heads to hand off to a nearby relay rather than transmitting across the field, the protocol keeps most transmissions in the cheaper `d^2` regime and reserves the `d^4` cost for a small, well-chosen set of relays that are additionally protected by mid-round re-election. The lower energy standard deviation reflects the coverage and spread terms in the fitness, which prevent any one region from being repeatedly over-served, and the energy term, which steers the role away from nodes that have already given a lot. Figure 4 shows total residual energy over time, and Figure 5 shows the standard deviation of per-node energy, on which a lower curve indicates better balance.

::center::**Figure 4.** Total residual network energy versus round. *(Generated by the simulator.)*
::center::**Figure 5.** Standard deviation of per-node energy versus round; lower is better. *(Generated by the simulator.)*

### 7.4 Throughput

Delivered packets track lifetime, since a network that survives longer and keeps more heads operational moves more data to the sink. In the representative run SGA-MS delivered roughly 38% more packets than LEACH over the horizon, consistent with its longer lifetime and its more reliable multi-sink final hop.

### 7.5 Multi-sink engagement dynamics

A useful diagnostic is how often the multi-sink tier is actually used. Early in a run, when many heads still have healthy batteries and the layout offers several heads near the sink, rounds are frequently resolved entirely through direct Path A transmissions, and the multi-sink stage is skipped. As the run progresses and batteries deplete, more heads fall into the relay pool and the multi-sink tier engages, at which point its cost savings matter most. This adaptivity, rather than a fixed relay structure, is what lets the protocol avoid overhead when it is not needed while providing relief when it is. The topology snapshots emitted by the simulator label each round as multi-sink-used or multi-sink-skipped, making this behaviour directly visible.

### 7.6 Effect of per-node harvesting

The per-node efficiency factor is what makes the solar-aware scoring operative. When two candidate heads or relays are otherwise comparable, the protocol now prefers the one on the sunnier site, since that node will recover its expenditure sooner. Under a uniform harvesting model this preference would be impossible to express, because the solar term would contribute the same amount to every candidate and drop out of the comparison. Making harvesting heterogeneous therefore does more than add realism: it is the precondition for the harvesting signal to influence any decision at all. A controlled ablation that disables the per-node factor, reducing the solar term to a network-wide constant, is expected to show the solar component losing all influence on selection while leaving the energy, coverage, and spread terms intact; we identify this ablation as part of the planned evaluation.

### 7.7 Parameter sensitivity

Several parameters offer intuitive control. Raising the head fraction *p* improves coverage but increases the number of nodes bearing forwarding costs, so there is a sweet spot rather than a monotone benefit. Increasing the relays served per multi-sink head *r* reduces the number of expensive final hops but enlarges each relay's receive-and-fuse burden. Larger peak harvest rates `H_max` lengthen lifetime across the board and make the solar terms more influential relative to the static energy term. A systematic sweep of these parameters, reported with confidence intervals, is a natural companion study.

### 7.8 Statistical analysis

The single-seed figures reported here demonstrate the direction and rough magnitude of the effect but do not establish statistical significance. The methodology of Section 6.6 is designed to close this gap: repeating each configuration over many seeds, reporting means with 95% confidence intervals, and applying a paired non-parametric test to the per-seed differences. We consider this the single most important next step before any external submission, and we make the claim explicitly here so that the present results are not over-read.

## 8. Limitations and Threats to Validity

We state the caveats plainly, because they bound the strength of the conclusions.

**Internal validity.** The results reported here are from a single seed. A single seed characterises one field layout and one realisation of the harvesting noise; sound practice, and our planned next step, is to average over many seeds and layouts and to report confidence intervals so that the differences are shown to be statistically significant rather than incidental.

**External validity.** The evaluation uses a single, deliberately simple baseline, LEACH. A fuller comparison against energy-aware and harvesting-aware protocols such as HEED, PEGASIS, and recent evolutionary schemes would more convincingly locate the protocol's standing. In addition, the nodes are stationary and the sink is fixed, so the findings do not speak to mobile scenarios.

**Construct validity.** The medium-access and physical layers are idealised: with no collisions, retransmissions, interference, or fading, the absolute lifetime figures are optimistic, although the relative comparison is fairer because both protocols enjoy the same idealisation. The energy model also omits the control-message overhead of disseminating the centrally computed head assignments; a distributed implementation would incur costs not captured here. Although harvesting is now node-specific, the panel-efficiency factor is static and drawn from a simple uniform distribution, whereas real irradiance varies with weather and season and would be better represented by empirical traces.

**Reproducibility.** All experiments use a fixed seed and the accompanying source, so the reported run is reproducible; changing the seed or configuration changes absolute figures while preserving the qualitative trends.

## 9. Conclusion and Future Work

We have presented SGA-MS, a solar-aware, genetic-algorithm-based, multi-sink data-aggregation protocol for energy-harvesting wireless sensor-assisted IoT networks. The protocol re-elects cluster heads each round with a genetic algorithm whose fitness balances energy, harvesting, coverage, and spread; routes each head along the cheapest viable path; and, when necessary, relays through a dynamically sized tier of multi-sink heads chosen for their battery, sunlight, and position and protected by mid-round re-election. A central methodological point is that harvesting must vary across candidates to influence selection, and we build that variation in through a per-node panel-efficiency factor. Under identical conditions the protocol outlasts a LEACH baseline, delivers more data, and balances energy more evenly in a representative run, and a vectorised implementation makes it practical at scale.

Future work follows directly from the limitations. The most pressing item is a multi-seed, multi-layout evaluation with confidence intervals, significance testing, and comparisons against stronger baselines including HEED and PEGASIS. Beyond that, we intend to replace the idealised link layer with a realistic medium-access and channel model, to drive the harvester from measured irradiance traces rather than a smooth analytic curve, to develop a distributed variant that accounts for control-message overhead, to explore mobility of both nodes and sinks, and to conduct the systematic parameter sweep and the harvesting-ablation study outlined above. We also see value in learning the fitness weights online rather than fixing them, allowing the protocol to adapt its priorities to the deployment in which it finds itself.

## References

1. W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan, "Energy-Efficient Communication Protocol for Wireless Microsensor Networks," in Proc. 33rd Hawaii Int. Conf. System Sciences (HICSS), 2000.
2. W. B. Heinzelman, A. P. Chandrakasan, and H. Balakrishnan, "An Application-Specific Protocol Architecture for Wireless Microsensor Networks," IEEE Trans. Wireless Communications, vol. 1, no. 4, pp. 660-670, 2002.
3. O. Younis and S. Fahmy, "HEED: A Hybrid, Energy-Efficient, Distributed Clustering Approach for Ad Hoc Sensor Networks," IEEE Trans. Mobile Computing, vol. 3, no. 4, pp. 366-379, 2004.
4. S. Lindsey and C. S. Raghavendra, "PEGASIS: Power-Efficient Gathering in Sensor Information Systems," in Proc. IEEE Aerospace Conf., 2002.
5. A. Manjeshwar and D. P. Agrawal, "TEEN: A Routing Protocol for Enhanced Efficiency in Wireless Sensor Networks," in Proc. Int. Parallel and Distributed Processing Symp. (IPDPS), 2001.
6. G. Smaragdakis, I. Matta, and A. Bestavros, "SEP: A Stable Election Protocol for Clustered Heterogeneous Wireless Sensor Networks," in Proc. Int. Workshop on SANPA, 2004.
7. L. Qing, Q. Zhu, and M. Wang, "Design of a Distributed Energy-Efficient Clustering Algorithm for Heterogeneous Wireless Sensor Networks (DEEC)," Computer Communications, vol. 29, no. 12, pp. 2230-2237, 2006.
8. J. H. Holland, Adaptation in Natural and Artificial Systems. Ann Arbor: University of Michigan Press, 1975.
9. D. E. Goldberg, Genetic Algorithms in Search, Optimization, and Machine Learning. Reading, MA: Addison-Wesley, 1989.
10. L. Kaufman and P. J. Rousseeuw, "Partitioning Around Medoids (Program PAM)," in Finding Groups in Data: An Introduction to Cluster Analysis. New York: Wiley, 1990.
11. A. Kansal, J. Hsu, S. Zahedi, and M. B. Srivastava, "Power Management in Energy Harvesting Sensor Networks," ACM Trans. Embedded Computing Systems, vol. 6, no. 4, 2007.
12. S. Sudevalayam and P. Kulkarni, "Energy Harvesting Sensor Nodes: Survey and Implications," IEEE Communications Surveys and Tutorials, vol. 13, no. 3, pp. 443-461, 2011.
13. P. S. Muruganantham and H. El-Ocla, "Routing Using Genetic Algorithm in a Wireless Sensor Network," Wireless Personal Communications, 2020.
14. J. Wu et al., "Genetic-Algorithm-Based Optimisation for Energy-Harvesting Wireless Sensor Networks," IET Wireless Sensor Systems, 2013.
15. "A Genetic Algorithm-Based Energy-Efficient Routing Protocol for Wireless Sensor Networks," ACSIJ Advances in Computer Science, 2014.
16. "Routing Optimisation in the Internet of Things Using Genetic Algorithms," 2023.

## Appendix A. Notation and Default Parameters

Table 2 defines the symbols; Table 3 lists the default values used in all experiments unless a section states otherwise. The two lifetime figures of merit are the round of first node death and the round of network death, as defined in Section 3.6.

## Appendix B. Implementation Notes

The reference implementation gathers live-node attributes into contiguous arrays once per round and evaluates the entire genetic population's fitness in a single batched tensor operation, with an automatic fallback to a per-chromosome loop when the temporary tensor would exceed a fixed element budget. Squared distances are used wherever only relative ordering matters, avoiding unnecessary square roots. Node objects use fixed slots to reduce memory footprint, and each node caches its distance to the sink at construction since it never changes. Fitness values are cached on chromosomes so that unchanged elites are not re-evaluated, energy-weighted seeding shortens convergence, and patience-based early stopping curtails the search once improvement ceases. Together these measures allow networks of thousands of nodes over hundreds of rounds to be simulated in seconds. The harvesting efficiency of each node is drawn once at construction from a fixed distribution and thereafter held constant, which both preserves the node layout across runs and provides the spatial heterogeneity on which the solar-aware scoring depends.

---

*Reproducibility note.* The protocol and baseline are implemented in the accompanying source files. All experiments use a fixed random seed so that the node layout and the search are reproducible; changing the seed or the configuration will change the absolute figures while preserving the qualitative trends discussed above. The quantitative results in Section 7 are from a single representative run and should be read as indicative pending the multi-seed validation described in Section 6.6.
