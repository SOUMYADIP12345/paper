# Comparing SGA-MS Against Modern Algorithms (Not Just LEACH)

This note answers three questions about the Solar-Aware GA Multi-Sink protocol
(**SGA-MS**) in this repo:

1. **Is LEACH a fair thing to compare against, and what should we use instead?**
2. **Why is our GA approach better, and where does it fall short?**
3. **A ready-to-run comparison against modern algorithms** (`compare_algorithms.py`).

---

## 1. Why LEACH alone is a weak baseline

LEACH (Heinzelman et al., 2000) is the *historical* reference for clustered WSNs,
but as a yardstick for a 2025 protocol it is very weak, for three reasons:

- **It is random.** LEACH elects cluster heads (CHs) by a coin flip; it ignores
  residual energy, position, and coverage entirely.
- **It is not a metaheuristic.** Beating LEACH with a genetic algorithm does not
  show the GA is good — it only shows "informed selection beats random," which
  nobody doubts.
- **It has no relay tier.** Every CH transmits directly to the base station, so
  LEACH is structurally guaranteed to lose to any multi-sink design.

Your own paper concedes this in Section 8 ("the evaluation uses a single
baseline, LEACH, which is a deliberately simple reference"). Reviewers of any
WSN venue will raise exactly this point.

---

## 2. What people actually compare against today (2023–2025)

Recent clustering papers benchmark against a mix of an **energy-aware classic**
and one or more **metaheuristics**. The most common, in rough order of how often
they appear as baselines:

| Algorithm | Class | Why it matters as a baseline |
|---|---|---|
| **HEED** (Younis & Fahmy, 2004) | Energy-aware, deterministic | The standard "fair, non-random" classic. If you only add one baseline, add this. |
| **PSO-based clustering** (PSO-C, DPFCP, FLS-PSO) | Metaheuristic (swarm) | The direct competitor to GA. Answers "is GA the right optimizer?" |
| **GWO / EECHIGWO** (Grey Wolf Optimizer) | Metaheuristic (swarm) | The single most-cited "modern" CH-selection metaheuristic in 2023–2025. |
| **ABC** (Artificial Bee Colony) | Metaheuristic | Common alternative swarm optimizer. |
| **Hybrid GWO–PSO / GA–PSO (GAPSO-H)** | Hybrid metaheuristic | State-of-the-art hybrids; harder to beat. |
| **PEGASIS**, **LEACH-C** | Classic | Chain-based / centralized LEACH variants; easy extra baselines. |

Selected recent sources (snippets rephrased for licensing compliance):

- Improved Grey Wolf Optimization for energy-efficient CH selection — [EECHIGWO, MDPI Computers 2023](https://www.mdpi.com/2073-431x/12/2/35)
- GWO clustering with enhanced CSMA/CA — [MDPI Sensors 2024](https://www.mdpi.com/1424-8220/24/16/5234)
- Distributed PSO-based fuzzy clustering (DPFCP), reporting large lifetime gains over LEACH and PSO variants — [MDPI Sensors 2023](https://www.mdpi.com/1424-8220/23/15/6699)
- Adaptive hybrid GWO–PSO CH selection vs LEACH/PSO-DE/QPSO-Fuzzy — [Springer 2026](https://link.springer.com/article/10.1007/s10791-026-10254-2)
- Artificial Bee Colony multi-hop clustering — [Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-12321-y)
- Optimized clustering for **energy-harvesting** WSN (closest to your solar setting) — [Nature Scientific Reports 2025](https://www.nature.com/articles/s41598-025-29453-w)
- Energy-optimization route + CH selection using M-PSO and GA — [Springer WPC 2024](https://link.springer.com/article/10.1007/s11277-024-11096-1)

*Content was rephrased for compliance with licensing restrictions.*

### The fairest possible comparison

The strongest contribution you can make is **not** "our protocol beats other
protocols" (too many confounding variables). It is:

> **"Holding our fitness function, energy model, solar model, and 3-tier
> multi-sink architecture fixed, the GENETIC ALGORITHM is the best optimizer for
> cluster-head selection — better than PSO and GWO."**

That isolates a single variable (the optimizer) and is very hard to argue with.
`compare_algorithms.py` is built to do exactly this.

---

## 3. Why the GA approach is strong (its real advantages)

1. **Discrete/combinatorial fit.** CH selection is "pick K distinct nodes." GA's
   set-based chromosome + duplicate-aware crossover represents this *natively*.
   PSO and GWO are continuous optimizers that must be *decoded* into a discrete
   set (with duplicate repair) — a genuine, defensible advantage for the GA.
2. **Multi-objective fitness done well.** Energy + solar + coverage + spread with
   node-specific solar makes the objective discriminative (your paper's central
   methodological point).
3. **Engineering quality.** Batched NumPy fitness, elitism, adaptive mutation,
   early stopping, energy-weighted seeding, fitness caching — this is a
   well-tuned GA, not a toy.
4. **The architecture, not just the optimizer.** The solar-aware multi-sink tier
   (Path A/B/C, k-medoids zoning, mid-round re-election) is where most of the
   energy savings actually come from, and it is optimizer-agnostic.

---

## 4. Where it falls short (be honest about these)

These are the points a reviewer will press on — better to pre-empt them:

1. **Single-seed results.** Every number in the paper is one run of seed 42.
   Metaheuristic comparisons are meaningless without **averaging over many seeds**
   and reporting **mean ± std / confidence intervals** (and ideally a
   significance test). This is the #1 fix.
2. **Per-round GA is expensive.** Running a full GA (`GA_POP × GA_GEN` evaluations)
   *every round* is heavy. HEED computes its heads in effectively O(N log N) with
   no iteration. You should report **runtime / convergence**, not just lifetime —
   a GA that wins by 2% but costs 50× the compute is a weak win.
3. **Idealized MAC/PHY.** No collisions, interference, retransmissions, or duty
   cycling. Both protocols share this, so relative comparison is fair, but
   absolute lifetimes are optimistic.
4. **Static, synthetic solar.** A smooth half-sine plus a fixed per-node
   efficiency is not real irradiance (weather, clouds, seasons). Real traces
   would strengthen the "solar-aware" claim.
5. **The "solar-aware" edge may be small.** Because `solar_eff ∈ [0.6, 1.0]` and
   `MAX_HARVEST` is tiny (0.002 J/round vs 0.5 J initial), harvesting barely moves
   the battery. Run an ablation: **SGA-MS with vs without the solar term** — if
   the gap is negligible, the headline claim needs softening or the harvest rate
   needs to be realistic-but-larger.
6. **GA vs PSO/GWO was never actually tested.** Until now the paper implies GA is
   a good choice without ever comparing it to another optimizer. `compare_algorithms.py`
   closes this gap — but be prepared for the honest outcome: on this specific
   problem, well-tuned PSO/GWO are often within noise of GA. If they tie, your
   real contribution is the **architecture**, and the paper should say so.

---

## 5. How to run the new comparison

`compare_algorithms.py` reuses **all** of `solar_ga_wsn.py` (same `Node`, energy
model, solar model, fitness function, and multi-sink tier) and only swaps the
cluster-head **optimizer**. Requires `numpy` and `matplotlib` (same as your
existing scripts) — run it in Colab or locally, not in this restricted sandbox.

```bash
# All five protocols, default config (50 nodes, 300 rounds):
python compare_algorithms.py

# Pick a subset:
SOLAR_GA_PROTOCOLS="GA,PSO,GWO,HEED,LEACH" python compare_algorithms.py
```

Or from a notebook / Python:

```python
import compare_algorithms as cmp
from solar_ga_wsn import default_config

cfg = default_config(NUM_NODES=200, NUM_ROUNDS=500)
results = cmp.main(cfg, protocols=["GA", "PSO", "GWO", "HEED", "LEACH"])
```

**Outputs**

- `comparison_all_algorithms.png` — alive nodes, residual energy, deaths, and
  energy balance, all protocols overlaid.
- `comparison_summary_bars.png` — bar chart of lifetime, first death, packets,
  and final residual energy.
- A printed summary table plus GA's relative lifetime gain vs each baseline.

**What each baseline is**

- **GA** — your SGA-MS, unchanged.
- **PSO** — Particle Swarm Optimization picks the CH set; *identical* fitness,
  same population size and iteration budget as the GA (fair compute).
- **GWO** — Grey Wolf Optimizer picks the CH set; same fitness and budget.
- **HEED** — energy-aware, spatially separated CHs (deterministic, no
  metaheuristic). A documented simplification of full HEED.
- **LEACH** — your existing random baseline, for continuity.

All four non-LEACH protocols run through the **same solar-aware multi-sink tier**,
so any difference is attributable to CH selection alone.

---

## 6. Suggested next steps for the paper

1. **Add HEED + PSO + GWO to the results tables** (done in code; run it).
2. **Multi-seed runs** (e.g., 20–30 seeds); report mean ± std and a Wilcoxon /
   t-test. This single change most improves credibility.
3. **Report compute cost** (wall-clock or fitness evaluations per round) so the
   GA's win is contextualized against its price.
4. **Ablation of the solar term** (SGA-MS with/without solar weighting) to prove
   the "solar-aware" label is earned in practice, not just in principle.
5. Reframe the contribution around the **architecture + the GA-vs-swarm study**,
   which is novel, rather than "we beat LEACH," which is not.
