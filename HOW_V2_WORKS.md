# How `solar_ga_wsn_v2.py` Works — Improvements Over v1

This document explains **what changed in v2**, **what actually happens when you run
it**, and **how each new mechanism works** under the hood. It is written so you can
put the explanation straight into your paper / report.

`solar_ga_wsn_v2.py` is a single self-contained file. It contains the original
protocol **plus** all the fixes below. Run it with `python solar_ga_wsn_v2.py`.

---

## 1. Why v2 exists

The earlier review of the project flagged six concrete weaknesses. Each one is a
question a paper reviewer would ask. v2 answers all of them **in code**:

| # | v1 weakness (the reviewer's question) | v2 answer |
|---|----------------------------------------|-----------|
| 1 | "You only ran seed 42. Is the result luck?" | Multi-seed **Monte-Carlo** + **mean ± std** + **significance test** |
| 2 | "How expensive is running a GA every round?" | **CH-election timing** + **convergence curves** |
| 3 | "Real radios drop packets and burn idle power. Where is that?" | **Packet loss + retransmissions + idle/sensing drain + PDR** |
| 4 | "Your solar curve is a perfect sine — unrealistic." | **Seasonal + cloud + noise model** (and optional real CSV trace) |
| 5 | "Does the solar-awareness actually help, or is it decoration?" | **Solar ablation** (run with vs without the solar term) |
| 6 | "You only beat LEACH. What about modern algorithms?" | **PSO / GWO / HEED** baselines kept from the earlier work |

> **Honest scope note:** a *simulation* can never have literally "zero" drawbacks —
> only a hardware testbed removes modelling assumptions entirely. v2 removes every
> **specific** drawback listed above and makes the models much more realistic.

---

## 2. Improvement 1 — Multi-seed Monte-Carlo + statistics

### The problem
v1 created the network with a hard-coded seed (`random.seed(42)`), so every reported
number came from one single random layout. One lucky or unlucky field can flip a
result. Metaheuristic comparisons are not credible without repetition.

### What v2 does
- `create_nodes(cfg, seed=42)` now takes a **seed argument**. Each seed produces a
  different-but-reproducible node field.
- `run_monte_carlo()` runs **every protocol across many seeds** (`MC_SEEDS`, default
  15) and collects six metrics per run: network lifetime, first-node-death round,
  packets to BS, final residual energy, PDR, and CH-election time.
- It prints a **mean ± std** table and then a **significance test** of GA vs each
  baseline on network lifetime.

### How the significance test works
`_significance(a, b)`:
- If **SciPy** is installed → runs a **Welch t-test** (unequal variances) *and* a
  **Mann-Whitney U** test (non-parametric, robust for lifetimes), reporting both
  p-values and flagging `SIGNIFICANT` when `p < 0.05`.
- If SciPy is **not** available → computes a manual Welch t-statistic and
  **Cohen's d** effect size, and honestly says a p-value needs SciPy.

### What you get
`comparison_monte_carlo.png` — bar charts with **error bars** (mean ± std) for
lifetime, PDR, packets, and election time, plus the printed tables. This is the
single biggest credibility upgrade for the paper.

---

## 3. Improvement 2 — Runtime cost + convergence

### The problem
Running a full GA (`GA_POP × GA_GEN` fitness evaluations) **every round** is
expensive, and v1 never measured it. A win of 2% that costs 50× the compute is a
weak win.

### What v2 does
- **Timing:** `run_protocol()` wraps the cluster-head selector in a timer and reports
  **average CH-election time in ms/round**. This shows up in the comparison table and
  the Monte-Carlo plot, so GA's cost is quantified next to PSO/GWO/HEED (HEED is
  effectively instant; GA is the most expensive — now you can *show* the trade-off).
- **Convergence:** a module-level recorder `_CONV_SINK` is filled by the GA, PSO, and
  GWO optimizers with their **best-fitness-so-far after each generation**.
  `plot_convergence()` runs one election with each optimizer on an identical field and
  draws the three curves under the same population/iteration budget →
  `comparison_convergence.png`.

### How the convergence hook works
Each optimizer has one guarded line, e.g. in the GA loop:
```python
if _CONV_SINK is not None:      # normal runs: None => zero overhead
    _CONV_SINK.append(best_score)
```
During normal simulation `_CONV_SINK` is `None`, so there is no cost. Only
`plot_convergence()` switches it on.

---

## 4. Improvement 3 — Realistic radio / MAC layer

### The problem
v1's radio was ideal: every packet always arrived, and a silent node spent no energy.
Real radios drop packets (collisions, fading), retransmit (burning extra energy), and
drain power just idle-listening and sensing.

### What v2 does — three additions inside the energy model

**(a) Packet loss + retransmissions (ARQ).** `Node.transmit()` now has a realistic
path controlled by config:
- `p_drop` grows with **distance** (`PACKET_LOSS × (1 + 0.5·d/DIRECT_DIST)`), because
  longer links are lossier.
- On a drop, the node **retries up to `MAX_RETX` times**; **every attempt costs full
  TX energy** (that is the real cost of ARQ).
- The outcome is stored in `node.delivered`.
- With `PACKET_LOSS = 0` this reduces **exactly** to the original loss-free behaviour,
  so v1 is a special case of v2.

**(b) Idle-listen + sensing drain.** `Node.idle_sense_drain()` subtracts
`IDLE_ENERGY + SENSING_ENERGY` from **every alive node, every round**, whether or not
it transmits. This models an always-on radio and periodic sensing. It also makes solar
harvesting genuinely matter — a node must out-harvest its idle cost to survive.

**(c) Packet Delivery Ratio (PDR).** Because packets can now fail, the simulator
tracks `packets_sent` vs `packets_delivered` per node and reports
**PDR = delivered / attempted**. Throughout the round logic, a receiver only spends
receive energy and a packet is only counted as reaching the BS **if it was actually
delivered** (the `if ch.delivered` / `if ms.delivered` / `if n.delivered` gates).

---

## 5. Improvement 4 — Realistic solar harvesting

### The problem
v1's harvest was a perfectly smooth, identical-every-day half-sine. Real harvesting
varies with season, weather, and noise.

### What v2 does
Two functions, split by purpose:
- `solar_rate_for_round()` — the **clear-sky forecast** (the old smooth curve). This is
  what the GA/PSO/GWO **plan against**, exactly like a real controller that has a
  forecast but not tomorrow's weather.
- `actual_solar_rate()` — what a panel **actually harvests** this round. On top of the
  clear-sky curve it layers:
  1. **Seasonal envelope** — a slow yearly sine (`SEASON_AMP`) so summer out-harvests winter.
  2. **Stochastic clouds** — with probability `CLOUD_PROB` the round is clouded, cutting output down to as little as `CLOUD_MIN` of clear-sky.
  3. **Measurement noise** — small Gaussian jitter.
- `load_solar_trace(path)` + the `SOLAR_TRACE` config option — if you point it at a CSV
  of measured harvest values, that **real data drives the simulation** and the analytic
  model is bypassed entirely.

### Why this design is stronger
Harvesting uses the *actual* (messy) value; planning uses the *forecast*. Modelling
that **forecast-vs-reality gap** is itself part of the added realism and is exactly how
a deployed energy-harvesting controller behaves.

---

## 6. Improvement 5 — Solar-awareness ablation

### The problem
The whole selling point is "solar-aware," but v1 never proved the solar term actually
changes outcomes. If it doesn't, the headline claim is decoration.

### What v2 does
- A config flag `USE_SOLAR_TERM` gates the solar contribution in **both** the GA fitness
  function (`_evaluate_population`) **and** the MS-CH score (`_solar_aware_score`). When
  it is `False`, the solar term drops to a neutral constant, so selection is no longer
  solar-aware.
- `run_solar_ablation()` runs GA **with** and **without** the solar term across all seeds
  and reports the mean lifetime difference **and its significance**.

### What you get
A number you can put in the paper: *"the solar term changes GA network lifetime by
X% (p = …)."* If X is large → the claim is earned. If X is tiny → you soften the claim
or increase `MAX_HARVEST` to a realistic-but-larger value. Either way the paper becomes
honest and defensible.

---

## 7. Improvement 6 — Modern baselines retained

v2 keeps the earlier work's fair baselines, all running through the **same** multi-sink
tier, energy model, and solar model — only the cluster-head **optimizer** changes:
- **PSO** (Particle Swarm) and **GWO** (Grey Wolf) — same fitness function, same
  population and iteration budget as the GA, so it's an apples-to-apples "is GA the best
  optimizer?" test.
- **HEED** — energy-aware, spatially-separated CHs (deterministic; a documented
  simplification of full HEED).
- **LEACH** — the original random baseline, for continuity.

---

## 8. What happens when you run it (execution flow)

```
python solar_ga_wsn_v2.py
        │
        ▼
   main_v2(cfg)         # cfg = default_config(): realistic radio + solar ON
        │
        ├─ run_monte_carlo(cfg)          # Improvement 1 + 2 (timing)
        │     for each seed 1..MC_SEEDS:
        │        for each protocol (GA, PSO, GWO, HEED, LEACH):
        │           run_protocol(cfg, protocol, seed, quiet=True)
        │              └─ per round: harvest(actual) → idle/sense drain →
        │                            reset roles → CH election (timed) →
        │                            path decision → MS-CH election →
        │                            sensors TX (lossy) → CH aggregate/forward →
        │                            MS-CH → BS → record stats
        │        collect lifetime, first-death, packets, residual, PDR, ms/round
        │     → mean ± std table + significance test + comparison_monte_carlo.png
        │
        ├─ plot_convergence(cfg)         # Improvement 2 (convergence)
        │     one CH election each for GA / PSO / GWO on an identical field
        │     → comparison_convergence.png
        │
        └─ run_solar_ablation(cfg)       # Improvement 5
              GA with vs without the solar term, across seeds
              → printed lifetime delta + significance
```

A single round (inside `simulate_round_generic`, shared by GA/PSO/GWO/HEED) now runs:
**realistic harvest → idle+sensing drain → role reset → (timed) CH election →
path decision → MS-CH election → lossy sensor transmission → CH aggregate & forward
(delivery-gated) → MS-CH → BS (delivery-gated) → stats**. LEACH uses the same physics
via `simulate_round_leach`.

---

## 9. How to run

```bash
# Full v2 pipeline: Monte-Carlo + convergence + ablation (Colab / local)
python solar_ga_wsn_v2.py

# Only part of it (via env var):
SOLAR_GA_STAGE=mc          python solar_ga_wsn_v2.py   # Monte-Carlo only
SOLAR_GA_STAGE=convergence python solar_ga_wsn_v2.py   # convergence only
SOLAR_GA_STAGE=ablation    python solar_ga_wsn_v2.py   # solar ablation only
SOLAR_GA_STAGE=single      python solar_ga_wsn_v2.py   # quick 1-seed comparison

# Choose which protocols to compare:
SOLAR_GA_PROTOCOLS="GA,PSO,GWO,HEED,LEACH" python solar_ga_wsn_v2.py
```

From a notebook, for faster runs use fewer seeds:
```python
import solar_ga_wsn_v2 as v2
cfg = v2.default_config(MC_SEEDS=5, NUM_ROUNDS=300)
v2.run_monte_carlo(cfg)
v2.plot_convergence(cfg)
v2.run_solar_ablation(cfg)
```

Requires `numpy` + `matplotlib` (same as the original scripts); **SciPy is optional**
(without it you get effect sizes instead of p-values).

---

## 10. Outputs

| File / output | Content |
|---|---|
| `comparison_monte_carlo.png` | Mean ± std bar charts: lifetime, PDR, packets, election time |
| `comparison_convergence.png` | GA vs PSO vs GWO best-fitness-per-generation curves |
| printed **mean ± std** table | All six metrics for every protocol |
| printed **significance** table | GA vs each baseline (Welch t / Mann-Whitney, or Cohen's d) |
| printed **ablation** summary | GA lifetime with vs without the solar term (+ significance) |

---

## 11. New / changed configuration knobs

All set in `default_config()`; override like `default_config(PACKET_LOSS=0.1)`.

| Key | Default | Meaning |
|---|---|---|
| `PACKET_LOSS` | `0.05` | Base per-transmission drop probability (scaled by distance) |
| `MAX_RETX` | `2` | Max automatic retransmissions per dropped packet |
| `IDLE_ENERGY` | `5e-5` | Idle-listen drain per alive node per round (J) |
| `SENSING_ENERGY` | `2e-5` | Sensing/CPU drain per alive node per round (J) |
| `SOLAR_MODEL` | `"realistic"` | `"clearsky"` (old sine) or `"realistic"` (season+cloud+noise) |
| `CLOUD_PROB` | `0.30` | Chance a round is clouded |
| `CLOUD_MIN` | `0.15` | Worst-case fraction of sun through a cloud |
| `SEASON_AMP` | `0.30` | Amplitude of the yearly seasonal envelope |
| `SOLAR_TRACE` | `None` | Path to a measured harvest CSV (overrides the analytic model) |
| `USE_SOLAR_TERM` | `True` | Turn solar-awareness on/off (for the ablation) |
| `MC_SEEDS` | `15` | Number of Monte-Carlo seeds to average over |

Set `PACKET_LOSS=0`, `IDLE_ENERGY=0`, `SENSING_ENERGY=0`, `SOLAR_MODEL="clearsky"` to
recover the exact v1 behaviour for a controlled before/after comparison.

---

## 12. Verification status

- `python -m py_compile solar_ga_wsn_v2.py` **passes**.
- The lossless path (`PACKET_LOSS = 0`) is written to reduce exactly to the original
  transmit behaviour, so v1 is a strict special case of v2.
- **Not executed in the build sandbox** (no numpy/matplotlib, installs blocked). Run it
  once in Colab/local; if anything surfaces, the traceback will pinpoint it.
