# How `ga_only_v2.py` Works — GA-Only Edition (v2)

This is the **GA-only** counterpart to `solar_ga_wsn_v2.py`. It runs *only* the
Solar-Aware GA Multi-Sink protocol — there is **no LEACH, no PSO/GWO/HEED** — but it
carries **all the same v2 realism and rigor fixes**. Run it with
`python ga_only_v2.py`.

> If you want the head-to-head comparison against modern algorithms, use
> `solar_ga_wsn_v2.py`. If you want to study, tune, and defend the **GA protocol
> itself** with credible statistics, this file is the one.

---

## 1. What `ga_only_v2.py` is

- It is `ga_only.py` (the GA-only runner) **upgraded** with every fix from the v2 work.
- Like the original GA-only file, its GA fitness uses **five** terms, not four — it adds
  a **BS-proximity** term that rewards cluster-head sets with members close enough to
  reach the base station directly (PATH A):

  | Weight | Term | Meaning |
  |---|---|---|
  | 0.20 | Energy | picked CHs have healthy batteries |
  | 0.20 | Solar | CHs are charged **and** currently harvesting sun (node-specific) |
  | 0.30 | Coverage | most sensors have a nearby CH |
  | 0.15 | Spread | CHs are spatially well-distributed |
  | 0.15 | **BS-proximity** | some CHs can reach the BS directly (fewer relays) |

- The 3-tier architecture (sensors → CH → MS-CH → BS) and the solar-aware MS-CH election
  are unchanged from the original.

---

## 2. The drawbacks it fixes (same six, GA-only framing)

| # | Old drawback | v2 fix in this file |
|---|--------------|---------------------|
| 1 | Only seed 42 | **Monte-Carlo** over many seeds → **mean ± std** for every metric |
| 2 | GA cost never measured | **CH-election timing** (ms/round) + a **GA convergence curve** |
| 3 | Idealized radio | **Packet loss + ARQ retransmissions + idle/sensing drain + PDR** |
| 4 | Fake smooth solar | **Seasonal + cloud + noise** model (+ optional measured CSV trace) |
| 5 | Solar edge unproven | **Solar ablation** — GA *with* vs *without* the solar term, **with a significance test** (this is the natural A/B experiment for a single-protocol study) |
| 6 | Only beat LEACH | N/A here by design — this is the GA-only file; use `solar_ga_wsn_v2.py` for cross-algorithm comparison |

> **Honest scope note:** a simulation can't be literally drawback-free (only a hardware
> testbed removes modelling assumptions). This file removes every *specific* weakness
> from the earlier review and makes the models much more realistic.

---

## 3. Improvement details

### 3.1 Multi-seed Monte-Carlo (fix 1)
- `create_nodes(cfg, seed=...)` lays out a different-but-reproducible field per seed.
- `run_monte_carlo()` runs the GA across `MC_SEEDS` seeds (default 15) and prints a
  **mean ± std** table for: network lifetime, first-node death, packets to BS, final
  residual energy, PDR, and CH-election time. It saves `ga_v2_monte_carlo.png`
  (bars with error bars).

### 3.2 Runtime + convergence (fix 2)
- **Timing:** each round records how long the GA cluster-head election took; the mean
  ms/round appears in the Monte-Carlo output. This quantifies the real cost of running a
  GA every round.
- **Convergence:** a guarded recorder (`_CONV_SINK`) captures the GA's best-fitness after
  each generation. `plot_convergence()` runs one election and draws the curve
  (`ga_v2_convergence.png`), showing how quickly the GA improves and where it plateaus
  (early-stop).

### 3.3 Realistic radio + PDR (fix 3)
- `Node.transmit()` now models **distance-scaled packet loss** and retries up to
  `MAX_RETX` times; **every attempt costs full TX energy** (real ARQ behaviour).
- `Node.idle_sense_drain()` subtracts a small `IDLE_ENERGY + SENSING_ENERGY` from every
  alive node each round (always-on radio + sensing).
- The round logic is **delivery-gated**: a receiver only spends receive energy, and a
  packet is only counted at the BS, **if it actually arrived** (`if …delivered`). The
  **Packet Delivery Ratio (PDR)** is reported per run.
- Setting `PACKET_LOSS = 0` reduces `transmit()` **exactly** to the original loss-free
  behaviour, so the old model is a strict special case.

### 3.4 Realistic solar (fix 4)
- `solar_rate_for_round()` is the **clear-sky forecast** — what the GA plans against.
- `actual_solar_rate()` is what a panel **actually harvests**: clear-sky × seasonal
  envelope (`SEASON_AMP`) × stochastic clouds (`CLOUD_PROB`, down to `CLOUD_MIN`) ×
  Gaussian noise. `SOLAR_TRACE` can point at a measured CSV to drive the sim from real
  data. Harvest uses the *actual* value; the GA plans on the *forecast* — modelling the
  real forecast-vs-reality gap.

### 3.5 Solar ablation (fix 5) — the key GA-only experiment
- `USE_SOLAR_TERM` gates the solar contribution in **both** the GA fitness and the MS-CH
  score; when off it becomes a neutral constant (no solar-awareness).
- `run_solar_ablation()` runs the GA **with** and **without** the solar term across all
  seeds, prints the mean lifetime difference **and its significance** (Welch t-test +
  Mann-Whitney U via SciPy, or Cohen's d without SciPy), and saves `ga_v2_ablation.png`.
- This produces a defensible sentence for the paper: *"the solar term changes GA network
  lifetime by X% (p = …)."*

---

## 4. What happens when you run it

```
python ga_only_v2.py
        │
        ▼
   main_v2(cfg)              # cfg = default_config(): realistic radio + solar ON
        │
        ├─ run_monte_carlo(cfg)      # fixes 1 + 2 (timing)
        │     for each seed 1..MC_SEEDS:
        │        run_ga_protocol(cfg, seed, quiet=True)
        │           └─ per round: realistic harvest → idle/sense drain →
        │                         reset roles → (timed) GA CH election →
        │                         path decision → MS-CH election →
        │                         lossy sensor TX → CH aggregate/forward →
        │                         MS-CH → BS (delivery-gated) → stats
        │     → mean ± std table + ga_v2_monte_carlo.png
        │
        ├─ plot_convergence(cfg)     # fix 2 (convergence) → ga_v2_convergence.png
        │
        └─ run_solar_ablation(cfg)   # fix 5 → printed delta + significance +
                                     #          ga_v2_ablation.png
```

---

## 5. How to run

```bash
python ga_only_v2.py                          # MC + convergence + ablation
SOLAR_GA_STAGE=mc          python ga_only_v2.py   # Monte-Carlo only
SOLAR_GA_STAGE=convergence python ga_only_v2.py   # convergence only
SOLAR_GA_STAGE=ablation    python ga_only_v2.py   # solar ablation only
SOLAR_GA_STAGE=single      python ga_only_v2.py   # one detailed seed-42 run
SOLAR_GA_STAGE=interactive python ga_only_v2.py   # original prompt-driven runner
```

From a notebook (faster with fewer seeds):
```python
import ga_only_v2 as ga
cfg = ga.default_config(MC_SEEDS=5, NUM_ROUNDS=300)
ga.run_monte_carlo(cfg)
ga.plot_convergence(cfg)
ga.run_solar_ablation(cfg)
```

Requires `numpy` + `matplotlib`; **SciPy is optional** (without it you get effect sizes
instead of p-values).

---

## 6. Outputs

| File / output | Content |
|---|---|
| `ga_v2_monte_carlo.png` | GA mean ± std bars: lifetime, PDR, packets, election time |
| `ga_v2_convergence.png` | GA best-fitness per generation |
| `ga_v2_ablation.png` | GA lifetime with vs without the solar term |
| printed **mean ± std** table | All six metrics across seeds |
| printed **ablation** summary | Solar on-vs-off lifetime delta + significance |

The original single-run outputs (`ga_results.png`, `ga_topology.png`, and
`topology_snapshots/`) are still produced by the `single` / `interactive` stages.

---

## 7. New / changed configuration knobs

Set in `default_config()`; override like `default_config(PACKET_LOSS=0.1, MC_SEEDS=30)`.

| Key | Default | Meaning |
|---|---|---|
| `PACKET_LOSS` | `0.05` | Base per-transmission drop probability (distance-scaled) |
| `MAX_RETX` | `2` | Max retransmissions per dropped packet |
| `IDLE_ENERGY` | `5e-5` | Idle-listen drain per alive node per round (J) |
| `SENSING_ENERGY` | `2e-5` | Sensing/CPU drain per alive node per round (J) |
| `SOLAR_MODEL` | `"realistic"` | `"clearsky"` (old sine) or `"realistic"` |
| `CLOUD_PROB` | `0.30` | Chance a round is clouded |
| `CLOUD_MIN` | `0.15` | Worst-case fraction of sun through a cloud |
| `SEASON_AMP` | `0.30` | Amplitude of the yearly seasonal envelope |
| `SOLAR_TRACE` | `None` | Path to a measured harvest CSV (overrides analytic model) |
| `USE_SOLAR_TERM` | `True` | Solar-awareness on/off (for the ablation) |
| `MC_SEEDS` | `15` | Number of Monte-Carlo seeds |
| `DIRECT_DIST` | `0.75·FIELD` | CH→BS direct-reach threshold (GA-only keeps the wider reach) |

To recover the original v1 behaviour for a controlled before/after: set
`PACKET_LOSS=0, IDLE_ENERGY=0, SENSING_ENERGY=0, SOLAR_MODEL="clearsky"`.

---

## 8. Difference vs `solar_ga_wsn_v2.py`

| | `ga_only_v2.py` | `solar_ga_wsn_v2.py` |
|---|---|---|
| Protocols run | GA only | GA + PSO + GWO + HEED + LEACH |
| Fitness terms | **5** (adds BS-proximity) | 4 |
| Headline experiment | **Solar ablation** (GA with/without solar) | GA-vs-baselines comparison |
| Convergence plot | GA curve | GA vs PSO vs GWO curves |
| Best for | Studying/tuning the GA itself | Positioning the GA against other algorithms |

Both share the identical realistic radio, realistic solar, Monte-Carlo, timing, and
significance machinery.

---

## 9. Verification status

- `python -m py_compile ga_only_v2.py` **passes**.
- The lossless path (`PACKET_LOSS = 0`) reduces exactly to the original transmit
  behaviour.
- **Not executed in the build sandbox** (no numpy/matplotlib, installs blocked). Run once
  in Colab/local; any traceback will pinpoint the line.
