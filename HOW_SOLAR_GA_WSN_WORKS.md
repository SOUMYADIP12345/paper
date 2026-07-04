# How `solar_ga_wsn.py` Works — A Plain English Guide

> A friendly explanation for people who don't write code.
> No programming knowledge needed!

---

## 1. The Big Picture (the head-to-head match)

This is the **main** file in the project. The companion file (`ga_only.py`) is a spin-off that runs just the GA strategy on its own. This combined file runs **both strategies side by side** — the Solar-Aware GA Multi-Sink protocol **and** the LEACH baseline (LEACH is implemented directly inside this file, not in a separate script) — in the same simulation, so you can directly compare them.

Think of it like a **boxing match**:

- **In the red corner:** **LEACH** — the classic, time-tested baseline. Picks team leaders at random.
- **In the blue corner:** **Solar-Aware GA Multi-Sink** — the modern, smarter challenger. Picks team leaders using evolution and adds an extra layer of "super leaders".

The script sets up the **same field, same sensors, same starting batteries, same sun cycle** for both fighters and runs them through hundreds of rounds. At the end, it prints a scorecard and draws graphs showing who lasted longer, who delivered more data, and who used energy more evenly.

---

## 2. What Happens in the Simulation

The basic setup is the same as in the spin-off files:

- A **square field** (default 100m × 100m)
- **50 sensor nodes** scattered randomly across it
- A **Base Station** sitting just outside the field
- Each sensor has a small **battery** and a tiny **solar panel**
- The simulated **sun rises and sets** in 24-round cycles, with a peak harvest at noon

Two complete simulations are run, one after the other:

1. **The LEACH simulation** — same scenario, dice-roll leader selection, 2 layers, every leader shouts directly at the Base Station.
2. **The GA simulation** — same scenario, leaders chosen by Genetic Algorithm, 3 layers including super-leaders for far-away clusters.

Because both use the same random seed and the same physical setup, the comparison is fair.

> If you want a deep explanation of the GA strategy on its own, see:
> - **`HOW_GA_ONLY_WORKS.md`** for the Solar-Aware GA Multi-Sink approach
> - **`SOLAR_GA_WSN_TECHNICAL_DEEP_DIVE.md`** for a formula-by-formula walkthrough
>
> The LEACH baseline is a short, self-contained function inside
> `solar_ga_wsn.py` (`simulate_round_leach`); it has no separate file.

---

## 3. The "Solar-Aware Multi-Sink" Innovation (in 60 seconds)

This file is the **research artifact** behind the project. The new idea has two parts:

### Part A — "Multi-Sink"
Most simple protocols (like LEACH) have **one drain point**: every team leader shouts at the Base Station. The "multi-sink" idea adds **intermediate super-leaders** (called **MS-CH**) that act as additional collection points. Far-away team leaders pass their bundles to a nearby super-leader, who then makes the long shout home. This saves a lot of energy because long shouts are *much* more expensive than short ones.

### Part B — "Solar-Aware"
Picking the super-leader carefully matters. This protocol scores candidate super-leaders on:

- **35%** — How full is the battery?
- **30%** — How much sun is *this* sensor harvesting right now? (Each sensor has its own fixed panel quality — full sun vs. partial shade — so this differs from sensor to sensor.)
- **20%** — Is it geographically central among the team leaders it would serve?
- **15%** — Is it close to the Base Station?

A sensor that is **half full but sitting in full sun** can outscore a fully charged sensor stuck in shade. That's the "solar-aware" twist — it doesn't just look at *now*, it looks at what's about to happen. It only bites because sensors genuinely harvest different amounts (a shared "is it daytime?" value alone couldn't tell two candidates apart).

---

## 4. The Genetic Algorithm Picks the Team Leaders

For team leader (CH) selection, this file uses a **Genetic Algorithm**, the same evolution-inspired trick described in detail in `HOW_GA_ONLY_WORKS.md`. In short:

1. Generate 30 random "candidate teams" of leaders.
2. Score each team on: energy + solar + coverage + spread.
3. Keep the best two, breed the rest with crossover and mutation.
4. Repeat for 50 generations.
5. Use the best-scoring team for this round.

This happens **every round**, because the field changes every round (batteries drain, the sun moves, sensors die).

---

## 5. One Round in the GA Simulation

This is the longest part of the file. Here's what happens in one heartbeat:

1. **Solar harvest** — every alive sensor picks up a bit of charge if the simulated sun is up.
2. **Reset roles** — everyone goes back to being a regular worker.
3. **GA elects team leaders** — runs evolution to choose the best ~10% of alive sensors as Cluster Heads.
4. **Path decisions** — each leader independently decides:
   - **Path A:** "I'm close enough to the Base Station and my battery is healthy → I'll shout directly."
   - **Path B:** "I'm far or weak → I'll wait for a super-leader to relay for me."
5. **Super-leaders are elected** — only among the Path-B leaders. The protocol uses a "k-medoids" technique (a way of geographically grouping the relay leaders into clusters) and picks one super-leader per cluster using the solar-aware score above.
6. **Workers whisper** to their nearest team leader.
7. **Team leaders bundle** the whispers.
8. **Team leaders forward** the bundles — Path A directly to the Base Station, Path B to their super-leader.
9. **Super-leaders bundle again** and make one big long shout to the Base Station.
10. **Emergency check** — if a super-leader's battery dropped below 15%, it's swapped out for a fresher peer mid-round.
11. **Snapshot taken** — every 50 rounds, a topology image is saved.

A round in LEACH is much shorter (no GA, no super-leaders, no path decision) — that's why LEACH is faster but less efficient.

---

## 6. What You Get When You Run It

When you run the file, it asks you setup questions (or you can use built-in defaults). Then it churns through **two full simulations** and produces:

### Terminal output
- Live progress bars for both LEACH and GA runs
- Reports of "first node died at round X" and "network died at round Y" for each
- A final **side-by-side comparison table**, e.g.:

```
  Metric                          LEACH      GA       Winner
  --------------------------------------------------------
  First node death (round)        87        134      GA
  Network lifetime (rounds)       245       298+     GA
  Packets delivered to BS         1,920     2,640    GA
  Final remaining energy (J)      0.04      1.82     GA
  Energy balance (lower=better)   0.087     0.041    GA
```

### Image files
- **`results_comparison.png`** — a multi-panel graph image showing both protocols overlaid: lifetime, residual energy, cumulative deaths, energy balance, dynamic leader counts, and the solar harvest cycle.
- **`topology_snapshots/`** folder — paired snapshots from both protocols every 50 rounds. The GA snapshots show the 3-tier paths (with relay arrows from team leaders to super leaders) and are flagged "MS-USED" or "MS-SKIPPED" depending on whether the super-leader tier was needed that round.

---

## 7. Why Both Protocols in One File?

Three good reasons:

1. **Fair comparison.** Running them in the same script with the same random seed and same configuration means any difference in performance is due to the *strategy*, not luck.
2. **Honest research.** A new protocol has to *prove* it beats the standard. By running LEACH alongside, the file makes the comparison explicit.
3. **Educational value.** Watching both protocols play out side by side makes it crystal clear *why* the smarter strategy wins (or, occasionally, doesn't).

---

## 8. The Story the Project Tells

If you read this file as a story, the message is:

> *"Old random methods like LEACH work, but they waste a lot of battery shouting across long distances. By thinking about who should be the leader (using evolution-style optimization), and by adding a smart relay layer that knows where the sun is shining, we can keep wireless sensor networks alive significantly longer — without changing the hardware at all."*

That's the contribution. The same physical sensors live longer just because they make smarter choices.

---

## 9. Real-World Applications

The same way as in the spin-off files, this kind of work matters for:

- **Smart farming** — sensors monitoring soil moisture across hectares of crops
- **Forest fire detection** — sensors in remote woodland watching for heat
- **Smart cities** — air quality, parking, traffic monitoring across an urban grid
- **Industrial monitoring** — sensors on pipelines, oil rigs, factory floors

In all these cases, **batteries are precious** and **replacing them is expensive or impossible**. Squeezing extra months or years out of a deployment is a big deal — and that's exactly what protocols like this aim to do.

---

## 10. Glossary (in case you got lost)

| Term | Meaning |
|---|---|
| **WSN** | Wireless Sensor Network — a bunch of sensors that talk wirelessly |
| **IoT** | Internet of Things — small connected gadgets, including WSNs |
| **Node / Sensor** | A small battery-powered device that takes measurements |
| **Base Station (BS)** | The central computer that collects all the data |
| **CH (Cluster Head)** | A "team leader" sensor that gathers data from neighbors |
| **MS-CH** | A "super leader" sensor that gathers from several team leaders |
| **LEACH** | The classic random-pick baseline protocol |
| **GA (Genetic Algorithm)** | Evolution-inspired optimization used to pick the best leaders |
| **Multi-Sink** | The "many drain points" idea — using super-leaders as extra collection points |
| **Solar-Aware** | Considering each sensor's own current sunshine (it has its own panel quality), not just battery level, when picking leaders |
| **Round** | One full cycle of the simulation |
| **Packet** | A bundle of data sent over the radio |
| **Path A / Path B** | Direct-to-Base or via-super-leader |

---

*Bottom line: this file is the **head-to-head championship bout** — the Solar-Aware GA Multi-Sink protocol versus the LEACH baseline, both run in the same script. The spin-off file `ga_only.py` is the individual training video for the GA strategy. Read that (and the technical deep dive) for depth, then run this one to see who wins.*
