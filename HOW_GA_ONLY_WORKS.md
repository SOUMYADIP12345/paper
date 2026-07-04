# How `ga_only.py` Works — A Plain English Guide

> A friendly explanation for people who don't write code.
> No programming knowledge needed!

---

## 1. The Big Picture (a story)

Imagine you scatter **50 tiny weather sensors** across a football field. Each sensor:

- Has a **small battery** (like a coin cell)
- Has a **mini solar panel** on top that slowly recharges it during the day
- Can **wirelessly talk** to other sensors and to a **Base Station** (a big computer at the edge of the field)
- Wants to send its readings (temperature, humidity, etc.) to that Base Station

There's one big problem: **radios drain batteries fast**, and the farther a sensor has to "shout" to reach someone, the more battery it burns. If every sensor shouted directly at the Base Station all day, they'd all die in a few hours.

This program is a **simulation** — it doesn't run real sensors, it pretends to, on a computer — that tries out a clever strategy to make the batteries last as long as possible.

---

## 2. The Clever Strategy (3 layers)

Instead of everyone shouting at the Base Station, the sensors organize themselves like a small company:

```
  Base Station (the boss)
        ^
        |
   [MS-CH]  <-- "super leader" who collects from many leaders
        ^
        |
    [ CH ]  <-- "team leader" who collects from nearby workers
        ^
        |
  [sensor] [sensor] [sensor]  <-- regular workers
```

### Layer 1 — Workers (regular sensors)
They just take a measurement and **whisper** it to their nearest team leader. Whispering is cheap (short distance = low battery cost).

### Layer 2 — Team Leaders (called **CH**, short for "Cluster Head")
A team leader gathers all the whispers from its team, **bundles them into one message**, and then has a choice:
- **Option A:** If the Base Station is close *and* my battery is healthy → shout straight to the Base Station.
- **Option B:** Otherwise → pass the bundle to a "super leader" (MS-CH) who is closer to the Base Station.

### Layer 3 — Super Leaders (called **MS-CH**, "Multi-Sink Cluster Head")
A super leader gathers bundles from several team leaders, combines them, and sends the giant bundle to the Base Station.

This way, only a few sensors do the expensive long-distance shouting. Everyone else conserves battery.

---

## 3. The Genetic Algorithm — The "Evolution" Trick

Here's the really cool part. **Who should be the team leaders?** Picking the wrong sensors as leaders (e.g., one with a nearly-dead battery, or one stuck in a corner) wastes energy.

So the program uses a **Genetic Algorithm (GA)** — a method inspired by **Darwinian evolution**.

Think of it like breeding the perfect racehorse:

### Step 1 — Create a generation of guesses
The computer makes 30 different **random guesses** ("what if these 5 sensors were the leaders? What if these other 5 were?"). Each guess is like a candidate team of leaders.

### Step 2 — Score each guess
Each candidate team is graded on four things, like a school report card:

| Grade | What it measures | Why it matters |
|---|---|---|
| **Energy** | Do the picked leaders have full batteries? | Tired leaders die fast |
| **Solar** | Are they getting good sunlight? | Recharging keeps them alive |
| **Coverage** | Can most workers reach a leader nearby? | Closer = cheaper whispers |
| **Spread** | Are the leaders well-spaced across the field? | No clumping in one corner |

The four grades are blended into one final score.

### Step 3 — Survival of the fittest
- The **best two** candidate teams automatically advance to the next round (called "elitism" — the champions live on).
- The rest are made by:
  - **Crossover**: take half of one good team + half of another (like breeding two parents)
  - **Mutation**: occasionally swap one leader for a random different sensor (like a small genetic mutation, in case it makes things better)

### Step 4 — Repeat
Do this 50 times. Each generation is slightly better than the last. After 50 generations, the **winning team of leaders** is chosen for this round.

This whole evolution happens **every single round** because batteries drain, the sun moves, and yesterday's best leader might be exhausted today.

---

## 4. The Solar-Aware Twist (the novel idea)

The "smart" part of this protocol — what makes it special — is that it doesn't just look at **current** battery levels. It also considers:

- **Is the sun shining right now?** (the simulation models a 24-hour day/night cycle, like a sine wave)
- **Will this sensor recharge soon?**
- **Is it physically central** to its peers (so others don't have to shout far)?
- **Is it close to the Base Station?**

A sensor with a half-empty battery but **strong sunlight** might be a better choice than a sensor with a fuller battery sitting in the shade. That's the "solar-aware" insight.

---

## 5. What Happens in One "Round" of Simulation

A round is like one heartbeat of the network. Here's the sequence:

1. **Sun rises/sets** — every alive sensor gets a small recharge if it's daytime.
2. **Reset roles** — yesterday's leaders go back to being regular workers.
3. **Run evolution** — the Genetic Algorithm picks today's best team of CHs.
4. **Decide paths** — each CH decides: send straight to Base Station, or use a super leader?
5. **Pick super leaders** — the program groups distant CHs and elects one MS-CH per group using the solar-aware score.
6. **Workers whisper** to their team leaders. Each whisper costs a bit of battery.
7. **Team leaders bundle** the whispers (also costs battery).
8. **Team leaders send** the bundle either to the Base Station or to a super leader.
9. **Super leaders forward** their giant bundle to the Base Station.
10. **Emergency check** — if a super leader's battery drops below 15%, it's immediately replaced by a healthier neighbor.
11. **Record statistics** — how many are alive? How much total energy is left?

The simulation runs **300 rounds by default** (or until everyone dies).

---

## 6. What You Get When You Run It

The program asks you a bunch of questions at the start (number of sensors, field size, energy settings, etc.). Press Enter to accept defaults.

Then it churns away and produces:

- **Status updates** in the terminal: how many sensors are alive, total remaining energy, etc.
- **Snapshot images** every 50 rounds (saved into a `topology_snapshots` folder) showing dots for sensors, with team leaders and super leaders highlighted differently — like a bird's-eye view of the field.
- **Final summary** showing:
  - Round when the **first sensor died** (network's "first death" — an important metric)
  - Round when the **whole network died**
  - Total packets delivered to the Base Station
  - How many emergency leader replacements happened

---

## 7. Why Does This Matter in Real Life?

Wireless sensor networks are used for:

- **Smart agriculture** — sensors in a field reporting soil moisture
- **Forest fire detection** — sensors in the woods watching for heat
- **Smart cities** — sensors monitoring traffic, air quality, parking
- **Industrial monitoring** — sensors in factories, pipelines, oil rigs

In all of these, you can't easily replace batteries on hundreds of sensors scattered over kilometres. So **stretching battery life from days to years is a huge deal**. Algorithms like this one are how engineers achieve that — by being smart about *who talks to whom* and *when*.

---

## 8. Glossary (in case you got lost)

| Term | Meaning |
|---|---|
| **Node / Sensor** | A small battery-powered device that takes measurements |
| **Base Station (BS)** | The central computer that collects all the data |
| **CH (Cluster Head)** | A "team leader" sensor that gathers data from neighbors |
| **MS-CH** | A "super leader" sensor that gathers from several team leaders |
| **Round** | One full cycle of the simulation (≈ a few seconds in real life) |
| **Genetic Algorithm** | An evolution-inspired method for finding good solutions |
| **Energy harvest** | Recharging the battery from the solar panel |
| **Packet** | A bundle of data sent over the radio |

---

*That's the whole story. The Python file is just a very precise, mathematical way of describing this scenario so a computer can act it out and we can measure how well the strategy works.*
