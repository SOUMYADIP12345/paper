# SGA-MS: A Solar-Aware Genetic-Algorithm Multi-Sink Data-Aggregation Protocol for Prolonging Lifetime in Energy-Harvesting IoT Sensor Networks

::center::**Soumyadip [Last Name]** *(corresponding author)*, **[Co-author Name]**, **[Supervisor Name]**
::center::Department of Computer Science and Engineering, *[Institution Name]*, *[City, Country]*
::center::Corresponding author: *[you@example.com]*   |   ORCID: *[0000-0000-0000-0000]*

---

## Abstract

Battery lifetime is the dominant constraint on the wireless sensor networks (WSNs) that support large-scale IoT deployments, because radio energy grows with the square, and beyond a crossover distance the fourth power, of transmission range. This paper presents SGA-MS, a solar-aware, genetic-algorithm-driven, multi-sink data-aggregation protocol for sensor networks whose nodes carry photovoltaic harvesters. A genetic algorithm re-elects cluster heads every round using a fitness that jointly rewards residual energy, present harvesting opportunity, coverage, and spatial spread; heads that cannot reach the base station cheaply relay through a dynamically sized tier of multi-sink cluster heads placed by k-medoids and chosen by a solar-aware score. Crucially, harvesting is modelled per node via a fixed panel-efficiency factor, so the solar signal genuinely discriminates between candidates rather than adding a decision-neutral constant. In representative simulations against a LEACH baseline under identical conditions, SGA-MS delayed the first node death by about 54% (round 87 to 134), delivered roughly 38% more packets to the base station, cut the spread of per-node energy by about half (0.087 to 0.041 J), and retained far more residual energy. The takeaway: pairing evolutionary, harvesting-aware cluster-head selection with an adaptive multi-sink relay tier can substantially extend the operational life of energy-harvesting sensor networks without any change to the hardware.

## Keywords

Wireless Sensor Networks; Internet of Things; Energy Harvesting; Genetic Algorithm; Cluster-Head Selection; Multi-Sink Routing; Data Aggregation; Network Lifetime

---

*Notes for the authors.* Replace every bracketed field above with the real details (name, co-authors, institution, email, and optional ORCID). The results quoted in the abstract come from a single representative simulation run under the default configuration; before submitting to a venue, re-run the experiments over multiple random seeds and report averages with confidence intervals so that these figures are statistically supported.
