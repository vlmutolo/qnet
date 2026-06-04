# TODO

- Add snapshot fields and analysis plots for virtual backlog, service deficit `H^R`, and swap deficit `H^mu` separately instead of only reporting the combined backlog.
- Decide whether the physical realization layer should remain stochastic or move to a greedier "flush queued requests when feasible" rule to get even closer to the paper's per-slot physical realization.
- Evaluate a next-reaction / indexed-priority Gillespie variant instead of the current direct-method sampler.
- Add an approximate batched mode such as tau-leaping for large experiments where exact event-by-event Gillespie fidelity is not required.
- Explore timescale-separation / quasi-steady-state approximations for fast subdynamics if exact continuous-time coupling remains too expensive.
