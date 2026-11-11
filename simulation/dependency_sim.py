#!/usr/bin/env python3
"""
Multi-component dependency simulation for H3 of the accompanying paper.

  H3  Incorporating dependency structure will improve the identification of
      relevant candidate components without necessarily improving causal
      accuracy unless additional causal evidence is available.

H3 separates two outcomes that AIOps work often conflates: narrowing the
candidate set, and correctly naming the cause. The settlement-queue model used
for H1, H2, H4 and H5 is a single server and contains no topology, so it cannot
address this. This model supplies the topology and nothing else; it is not a
queueing model and makes no settlement claims.

Model
-----
A layered directed acyclic graph of services. An edge u -> v means v depends on
u, so a fault at u can surface as an anomaly at v.

One service is the root cause. Its anomaly propagates to descendants with a
probability that decays with hop distance, and the observed magnitude decays
with it. Independently, any service can produce a spontaneous anomaly that has
nothing to do with the fault. Those noise anomalies are the crux of the
hypothesis: a noise anomaly sitting upstream of the true cause will be selected
by any purely structural heuristic, because structure encodes reachability and
not causation.

Onset times are recorded. The true cause starts first; propagated anomalies
follow after a per-hop lag; noise anomalies start at an arbitrary time, which
may precede the fault.

Arms
----
no_graph      Candidate set is every anomalous service. Ranked by magnitude.
              This is the service-centric baseline: no dependency information.

graph         Candidate set is the minimal anomalous set - anomalous services
              with no anomalous ancestor. Ranked by magnitude. Dependency
              structure is used for localisation only.

graph_causal  Same candidate set as graph, ranked by earliest onset. Temporal
              precedence is the additional causal evidence H3 refers to.

Metrics
-------
candidates    Mean size of the candidate set. Lower is better localisation.
recall        True cause is inside the candidate set. Guards against a method
              that shrinks the set by discarding the answer.
top1          The highest-ranked candidate is the true cause. This is causal
              accuracy, and it is what H3 predicts structure alone will not fix.

Stdlib only. No external data. All parameters are stated assumptions.
"""

from __future__ import annotations

import csv
import hashlib
import random
import statistics
from pathlib import Path

# --------------------------------------------------------------------------
# Parameters (assumptions, not measurements)
# --------------------------------------------------------------------------

N_LAYERS = 5              # depth of the dependency graph
PER_LAYER = 5             # services per layer  -> 25 services
MAX_PARENTS = 2           # upper bound on in-degree

P_PROPAGATE = 0.65        # per-hop probability a fault surfaces downstream
HOP_LAG = 2.4             # minutes of delay per hop
JITTER = 0.8              # minutes of noise on propagated onset

# Observed magnitude is not a causal signal. A saturating downstream dependency
# frequently presents a larger deviation than the component that caused it,
# because queueing amplifies. The per-hop factor is therefore centred above 1.0.
# An earlier parameterisation of this model decayed magnitude downstream and set
# the cause's magnitude to a fixed 1.0, which made "rank by magnitude" a near
# oracle and produced a no-graph baseline of 77% top-1 accuracy. That baseline
# was an artefact of the parameterisation, not a property of the problem.
CAUSE_MAG = (0.60, 1.00)  # magnitude at the faulting component
AMPLIFY = (0.85, 1.55)    # per-hop magnitude factor, mean above 1
NOISE_MAG = (0.40, 1.40)  # unrelated anomalies are not smaller as a class

P_NOISE = 0.06            # per-service probability of an unrelated anomaly
WINDOW = 30.0             # observation window, minutes
NOISE_ONSET = (-10.0, 30.0)   # unrelated anomalies may precede the fault

ARMS = ["no_graph", "graph", "graph_causal"]


def stable_seed(*parts) -> int:
    key = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


class Graph:
    """Layered DAG. Parents of a node are drawn from the previous layer."""

    def __init__(self, rng: random.Random):
        self.nodes = list(range(N_LAYERS * PER_LAYER))
        self.parents: dict[int, list[int]] = {n: [] for n in self.nodes}
        self.children: dict[int, list[int]] = {n: [] for n in self.nodes}
        for layer in range(1, N_LAYERS):
            prev = [(layer - 1) * PER_LAYER + i for i in range(PER_LAYER)]
            for i in range(PER_LAYER):
                v = layer * PER_LAYER + i
                for u in rng.sample(prev, rng.randint(1, MAX_PARENTS)):
                    self.parents[v].append(u)
                    self.children[u].append(v)

    def descendants_with_distance(self, r: int) -> dict[int, int]:
        """Hop distance from r to each reachable descendant."""
        out: dict[int, int] = {r: 0}
        frontier = [r]
        while frontier:
            nxt = []
            for u in frontier:
                for v in self.children[u]:
                    if v not in out:
                        out[v] = out[u] + 1
                        nxt.append(v)
            frontier = nxt
        return out

    def ancestors(self, v: int) -> set[int]:
        seen: set[int] = set()
        frontier = list(self.parents[v])
        while frontier:
            u = frontier.pop()
            if u in seen:
                continue
            seen.add(u)
            frontier.extend(self.parents[u])
        return seen


def one_run(seed: int) -> dict:
    rng = random.Random(seed)
    g = Graph(rng)

    cause = rng.choice(g.nodes)
    dist = g.descendants_with_distance(cause)

    # anomaly -> (magnitude, onset, is_fault_related)
    anom: dict[int, tuple[float, float, bool]] = {}

    # the fault itself
    base = rng.uniform(*CAUSE_MAG)
    anom[cause] = (base, 0.0, True)

    # propagation to descendants, with amplification rather than decay
    for v, d in dist.items():
        if v == cause:
            continue
        if rng.random() < P_PROPAGATE ** d:
            mag = base
            for _ in range(d):
                mag *= rng.uniform(*AMPLIFY)
            onset = d * HOP_LAG + rng.gauss(0, JITTER)
            anom[v] = (mag, onset, True)

    # unrelated anomalies anywhere in the graph, at arbitrary times
    for v in g.nodes:
        if v in anom:
            continue
        if rng.random() < P_NOISE:
            anom[v] = (rng.uniform(*NOISE_MAG),
                       rng.uniform(*NOISE_ONSET), False)

    anomalous = set(anom)

    # ---- arm 1: no dependency information
    cand_ng = sorted(anomalous)
    rank_ng = sorted(cand_ng, key=lambda v: -anom[v][0])

    # ---- arms 2 and 3: minimal anomalous set
    minimal = [v for v in anomalous if not (g.ancestors(v) & anomalous)]
    if not minimal:                     # pathological; fall back
        minimal = cand_ng
    rank_g = sorted(minimal, key=lambda v: -anom[v][0])

    # Arm 3's additional causal evidence is propagation-pattern fit, not raw
    # earliest onset. For each candidate, ask how well the onsets of the
    # anomalous services it reaches match the delay pattern expected if that
    # candidate were the cause. A candidate that explains the observed timing
    # of its own downstream anomalies is preferred. Ranking on earliest onset
    # alone was tried first and is a weaker test, because it cannot separate
    # the cause from an unrelated anomaly that merely started earlier.
    def fit(v: int) -> tuple[float, float]:
        reach = g.descendants_with_distance(v)
        expl = [(u, d) for u, d in reach.items() if u in anomalous and u != v]
        if not expl:
            return (float("inf"), anom[v][1])
        err = statistics.fmean(
            abs((anom[u][1] - anom[v][1]) - d * HOP_LAG) for u, d in expl)
        # reward candidates that account for more of what is anomalous
        coverage = len(expl) / max(len(anomalous) - 1, 1)
        return (err - 2.0 * coverage, anom[v][1])

    rank_gc = sorted(minimal, key=fit)

    res = {"seed": seed, "cause": cause, "n_anomalous": len(anomalous),
           "n_fault_related": sum(1 for v in anom if anom[v][2]),
           "n_noise": sum(1 for v in anom if not anom[v][2])}
    for name, cand, rank in (("no_graph", cand_ng, rank_ng),
                             ("graph", minimal, rank_g),
                             ("graph_causal", minimal, rank_gc)):
        res[f"{name}_candidates"] = len(cand)
        res[f"{name}_recall"] = int(cause in cand)
        res[f"{name}_top1"] = int(bool(rank) and rank[0] == cause)
        res[f"{name}_top3"] = int(cause in rank[:3])
    return res


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5


def experiment(reps: int = 3000, out_dir: Path | None = None,
               p_noise: float | None = None) -> list[dict]:
    global P_NOISE
    keep = P_NOISE
    if p_noise is not None:
        P_NOISE = p_noise
    rows = [one_run(stable_seed("h3", p_noise, i)) for i in range(reps)]
    P_NOISE = keep
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "dependency_results.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def report(rows: list[dict], reps: int) -> None:
    print(f"\n{'='*104}")
    print(f"H3  DEPENDENCY STRUCTURE vs CAUSAL ACCURACY   reps={reps}   "
          f"{N_LAYERS*PER_LAYER} services, {N_LAYERS} layers")
    print(f"{'='*104}")
    a, _ = mean_ci([r["n_anomalous"] for r in rows])
    f, _ = mean_ci([r["n_fault_related"] for r in rows])
    n, _ = mean_ci([r["n_noise"] for r in rows])
    print(f"  anomalous services per run: {a:.2f}  "
          f"(fault-related {f:.2f}, unrelated {n:.2f})")
    print(f"\n{'arm':<14} | {'candidate set':>20} | {'recall %':>17} | "
          f"{'top-1 causal %':>20} | {'top-3 %':>8}")
    print("-" * 104)
    out = {}
    for arm in ARMS:
        c, cci = mean_ci([r[f"{arm}_candidates"] for r in rows])
        rc, rci = mean_ci([r[f"{arm}_recall"] for r in rows])
        t1, t1ci = mean_ci([r[f"{arm}_top1"] for r in rows])
        t3, _ = mean_ci([r[f"{arm}_top3"] for r in rows])
        out[arm] = (c, cci, rc, rci, t1, t1ci, t3)
        print(f"{arm:<14} | {c:>9.2f} [{c-cci:>4.2f},{c+cci:>5.2f}] | "
              f"{100*rc:>8.1f} [{100*(rc-rci):>4.1f}] | "
              f"{100*t1:>10.1f} [{100*(t1-t1ci):>4.1f},{100*(t1+t1ci):>5.1f}] | "
              f"{100*t3:>7.1f}")
    print("-" * 104)
    ng, gr, gc = out["no_graph"], out["graph"], out["graph_causal"]
    print(f"\n  localisation: candidate set {ng[0]:.2f} -> {gr[0]:.2f}  "
          f"({100*(ng[0]-gr[0])/ng[0]:.0f}% smaller), "
          f"recall {100*ng[2]:.1f}% -> {100*gr[2]:.1f}%")
    d1 = gr[4] - ng[4]
    sig1 = abs(d1) > (gr[5] + ng[5])
    print(f"  structure alone, causal accuracy: {100*ng[4]:.1f}% -> {100*gr[4]:.1f}% "
          f"({100*d1:+.1f} pp, {'significant' if sig1 else 'not significant'})")
    d2 = gc[4] - gr[4]
    sig2 = abs(d2) > (gc[5] + gr[5])
    print(f"  adding temporal precedence:       {100*gr[4]:.1f}% -> {100*gc[4]:.1f}% "
          f"({100*d2:+.1f} pp, {'significant' if sig2 else 'not significant'})")
    print(f"\n  H3 asks for: localisation improves, causal accuracy does not improve "
          f"much on structure\n  alone, and does improve once causal evidence is added.")


def noise_sweep(reps: int = 1500) -> None:
    """The noise rate is the parameter H3 turns on: an unrelated anomaly sitting
    upstream of the true cause is what defeats a purely structural heuristic."""
    print(f"\n{'='*104}")
    print(f"SENSITIVITY: unrelated-anomaly rate   reps={reps}/cell")
    print(f"{'='*104}")
    print(f"{'P(noise)':>9} | {'cand no_graph':>13} | {'cand graph':>11} | "
          f"{'top1 no_graph':>13} | {'top1 graph':>11} | {'top1 graph+time':>15}")
    print("-" * 104)
    for pn in (0.00, 0.03, 0.06, 0.12, 0.20):
        rows = experiment(reps=reps, p_noise=pn)
        f = lambda k: mean_ci([r[k] for r in rows])[0]
        print(f"{pn:>9.2f} | {f('no_graph_candidates'):>13.2f} | {f('graph_candidates'):>11.2f} | "
              f"{100*f('no_graph_top1'):>12.1f}% | {100*f('graph_top1'):>10.1f}% | "
              f"{100*f('graph_causal_top1'):>14.1f}%")
    print()


if __name__ == "__main__":
    out = Path(__file__).parent / "results"
    REPS = 3000
    rows = experiment(reps=REPS, out_dir=out)
    report(rows, REPS)
    noise_sweep(reps=1500)
    print(f"per-replication results written to {out/'dependency_results.csv'}")
