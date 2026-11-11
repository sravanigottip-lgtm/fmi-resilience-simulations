#!/usr/bin/env python3
"""
Release admission against settlement finality: the rollback-horizon simulation.

Premise
-------
Conventional release governance assumes rollback is always available, so a
change can be admitted whenever pre-deployment signals look acceptable. In a
settlement system that assumption fails on timing. Rolling a change back costs
time - detect, decide, execute, recover - and that cost must be paid before the
finality boundary of the obligations currently in flight. Past a certain
instant, initiating a rollback is itself the thing that causes obligations to
miss their deadlines. Rollback capability therefore expires before the
settlement deadline does.

The instant at which it expires is what this model calls the ROLLBACK HORIZON.
Admitting a change after its horizon has passed means accepting that if the
change is bad, the safe remedy is unavailable.

Hypotheses
----------
H1  A blanket change freeze and a rollback-horizon policy admit different sets
    of deployments, not merely fewer. Measured as Jaccard overlap of admitted
    deployment sets on identical arrival streams.

H2  Rollback-horizon admission reduces finality breaches relative to a
    volume-matched control that defers the same proportion of changes at
    random. This is the form that matters: if the horizon policy only wins by
    deploying less, it is not a contribution.

H3  A blanket freeze costs throughput out of proportion to the safety it buys,
    relative to horizon-aware admission.

H4  The mediating mechanism is the unrollbackable-deployment rate: changes whose
    defect is detected after the horizon has passed. Horizon-aware admission
    should reduce this sharply, and it should account for the H2 effect.

H5  The advantage depends on how predictable rollback duration is. As rollback
    duration variance grows, the horizon cannot be computed reliably and the
    policies should converge toward freezing.

H6  NULL CONDITION. In a workload whose finality boundaries never bind,
    rollback-horizon admission should degenerate to error-budget admission and
    show no advantage. If an advantage still appears under LOOSE, the model is
    rigged rather than the hypothesis confirmed.

Model
-----
Single-server settlement queue over one processing session. Obligations arrive
Poisson with exponential service demand and carry a finality deadline. A
fraction are linked pairs sharing the tighter deadline, so either side missing
fails both.

Change requests become ready at intervals. An admission policy accepts or defers
each one. An admitted change carries a latent defect with probability P_DEFECT.
A defect manifests after a delay and then multiplies service time until it is
remediated. Detection takes DETECT_LAG after manifestation.

On detection the operator rolls back. Rollback takes ROLLBACK_MEAN minutes with
variability, disrupts work in progress, and is followed by a recovery interval.
A rollback that completes after the earliest at-risk finality boundary has not
saved those obligations; the deployment is recorded as unrollbackable.

Policies
--------
always_deploy     Admit every ready change. No timing awareness.
freeze_window     Admit unless within FREEZE_MIN of the session finality cutoff.
                  The conventional conservative practice.
error_budget      Admit while recent deadline misses stay under a budget.
                  Service-reliability style admission with no timing model.
rollback_horizon  Admit only if the projected worst-case rollback completion -
                  manifestation, detection, rollback, recovery - precedes the
                  earliest finality boundary among in-flight obligations, with
                  a safety margin. Uses observable quantities only.

Stdlib only. No external data. All parameters are stated assumptions.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Parameters (assumptions, not measurements)
# --------------------------------------------------------------------------

HORIZON = 480.0           # minutes in the processing session
ARRIVAL_RATE = 0.50       # obligations per minute
BASE_DEMAND = 0.9         # mean service minutes per obligation
LINK_FRACTION = 0.30      # fraction of obligations in linked pairs

CHANGE_INTERVAL = 45.0    # a change becomes ready this often
P_DEFECT = 0.15           # probability an admitted change carries a defect
MANIFEST_LAG = (5.0, 40.0)   # uniform delay from deploy to first symptom
DETECT_LAG = 6.0          # minutes from manifestation to detection
DEFECT_FACTOR = 6.0       # service-time multiplier while a defect is live

ROLLBACK_MEAN = 12.0      # minutes to execute a rollback
ROLLBACK_CV = 0.35        # coefficient of variation on rollback duration
RECOVERY_TIME = 5.0       # minutes after rollback before throughput normalises
ROLLBACK_DISRUPT = 2.5    # minutes charged to each queued obligation

FREEZE_MIN = 90.0         # blanket freeze this long before the session cutoff
ERROR_BUDGET = 0.02       # tolerated recent miss fraction for error_budget
BUDGET_WINDOW = 60        # obligations in the error-budget lookback
HORIZON_MARGIN = 1.25     # safety multiplier on projected rollback cost

SLACK = {"tight": (25.0, 70.0), "loose": (600.0, 900.0)}

# random_defer is a volume-matched control, not a practice anyone proposes. It
# admits changes at the same rate the horizon policy happens to admit them, so
# that any advantage the horizon policy shows cannot be attributed to simply
# deploying less often. Without this arm the headline result is uninterpretable.
POLICIES = ["always_deploy", "freeze_window", "error_budget",
            "random_defer", "rollback_horizon"]
RANDOM_ADMIT_RATE = 0.50

DEFAULTS = {
    "p_defect": P_DEFECT,
    "rollback_mean": ROLLBACK_MEAN,
    "rollback_cv": ROLLBACK_CV,
    "freeze_min": FREEZE_MIN,
    "horizon_margin": HORIZON_MARGIN,
    "defect_factor": DEFECT_FACTOR,
    "random_rate": RANDOM_ADMIT_RATE,
    "manifest_lo": MANIFEST_LAG[0],
    "manifest_hi": MANIFEST_LAG[1],
}


def cfg(**over):
    c = dict(DEFAULTS)
    c.update(over)
    return c


def stable_seed(*parts) -> int:
    key = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


(ARRIVAL, COMPLETE, CHANGE_READY, MANIFEST, DETECT,
 ROLLBACK_DONE, RECOVERED) = range(7)


@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: int = field(compare=False)
    ref: int | None = field(compare=False, default=None)


@dataclass
class Obligation:
    oid: int
    arrival: float
    demand: float
    deadline: float
    partner: int | None = None
    completed: float | None = None
    penalty: float = 0.0

    def on_time(self) -> bool:
        return self.completed is not None and self.completed <= self.deadline


@dataclass
class Change:
    cid: int
    ready: float
    admitted: bool = False
    defective: bool = False
    deployed_at: float | None = None
    manifested_at: float | None = None
    detected_at: float | None = None
    rollback_started: float | None = None
    rollback_done: float | None = None
    unrollbackable: bool = False
    horizon_at_admission: float | None = None


class Sim:
    def __init__(self, policy: str, workload: str, seed: int, conf: dict | None = None):
        self.cfg = conf or cfg()
        self.policy = policy
        self.workload = workload
        self.rng = random.Random(seed)

        self.t = 0.0
        self.seq = 0
        self.eq: list[Event] = []

        self.obs: dict[int, Obligation] = {}
        self.queue: list[int] = []
        self.in_service: int | None = None
        self.service_ends: float | None = None

        self.multiplier = 1.0
        self.changes: dict[int, Change] = {}
        self.live_defect: int | None = None
        self.rolling_back = False

        self.completed_order: list[bool] = []   # on-time flags, for error budget
        self.n_rollbacks = 0
        self.n_disrupted = 0

    def push(self, time: float, kind: int, ref: int | None = None) -> None:
        self.seq += 1
        heapq.heappush(self.eq, Event(time, self.seq, kind, ref))

    # ---------------- setup ----------------

    def _generate(self) -> None:
        lo, hi = SLACK[self.workload]
        t = 0.0
        oid = 0
        pending: int | None = None
        while True:
            t += self.rng.expovariate(ARRIVAL_RATE)
            if t >= HORIZON:
                break
            o = Obligation(oid, t, self.rng.expovariate(1.0 / BASE_DEMAND),
                           t + self.rng.uniform(lo, hi))
            if pending is None and self.rng.random() < LINK_FRACTION:
                pending = oid
            elif pending is not None:
                o.partner = pending
                self.obs[pending].partner = oid
                d = min(o.deadline, self.obs[pending].deadline)
                o.deadline = self.obs[pending].deadline = d
                pending = None
            self.obs[oid] = o
            self.push(t, ARRIVAL, oid)
            oid += 1

        cid = 0
        ct = CHANGE_INTERVAL
        while ct < HORIZON:
            self.changes[cid] = Change(cid, ct)
            self.push(ct, CHANGE_READY, cid)
            cid += 1
            ct += CHANGE_INTERVAL

    # ---------------- service ----------------

    def _start_service(self) -> None:
        if self.in_service is not None or not self.queue or self.rolling_back:
            return
        oid = self.queue.pop(0)
        self.in_service = oid
        o = self.obs[oid]
        dur = o.demand * self.multiplier + o.penalty
        o.penalty = 0.0
        self.service_ends = self.t + dur
        self.push(self.service_ends, COMPLETE, oid)

    # ---------------- admission ----------------

    def _session_cutoff(self) -> float:
        """Latest finality boundary that still binds within the session."""
        return HORIZON

    def _earliest_at_risk_boundary(self) -> float | None:
        order = ([self.in_service] if self.in_service is not None else []) + self.queue
        if not order:
            return None
        return min(self.obs[o].deadline for o in order)

    def _projected_rollback_cost(self) -> float:
        """Worst-case time from deploying now to being recovered from a rollback.

        Uses only quantities an operator can know before deploying: the
        distribution of manifestation delay, the detection lag, the mean
        rollback duration inflated for variability, and the recovery interval.
        """
        c = self.cfg
        manifest = self.cfg["manifest_hi"]
        spread = c["rollback_mean"] * c["rollback_cv"]
        return (manifest + DETECT_LAG + c["rollback_mean"] + 2.0 * spread
                + RECOVERY_TIME) * c["horizon_margin"]

    def _recent_miss_rate(self) -> float:
        w = self.completed_order[-BUDGET_WINDOW:]
        if len(w) < 10:
            return 0.0
        return sum(1 for ok in w if not ok) / len(w)

    def _admit(self, ch: Change) -> bool:
        if self.policy == "always_deploy":
            return True
        if self.policy == "freeze_window":
            return self.t < self._session_cutoff() - self.cfg["freeze_min"]
        if self.policy == "error_budget":
            return self._recent_miss_rate() <= ERROR_BUDGET
        if self.policy == "random_defer":
            return self.rng.random() < self.cfg["random_rate"]
        # rollback_horizon
        boundary = self._earliest_at_risk_boundary()
        if boundary is None:
            return True
        cost = self._projected_rollback_cost()
        ch.horizon_at_admission = boundary - cost
        return self.t <= ch.horizon_at_admission

    # ---------------- defect lifecycle ----------------

    def _deploy(self, ch: Change) -> None:
        ch.admitted = True
        ch.deployed_at = self.t
        if self.rng.random() < self.cfg["p_defect"]:
            ch.defective = True
            self.push(self.t + self.rng.uniform(self.cfg["manifest_lo"],
                                               self.cfg["manifest_hi"]), MANIFEST, ch.cid)

    def _rollback(self, ch: Change) -> None:
        self.n_rollbacks += 1
        ch.rollback_started = self.t
        dur = max(1.0, self.rng.gauss(self.cfg["rollback_mean"],
                                      self.cfg["rollback_mean"] * self.cfg["rollback_cv"]))
        # Was the safe remedy still available when we reached for it?
        boundary = self._earliest_at_risk_boundary()
        if boundary is not None and self.t + dur + RECOVERY_TIME > boundary:
            ch.unrollbackable = True

        self.rolling_back = True
        if self.in_service is not None:
            self.queue.insert(0, self.in_service)
            self.n_disrupted += 1
            self.in_service = None
            self.service_ends = None
        for oid in self.queue:
            self.obs[oid].penalty += ROLLBACK_DISRUPT
            self.n_disrupted += 1
        self.push(self.t + dur, ROLLBACK_DONE, ch.cid)

    # ---------------- main loop ----------------

    def run(self) -> dict:
        self._generate()
        while self.eq:
            ev = heapq.heappop(self.eq)
            self.t = ev.time
            if self.t > HORIZON * 3:
                break

            if ev.kind == ARRIVAL:
                self.queue.append(ev.ref)
                self._start_service()

            elif ev.kind == COMPLETE:
                if self.in_service != ev.ref or abs(self.t - (self.service_ends or -1)) > 1e-9:
                    continue
                o = self.obs[ev.ref]
                o.completed = self.t
                self.completed_order.append(o.on_time())
                self.in_service = None
                self.service_ends = None
                self._start_service()

            elif ev.kind == CHANGE_READY:
                ch = self.changes[ev.ref]
                if self._admit(ch):
                    self._deploy(ch)

            elif ev.kind == MANIFEST:
                ch = self.changes[ev.ref]
                if self.live_defect is None:
                    ch.manifested_at = self.t
                    self.live_defect = ch.cid
                    self.multiplier = self.cfg["defect_factor"]
                    self.push(self.t + DETECT_LAG, DETECT, ch.cid)

            elif ev.kind == DETECT:
                ch = self.changes[ev.ref]
                if self.live_defect == ch.cid and not self.rolling_back:
                    ch.detected_at = self.t
                    self._rollback(ch)

            elif ev.kind == ROLLBACK_DONE:
                ch = self.changes[ev.ref]
                ch.rollback_done = self.t
                self.live_defect = None
                self.multiplier = 1.0
                self.push(self.t + RECOVERY_TIME, RECOVERED, ch.cid)

            elif ev.kind == RECOVERED:
                self.rolling_back = False
                self._start_service()

        return self._results()

    # ---------------- results ----------------

    def _results(self) -> dict:
        settled = {o.oid: o.on_time() for o in self.obs.values()}
        for o in self.obs.values():
            if o.partner is not None and not settled.get(o.partner, False):
                settled[o.oid] = False
        total = len(self.obs) or 1
        missed = sum(1 for v in settled.values() if not v)

        admitted = [c for c in self.changes.values() if c.admitted]
        surfaced = [c for c in self.changes.values() if c.manifested_at is not None]
        late = [c for c in self.changes.values()
                if c.defective and c.manifested_at is None]
        defective = [c for c in admitted if c.defective]
        unroll = [c for c in defective if c.unrollbackable]
        return {
            "policy": self.policy,
            "workload": self.workload,
            "obligations": total,
            "missed": missed,
            "miss_rate": missed / total,
            "changes_ready": len(self.changes),
            "admitted": len(admitted),
            "throughput": len(admitted) / max(len(self.changes), 1),
            "defective_admitted": len(defective),
            "unrollbackable": len(unroll),
            "unroll_rate_of_admitted": len(unroll) / max(len(admitted), 1),
            "surfaced": len(surfaced),
            "defect_never_surfaced": len(late),
            "rollbacks": self.n_rollbacks,
            "disrupted": self.n_disrupted,
            "admitted_ids": frozenset(c.cid for c in admitted),
        }


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5


def run_experiment(reps=500, out_dir: Path | None = None, conf=None,
                   workloads=None, out_name="rollback_horizon_results.csv"):
    rows, jac = [], {}
    for wl in (workloads or list(SLACK)):
        for seed in range(reps):
            per = {}
            for pol in POLICIES:
                r = Sim(pol, wl, stable_seed(wl, seed) & 0xFFFFFF, conf).run()
                per[pol] = r
                rows.append({k: v for k, v in r.items() if k != "admitted_ids"})
            a = per["freeze_window"]["admitted_ids"]
            b = per["rollback_horizon"]["admitted_ids"]
            if a or b:
                jac.setdefault(wl, []).append(len(a & b) / len(a | b))
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / out_name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return {"rows": rows, "jaccard": jac, "reps": reps}


def summarize(exp):
    rows, jac, reps = exp["rows"], exp["jaccard"], exp["reps"]
    sel = lambda wl, pol, f: [r[f] for r in rows if r["workload"] == wl and r["policy"] == pol]
    print(f"\n{'='*112}")
    print(f"RELEASE ADMISSION AGAINST SETTLEMENT FINALITY   reps={reps}/cell   "
          f"session {HORIZON:.0f} min, change every {CHANGE_INTERVAL:.0f} min")
    print(f"{'='*112}")
    for wl in SLACK:
        tag = ("TIGHT - finality boundaries bind" if wl == "tight"
               else "LOOSE - boundaries never bind (null condition for H6)")
        print(f"\n### {wl.upper()}: {tag}")
        print(f"{'policy':<18} | {'miss rate %':>18} | {'admitted':>9} | {'throughput':>10} | "
              f"{'unrollbackable':>14} | {'rollbacks':>9}")
        print("-" * 112)
        for pol in POLICIES:
            mr, ci = mean_ci(sel(wl, pol, "miss_rate"))
            ad, _ = mean_ci(sel(wl, pol, "admitted"))
            tp, _ = mean_ci(sel(wl, pol, "throughput"))
            ur, _ = mean_ci(sel(wl, pol, "unrollbackable"))
            rb, _ = mean_ci(sel(wl, pol, "rollbacks"))
            print(f"{pol:<18} | {100*mr:>9.2f} [{100*(mr-ci):>4.2f},{100*(mr+ci):>5.2f}] | "
                  f"{ad:>9.2f} | {100*tp:>9.1f}% | {ur:>14.3f} | {rb:>9.2f}")

    print("\n### H1  overlap of admitted deployment sets: freeze_window vs rollback_horizon")
    for wl in SLACK:
        m, ci = mean_ci(jac.get(wl, []))
        print(f"  {wl:>6}: Jaccard {m:.3f} [{m-ci:.3f}, {m+ci:.3f}]")

    print("\n### H2  volume-matched comparison: is the gain from timing or from deploying less?")
    rh_m, rh_ci = mean_ci(sel('tight', 'rollback_horizon', 'miss_rate'))
    rd_m, rd_ci = mean_ci(sel('tight', 'random_defer', 'miss_rate'))
    rh_t = mean_ci(sel('tight', 'rollback_horizon', 'throughput'))[0]
    rd_t = mean_ci(sel('tight', 'random_defer', 'throughput'))[0]
    print(f"  random_defer      miss {100*rd_m:>5.2f}% [{100*(rd_m-rd_ci):.2f},{100*(rd_m+rd_ci):.2f}]"
          f"   throughput {100*rd_t:.1f}%")
    print(f"  rollback_horizon  miss {100*rh_m:>5.2f}% [{100*(rh_m-rh_ci):.2f},{100*(rh_m+rh_ci):.2f}]"
          f"   throughput {100*rh_t:.1f}%")
    dis = abs(rd_m-rh_m) > (rd_ci+rh_ci)
    print(f"  difference {100*(rd_m-rh_m):+.2f} pp at {100*(rd_t-rh_t):+.1f} pp throughput gap, "
          f"intervals {'disjoint' if dis else 'OVERLAP - mechanism not established'}")

    print("\n### H3  safety and throughput, TIGHT")
    base = {p: (mean_ci(sel('tight', p, 'miss_rate'))[0],
                mean_ci(sel('tight', p, 'throughput'))[0]) for p in POLICIES}
    print(f"{'policy':<18} | {'miss %':>8} | {'throughput':>11} | {'vs error_budget':>32}")
    print("-" * 82)
    eb_m, eb_t = base['error_budget']
    for p in POLICIES:
        m, t = base[p]
        print(f"{p:<18} | {100*m:>8.2f} | {100*t:>10.1f}% | "
              f"{100*(m-eb_m):>+8.2f} pp miss, {100*(t-eb_t):>+7.1f} pp throughput")

    print("\n### H4  unrollbackable rate as share of admitted deployments, TIGHT")
    for p in POLICIES:
        m, ci = mean_ci(sel('tight', p, 'unroll_rate_of_admitted'))
        print(f"  {p:<18} {100*m:>6.2f}% [{100*(m-ci):>5.2f}, {100*(m+ci):>5.2f}]")
    print()


def cv_sweep(reps=300):
    """H5: as rollback duration becomes less predictable, does the horizon
    policy lose its advantage and converge toward freezing?"""
    print(f"\n{'='*104}")
    print(f"H5  SENSITIVITY TO ROLLBACK-DURATION VARIABILITY   reps={reps}/cell, TIGHT")
    print(f"{'='*104}")
    print(f"{'CV':>6} | {'freeze miss%':>12} | {'budget miss%':>12} | {'horizon miss%':>13} | "
          f"{'freeze tp':>10} | {'horizon tp':>11} | {'horizon best?':>13}")
    print("-" * 104)
    for cv in (0.10, 0.35, 0.60, 1.00):
        exp = run_experiment(reps=reps, conf=cfg(rollback_cv=cv), workloads=["tight"])
        rows = exp["rows"]
        g = lambda p, f: mean_ci([r[f] for r in rows if r["policy"] == p])[0]
        fm, bm, hm = g('freeze_window', 'miss_rate'), g('error_budget', 'miss_rate'), g('rollback_horizon', 'miss_rate')
        ft, ht = g('freeze_window', 'throughput'), g('rollback_horizon', 'throughput')
        best = hm <= min(fm, bm)
        print(f"{cv:>6.2f} | {100*fm:>12.2f} | {100*bm:>12.2f} | {100*hm:>13.2f} | "
              f"{100*ft:>9.1f}% | {100*ht:>10.1f}% | {'yes' if best else 'NO':>13}")
    print()


if __name__ == "__main__":
    out = Path(__file__).parent / "results"
    summarize(run_experiment(reps=500, out_dir=out))
    cv_sweep(reps=300)
    print(f"per-replication results written to {out/'rollback_horizon_results.csv'}")
