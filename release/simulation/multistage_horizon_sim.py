#!/usr/bin/env python3
"""
Second model for the rollback-horizon paper: a multi-stage settlement path.

Why a second model
------------------
The single-server model treats the cost of withdrawing a change, R, as one
number. That is defensible when a change lands on the only thing there is. It
stops being defensible the moment an obligation has to traverse several
processing stages, because then the cost of withdrawing a change depends on
which stage it landed on and on where each in-flight obligation currently sits
relative to that stage. An obligation that has already cleared the affected
stage is unharmed by a defect there. An obligation that has not yet reached it
will be harmed, and will also be harmed by the rollback. R becomes a
distribution over paths rather than a scalar, and the question is whether a
horizon computed from the scalar still protects anything.

  H7  A horizon computed from a single system-wide withdrawal cost will
      under-protect in a multi-stage path, because the effective cost varies by
      which stage was changed. A stage-aware horizon, computed from the
      obligations that must still traverse the affected stage, will outperform
      it at matched deployment volume.

H7 can fail in either direction. If stage awareness gives nothing, the scalar
horizon of the first model generalises and the extra machinery is unjustified.
If it gives a great deal, the first model's headline numbers are optimistic and
should be read as a lower bound on what a real path costs.

Model
-----
Obligations traverse STAGES tandem queues in order. Each stage has its own
exponential service demand. An obligation carries one finality deadline that
applies to completing the whole path.

A change is deployed to exactly one stage, chosen uniformly. If defective, the
defect appears after a delay and multiplies that stage's service time only.
Detection follows. Rollback restores the stage, costs time, and disrupts work
sitting in or before the affected stage; obligations already downstream of it
are untouched.

Policies
--------
always_deploy        admit everything
scalar_horizon       the first model's rule: one system-wide R, compared against
                     the earliest deadline among all in-flight obligations
stage_aware_horizon  R computed for the target stage, compared against the
                     earliest deadline among obligations that have not yet
                     cleared that stage
random_defer         volume-matched control, rate set to match stage_aware

Stdlib only. Parameters are stated assumptions.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

HORIZON = 480.0
ARRIVAL_RATE = 0.42
STAGES = 4
STAGE_DEMAND = [0.55, 0.45, 0.50, 0.40]   # mean service minutes per stage
LINK_FRACTION = 0.30
SLACK = {"tight": (30.0, 85.0), "loose": (600.0, 900.0)}

CHANGE_INTERVAL = 45.0
P_DEFECT = 0.15
MANIFEST_LAG = (5.0, 40.0)
DETECT_LAG = 6.0
DEFECT_FACTOR = 6.0

ROLLBACK_MEAN = 12.0
ROLLBACK_CV = 0.35
RECOVERY_TIME = 5.0
ROLLBACK_DISRUPT = 2.5
HORIZON_MARGIN = 1.25

POLICIES = ["always_deploy", "scalar_horizon", "stage_aware_horizon", "random_defer"]

DEFAULTS = {"p_defect": P_DEFECT, "rollback_mean": ROLLBACK_MEAN,
            "rollback_cv": ROLLBACK_CV, "horizon_margin": HORIZON_MARGIN,
            "defect_factor": DEFECT_FACTOR, "random_rate": 0.50,
            "stages": STAGES}


def cfg(**o):
    c = dict(DEFAULTS); c.update(o); return c


def stable_seed(*parts):
    k = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(k, digest_size=4).digest(), "big")


ARRIVAL, DEPART, CHANGE_READY, MANIFEST, DETECT, ROLLBACK_DONE, RECOVERED = range(7)


@dataclass(order=True)
class Ev:
    time: float
    seq: int
    kind: int = field(compare=False)
    ref: int | None = field(compare=False, default=None)
    stage: int | None = field(compare=False, default=None)


@dataclass
class Ob:
    oid: int
    arrival: float
    deadline: float
    demand: list
    stage: int = 0                # next stage to be served by
    partner: int | None = None
    done: float | None = None
    penalty: float = 0.0

    def on_time(self):
        return self.done is not None and self.done <= self.deadline


@dataclass
class Ch:
    cid: int
    ready: float
    target: int
    admitted: bool = False
    defective: bool = False
    manifested: float | None = None
    detected: float | None = None
    unrollbackable: bool = False


class Sim:
    def __init__(self, policy, workload, seed, conf=None):
        self.cfg = conf or cfg()
        self.S = self.cfg["stages"]
        self.policy = policy
        self.workload = workload
        self.rng = random.Random(seed)
        self.t = 0.0; self.seq = 0; self.eq = []
        self.obs = {}; self.q = [[] for _ in range(self.S)]
        self.busy = [None] * self.S; self.ends = [None] * self.S
        self.mult = [1.0] * self.S
        self.changes = {}; self.live = None; self.rolling = False
        self.n_rb = 0

    def push(self, t, k, ref=None, stage=None):
        self.seq += 1
        heapq.heappush(self.eq, Ev(t, self.seq, k, ref, stage))

    def _gen(self):
        lo, hi = SLACK[self.workload]
        t = 0.0; oid = 0; pend = None
        while True:
            t += self.rng.expovariate(ARRIVAL_RATE)
            if t >= HORIZON: break
            o = Ob(oid, t, t + self.rng.uniform(lo, hi),
                   [self.rng.expovariate(1.0 / d) for d in STAGE_DEMAND[:self.S]])
            if pend is None and self.rng.random() < LINK_FRACTION:
                pend = oid
            elif pend is not None:
                o.partner = pend; self.obs[pend].partner = oid
                d = min(o.deadline, self.obs[pend].deadline)
                o.deadline = self.obs[pend].deadline = d; pend = None
            self.obs[oid] = o; self.push(t, ARRIVAL, oid); oid += 1
        cid = 0; ct = CHANGE_INTERVAL
        while ct < HORIZON:
            self.changes[cid] = Ch(cid, ct, self.rng.randrange(self.S))
            self.push(ct, CHANGE_READY, cid); cid += 1; ct += CHANGE_INTERVAL

    def _start(self, s):
        if self.busy[s] is not None or not self.q[s] or self.rolling: return
        oid = self.q[s].pop(0); self.busy[s] = oid; o = self.obs[oid]
        dur = o.demand[s] * self.mult[s] + o.penalty; o.penalty = 0.0
        self.ends[s] = self.t + dur
        self.push(self.ends[s], DEPART, oid, s)

    def _inflight(self):
        out = []
        for s in range(self.S):
            if self.busy[s] is not None: out.append((self.busy[s], s))
            out += [(x, s) for x in self.q[s]]
        return out

    def _cost(self):
        c = self.cfg
        sp = c["rollback_mean"] * c["rollback_cv"]
        return (MANIFEST_LAG[1] + DETECT_LAG + c["rollback_mean"] + 2.0 * sp
                + RECOVERY_TIME) * c["horizon_margin"]

    def _admit(self, ch):
        if self.policy == "always_deploy": return True
        if self.policy == "random_defer":
            return self.rng.random() < self.cfg["random_rate"]
        fl = self._inflight()
        if not fl: return True
        if self.policy == "scalar_horizon":
            # one system-wide cost, earliest deadline anywhere in the system
            b = min(self.obs[o].deadline for o, _ in fl)
            return self.t <= b - self._cost()
        # stage_aware_horizon: only obligations that must still cross the target
        exposed = [o for o, s in fl if s <= ch.target]
        if not exposed: return True
        b = min(self.obs[o].deadline for o in exposed)
        return self.t <= b - self._cost()

    def _rollback(self, ch):
        self.n_rb += 1
        dur = max(1.0, self.rng.gauss(self.cfg["rollback_mean"],
                                      self.cfg["rollback_mean"] * self.cfg["rollback_cv"]))
        exposed = [o for o, s in self._inflight() if s <= ch.target]
        if exposed:
            b = min(self.obs[o].deadline for o in exposed)
            if self.t + dur + RECOVERY_TIME > b:
                ch.unrollbackable = True
        self.rolling = True
        for s in range(ch.target + 1):
            if self.busy[s] is not None:
                self.q[s].insert(0, self.busy[s]); self.busy[s] = None; self.ends[s] = None
            for oid in self.q[s]:
                self.obs[oid].penalty += ROLLBACK_DISRUPT
        self.push(self.t + dur, ROLLBACK_DONE, ch.cid)

    def run(self):
        self._gen()
        while self.eq:
            e = heapq.heappop(self.eq); self.t = e.time
            if self.t > HORIZON * 3: break
            if e.kind == ARRIVAL:
                self.q[0].append(e.ref); self._start(0)
            elif e.kind == DEPART:
                s = e.stage
                if self.busy[s] != e.ref or abs(self.t - (self.ends[s] or -1)) > 1e-9: continue
                o = self.obs[e.ref]; self.busy[s] = None; self.ends[s] = None
                o.stage = s + 1
                if o.stage >= self.S: o.done = self.t
                else:
                    self.q[o.stage].append(o.oid); self._start(o.stage)
                self._start(s)
            elif e.kind == CHANGE_READY:
                ch = self.changes[e.ref]
                if self._admit(ch):
                    ch.admitted = True
                    if self.rng.random() < self.cfg["p_defect"]:
                        ch.defective = True
                        self.push(self.t + self.rng.uniform(*MANIFEST_LAG), MANIFEST, ch.cid)
            elif e.kind == MANIFEST:
                ch = self.changes[e.ref]
                if self.live is None:
                    ch.manifested = self.t; self.live = ch.cid
                    self.mult[ch.target] = self.cfg["defect_factor"]
                    self.push(self.t + DETECT_LAG, DETECT, ch.cid)
            elif e.kind == DETECT:
                ch = self.changes[e.ref]
                if self.live == ch.cid and not self.rolling:
                    ch.detected = self.t; self._rollback(ch)
            elif e.kind == ROLLBACK_DONE:
                ch = self.changes[e.ref]
                self.live = None; self.mult[ch.target] = 1.0
                self.push(self.t + RECOVERY_TIME, RECOVERED, ch.cid)
            elif e.kind == RECOVERED:
                self.rolling = False
                for s in range(self.S): self._start(s)
        return self._res()

    def _res(self):
        settled = {o.oid: o.on_time() for o in self.obs.values()}
        for o in self.obs.values():
            if o.partner is not None and not settled.get(o.partner, False):
                settled[o.oid] = False
        tot = len(self.obs) or 1
        missed = sum(1 for v in settled.values() if not v)
        adm = [c for c in self.changes.values() if c.admitted]
        unr = [c for c in adm if c.unrollbackable]
        return {"policy": self.policy, "workload": self.workload, "obligations": tot,
                "missed": missed, "miss_rate": missed / tot,
                "admitted": len(adm), "changes": len(self.changes),
                "throughput": len(adm) / max(len(self.changes), 1),
                "unrollbackable": len(unr), "rollbacks": self.n_rb}


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return float("nan"), float("nan")
    m = statistics.mean(xs)
    if len(xs) < 2: return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5


def run_exp(reps=3000, conf=None, workloads=None, out_dir=None):
    rows = []
    for wl in (workloads or ["tight"]):
        for i in range(reps):
            for pol in POLICIES:
                rows.append(Sim(pol, wl, stable_seed("ms", wl, i) & 0xFFFFFF, conf).run())
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "multistage_results.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    out = Path(__file__).parent / "results"
    REPS = 3000
    print(f"MULTI-STAGE PATH, {STAGES} stages, reps={REPS}")
    rows = run_exp(REPS, out_dir=out)
    g = lambda p, f: [r[f] for r in rows if r["policy"] == p]
    # volume-match the control to stage_aware
    rate = statistics.mean(g("stage_aware_horizon", "throughput"))
    rows2 = run_exp(REPS, conf=cfg(random_rate=rate))
    g2 = lambda p, f: [r[f] for r in rows2 if r["policy"] == p]
    print(f"\n{'policy':<22}{'miss %':>18}{'throughput':>12}{'unroll':>9}")
    print("-" * 62)
    for p in POLICIES:
        src = g2 if p == "random_defer" else g
        m, c = mean_ci(src(p, "miss_rate"))
        print(f"{p:<22}{100*m:>9.2f} +-{100*c:<5.2f}{100*statistics.mean(src(p,'throughput')):>10.1f}%"
              f"{statistics.mean(src(p,'unrollbackable')):>9.3f}")
    sa, sac = mean_ci(g("stage_aware_horizon", "miss_rate"))
    sc, scc = mean_ci(g("scalar_horizon", "miss_rate"))
    rd, rdc = mean_ci(g2("random_defer", "miss_rate"))
    print(f"\nH7  stage_aware vs scalar : {100*sc:.2f} -> {100*sa:.2f}  "
          f"gap {100*(sc-sa):+.2f} pp, threshold {100*(sac+scc):.2f} -> "
          f"{'SUPPORTED' if (sc-sa)>(sac+scc) else 'NOT SUPPORTED'}")
    print(f"    stage_aware vs control: {100*rd:.2f} -> {100*sa:.2f}  "
          f"gap {100*(rd-sa):+.2f} pp, threshold {100*(sac+rdc):.2f} -> "
          f"{'mechanism holds' if (rd-sa)>(sac+rdc) else 'MECHANISM NOT ESTABLISHED'}")
    print(f"    throughput stage_aware {100*statistics.mean(g('stage_aware_horizon','throughput')):.1f}% "
          f"vs control {100*statistics.mean(g2('random_defer','throughput')):.1f}%")
