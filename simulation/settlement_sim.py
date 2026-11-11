#!/usr/bin/env python3
"""
Discrete-event simulation of a settlement-processing queue under injected
latency degradation, comparing three operational decision policies.

Purpose
-------
Tests two hypotheses from the accompanying paper:

  H1  A monitoring system incorporating transaction state and timing flags a
      different set of operationally significant events than an otherwise
      equivalent service-centric system.

  H5  The benefit of transaction- and timing-aware evidence authorization is
      greater for workloads containing settlement deadlines than for workloads
      in which service availability is a sufficient proxy for operational
      success.

H5 is falsifiable by construction: the LOOSE workload has deadlines so distant
that they are never binding, so a transaction-aware policy has no information
advantage there. If the proposed policy still appeared to win under LOOSE, that
would indicate the model is rigged rather than that the hypothesis holds.

Model
-----
Single-server FIFO settlement service. Obligations arrive Poisson(lambda) with
exponential base service demand. Each obligation carries a settlement deadline.
A fraction are LINKED pairs: under the exchange-of-value principle, if either
side misses its deadline, both are recorded as failed.

A degradation episode multiplies service time by DEGRADE_FACTOR for a bounded
interval. The degradation has a hidden cause:
    internal  - an intervention (restart / rollback) clears it
    external  - an intervention cannot clear it; the episode runs its course
Interventions are not free. Executing one disrupts work in progress: the
in-service obligation restarts, and queued obligations lose DISRUPT_PENALTY
minutes. An intervention aimed at an external cause therefore pays the cost
and gains nothing.

Policies observe the system every TICK minutes and return one of
    'none' | 'act' | 'escalate'
'escalate' routes to a human and takes effect after HUMAN_DELAY minutes.

Policies
--------
service_centric   Rolling p95 completion latency against an SLO threshold,
                  requiring SUSTAIN consecutive breaching ticks. No transaction
                  state. Acts on breach.

confidence_only   Same detector, acts on the first breaching tick. Represents
                  automation triggered by model confidence with no evidence
                  test.

transaction_aware Projects completion time for every in-flight and queued
                  obligation at the currently observed service rate and counts
                  those projected to miss their deadline. Flags when any
                  obligation is projected to miss. Then applies an evidence
                  test before acting: it acts automatically only when
                  transaction state is instrumented AND no obligation already
                  inside FINALITY_GUARD minutes of its deadline would be
                  disrupted by the action. Otherwise it escalates.

Stdlib only. No external data. All parameters are stated assumptions.
"""

from __future__ import annotations

import csv
import heapq
import hashlib
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path


def stable_seed(*parts) -> int:
    """Deterministic seed from arbitrary parts.

    The built-in hash() is randomized per process for str inputs unless
    PYTHONHASHSEED is set, which made earlier revisions of this script
    irreproducible between runs. Results were statistically consistent but not
    identical. A digest keeps the paired-seed design and makes every reported
    figure reproducible without environment configuration.
    """
    key = "|".join(repr(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")

# --------------------------------------------------------------------------
# Parameters (assumptions, not measurements)
# --------------------------------------------------------------------------

HORIZON = 480.0          # simulated minutes (one processing session)
ARRIVAL_RATE = 0.50       # obligations per minute (rho = 0.45 at baseline)
BASE_DEMAND = 0.9         # mean service minutes per obligation
LINK_FRACTION = 0.30      # fraction of obligations in linked pairs

DEGRADE_FACTOR = 7.0      # service-time multiplier during episode
DEGRADE_DURATION = 40.0   # minutes, if never cleared
P_INTERNAL_CAUSE = 0.5    # probability the cause is clearable by intervention

REPAIR_TIME = 3.0         # minutes for an intervention to take effect
HUMAN_DELAY = 12.0        # additional minutes when escalated
DISRUPT_PENALTY = 2.5     # minutes added to queued work by an intervention

TICK = 1.0                # monitoring interval
WINDOW = 25               # completions in the rolling latency window
SLO_LATENCY = 6.0         # p95 latency threshold (minutes)
SUSTAIN = 5               # consecutive breaching ticks for service_centric
FINALITY_GUARD = 6.0      # minutes before deadline treated as finality-critical
INSTRUMENTATION = 0.85    # probability transaction state is fully available

# H4: verification. An intervention issues a command. The command can succeed
# mechanically while the operational condition persists, which is the gap the
# hypothesis concerns.
P_CMD_SUCCESS = 0.95      # probability the remediation command itself returns success
VERIFY_WINDOW = 10.0      # minutes after action at which dependency state is sampled

# Slack = deadline - arrival. TIGHT workload has binding deadlines; LOOSE does
# not, and serves as the null condition for H5.
SLACK = {
    "tight": (18.0, 55.0),
    "loose": (600.0, 900.0),
}

# Degradation onset placed this many minutes before the deadline of the
# obligations then in the system. Four experimental distances.
BOUNDARY_OFFSETS = [90.0, 45.0, 20.0, 8.0]

POLICIES = ["no_action", "service_centric", "confidence_only", "transaction_aware"]
INTERVENING = [p for p in POLICIES if p != "no_action"]

# H2 uses a separate policy list so that the H1 and H5 figures reported in the
# paper are unaffected by the addition of a fourth arm. evidence_gated shares
# the confidence_only detector exactly; the evidence test is the only
# difference between them, which is what makes the pair an ablation.
H2_POLICIES = ["no_action", "confidence_only", "evidence_gated"]

# Parameters varied in the sensitivity analysis (Section: sweep). Defaults
# reproduce the main experiment.
DEFAULTS = {
    "degrade_factor": DEGRADE_FACTOR,
    "degrade_duration": DEGRADE_DURATION,
    "p_internal": P_INTERNAL_CAUSE,
    "disrupt_penalty": DISRUPT_PENALTY,
    "human_delay": HUMAN_DELAY,
    "instrumentation": INSTRUMENTATION,
    # The main experiment assumes remediation commands always execute
    # successfully, so that the H1 and H5 figures isolate policy behaviour from
    # execution reliability. The H4 experiment relaxes this to P_CMD_SUCCESS.
    "p_cmd_success": 1.0,
}


def cfg(**over):
    c = dict(DEFAULTS)
    c.update(over)
    return c

ARRIVAL, COMPLETE, TICK_EV, DEG_START, DEG_END, ACTION, VERIFY = range(7)


@dataclass(order=True)
class Event:
    time: float
    seq: int
    kind: int = field(compare=False)
    oid: int | None = field(compare=False, default=None)


@dataclass
class Obligation:
    oid: int
    arrival: float
    demand: float
    deadline: float
    partner: int | None = None
    completed: float | None = None
    penalty: float = 0.0

    def settled_on_time(self) -> bool:
        return self.completed is not None and self.completed <= self.deadline


class Sim:
    def __init__(self, policy: str, workload: str, offset: float, seed: int,
                 conf: dict | None = None):
        self.cfg = conf or cfg()
        self.policy = policy
        self.workload = workload
        self.offset = offset
        self.rng = random.Random(seed)

        self.t = 0.0
        self.seq = 0
        self.eq: list[Event] = []

        self.obs: dict[int, Obligation] = {}
        self.queue: list[int] = []
        self.in_service: int | None = None
        self.service_ends: float | None = None

        self.multiplier = 1.0
        self.degrading = False
        self.cause: str | None = None
        self.degrade_start: float | None = None
        self.cleared = False

        self.latencies: list[float] = []
        self.breach_streak = 0
        self.acted = False
        self.pending_action = False

        # instrumentation is a property of the deployment, drawn once
        self.instrumented = self.rng.random() < self.cfg["instrumentation"]

        # metrics
        self.flagged: set[int] = set()   # obligations flagged as at-risk
        self.n_interventions = 0
        self.n_wasted = 0                # acted but cause was external
        self.n_disrupted = 0             # obligations paying disruption cost
        self.flag_time: float | None = None
        self.escalated = False

        # H2: was the authorizing decision automatic, and was transaction-state
        # evidence actually available to it at that moment?
        self.auto_acted = False
        self.acted_without_evidence = False
        # which evidence test blocked automatic authorization, if any
        self.block_instr = False
        self.block_corrob = False
        self.block_finality = False

        # H4: verification. Recorded at authorization and sampled afterwards.
        self.cmd_ok: bool | None = None          # command-level outcome
        self.dep_cleared: bool | None = None     # dependency-level, at +VERIFY_WINDOW
        self.at_risk_at_action: frozenset[int] = frozenset()
        self.action_time: float | None = None

    # ---------------- event helpers ----------------

    def push(self, time: float, kind: int, oid: int | None = None) -> None:
        self.seq += 1
        heapq.heappush(self.eq, Event(time, self.seq, kind, oid))

    # ---------------- setup ----------------

    def _generate_arrivals(self) -> None:
        lo, hi = SLACK[self.workload]
        t = 0.0
        oid = 0
        pending_link: int | None = None
        while t < HORIZON:
            t += self.rng.expovariate(ARRIVAL_RATE)
            if t >= HORIZON:
                break
            slack = self.rng.uniform(lo, hi)
            o = Obligation(
                oid=oid,
                arrival=t,
                demand=self.rng.expovariate(1.0 / BASE_DEMAND),
                deadline=t + slack,
            )
            # pair a fraction of obligations into linked couples
            if pending_link is None and self.rng.random() < LINK_FRACTION:
                pending_link = oid
            elif pending_link is not None:
                o.partner = pending_link
                self.obs[pending_link].partner = oid
                # linked obligations share the tighter deadline
                d = min(o.deadline, self.obs[pending_link].deadline)
                o.deadline = self.obs[pending_link].deadline = d
                pending_link = None
            self.obs[oid] = o
            self.push(t, ARRIVAL, oid)
            oid += 1

    def _schedule_degradation(self) -> None:
        """Place onset so that obligations in the system are `offset` minutes
        from their deadlines when degradation begins."""
        mid = HORIZON * 0.5
        candidates = [o for o in self.obs.values() if o.arrival <= mid]
        if candidates:
            med_deadline = statistics.median(o.deadline for o in candidates)
            onset = med_deadline - self.offset
        else:
            onset = mid
        onset = min(max(onset, 30.0), HORIZON - self.cfg["degrade_duration"] - 10.0)
        self.degrade_start = onset
        self.cause = "internal" if self.rng.random() < self.cfg["p_internal"] else "external"
        self.push(onset, DEG_START)
        self.push(onset + self.cfg["degrade_duration"], DEG_END)

    # ---------------- service ----------------

    def _start_service(self) -> None:
        if self.in_service is not None or not self.queue:
            return
        oid = self.queue.pop(0)
        self.in_service = oid
        o = self.obs[oid]
        dur = o.demand * self.multiplier + o.penalty
        o.penalty = 0.0
        self.service_ends = self.t + dur
        self.push(self.service_ends, COMPLETE, oid)

    def _apply_intervention(self) -> None:
        self.n_interventions += 1
        self.action_time = self.t
        # The command either executes or it does not. Execution failure is
        # visible to command-level verification; everything below it is not.
        self.cmd_ok = self.rng.random() < self.cfg["p_cmd_success"]
        if not self.cmd_ok or self.cause == "external" or not self.degrading:
            self.n_wasted += 1
        else:
            self.cleared = True
            self.degrading = False
            self.multiplier = 1.0
        self.push(self.t + VERIFY_WINDOW, VERIFY)
        # disruption: in-service obligation restarts, queue pays a penalty
        if self.in_service is not None:
            o = self.obs[self.in_service]
            self.queue.insert(0, o.oid)
            self.n_disrupted += 1
            self.in_service = None
            self.service_ends = None
        for oid in self.queue:
            self.obs[oid].penalty += self.cfg["disrupt_penalty"]
            self.n_disrupted += 1
        self._start_service()

    # ---------------- monitoring ----------------

    def _p95(self) -> float:
        w = self.latencies[-WINDOW:]
        if len(w) < 5:
            return 0.0
        s = sorted(w)
        return s[min(len(s) - 1, int(0.95 * len(s)))]

    def _observed_rate(self) -> float:
        """Service-time inflation, as currently observable.

        Uses completed-work latency plus, if a request is currently in service
        and already running long, its elapsed time. The in-flight term is
        observable in practice and keeps the estimate from lagging a step
        change by a full window. It confers no knowledge of the cause, the
        multiplier, or the episode duration.
        """
        w = self.latencies[-WINDOW:]
        hist = max(1.0, statistics.mean(w) / max(BASE_DEMAND, 1e-9)) if len(w) >= 5 else 1.0
        live = 1.0
        if self.in_service is not None:
            elapsed = self.t - (self.service_ends - self.obs[self.in_service].demand
                                * self.multiplier) if self.service_ends else 0.0
            if elapsed > 0:
                live = max(1.0, elapsed / max(self.obs[self.in_service].demand, 1e-9))
        return max(hist, live)

    def _projected_misses(self) -> set[int]:
        """Obligations projected to miss their deadline at the observed rate."""
        rate = self._observed_rate()
        at_risk: set[int] = set()
        clock = self.t
        order = ([self.in_service] if self.in_service is not None else []) + self.queue
        for oid in order:
            o = self.obs[oid]
            clock += o.demand * rate
            if clock > o.deadline:
                at_risk.add(oid)
        return at_risk

    def _would_disrupt_finality_critical(self) -> bool:
        order = ([self.in_service] if self.in_service is not None else []) + self.queue
        return any(self.obs[oid].deadline - self.t <= FINALITY_GUARD for oid in order)

    def _monitor(self) -> None:
        if self.policy == "no_action" or self.acted or self.pending_action:
            return

        breach = self._p95() > SLO_LATENCY

        if self.policy == "confidence_only":
            if breach:
                self._flag(self._projected_misses() or set(self.queue))
                self._commit("act")
            return

        if self.policy == "evidence_gated":
            # Identical detector and identical trigger instant as
            # confidence_only. The only difference is that authorization
            # requires the financial-process evidence to be adequate:
            #   (a) transaction state must actually be instrumented,
            #   (b) an independent evidence source must corroborate the
            #       detector by projecting at least one obligation to miss,
            #   (c) the action must not disrupt an obligation already inside
            #       the finality guard.
            # Failing any of these, the decision escalates rather than acting.
            if not breach:
                return
            at_risk = self._projected_misses()
            self._flag(at_risk or set(self.queue))
            no_instr = not self.instrumented
            no_corrob = not at_risk
            finality = self._would_disrupt_finality_critical()
            if not (no_instr or no_corrob or finality):
                self._commit("act")
            else:
                # Record which test blocked authorization, so that the gate's
                # conservatism can be attributed rather than assumed.
                self.block_instr = no_instr
                self.block_corrob = no_corrob
                self.block_finality = finality
                self.escalated = True
                self._commit("escalate")
            return

        if self.policy == "service_centric":
            self.breach_streak = self.breach_streak + 1 if breach else 0
            if self.breach_streak >= SUSTAIN:
                self._flag(set(self.queue))
                self._commit("act")
            return

        # transaction_aware
        at_risk = self._projected_misses()
        if not at_risk:
            return
        self._flag(at_risk)
        if self.instrumented and not self._would_disrupt_finality_critical():
            self._commit("act")
        else:
            self.escalated = True
            self._commit("escalate")

    def _flag(self, oids: set[int]) -> None:
        if self.flag_time is None and oids:
            self.flag_time = self.t
        self.flagged |= oids

    def _commit(self, decision: str) -> None:
        delay = REPAIR_TIME + (self.cfg["human_delay"] if decision == "escalate" else 0.0)
        self.pending_action = True
        # Recorded for H2 and H4: whether the action was authorized
        # automatically, whether transaction-state evidence was available to
        # the authorizing decision, and which obligations that decision
        # believed were at risk.
        if decision == "act":
            self.auto_acted = True
            has_evidence = self.instrumented and self.policy in (
                "transaction_aware", "evidence_gated")
            self.acted_without_evidence = not has_evidence
        self.at_risk_at_action = frozenset(self._projected_misses())
        self.push(self.t + delay, ACTION)

    # ---------------- main loop ----------------

    def run(self) -> dict:
        self._generate_arrivals()
        self._schedule_degradation()
        for k in range(1, int(HORIZON / TICK) + 1):
            self.push(k * TICK, TICK_EV)

        while self.eq:
            ev = heapq.heappop(self.eq)
            self.t = ev.time
            if self.t > HORIZON * 3:
                break

            if ev.kind == ARRIVAL:
                self.queue.append(ev.oid)
                self._start_service()

            elif ev.kind == COMPLETE:
                # stale completion after a disruption
                if self.in_service != ev.oid or abs(self.t - (self.service_ends or -1)) > 1e-9:
                    continue
                o = self.obs[ev.oid]
                o.completed = self.t
                self.latencies.append(self.t - o.arrival)
                self.in_service = None
                self.service_ends = None
                self._start_service()

            elif ev.kind == DEG_START:
                if not self.cleared:
                    self.degrading = True
                    self.multiplier = self.cfg["degrade_factor"]

            elif ev.kind == DEG_END:
                self.degrading = False
                self.multiplier = 1.0

            elif ev.kind == ACTION:
                self.pending_action = False
                self.acted = True
                self._apply_intervention()

            elif ev.kind == VERIFY:
                # Dependency-level verification: is the degradation actually
                # gone a short window after the action reported success?
                if self.dep_cleared is None:
                    self.dep_cleared = not self.degrading

            elif ev.kind == TICK_EV:
                self._monitor()

        return self._results()

    # ---------------- results ----------------

    def _results(self) -> dict:
        settled = {o.oid: o.settled_on_time() for o in self.obs.values()}
        # exchange-of-value: a linked pair fails together
        for o in self.obs.values():
            if o.partner is not None and not settled.get(o.partner, False):
                settled[o.oid] = False
        total = len(self.obs)
        missed = sum(1 for v in settled.values() if not v)
        lead = (self.flag_time - self.degrade_start
                if self.flag_time is not None and self.degrade_start is not None
                else None)

        # H4 verification outcomes, defined only where an action was taken.
        #   command-level    : the command returned success
        #   dependency-level : the degradation was gone at +VERIFY_WINDOW
        #   transaction-level: every obligation the authorizing decision
        #                      believed to be at risk did in fact settle on time
        txn_ok = None
        if self.n_interventions:
            watched = self.at_risk_at_action
            txn_ok = all(settled.get(o, False) for o in watched) if watched else True

        return {
            "policy": self.policy,
            "workload": self.workload,
            "offset": self.offset,
            "total": total,
            "missed": missed,
            "miss_rate": missed / total if total else 0.0,
            "interventions": self.n_interventions,
            "wasted": self.n_wasted,
            "disrupted": self.n_disrupted,
            "escalated": int(self.escalated),
            "flag_lead": lead,
            "auto_acted": int(self.auto_acted),
            "acted_without_evidence": int(self.acted_without_evidence),
            "block_instr": int(self.block_instr),
            "block_corrob": int(self.block_corrob),
            "block_finality": int(self.block_finality),
            "cmd_ok": None if self.cmd_ok is None else int(self.cmd_ok),
            "dep_cleared": None if self.dep_cleared is None else int(self.dep_cleared),
            "txn_ok": None if txn_ok is None else int(txn_ok),
            "n_watched": len(self.at_risk_at_action),
            "cause": self.cause,
            "flagged": frozenset(self.flagged),
        }


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% normal-approximation interval."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5


def run_experiment(reps: int = 400, out_dir: Path | None = None,
                   conf: dict | None = None, workloads=None,
                   policies=None, out_name: str = "raw_results.csv") -> dict:
    rows: list[dict] = []
    jaccard: dict[tuple[str, float], list[float]] = {}
    pols = policies or POLICIES
    paired: list[dict] = []

    for workload in (workloads or list(SLACK)):
        for offset in BOUNDARY_OFFSETS:
            for seed in range(reps):
                per_policy = {}
                for pol in pols:
                    # identical seed => identical arrivals and cause across policies
                    r = Sim(pol, workload, offset,
                            seed=stable_seed(workload, offset, seed) & 0xFFFFFF,
                            conf=conf).run()
                    per_policy[pol] = r
                    rows.append({k: v for k, v in r.items() if k != "flagged"})
                if "service_centric" in pols and "transaction_aware" in pols:
                    a = per_policy["service_centric"]["flagged"]
                    b = per_policy["transaction_aware"]["flagged"]
                    if a or b:
                        j = len(a & b) / len(a | b)
                        jaccard.setdefault((workload, offset), []).append(j)
                # Retain the matched set so that an intervention can be judged
                # against what the same arrival stream and the same hidden
                # cause produced with no intervention at all.
                if "no_action" in pols:
                    paired.append({
                        "workload": workload, "offset": offset, "seed": seed,
                        "baseline": per_policy["no_action"]["missed"],
                        **{p: per_policy[p] for p in pols if p != "no_action"},
                    })

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / out_name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return {"rows": rows, "jaccard": jaccard, "reps": reps, "paired": paired}


def summarize(exp: dict) -> None:
    rows, jac, reps = exp["rows"], exp["jaccard"], exp["reps"]

    def sel(workload, offset, pol, field):
        return [r[field] for r in rows
                if r["workload"] == workload and r["offset"] == offset and r["policy"] == pol]

    print(f"\n{'='*104}")
    print(f"SETTLEMENT-QUEUE SIMULATION   reps={reps} per cell   horizon={HORIZON:.0f} min"
          f"   degrade x{DEGRADE_FACTOR:.0f} for {DEGRADE_DURATION:.0f} min")
    print(f"{'='*104}")

    for workload in SLACK:
        tag = ("TIGHT - binding settlement deadlines" if workload == "tight"
               else "LOOSE - deadlines never binding (null condition for H5)")
        print(f"\n### {workload.upper()} workload: {tag}")
        print(f"{'onset before':>13} | {'policy':<18} | {'miss rate %':>18} | "
              f"{'interv.':>8} | {'wasted':>7} | {'disrupted':>9} | {'flag lead min':>13}")
        print(f"{'boundary (min)':>13} | {'':<18} | {'mean [95% CI]':>18} | "
              f"{'':>8} | {'':>7} | {'':>9} | {'':>13}")
        print("-" * 104)
        for offset in BOUNDARY_OFFSETS:
            for pol in POLICIES:
                mr, ci = mean_ci(sel(workload, offset, pol, "miss_rate"))
                iv, _ = mean_ci(sel(workload, offset, pol, "interventions"))
                wa, _ = mean_ci(sel(workload, offset, pol, "wasted"))
                ds, _ = mean_ci(sel(workload, offset, pol, "disrupted"))
                fl, _ = mean_ci(sel(workload, offset, pol, "flag_lead"))
                print(f"{offset:>13.0f} | {pol:<18} | {100*mr:>9.2f} [{100*(mr-ci):>4.2f},{100*(mr+ci):>5.2f}] | "
                      f"{iv:>8.2f} | {wa:>7.2f} | {ds:>9.1f} | {fl:>13.1f}")
            print("-" * 104)

    print("\n### H1  overlap of flagged obligation sets: service_centric vs transaction_aware")
    print(f"{'workload':>10} | {'onset (min)':>11} | {'Jaccard mean [95% CI]':>28}")
    print("-" * 56)
    for workload in SLACK:
        for offset in BOUNDARY_OFFSETS:
            m, ci = mean_ci(jac.get((workload, offset), []))
            print(f"{workload:>10} | {offset:>11.0f} | {m:>14.3f} [{m-ci:.3f}, {m+ci:.3f}]")

    print("\n### H5  transaction_aware vs best intervening alternative (no_action shown for reference)")
    print(f"{'workload':>10} | {'onset':>6} | {'TA %':>7} | {'best interv. alt':>19} | "
          f"{'reduction (pp)':>15} | {'sig':>4} | {'no_action %':>11}")
    print("-" * 92)
    for workload in SLACK:
        for offset in BOUNDARY_OFFSETS:
            ta, ta_ci = mean_ci(sel(workload, offset, "transaction_aware", "miss_rate"))
            alts = {p: mean_ci(sel(workload, offset, p, "miss_rate"))
                    for p in INTERVENING if p != "transaction_aware"}
            bp = min(alts, key=lambda p: alts[p][0])
            bm, bm_ci = alts[bp]
            diff = bm - ta
            sig = abs(diff) > (ta_ci + bm_ci)
            na, _ = mean_ci(sel(workload, offset, "no_action", "miss_rate"))
            print(f"{workload:>10} | {offset:>6.0f} | {100*ta:>7.2f} | {bp:>19} | "
                  f"{100*diff:>15.2f} | {'yes' if sig else 'no':>4} | {100*na:>11.2f}")
    print()




# --------------------------------------------------------------------------
# Sensitivity analysis
# --------------------------------------------------------------------------

def sweep(reps: int = 250) -> None:
    """Vary the parameters that govern whether intervening pays at all.

    The main experiment uses a 40-minute episode that is clearable only half
    the time, with disruption charged on every queued obligation. Under those
    assumptions no intervention dominates, so the interesting question is
    where that ordering reverses and whether transaction-aware authorization
    remains the best intervening policy once it does.
    """
    grid = [
        (dur, pi, dp)
        for dur in (40.0, 90.0, 150.0)
        for pi in (0.5, 0.8)
        for dp in (2.5, 0.8)
    ]
    print(f"\n{'='*112}")
    print(f"SENSITIVITY: does intervening pay, and is transaction_aware still best when it does?"
          f"   reps={reps}/cell, TIGHT workload, onset 20 min")
    print(f"{'='*112}")
    print(f"{'episode':>8} | {'p(int)':>6} | {'disrupt':>7} | {'no_action':>9} | {'svc-centric':>11} | "
          f"{'conf-only':>9} | {'txn-aware':>9} | {'best':>13} | {'TA best of':>11}")
    print(f"{'(min)':>8} | {'':>6} | {'(min)':>7} | {'miss %':>9} | {'miss %':>11} | "
          f"{'miss %':>9} | {'miss %':>9} | {'overall':>13} | {'intervening':>11}")
    print("-" * 112)
    for dur, pi, dp in grid:
        c = cfg(degrade_duration=dur, p_internal=pi, disrupt_penalty=dp)
        exp = run_experiment(reps=reps, conf=c, workloads=["tight"])
        rows = [r for r in exp["rows"] if r["offset"] == 20.0]
        m = {}
        for pol in POLICIES:
            vals = [r["miss_rate"] for r in rows if r["policy"] == pol]
            m[pol] = mean_ci(vals)
        best = min(POLICIES, key=lambda p: m[p][0])
        best_int = min(INTERVENING, key=lambda p: m[p][0])
        print(f"{dur:>8.0f} | {pi:>6.1f} | {dp:>7.1f} | "
              f"{100*m['no_action'][0]:>9.2f} | {100*m['service_centric'][0]:>11.2f} | "
              f"{100*m['confidence_only'][0]:>9.2f} | {100*m['transaction_aware'][0]:>9.2f} | "
              f"{best:>13} | {'yes' if best_int=='transaction_aware' else 'NO':>11}")
    print()

# --------------------------------------------------------------------------
# H2: evidence authorization
# --------------------------------------------------------------------------

def h2_experiment(reps: int = 400, out_dir: Path | None = None) -> None:
    """Does gating authorization on evidence adequacy reduce interventions that
    were not warranted, and what does the gate cost?

    confidence_only and evidence_gated share a detector, a threshold and a
    trigger instant. The evidence test is the only difference. Instrumentation
    completeness is swept because H2 is conditioned on evidence being
    incomplete: if the gate has any value it should grow as evidence degrades.

    An automated intervention is counted as UNWARRANTED when the matched
    replication - same arrivals, same hidden cause - finished with no more
    missed obligations under no_action. That is a counterfactual test against
    ground truth rather than a restatement of the policy's own reasoning.
    """
    levels = [0.25, 0.50, 0.85, 1.00]
    print(f"\n{'='*118}")
    print("H2  EVIDENCE AUTHORIZATION: confidence_only vs evidence_gated "
          f"(identical detector)   reps={reps}/cell, TIGHT, onset 20 min")
    print(f"{'='*118}")
    print("NOTE: at the default 40-minute episode no intervening policy beats "
          "no_action (see sweep). The\n      episode length is therefore also "
          "varied below, so that the gate is judged in a regime where\n"
          "      intervening pays at all.")
    print(f"{'instrumented':>12} | {'policy':<16} | {'auto-acted':>10} | {'unwarranted':>12} | "
          f"{'unwarranted':>12} | {'escalated':>9} | {'miss rate %':>18}")
    print(f"{'fraction':>12} | {'':<16} | {'% of reps':>10} | {'% of reps':>12} | "
          f"{'% of actions':>12} | {'% of reps':>9} | {'mean [95% CI]':>18}")
    print("-" * 118)

    summary = {}
    for lv in levels:
        exp = run_experiment(reps=reps, conf=cfg(instrumentation=lv),
                             workloads=["tight"], policies=H2_POLICIES,
                             out_dir=out_dir if lv == 0.85 else None,
                             out_name="h2_results.csv")
        pr = [p for p in exp["paired"] if p["offset"] == 20.0]
        for pol in ("confidence_only", "evidence_gated"):
            auto = [p for p in pr if p[pol]["auto_acted"]]
            unwar = [p for p in auto if p[pol]["missed"] >= p["baseline"]]
            esc = [p for p in pr if p[pol]["escalated"]]
            mr, ci = mean_ci([p[pol]["miss_rate"] for p in pr])
            n = len(pr)
            summary[(lv, pol)] = (100*len(auto)/n, 100*len(unwar)/n,
                                  100*len(unwar)/max(len(auto), 1), 100*len(esc)/n, mr, ci)
            a, u, ua, e, m, c = summary[(lv, pol)]
            print(f"{lv:>12.2f} | {pol:<16} | {a:>10.1f} | {u:>12.1f} | {ua:>12.1f} | "
                  f"{e:>9.1f} | {100*m:>9.2f} [{100*(m-c):>4.2f},{100*(m+c):>5.2f}]")
        print("-" * 118)

    print("\n### H2 verdict input: reduction in unwarranted automated interventions, "
          "and the miss-rate cost of the gate")
    print(f"{'instrumented':>12} | {'unwarranted (% of reps)':>26} | {'relative':>9} | "
          f"{'miss rate % (CO -> EG)':>24} | {'gate helps':>10}")
    print("-" * 94)
    for lv in levels:
        co = summary[(lv, "confidence_only")]
        eg = summary[(lv, "evidence_gated")]
        rel = (co[1] - eg[1]) / co[1] * 100 if co[1] else float("nan")
        better = eg[4] < co[4] - (eg[5] + co[5])
        worse = eg[4] > co[4] + (eg[5] + co[5])
        verdict = "yes" if better else ("costs" if worse else "neutral")
        print(f"{lv:>12.2f} | {co[1]:>11.1f} -> {eg[1]:>11.1f} | {rel:>8.1f}% | "
              f"{100*co[4]:>10.2f} -> {100*eg[4]:>10.2f} | {verdict:>10}")

    # Which of the three evidence tests actually makes the gate conservative?
    print("\n### Attribution: which evidence test blocked automatic authorization")
    print(f"{'instrumented':>12} | {'escalations':>11} | {'blocked by':>11} | "
          f"{'blocked by':>11} | {'blocked by':>11}")
    print(f"{'fraction':>12} | {'% of reps':>11} | {'instr. %':>11} | "
          f"{'no corrob %':>11} | {'finality %':>11}")
    print("-" * 66)
    for lv in levels:
        exp = run_experiment(reps=reps, conf=cfg(instrumentation=lv),
                             workloads=["tight"], policies=H2_POLICIES)
        rs = [r for r in exp["rows"] if r["policy"] == "evidence_gated"
              and r["offset"] == 20.0]
        esc = [r for r in rs if r["escalated"]]
        n = max(len(esc), 1)
        print(f"{lv:>12.2f} | {100*len(esc)/len(rs):>11.1f} | "
              f"{100*sum(r['block_instr'] for r in esc)/n:>11.1f} | "
              f"{100*sum(r['block_corrob'] for r in esc)/n:>11.1f} | "
              f"{100*sum(r['block_finality'] for r in esc)/n:>11.1f}")

    # The regime question: judge the gate where intervening is worthwhile.
    print("\n### The gate judged in a regime where intervening pays "
          "(90-minute episode, instrumentation 0.85)")
    print(f"{'episode':>8} | {'policy':<16} | {'miss rate %':>18} | "
          f"{'unwarranted':>12} | {'vs no_action':>12}")
    print("-" * 78)
    for dur in (40.0, 90.0, 150.0):
        exp = run_experiment(reps=reps, conf=cfg(degrade_duration=dur),
                             workloads=["tight"], policies=H2_POLICIES)
        pr = [p for p in exp["paired"] if p["offset"] == 20.0]
        na, _ = mean_ci([p["baseline"] / max(p["confidence_only"]["total"], 1) for p in pr])
        print(f"{dur:>8.0f} | {'no_action':<16} | {100*na:>9.2f} {'':>8} | "
              f"{'-':>12} | {'-':>12}")
        for pol in ("confidence_only", "evidence_gated"):
            mr, ci = mean_ci([p[pol]["miss_rate"] for p in pr])
            auto = [p for p in pr if p[pol]["auto_acted"]]
            unwar = [p for p in auto if p[pol]["missed"] >= p["baseline"]]
            print(f"{'':>8} | {pol:<16} | {100*mr:>9.2f} [{100*(mr-ci):>4.2f},{100*(mr+ci):>5.2f}] | "
                  f"{100*len(unwar)/len(pr):>11.1f}% | {100*(mr-na):>+11.2f}pp")
        print("-" * 78)
    print()


# --------------------------------------------------------------------------
# H4: verification
# --------------------------------------------------------------------------

def h4_experiment(reps: int = 400, out_dir: Path | None = None) -> None:
    """Is there a class of recoveries that command-level verification records as
    successful and transaction-level verification does not?

    Three verification levels are applied to the same interventions:
        command      the command returned success
        dependency   the degradation was gone VERIFY_WINDOW minutes later
        transaction  every obligation the authorizing decision believed to be
                     at risk did settle on time

    The class of interest is command-success with transaction-failure. It is
    then split by whether dependency-level verification would also have caught
    it, because only the remainder is evidence that transaction-level
    verification adds something a dependency check does not.
    """
    print(f"\n{'='*116}")
    print(f"H4  VERIFICATION LEVELS   reps={reps}/cell, TIGHT workload, "
          f"P(command succeeds)={P_CMD_SUCCESS:.2f}, verify window {VERIFY_WINDOW:.0f} min")
    print(f"{'='*116}")
    exp = run_experiment(reps=reps, conf=cfg(p_cmd_success=P_CMD_SUCCESS),
                         workloads=["tight"], out_dir=out_dir,
                         out_name="h4_results.csv")
    rows = [r for r in exp["rows"]
            if r["policy"] in INTERVENING and r["interventions"] and r["cmd_ok"] is not None]

    print(f"{'policy':<18} | {'onset':>5} | {'n':>5} | {'cmd ok':>7} | {'dep ok':>7} | "
          f"{'txn ok':>7} | {'cmd ok but':>11} | {'of those,':>11}")
    print(f"{'':<18} | {'(min)':>5} | {'':>5} | {'%':>7} | {'%':>7} | {'%':>7} | "
          f"{'txn failed %':>11} | {'dep ok too %':>11}")
    print("-" * 116)
    totals = {}
    for pol in INTERVENING:
        for off in BOUNDARY_OFFSETS:
            rs = [r for r in rows if r["policy"] == pol and r["offset"] == off]
            if not rs:
                continue
            n = len(rs)
            cmd = sum(r["cmd_ok"] for r in rs)
            dep = sum(r["dep_cleared"] or 0 for r in rs)
            txn = sum(r["txn_ok"] for r in rs)
            gap = [r for r in rs if r["cmd_ok"] and not r["txn_ok"]]
            hidden = [r for r in gap if r["dep_cleared"]]
            print(f"{pol:<18} | {off:>5.0f} | {n:>5} | {100*cmd/n:>7.1f} | {100*dep/n:>7.1f} | "
                  f"{100*txn/n:>7.1f} | {100*len(gap)/n:>11.1f} | "
                  f"{100*len(hidden)/max(len(gap),1):>11.1f}")
        print("-" * 116)

    print("\n### H4 verdict input: pooled across onsets")
    print(f"{'policy':<18} | {'n':>5} | {'cmd-level says success':>22} | "
          f"{'and txn-level says failure':>26} | {'invisible to dep-level':>22}")
    print("-" * 104)
    for pol in INTERVENING:
        rs = [r for r in rows if r["policy"] == pol]
        if not rs:
            continue
        ok = [r for r in rs if r["cmd_ok"]]
        gap = [r for r in ok if not r["txn_ok"]]
        hidden = [r for r in gap if r["dep_cleared"]]
        totals[pol] = (len(rs), len(ok), len(gap), len(hidden))
        print(f"{pol:<18} | {len(rs):>5} | {100*len(ok)/len(rs):>21.1f}% | "
              f"{100*len(gap)/len(rs):>25.1f}% | {100*len(hidden)/len(rs):>21.1f}%")
    print("\nThe final column is the part of the claim that is not true by construction: "
          "\nrecoveries where the command succeeded, the dependency recovered, and "
          "obligations \nstill missed their deadlines. Only transaction-level "
          "verification detects these.")
    print()


if __name__ == "__main__":
    out = Path(__file__).parent / "results"
    summarize(run_experiment(reps=400, out_dir=out))
    sweep(reps=250)
    h2_experiment(reps=400, out_dir=out)
    h4_experiment(reps=400, out_dir=out)
    print(f"raw per-replication results written to {out}")
