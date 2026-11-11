#!/usr/bin/env python3
"""
Reference implementation of horizon-aware release admission.

This is the control described in Section 4.4 of the accompanying paper, built as
a working artifact rather than a sketch. It is deliberately small and has no
dependencies outside the standard library, because the point is that the
arithmetic is simple and the difficulty lies entirely in sourcing its inputs.

The five elements of the architecture map to five classes:

    IncidentRecord / CostModel   the four intervals, fitted from history
    Obligation / SettlementState the deadlines of work currently in flight
    HorizonEvaluator             computes B - R and compares it against now
    AdmissionDecision            the auditable record of one decision
    AdmissionController          ties them together and exposes one method

What an institution has to supply
---------------------------------
1. A list of IncidentRecord objects covering past change-induced incidents.
   Every field is recoverable from an incident timeline that most institutions
   already keep: when the change deployed, when the anomaly began, when a
   monitor fired, when remediation started, when it finished, when throughput
   normalised.

2. An implementation of SettlementState.in_flight() returning the obligations
   currently in the system with their finality deadlines. This is the hard part,
   and the paper says so: deployment tooling and settlement state usually live
   in different systems.

Everything else is here.

Run this file directly to execute the self-tests and a worked example.
"""

from __future__ import annotations

import json
import statistics
import unittest
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Protocol, Sequence

__all__ = [
    "IncidentRecord", "CostModel", "Obligation", "SettlementState",
    "Outcome", "AdmissionDecision", "HorizonEvaluator", "AdmissionController",
]

UTC = timezone.utc


# ---------------------------------------------------------------------------
# 1. Cost model: the four intervals, fitted from incident history
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IncidentRecord:
    """One past change-induced incident, as an institution would have logged it.

    All five timestamps are required. A record where remediation never ran
    cannot inform a rollback-duration estimate and should be excluded by the
    caller rather than passed with invented values.
    """
    incident_id: str
    deployed_at: datetime
    anomaly_began_at: datetime
    detected_at: datetime
    remediation_started_at: datetime
    remediation_finished_at: datetime
    throughput_normal_at: datetime

    def __post_init__(self) -> None:
        seq = [self.deployed_at, self.anomaly_began_at, self.detected_at,
               self.remediation_started_at, self.remediation_finished_at,
               self.throughput_normal_at]
        if any(b < a for a, b in zip(seq, seq[1:])):
            raise ValueError(f"{self.incident_id}: timestamps are not monotonic")

    # the four intervals, in minutes
    @property
    def appearance(self) -> float:
        return (self.anomaly_began_at - self.deployed_at).total_seconds() / 60.0

    @property
    def detection(self) -> float:
        return (self.detected_at - self.anomaly_began_at).total_seconds() / 60.0

    @property
    def rollback(self) -> float:
        return (self.remediation_finished_at - self.remediation_started_at).total_seconds() / 60.0

    @property
    def recovery(self) -> float:
        return (self.throughput_normal_at - self.remediation_finished_at).total_seconds() / 60.0


@dataclass(frozen=True)
class CostModel:
    """The withdrawal cost R, with the provenance of each term retained.

    Deliberately conservative in two ways, both stated in Section 4.1 of the
    paper. Appearance delay uses a high quantile rather than the mean, because a
    defect that surfaces late is the one that consumes margin before anyone
    knows there is a problem. Rollback duration carries a variability allowance
    on top of its mean.
    """
    appearance_p90: float
    detection_mean: float
    rollback_mean: float
    rollback_sd: float
    recovery_mean: float
    margin: float = 1.25
    sample_size: int = 0
    fitted_at: datetime | None = None
    sd_allowance: float = 2.0

    @classmethod
    def fit(cls, records: Sequence[IncidentRecord], margin: float = 1.25,
            sd_allowance: float = 2.0) -> "CostModel":
        if len(records) < 3:
            raise ValueError("at least three incident records are required to fit a cost model")
        app = sorted(r.appearance for r in records)
        p90 = app[min(len(app) - 1, int(0.90 * len(app)))]
        rb = [r.rollback for r in records]
        return cls(
            appearance_p90=p90,
            detection_mean=statistics.fmean(r.detection for r in records),
            rollback_mean=statistics.fmean(rb),
            rollback_sd=statistics.pstdev(rb) if len(rb) > 1 else 0.0,
            recovery_mean=statistics.fmean(r.recovery for r in records),
            margin=margin,
            sample_size=len(records),
            fitted_at=datetime.now(UTC),
            sd_allowance=sd_allowance,
        )

    @property
    def total_minutes(self) -> float:
        """R: worst-case time from deploying now to being recovered."""
        base = (self.appearance_p90 + self.detection_mean
                + self.rollback_mean + self.sd_allowance * self.rollback_sd
                + self.recovery_mean)
        return base * self.margin

    def breakdown(self) -> dict:
        return {"appearance_p90": round(self.appearance_p90, 2),
                "detection_mean": round(self.detection_mean, 2),
                "rollback_mean": round(self.rollback_mean, 2),
                "rollback_sd": round(self.rollback_sd, 2),
                "recovery_mean": round(self.recovery_mean, 2),
                "sd_allowance": self.sd_allowance,
                "margin": self.margin,
                "total_minutes": round(self.total_minutes, 2),
                "sample_size": self.sample_size}

    def version(self) -> str:
        """Stable identifier so a decision record can name the model that made it."""
        payload = json.dumps(self.breakdown(), sort_keys=True).encode()
        import hashlib
        return hashlib.blake2b(payload, digest_size=6).hexdigest()

    def staleness_days(self, now: datetime | None = None) -> float | None:
        if self.fitted_at is None:
            return None
        return ((now or datetime.now(UTC)) - self.fitted_at).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# 2. Settlement state: what is in flight and when it must be final
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Obligation:
    obligation_id: str
    finality_deadline: datetime
    stage: int = 0
    linked_to: str | None = None


class SettlementState(Protocol):
    """The institution supplies this. It is the integration the paper calls the
    principal implementation risk."""

    def in_flight(self) -> Iterable[Obligation]:
        ...


# ---------------------------------------------------------------------------
# 3 and 4. Evaluation and the decision record
# ---------------------------------------------------------------------------

class Outcome(str, Enum):
    ADMIT = "admit"
    DEFER = "defer"
    ESCALATE = "escalate"
    ADMIT_NO_EXPOSURE = "admit_no_exposure"


@dataclass
class AdmissionDecision:
    """Everything needed to reconstruct one admission after an incident."""
    decision_id: str
    change_id: str
    decided_at: datetime
    outcome: Outcome
    reason: str
    horizon: datetime | None
    binding_deadline: datetime | None
    binding_obligation: str | None
    margin_remaining_minutes: float | None
    cost_minutes: float
    cost_model_version: str
    cost_breakdown: dict
    exposed_obligations: int
    target_stage: int | None = None
    overridden_by: str | None = None
    override_justification: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        for k in ("decided_at", "horizon", "binding_deadline"):
            d[k] = d[k].isoformat() if d[k] else None
        return json.dumps(d, sort_keys=True)

    def record_override(self, who: str, justification: str) -> None:
        if not who or not justification:
            raise ValueError("an override requires both an identity and a justification")
        self.overridden_by = who
        self.override_justification = justification


class HorizonEvaluator:
    """Computes B - R and compares it against the present instant.

    stage_aware reflects the Section 8.7 finding: restricting the exposed set to
    obligations that have not yet cleared the target stage reached the same
    safety while admitting materially more changes. It is off by default because
    it requires per-obligation stage information that many institutions will not
    have.
    """

    def __init__(self, cost_model: CostModel, stage_aware: bool = False,
                 max_cost_model_age_days: float | None = 90.0):
        self.cost_model = cost_model
        self.stage_aware = stage_aware
        self.max_cost_model_age_days = max_cost_model_age_days

    def _exposed(self, obligations: Sequence[Obligation],
                 target_stage: int | None) -> list[Obligation]:
        if not self.stage_aware or target_stage is None:
            return list(obligations)
        return [o for o in obligations if o.stage <= target_stage]

    def evaluate(self, change_id: str, obligations: Sequence[Obligation],
                 now: datetime | None = None,
                 target_stage: int | None = None) -> AdmissionDecision:
        now = now or datetime.now(UTC)
        cm = self.cost_model
        cost = cm.total_minutes

        def record(outcome: Outcome, reason: str, horizon=None, bd=None,
                   bo=None, margin=None, exposed=0) -> AdmissionDecision:
            return AdmissionDecision(
                decision_id=str(uuid.uuid4()), change_id=change_id, decided_at=now,
                outcome=outcome, reason=reason, horizon=horizon, binding_deadline=bd,
                binding_obligation=bo, margin_remaining_minutes=margin,
                cost_minutes=round(cost, 2), cost_model_version=cm.version(),
                cost_breakdown=cm.breakdown(), exposed_obligations=exposed,
                target_stage=target_stage)

        age = cm.staleness_days(now)
        if (self.max_cost_model_age_days is not None and age is not None
                and age > self.max_cost_model_age_days):
            return record(Outcome.ESCALATE,
                          f"cost model is {age:.0f} days old, beyond the "
                          f"{self.max_cost_model_age_days:.0f}-day limit; the horizon "
                          f"cannot be trusted until it is refitted")

        exposed = self._exposed(obligations, target_stage)
        if not exposed:
            return record(Outcome.ADMIT_NO_EXPOSURE,
                          "no obligation in flight is exposed to this change",
                          exposed=0)

        binding = min(exposed, key=lambda o: o.finality_deadline)
        horizon = binding.finality_deadline - timedelta(minutes=cost)
        margin = (horizon - now).total_seconds() / 60.0

        if now <= horizon:
            return record(Outcome.ADMIT,
                          f"{margin:.1f} minutes of rollback margin remain before the "
                          f"binding finality boundary",
                          horizon, binding.finality_deadline, binding.obligation_id,
                          round(margin, 2), len(exposed))
        return record(Outcome.ESCALATE,
                      f"the horizon passed {abs(margin):.1f} minutes ago; admitting now "
                      f"means accepting that no rollback can complete before finality",
                      horizon, binding.finality_deadline, binding.obligation_id,
                      round(margin, 2), len(exposed))


# ---------------------------------------------------------------------------
# 5. Controller
# ---------------------------------------------------------------------------

class AdmissionController:
    def __init__(self, state: SettlementState, evaluator: HorizonEvaluator):
        self.state = state
        self.evaluator = evaluator
        self.journal: list[AdmissionDecision] = []

    def request(self, change_id: str, now: datetime | None = None,
                target_stage: int | None = None) -> AdmissionDecision:
        obs = list(self.state.in_flight())
        d = self.evaluator.evaluate(change_id, obs, now=now, target_stage=target_stage)
        self.journal.append(d)
        return d

    def export_journal(self) -> str:
        return "\n".join(d.to_json() for d in self.journal)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _rec(i, dep, app, det, rbs, rbe, norm) -> IncidentRecord:
    t0 = datetime(2026, 1, 1, 9, 0, tzinfo=UTC) + timedelta(days=i)
    m = lambda x: t0 + timedelta(minutes=x)
    return IncidentRecord(f"INC-{i:03d}", m(dep), m(app), m(det), m(rbs), m(rbe), m(norm))


SAMPLE = [_rec(0, 0, 12, 18, 20, 31, 36), _rec(1, 0, 25, 33, 35, 44, 48),
          _rec(2, 0, 8, 15, 16, 30, 34), _rec(3, 0, 38, 45, 47, 61, 67),
          _rec(4, 0, 18, 23, 25, 36, 40), _rec(5, 0, 30, 39, 41, 55, 60)]


class _Fixed:
    def __init__(self, obs): self._obs = obs
    def in_flight(self): return self._obs


class TestCostModel(unittest.TestCase):
    def test_intervals(self):
        r = SAMPLE[0]
        self.assertAlmostEqual(r.appearance, 12.0)
        self.assertAlmostEqual(r.detection, 6.0)
        self.assertAlmostEqual(r.rollback, 11.0)
        self.assertAlmostEqual(r.recovery, 5.0)

    def test_non_monotonic_rejected(self):
        with self.assertRaises(ValueError):
            _rec(9, 10, 5, 15, 20, 30, 35)

    def test_too_few_records(self):
        with self.assertRaises(ValueError):
            CostModel.fit(SAMPLE[:2])

    def test_uses_high_quantile_not_mean(self):
        cm = CostModel.fit(SAMPLE)
        mean_app = statistics.fmean(r.appearance for r in SAMPLE)
        self.assertGreater(cm.appearance_p90, mean_app)

    def test_version_is_stable_and_sensitive(self):
        a = CostModel.fit(SAMPLE)
        self.assertEqual(a.version(), CostModel.fit(SAMPLE).version())
        self.assertNotEqual(a.version(), CostModel.fit(SAMPLE, margin=1.5).version())


class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.cm = CostModel.fit(SAMPLE)
        self.now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)

    def _obs(self, *mins):
        return [Obligation(f"OB-{i}", self.now + timedelta(minutes=m), stage=i)
                for i, m in enumerate(mins)]

    def test_admits_with_ample_margin(self):
        ev = HorizonEvaluator(self.cm)
        d = ev.evaluate("CHG-1", self._obs(600, 700), now=self.now)
        self.assertEqual(d.outcome, Outcome.ADMIT)
        self.assertGreater(d.margin_remaining_minutes, 0)

    def test_escalates_inside_the_horizon(self):
        ev = HorizonEvaluator(self.cm)
        d = ev.evaluate("CHG-2", self._obs(20), now=self.now)
        self.assertEqual(d.outcome, Outcome.ESCALATE)
        self.assertLess(d.margin_remaining_minutes, 0)

    def test_binding_obligation_is_the_earliest(self):
        ev = HorizonEvaluator(self.cm)
        d = ev.evaluate("CHG-3", self._obs(600, 90, 800), now=self.now)
        self.assertEqual(d.binding_obligation, "OB-1")

    def test_empty_system_admits(self):
        ev = HorizonEvaluator(self.cm)
        d = ev.evaluate("CHG-4", [], now=self.now)
        self.assertEqual(d.outcome, Outcome.ADMIT_NO_EXPOSURE)

    def test_stage_aware_ignores_cleared_obligations(self):
        """Section 8.7: an obligation past the target stage is not exposed."""
        obs = [Obligation("early", self.now + timedelta(minutes=20), stage=3),
               Obligation("later", self.now + timedelta(minutes=900), stage=0)]
        naive = HorizonEvaluator(self.cm, stage_aware=False).evaluate(
            "CHG-5", obs, now=self.now, target_stage=1)
        aware = HorizonEvaluator(self.cm, stage_aware=True).evaluate(
            "CHG-5", obs, now=self.now, target_stage=1)
        self.assertEqual(naive.outcome, Outcome.ESCALATE)
        self.assertEqual(aware.outcome, Outcome.ADMIT)
        self.assertEqual(aware.exposed_obligations, 1)

    def test_stale_cost_model_escalates(self):
        old = CostModel.fit(SAMPLE)
        object.__setattr__(old, "fitted_at", self.now - timedelta(days=200))
        d = HorizonEvaluator(old).evaluate("CHG-6", self._obs(900), now=self.now)
        self.assertEqual(d.outcome, Outcome.ESCALATE)
        self.assertIn("cost model", d.reason)


class TestRecordAndController(unittest.TestCase):
    def test_record_round_trips_and_names_its_model(self):
        cm = CostModel.fit(SAMPLE)
        now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        ctl = AdmissionController(
            _Fixed([Obligation("OB-1", now + timedelta(minutes=900))]),
            HorizonEvaluator(cm))
        d = ctl.request("CHG-7", now=now)
        back = json.loads(d.to_json())
        self.assertEqual(back["cost_model_version"], cm.version())
        self.assertEqual(back["outcome"], "admit")
        self.assertEqual(len(ctl.journal), 1)

    def test_override_requires_identity_and_reason(self):
        cm = CostModel.fit(SAMPLE)
        now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        d = HorizonEvaluator(cm).evaluate(
            "CHG-8", [Obligation("OB", now + timedelta(minutes=5))], now=now)
        with self.assertRaises(ValueError):
            d.record_override("", "no identity")
        d.record_override("release-manager@example", "hotfix for a live settlement defect")
        self.assertIn("hotfix", d.override_justification)


def _demo() -> None:
    cm = CostModel.fit(SAMPLE)
    print("COST MODEL fitted from 6 incident records")
    for k, v in cm.breakdown().items():
        print(f"  {k:<18} {v}")
    print(f"  version            {cm.version()}")

    now = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
    obs = [Obligation("SETT-001", now + timedelta(minutes=240), stage=0),
           Obligation("SETT-002", now + timedelta(minutes=95), stage=2),
           Obligation("SETT-003", now + timedelta(minutes=480), stage=1)]
    print(f"\nIN FLIGHT at {now:%H:%M} UTC")
    for o in obs:
        print(f"  {o.obligation_id}  stage {o.stage}  final by "
              f"{o.finality_deadline:%H:%M}  ({(o.finality_deadline-now).total_seconds()/60:.0f} min)")

    for label, ev, stage in [
            ("system-wide horizon", HorizonEvaluator(cm), None),
            ("stage-aware, change targets stage 1", HorizonEvaluator(cm, stage_aware=True), 1)]:
        d = ev.evaluate("CHG-2026-0601-17", obs, now=now, target_stage=stage)
        print(f"\n{label}")
        print(f"  outcome   {d.outcome.value}")
        print(f"  reason    {d.reason}")
        print(f"  exposed   {d.exposed_obligations} of {len(obs)} obligations")
        if d.horizon:
            print(f"  horizon   {d.horizon:%H:%M} UTC   (binding: {d.binding_obligation})")


if __name__ == "__main__":
    _demo()
    print("\n" + "=" * 68)
    unittest.main(argv=["-q"], exit=False, verbosity=2)
