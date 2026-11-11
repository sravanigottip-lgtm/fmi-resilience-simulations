#!/usr/bin/env python3
"""
Evaluation of the risk-adaptive release-governance framework (Paper 3, Section 7).

Four governance conditions over an identical change corpus with identical
injected defects:

  A  uniform gating        same validation regime for every change;
                           component-level verification only
  B  advisory assessment   model estimates change risk; the estimate has no
                           authorisation effect; regime unchanged from A
  C  policy-governed       deterministic policy selects the release mode from
                           the model assessment
  D  full framework        mode selection plus progressive deployment and
                           transaction-aware post-release verification

Hypotheses H1-H6 of Section 7.4, a five-configuration ablation from Section
7.5, and a policy-sensitivity sweep.

Several hypotheses can fail. H5 is two-sided: validation effort must fall
*and* escaped defects must not rise. H4 asks whether historical change
evidence adds accuracy over the other three evidence classes, which the
framework's own Section 4.4 gives reason to doubt. H6 is satisfied by
construction and is reported as such rather than as a finding.

Stdlib only. No operational data. All parameters are stated assumptions.
"""

from __future__ import annotations

import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

# ------------------------------------------------------------------ parameters

N_CHANGES = 800
REPS = 300

# blast-radius axis distributions
P_TOUCHES_CONTRACT = 0.18      # interface or schema contract affected
P_SETTLEMENT_PATH = 0.22       # change lies on a settlement-critical path
P_EXTERNAL_EXPOSURE = 0.30     # organisational axis: external parties exposed
P_CONCURRENT = 0.20            # concurrent change to a related component
P_NO_ROLLBACK = 0.10           # rollback path unavailable at execution

# seven injected defect scenarios (Section 7.2)
SCENARIOS = ["hidden_dependency", "resource_within_tolerance", "transaction_class",
             "config_interaction", "transitive_security", "concurrent_interaction",
             "rollback_unavailable"]
SCEN_W = [0.17, 0.14, 0.18, 0.15, 0.12, 0.14, 0.10]
# scenarios invisible to component-level verification
TRANSACTION_ONLY = {"transaction_class", "resource_within_tolerance"}

P_DEFECT_BASE = 0.14           # probability a change carries a latent defect

# validation effort (engineer-hours, arbitrary but consistent unit)
EFFORT = {"standard": 3.0, "enhanced": 7.0, "controlled": 13.0, "restricted": 26.0}
EFFORT_UNIFORM = 12.0          # Condition A applies one regime to everything

# detection probability of the pre-release validation regime
DETECT = {"standard": 0.30, "enhanced": 0.55, "controlled": 0.70, "restricted": 0.85}
DETECT_UNIFORM = 0.62

# post-release verification
P_VERIFY_COMPONENT = 0.72      # component-level verification catches a visible defect
P_VERIFY_TXN = 0.88            # transaction-aware verification catches a txn defect
EXPOSURE_PROGRESSIVE = (4.0, 18.0)    # minutes exposed before withdrawal
EXPOSURE_FULL = (35.0, 180.0)         # full-fleet release, later discovery

LEAD_TIME = {"standard": 6.0, "enhanced": 14.0, "controlled": 30.0, "restricted": 72.0}
LEAD_UNIFORM = 26.0

MODES = ["standard", "enhanced", "controlled", "restricted"]
CONDITIONS = ["A_uniform", "B_advisory", "C_policy", "D_full"]
ABLATIONS = ["full", "no_historical", "no_blast_analysis",
             "no_progressive_verification", "confidence_gating"]


@dataclass
class Change:
    high_risk: bool            # ground truth: carries a latent defect
    scenario: str
    contract: bool
    settlement: bool
    external: bool
    concurrent: bool
    no_rollback: bool
    # evidence-class signals, each partially informative about high_risk
    ev_change: float
    ev_validation: float
    ev_operational: float
    ev_historical: float


def make_change(rng: random.Random) -> Change:
    contract = rng.random() < P_TOUCHES_CONTRACT
    settlement = rng.random() < P_SETTLEMENT_PATH
    external = rng.random() < P_EXTERNAL_EXPOSURE
    concurrent = rng.random() < P_CONCURRENT
    # latent-defect probability is raised by contract changes and concurrency
    p = P_DEFECT_BASE * (1.9 if contract else 1.0) * (1.5 if concurrent else 1.0)
    high = rng.random() < min(p, 0.85)
    sc = rng.choices(SCENARIOS, SCEN_W)[0]

    def signal(strength):
        """Noisy evidence: higher when high_risk, with overlap."""
        base = rng.gauss(0.62 if high else 0.38, 0.17)
        return max(0.0, min(1.0, base * strength + rng.gauss(0, 0.06)))

    return Change(high, sc, contract, settlement, external, concurrent,
                  rng.random() < P_NO_ROLLBACK,
                  signal(1.00), signal(0.95), signal(0.85), signal(0.70))


def assess(ch: Change, cfg: str) -> float:
    """Model risk estimate over the enabled evidence classes."""
    parts = [(ch.ev_change, 0.30), (ch.ev_validation, 0.28), (ch.ev_operational, 0.22)]
    if cfg != "no_historical":
        parts.append((ch.ev_historical, 0.20))
    tot = sum(w for _, w in parts)
    return sum(v * w for v, w in parts) / tot


def blast_floor(ch: Change, cfg: str) -> str:
    """Policy floor: minimum mode implied by blast radius, independent of the model."""
    if cfg == "no_blast_analysis":
        return "standard"
    if ch.settlement and ch.external:
        return "restricted"
    if ch.settlement or (ch.external and ch.contract):
        return "controlled"
    if ch.contract or ch.external:
        return "enhanced"
    return "standard"


def select_mode(ch: Change, est: float, cfg: str) -> str:
    """Deterministic policy over the assessment and the blast-radius floor."""
    if est >= 0.62:
        m = "restricted"
    elif est >= 0.52:
        m = "controlled"
    elif est >= 0.44:
        m = "enhanced"
    else:
        m = "standard"
    floor = blast_floor(ch, cfg)
    return MODES[max(MODES.index(m), MODES.index(floor))]


def confidence_mode(est: float) -> str:
    """Ablation: model gates itself, no policy floor."""
    if est >= 0.62:
        return "restricted"
    if est >= 0.52:
        return "controlled"
    if est >= 0.44:
        return "enhanced"
    return "standard"


def run(cond: str, cfg: str, seed: int, thresh_shift: float = 0.0) -> dict:
    rng = random.Random(seed)
    m = dict(n=0, tp=0, fp=0, fn=0, tn=0, failures=0, escaped=0, unsafe_auth=0,
             effort=0.0, lead=0.0, records=0, exposure=[], txn_caught=0, txn_total=0)
    for _ in range(N_CHANGES):
        ch = make_change(rng)
        m['n'] += 1
        est = assess(ch, cfg) + thresh_shift

        # --- regime selection ---
        if cond in ("A_uniform", "B_advisory"):
            mode, detect, effort, lead = None, DETECT_UNIFORM, EFFORT_UNIFORM, LEAD_UNIFORM
        else:
            mode = (confidence_mode(est) if cfg == "confidence_gating"
                    else select_mode(ch, est, cfg))
            detect, effort, lead = DETECT[mode], EFFORT[mode], LEAD_TIME[mode]
        m['effort'] += effort
        m['lead'] += lead

        # --- assessment quality (B, C, D produce an estimate) ---
        if cond != "A_uniform":
            pred = est >= 0.50
            if ch.high_risk and pred: m['tp'] += 1
            elif ch.high_risk: m['fn'] += 1
            elif pred: m['fp'] += 1
            else: m['tn'] += 1

        # --- unsafe authorisation: a high-blast change promoted below its floor ---
        if cond in ("C_policy", "D_full"):
            floor = blast_floor(ch, cfg)
            if mode is not None and MODES.index(mode) < MODES.index(floor):
                m['unsafe_auth'] += 1
        elif cond == "B_advisory":
            # advisory: the estimate is recorded but uniform gating still applies,
            # so any change whose floor exceeds the uniform regime is unsafely promoted
            if MODES.index(blast_floor(ch, cfg)) > MODES.index("enhanced"):
                m['unsafe_auth'] += 1

        # --- audit record ---
        if cond in ("C_policy", "D_full"):
            m['records'] += 1

        if not ch.high_risk:
            continue

        # --- pre-release validation ---
        if rng.random() < detect:
            continue                                  # defect caught before release

        m['failures'] += 1
        txn_only = ch.scenario in TRANSACTION_ONLY
        if txn_only:
            m['txn_total'] += 1

        # --- post-release verification ---
        if cond == "D_full" and cfg != "no_progressive_verification":
            p = P_VERIFY_TXN if txn_only else P_VERIFY_COMPONENT
            caught = rng.random() < p
            if caught and txn_only:
                m['txn_caught'] += 1
            if caught and not ch.no_rollback:
                m['exposure'].append(rng.uniform(*EXPOSURE_PROGRESSIVE))
                continue
            m['escaped'] += 1
            m['exposure'].append(rng.uniform(*EXPOSURE_FULL))
        else:
            # component-level verification only: transaction-only defects invisible
            caught = (not txn_only) and rng.random() < P_VERIFY_COMPONENT
            if caught and not ch.no_rollback:
                m['exposure'].append(rng.uniform(*EXPOSURE_PROGRESSIVE))
                continue
            m['escaped'] += 1
            m['exposure'].append(rng.uniform(*EXPOSURE_FULL))

    tp, fp, fn = m['tp'], m['fp'], m['fn']
    return {
        'condition': cond, 'cfg': cfg, 'shift': thresh_shift,
        'precision': tp / (tp + fp) if (tp + fp) else float('nan'),
        'recall': tp / (tp + fn) if (tp + fn) else float('nan'),
        'change_failure_rate': m['failures'] / m['n'],
        'escaped_rate': m['escaped'] / m['n'],
        'txn_detect_rate': (m['txn_caught'] / m['txn_total']) if m['txn_total'] else float('nan'),
        'unsafe_auth_rate': m['unsafe_auth'] / m['n'],
        'record_completeness': m['records'] / m['n'],
        'effort_per_change': m['effort'] / m['n'],
        'lead_time': m['lead'] / m['n'],
        'mean_exposure': statistics.mean(m['exposure']) if m['exposure'] else float('nan'),
    }


def ci(xs):
    xs = [x for x in xs if x == x]
    if not xs: return float('nan'), float('nan')
    mu = statistics.mean(xs)
    return (mu, 1.96 * statistics.stdev(xs) / len(xs) ** 0.5) if len(xs) > 1 else (mu, 0.0)


def main():
    rows = []
    for c in CONDITIONS:
        for r in range(REPS):
            rows.append(run(c, "full", 5000 + r))
    for cfg in ABLATIONS[1:]:
        for r in range(REPS):
            rows.append(run("D_full", cfg, 5000 + r))
    sens = []
    for sh in (-0.06, -0.03, 0.0, 0.03, 0.06):
        for r in range(150):
            sens.append(run("D_full", "full", 5000 + r, thresh_shift=sh))

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    with (out / "release_governance_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows + sens)

    sel = lambda c, g, f: [r[f] for r in rows if r['condition'] == c and r['cfg'] == g]
    F = ['precision', 'recall', 'change_failure_rate', 'escaped_rate', 'txn_detect_rate',
         'unsafe_auth_rate', 'record_completeness', 'effort_per_change', 'lead_time',
         'mean_exposure']

    print("=" * 120)
    print(f"RISK-ADAPTIVE RELEASE GOVERNANCE — EVALUATION   {REPS} reps x {N_CHANGES} changes per condition")
    print("=" * 120)
    print(f"{'metric':<24}" + "".join(f"{c:>24}" for c in CONDITIONS))
    print("-" * 120)
    for f in F:
        line = f"{f:<24}"
        for c in CONDITIONS:
            mu, h = ci(sel(c, 'full', f))
            line += f"{mu:>17.3f}±{h:<6.3f}" if mu == mu else f"{'n/a':>24}"
        print(line)

    print("\n" + "=" * 120)
    print("ABLATION (Condition D)")
    print("=" * 120)
    print(f"{'configuration':<30}{'precision':>12}{'recall':>10}{'CFR':>10}"
          f"{'escaped':>10}{'unsafe auth':>13}{'effort':>10}")
    print("-" * 120)
    for cfg in ABLATIONS:
        p, _ = ci(sel('D_full', cfg, 'precision'))
        rc, _ = ci(sel('D_full', cfg, 'recall'))
        cf, _ = ci(sel('D_full', cfg, 'change_failure_rate'))
        es, _ = ci(sel('D_full', cfg, 'escaped_rate'))
        ua, _ = ci(sel('D_full', cfg, 'unsafe_auth_rate'))
        ef, _ = ci(sel('D_full', cfg, 'effort_per_change'))
        print(f"{cfg:<30}{p:>12.3f}{rc:>10.3f}{cf:>10.3f}{es:>10.3f}{ua:>13.3f}{ef:>10.1f}")

    print("\n" + "=" * 120)
    print("POLICY SENSITIVITY (Condition D, threshold shift)")
    print("=" * 120)
    print(f"{'shift':>8}{'escaped rate':>16}{'effort/change':>16}{'unsafe auth':>14}")
    print("-" * 120)
    for sh in (-0.06, -0.03, 0.0, 0.03, 0.06):
        e, _ = ci([r['escaped_rate'] for r in sens if r['shift'] == sh])
        ef, _ = ci([r['effort_per_change'] for r in sens if r['shift'] == sh])
        ua, _ = ci([r['unsafe_auth_rate'] for r in sens if r['shift'] == sh])
        print(f"{sh:>+8.2f}{e:>16.3f}{ef:>16.1f}{ua:>14.3f}")

    print("\n" + "=" * 120)
    print("HYPOTHESIS DISPOSITION")
    print("=" * 120)
    cfA, hA = ci(sel('A_uniform', 'full', 'change_failure_rate'))
    cfD, hD = ci(sel('D_full', 'full', 'change_failure_rate'))
    ltA, _ = ci(sel('A_uniform', 'full', 'lead_time'))
    ltD, _ = ci(sel('D_full', 'full', 'lead_time'))
    print(f"H1 CFR D<A at equal/greater throughput")
    print(f"   A={cfA:.3f}±{hA:.3f}  D={cfD:.3f}±{hD:.3f} | lead time A={ltA:.1f}h D={ltD:.1f}h"
          f"  -> {'SUPPORTED' if cfD < cfA and (cfA-hA) > (cfD+hD) and ltD <= ltA else 'NOT SUPPORTED'}")
    uB, hB = ci(sel('B_advisory', 'full', 'unsafe_auth_rate'))
    uC, hC = ci(sel('C_policy', 'full', 'unsafe_auth_rate'))
    print(f"H2 unsafe authorisation C<B")
    print(f"   B={uB:.3f}±{hB:.3f}  C={uC:.3f}±{hC:.3f}"
          f"  -> {'SUPPORTED' if uC < uB and (uB-hB) > (uC+hC) else 'NOT SUPPORTED'}")
    tD, htD = ci(sel('D_full', 'full', 'txn_detect_rate'))
    tN, htN = ci(sel('D_full', 'no_progressive_verification', 'txn_detect_rate'))
    print(f"H3 transaction-aware verification detects txn-class regressions")
    print(f"   D with txn-aware={tD:.3f}±{htD:.3f}  component-only={0.0 if tN != tN else tN:.3f}"
          f"  -> {'SUPPORTED' if tD == tD and (tN != tN or tD > tN) else 'NOT SUPPORTED'}")
    pF, hpF = ci(sel('D_full', 'full', 'precision'))
    pN, hpN = ci(sel('D_full', 'no_historical', 'precision'))
    rF, _ = ci(sel('D_full', 'full', 'recall'))
    rN, _ = ci(sel('D_full', 'no_historical', 'recall'))
    print(f"H4 historical evidence adds assessment accuracy")
    print(f"   precision full={pF:.3f}±{hpF:.3f} vs no-historical={pN:.3f}±{hpN:.3f} | "
          f"recall {rF:.3f} vs {rN:.3f}"
          f"  -> {'SUPPORTED' if pF > pN and (pF-hpF) > (pN+hpN) else 'NOT SUPPORTED'}")
    efA, _ = ci(sel('A_uniform', 'full', 'effort_per_change'))
    efD, _ = ci(sel('D_full', 'full', 'effort_per_change'))
    esA, hesA = ci(sel('A_uniform', 'full', 'escaped_rate'))
    esD, hesD = ci(sel('D_full', 'full', 'escaped_rate'))
    print(f"H5 effort D<A while escaped defects not higher")
    print(f"   effort A={efA:.1f} D={efD:.1f} | escaped A={esA:.3f}±{hesA:.3f} D={esD:.3f}±{hesD:.3f}"
          f"  -> {'SUPPORTED' if efD < efA and (esD - hesD) <= (esA + hesA) else 'NOT SUPPORTED'}")
    rcA, _ = ci(sel('A_uniform', 'full', 'record_completeness'))
    rcD, _ = ci(sel('D_full', 'full', 'record_completeness'))
    print(f"H6 record completeness C/D > A/B")
    print(f"   A={rcA:.3f}  D={rcD:.3f}  -> SUPPORTED BY CONSTRUCTION (not an empirical finding)")
    print(f"\nraw results: {out/'release_governance_results.csv'}")


if __name__ == "__main__":
    main()
