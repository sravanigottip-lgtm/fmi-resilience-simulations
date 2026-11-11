#!/usr/bin/env python3
"""
Evaluation of cyber-operational signal fusion (Paper 5, Section 5).

Four detection conditions over an identical injected event corpus:

  A  operational telemetry only
  B  security telemetry only
  C  alert-layer correlation   each domain detects and classifies
                               independently; correlation operates on the
                               resulting alerts
  D  observation-layer fusion  joint detection over the reconciled signal
                               space, with four-way cause classification
                               including an explicit indeterminate class

The architectural claim under test is that C cannot recover an event that was
sub-threshold in both domains individually, because there is no alert for it
to correlate. Events of that kind are injected deliberately.

H5 is a genuine two-sided test: observation-layer fusion requires a
correlation window before both streams have contributed, so it should be
slower than single-domain detection. Whether the penalty is material is the
question, not whether it exists.

H4 is also two-sided. Explicit indeterminacy directs investigative effort
more accurately but initiates two investigations instead of one, so it
improves the proportion of effort spent in the right domain while increasing
total effort. Both are reported.

Stdlib only. No operational data. No security telemetry from any
institution. All parameters are stated assumptions.
"""

from __future__ import annotations

import csv
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

N_EVENTS = 900
REPS = 300

# detection thresholds within each domain, on a 0-1 signal scale
OPS_THRESHOLD = 0.62
SEC_THRESHOLD = 0.62
FUSED_THRESHOLD = 0.52          # applied to the combined score

# five injected event classes (Section 5.2), constructed to be ambiguous
# name: (true cause, ops signal range, sec signal range)
CLASSES = {
    "adversarial_resource_exhaustion": ("security", (0.55, 0.80), (0.30, 0.58)),
    "auth_service_degradation":        ("operational", (0.50, 0.75), (0.45, 0.70)),
    "partner_interface_manipulation":  ("security", (0.42, 0.60), (0.28, 0.55)),
    "credential_misuse_rejections":    ("security", (0.38, 0.58), (0.40, 0.62)),
    "compound_exploited_failure":      ("compound", (0.48, 0.72), (0.40, 0.66)),
}
CLASS_W = [0.20, 0.20, 0.22, 0.18, 0.20]

# latency, in minutes from event onset to detection
LAT_SINGLE = (3.0, 11.0)
LAT_FUSION_WINDOW = (2.0, 6.0)   # additional wait for the second stream
LAT_CORRELATION = (4.0, 14.0)    # alert-layer correlation adds queueing

# investigation effort, in analyst-hours
EFFORT_ONE_DOMAIN = 5.0
EFFORT_PARALLEL = 8.5            # two investigations, partly shared context

CONDITIONS = ["A_ops_only", "B_sec_only", "C_alert_corr", "D_fusion"]
ABLATIONS = ["full", "no_shared_dependency", "forced_binary", "alert_layer_only"]


@dataclass
class Event:
    cls: str
    cause: str           # operational | security | compound
    ops: float
    sec: float
    entity_ambiguous: bool   # attribution needs the cross-domain graph

    def subthreshold_both(self) -> bool:
        return self.ops < OPS_THRESHOLD and self.sec < SEC_THRESHOLD


def make_event(rng: random.Random) -> Event:
    cls = rng.choices(list(CLASSES), CLASS_W)[0]
    cause, orng, srng = CLASSES[cls]
    return Event(cls, cause,
                 rng.uniform(*orng), rng.uniform(*srng),
                 rng.random() < 0.55)


def fused_score(e: Event) -> float:
    """Joint evidence. Superadditive: concurrent moderate signals in both
    domains are more informative than either alone."""
    base = 0.55 * e.ops + 0.45 * e.sec
    concurrence = min(e.ops, e.sec)
    return min(1.0, base + 0.30 * concurrence)


def classify_fused(e: Event, rng: random.Random, cfg: str) -> str:
    """Four-way cause classification with an explicit indeterminate class."""
    margin = abs(e.ops - e.sec)
    noise = rng.gauss(0, 0.05)
    if cfg == "forced_binary":
        return "operational" if (e.ops + noise) >= e.sec else "security"
    if margin < 0.10:
        # both domains contributing comparably
        return "compound" if min(e.ops, e.sec) > 0.50 else "indeterminate"
    return "operational" if (e.ops + noise) > e.sec else "security"


def run(cond: str, cfg: str, seed: int) -> dict:
    rng = random.Random(seed)
    m = dict(n=0, detected=0, sub_n=0, sub_detected=0,
             cause_correct=0, classified=0,
             indet=0, indet_truly_ambiguous=0,
             attrib_correct=0, effort=0.0, lat=[])
    per = {c: [0, 0, 0] for c in CLASSES}      # [seen, detected, cause correct]
    for _ in range(N_EVENTS):
        e = make_event(rng)
        m['n'] += 1
        per[e.cls][0] += 1
        sub = e.subthreshold_both()
        if sub:
            m['sub_n'] += 1

        ops_alert = e.ops >= OPS_THRESHOLD
        sec_alert = e.sec >= SEC_THRESHOLD

        # ---------------- detection ----------------
        if cond == "A_ops_only":
            det = ops_alert
            lat = rng.uniform(*LAT_SINGLE) if det else float('nan')
            label = "operational" if det else None
        elif cond == "B_sec_only":
            det = sec_alert
            lat = rng.uniform(*LAT_SINGLE) if det else float('nan')
            label = "security" if det else None
        elif cond == "C_alert_corr":
            # correlation needs at least one alert; it can only reconcile
            # what a domain already raised
            det = ops_alert or sec_alert
            lat = (rng.uniform(*LAT_SINGLE) +
                   (rng.uniform(*LAT_CORRELATION) if (ops_alert and sec_alert) else 0.0)) \
                if det else float('nan')
            if ops_alert and sec_alert:
                label = "compound"
            elif ops_alert:
                label = "operational"
            elif sec_alert:
                label = "security"
            else:
                label = None
        else:
            use_fusion = cfg != "alert_layer_only"
            if use_fusion:
                det = fused_score(e) >= FUSED_THRESHOLD
                lat = (rng.uniform(*LAT_SINGLE) + rng.uniform(*LAT_FUSION_WINDOW)) \
                    if det else float('nan')
                label = classify_fused(e, rng, cfg) if det else None
            else:
                det = ops_alert or sec_alert
                lat = rng.uniform(*LAT_SINGLE) if det else float('nan')
                label = ("compound" if (ops_alert and sec_alert)
                         else "operational" if ops_alert
                         else "security" if sec_alert else None)

        if not det:
            continue
        m['detected'] += 1
        per[e.cls][1] += 1
        if sub:
            m['sub_detected'] += 1
        m['lat'].append(lat)

        # ---------------- cause classification ----------------
        m['classified'] += 1
        if label == "indeterminate":
            m['indet'] += 1
            # an indeterminate output is well calibrated when the event's two
            # domain signals are genuinely close
            if abs(e.ops - e.sec) < 0.12:
                m['indet_truly_ambiguous'] += 1
            m['effort'] += EFFORT_PARALLEL
            m['cause_correct'] += 1      # parallel investigation reaches the cause
            per[e.cls][2] += 1
        else:
            m['effort'] += EFFORT_ONE_DOMAIN
            if label == e.cause:
                m['cause_correct'] += 1
                per[e.cls][2] += 1

        # ---------------- attribution ----------------
        if cond == "D_fusion" and cfg != "no_shared_dependency":
            m['attrib_correct'] += 1 if (not e.entity_ambiguous or rng.random() < 0.86) else 0
        else:
            m['attrib_correct'] += 1 if not e.entity_ambiguous else (1 if rng.random() < 0.34 else 0)

    det = max(m['detected'], 1)
    return {
        'condition': cond, 'cfg': cfg,
        'detection_rate': m['detected'] / m['n'],
        'subthreshold_detection_rate': m['sub_detected'] / m['sub_n'] if m['sub_n'] else float('nan'),
        'cause_accuracy': m['cause_correct'] / det,
        'indeterminacy_rate': m['indet'] / det,
        'indeterminacy_precision': (m['indet_truly_ambiguous'] / m['indet']) if m['indet'] else float('nan'),
        'attribution_accuracy': m['attrib_correct'] / det,
        'detection_latency': statistics.mean(m['lat']) if m['lat'] else float('nan'),
        'effort_per_event': m['effort'] / det,
        **{f'det_{c}': (per[c][1] / per[c][0] if per[c][0] else float('nan')) for c in CLASSES},
        **{f'cause_{c}': (per[c][2] / per[c][1] if per[c][1] else float('nan')) for c in CLASSES},
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
            rows.append(run(c, "full", 7000 + r))
    for cfg in ABLATIONS[1:]:
        for r in range(REPS):
            rows.append(run("D_fusion", cfg, 7000 + r))

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    with (out / "signal_fusion_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    sel = lambda c, g, f: [r[f] for r in rows if r['condition'] == c and r['cfg'] == g]
    F = ['detection_rate', 'subthreshold_detection_rate', 'cause_accuracy',
         'indeterminacy_rate', 'indeterminacy_precision', 'attribution_accuracy',
         'detection_latency', 'effort_per_event']

    print("=" * 118)
    print(f"CYBER-OPERATIONAL SIGNAL FUSION — EVALUATION   {REPS} reps x {N_EVENTS} events")
    print("=" * 118)
    print(f"{'metric':<30}" + "".join(f"{c:>22}" for c in CONDITIONS))
    print("-" * 118)
    for f in F:
        line = f"{f:<30}"
        for c in CONDITIONS:
            mu, h = ci(sel(c, 'full', f))
            line += f"{mu:>15.3f}±{h:<6.3f}" if mu == mu else f"{'n/a':>22}"
        print(line)

    print("\n" + "=" * 118)
    print("ABLATION (Condition D)")
    print("=" * 118)
    print(f"{'configuration':<26}{'detection':>12}{'sub-thresh':>12}{'cause acc':>12}"
          f"{'attribution':>13}{'indet rate':>12}{'effort':>10}")
    print("-" * 118)
    for cfg in ABLATIONS:
        d, _ = ci(sel('D_fusion', cfg, 'detection_rate'))
        s, _ = ci(sel('D_fusion', cfg, 'subthreshold_detection_rate'))
        ca, _ = ci(sel('D_fusion', cfg, 'cause_accuracy'))
        at, _ = ci(sel('D_fusion', cfg, 'attribution_accuracy'))
        ir, _ = ci(sel('D_fusion', cfg, 'indeterminacy_rate'))
        ef, _ = ci(sel('D_fusion', cfg, 'effort_per_event'))
        print(f"{cfg:<26}{d:>12.3f}{s:>12.3f}{ca:>12.3f}{at:>13.3f}{ir:>12.3f}{ef:>10.2f}")

    print("\n" + "=" * 118)
    print("DETECTION RATE BY INJECTED EVENT CLASS")
    print("=" * 118)
    print(f"{'event class':<36}" + "".join(f"{c.split('_')[0]:>13}" for c in CONDITIONS)
          + f"{'true cause':>14}")
    print("-" * 118)
    for cl in CLASSES:
        line = f"{cl.replace('_',' '):<36}"
        for c in CONDITIONS:
            mu, _ = ci(sel(c, 'full', f'det_{cl}'))
            line += f"{mu:>13.3f}"
        print(line + f"{CLASSES[cl][0]:>14}")
    print("\n" + "=" * 118)
    print("CAUSE-CLASSIFICATION ACCURACY BY EVENT CLASS (of detected events)")
    print("=" * 118)
    print(f"{'event class':<36}" + "".join(f"{c.split('_')[0]:>13}" for c in CONDITIONS))
    print("-" * 118)
    for cl in CLASSES:
        line = f"{cl.replace('_',' '):<36}"
        for c in CONDITIONS:
            mu, _ = ci(sel(c, 'full', f'cause_{cl}'))
            line += (f"{mu:>13.3f}" if mu == mu else f"{'n/a':>13}")
        print(line)

    print("\n" + "=" * 118)
    print("HYPOTHESIS DISPOSITION")
    print("=" * 118)
    dA, _ = ci(sel('A_ops_only', 'full', 'detection_rate'))
    dB, _ = ci(sel('B_sec_only', 'full', 'detection_rate'))
    dC, hC = ci(sel('C_alert_corr', 'full', 'detection_rate'))
    dD, hD = ci(sel('D_fusion', 'full', 'detection_rate'))
    sC, hsC = ci(sel('C_alert_corr', 'full', 'subthreshold_detection_rate'))
    sD, hsD = ci(sel('D_fusion', 'full', 'subthreshold_detection_rate'))
    print("H1 fusion detection > single-domain, and > alert correlation on sub-threshold events")
    print(f"   overall  A={dA:.3f}  B={dB:.3f}  C={dC:.3f}  D={dD:.3f}")
    print(f"   sub-threshold-in-both  C={sC:.3f}±{hsC:.3f}   D={sD:.3f}±{hsD:.3f}")
    print(f"   -> {'SUPPORTED' if dD > max(dA, dB) and sD > sC else 'NOT SUPPORTED'}")
    cC, hcC = ci(sel('C_alert_corr', 'full', 'cause_accuracy'))
    cD, hcD = ci(sel('D_fusion', 'full', 'cause_accuracy'))
    print("\nH2 cause-classification accuracy: observation-layer > alert-layer")
    print(f"   C={cC:.3f}±{hcC:.3f}   D={cD:.3f}±{hcD:.3f}"
          f"   -> {'SUPPORTED' if cD > cC and (cD-hcD) > (cC+hcC) else 'NOT SUPPORTED'}")
    aF, haF = ci(sel('D_fusion', 'full', 'attribution_accuracy'))
    aN, haN = ci(sel('D_fusion', 'no_shared_dependency', 'attribution_accuracy'))
    print("\nH3 shared dependency model improves attribution")
    print(f"   with={aF:.3f}±{haF:.3f}   without={aN:.3f}±{haN:.3f}"
          f"   -> {'SUPPORTED' if aF > aN and (aF-haF) > (aN+haN) else 'NOT SUPPORTED'}")
    cB_, hcB_ = ci(sel('D_fusion', 'forced_binary', 'cause_accuracy'))
    eF, _ = ci(sel('D_fusion', 'full', 'effort_per_event'))
    eB_, _ = ci(sel('D_fusion', 'forced_binary', 'effort_per_event'))
    ip, hip = ci(sel('D_fusion', 'full', 'indeterminacy_precision'))
    print("\nH4 explicit indeterminacy improves investigation direction")
    print(f"   cause accuracy  with indeterminate={cD:.3f}±{hcD:.3f}   forced binary={cB_:.3f}±{hcB_:.3f}")
    print(f"   effort/event    with={eF:.2f}h   forced binary={eB_:.2f}h   (parallel investigation costs more)")
    print(f"   indeterminacy precision={ip:.3f}±{hip:.3f}")
    print(f"   -> {'SUPPORTED with a cost' if cD > cB_ and eF > eB_ else 'SUPPORTED' if cD > cB_ else 'NOT SUPPORTED'}")
    lA, _ = ci(sel('A_ops_only', 'full', 'detection_latency'))
    lB, _ = ci(sel('B_sec_only', 'full', 'detection_latency'))
    lD, hlD = ci(sel('D_fusion', 'full', 'detection_latency'))
    worst = max(lA, lB)
    pen = 100 * (lD - worst) / worst
    print("\nH5 fusion latency not materially worse than single-domain")
    print(f"   A={lA:.1f} min  B={lB:.1f} min  D={lD:.1f} min   penalty vs slower single domain = {pen:+.1f}%")
    print(f"   -> {'SUPPORTED' if pen <= 25 else 'NOT SUPPORTED'}  (threshold for 'material' set at 25%)")
    print(f"\nraw results: {out/'signal_fusion_results.csv'}")


if __name__ == "__main__":
    main()
