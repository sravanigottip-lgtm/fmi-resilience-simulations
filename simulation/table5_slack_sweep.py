"""Reproduces Table 5 of the rollback-horizon paper: admission outcomes as the
settlement window (slack between arrival and finality) compresses.

Run from this folder:  python3 table5_slack_sweep.py        (about 2 minutes)
Writes results/table5_slack_sweep.csv. Uses 6,000 replications per band; the
volume-matched control's admission rate is re-matched to the horizon policy's
throughput in each band, as the paper describes.
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import rollback_horizon_sim as m

REPS = 6000
# (label, slack range in minutes, control admission rate matched to the horizon)
BANDS = [
    ("generous", (90.0, 180.0), 0.908),
    ("moderate", (50.0, 110.0), 0.563),
    ("tight", (25.0, 70.0), 0.503),
    ("compressed", (12.0, 35.0), 0.503),
]


def main() -> None:
    out_rows = []
    for name, (lo, hi), rate in BANDS:
        m.SLACK["tight"] = (lo, hi)
        exp = m.run_experiment(reps=REPS, conf=m.cfg(random_rate=rate), workloads=["tight"])
        g = defaultdict(list)
        for r in exp["rows"]:
            g[r["policy"]].append((r["miss_rate"], r["throughput"]))
        mean = {p: (st.mean(a for a, _ in v) * 100, st.mean(b for _, b in v) * 100) for p, v in g.items()}
        out_rows.append([
            f"{lo:.0f} to {hi:.0f}", name,
            round(mean["always_deploy"][0], 2),
            round(mean["rollback_horizon"][0], 2),
            round(mean["rollback_horizon"][1], 1),
            round(mean["random_defer"][0], 2),
            round(mean["random_defer"][1], 1),
        ])
        print(out_rows[-1])
    out = Path(__file__).parent / "results" / "table5_slack_sweep.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slack_min", "band", "unrestricted_miss_pct", "horizon_miss_pct",
                    "horizon_throughput_pct", "volume_matched_control_miss_pct", "control_admit_pct"])
        w.writerows(out_rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
