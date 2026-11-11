# Simulation code for four papers on operational resilience in financial market infrastructure

Code, raw results and a reference implementation that accompany four manuscripts by Sravani Gottipati (Independent Researcher, ORCID 0009-0001-0385-3872):

| Paper | Topic | Code in this repository |
|---|---|---|
| 1 | AI-driven observability and policy-governed automation (evidence-adequacy authorization) | `simulation/settlement_sim.py` (H1, H2, H4, H5), `simulation/dependency_sim.py` (H3) |
| 3 | AI-assisted, risk-adaptive release governance under deterministic policy control | `simulation/release_governance_sim.py` |
| 5 | Cyber-operational signal fusion | `simulation/signal_fusion_sim.py` |
| 6 | Release admission and the rollback horizon | `simulation/rollback_horizon_sim.py`, `simulation/multistage_horizon_sim.py`, `implementation/rollback_horizon.py` |

All studies are **simulations**. No data from any institution is used, and nothing here has been validated in production. The papers state this too.

## Requirements

Python 3.11 or newer. Standard library only; there is nothing to install.
(Checked with Python 3.14.)

## Running

Run each script from the `simulation/` directory. It writes per-replication CSV files to `simulation/results/` and prints the hypothesis summary.

```bash
cd simulation
python3 signal_fusion_sim.py          # Paper 5, a few seconds
python3 release_governance_sim.py     # Paper 3, about 12 seconds
python3 dependency_sim.py             # Paper 1 (H3), about 1 second
python3 settlement_sim.py             # Paper 1 (H1, H2, H4, H5)
python3 rollback_horizon_sim.py       # Paper 6, single-stage model
python3 multistage_horizon_sim.py     # Paper 6, four-stage model, about 1 minute

cd ../implementation
python3 rollback_horizon.py           # Paper 6: worked example and 13 unit tests
```

Every run is seeded, so the same replication count gives the same numbers every time.

## Replication counts

Some scripts default to fewer replications than the paper reports for its main tables. To reproduce a paper's figures, change the replication count in the `if __name__ == "__main__":` block at the bottom of the script.

| Script | Default in code | Paper's main tables |
|---|---|---|
| `signal_fusion_sim.py` | 300 | 300 |
| `release_governance_sim.py` | 300 | 300 |
| `dependency_sim.py` | 3,000 | 3,000 |
| `multistage_horizon_sim.py` | 3,000 | 3,000 |
| `settlement_sim.py` | 400 | 2,000 (Paper 1) |
| `rollback_horizon_sim.py` | 500 | 3,000 to 4,000 (Paper 6) |

With the defaults, the estimates are close to the papers' but not identical, and confidence intervals are wider.

## What was checked

The author's scripts were rerun from a clean copy with the defaults above. Headline results matched the manuscripts:

- Signal fusion: detection of events below threshold in both domains 0.943 (alert correlation 0.000); attribution 0.923 with the shared dependency model vs 0.637 without; latency penalty +57%.
- Dependency model: top-1 root-cause accuracy 20.0%, 47.2% and 70.3% for the three arms; candidate set 4.01 vs 1.81.
- Release governance (300 replications): runs to completion and prints the hypothesis dispositions.
- Reference implementation: 13 unit tests pass.

`settlement_sim.py` and `rollback_horizon_sim.py` were not rerun at the papers' replication counts.

## Files

- `simulation/results/`: raw per-replication outputs produced by the scripts (CSV).
- Comments inside the scripts cite section numbers from earlier drafts of the papers; the structure is the same, but the numbers may differ from the final manuscripts.

## Citing

See `CITATION.cff`. Once the Zenodo record exists, cite the DOI shown on the repository page.

## Licence

MIT (see `LICENSE`).
