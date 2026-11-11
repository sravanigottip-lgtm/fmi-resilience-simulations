# Reproducing the numbers in the four papers

All scripts use only the Python standard library (tested with Python 3.14) and are deterministic: the same command gives the same output every time. Run each from the `simulation/` folder.

| Paper | Command | What it reproduces | Time |
|---|---|---|---|
| 1 (observability) | `python3 settlement_sim.py` | Jaccard overlap of flagged obligations under TIGHT: 0.077 to 0.090 across onsets (paper reports 0.077 to 0.089); the H2/H4 comparisons | about 5 min |
| 1 (observability) | `python3 dependency_sim.py` | Top-1 attribution 20.0% (no graph), 47.2% (graph), 70.3% (graph with causal filter) | under 1 s |
| 3 (release governance) | `python3 release_governance_sim.py` | Change-failure 6.8% to 5.5%; escaped defects 3.8% to 1.7%; transaction-class detection 0.886 +/- 0.010 (300 replications) | about 15 s |
| 5 (signal fusion) | `python3 signal_fusion_sim.py` | Sub-threshold detection 0.943; attribution accuracy 0.639 to 0.923; detection latency about 7.0 to 11.0 minutes (+57%) | about 5 s |
| 6 (rollback horizon) | see below | Headline: miss rate 0.41% (horizon) against 0.69% (volume-matched control) | about 20 s |
| 6 (rollback horizon) | `python3 table5_slack_sweep.py` | Table 5 (settlement window compression) | about 2 min |

Paper 6 headline (4,000 replications, TIGHT workload):

    python3 -I -c "import rollback_horizon_sim as m; \
    exp=m.run_experiment(reps=4000, workloads=['tight']); \
    import statistics as st, collections; g=collections.defaultdict(list); \
    [g[r['policy']].append(r['miss_rate']) for r in exp['rows']]; \
    print({k: round(st.mean(v)*100,3) for k,v in g.items()})"

Expected: rollback_horizon 0.407, random_defer 0.693, always_deploy 2.665, freeze_window 2.351, error_budget 2.146.

## Notes on the saved result files

- `results/rollback_horizon_results.csv` is exactly the 4,000-replication TIGHT run above (20,000 rows).
- `results/dependency_results.csv`, `h2_results.csv`, `h4_results.csv` and `multistage_results.csv` are regenerated identically by the scripts above.
- `results/release_governance_results.csv`, `signal_fusion_results.csv` and `raw_results.csv` come from runs with more replications than the script defaults, so they have more rows than a default run writes. Their means agree with the papers within the stated confidence intervals (for example, transaction-class detection is 0.880 in the saved file against 0.886 +/- 0.010 in the 300-replication run). Running the scripts as above writes a default-size file over them.
- `results/table5_slack_sweep.csv` is written by `table5_slack_sweep.py` and matches Table 5 as printed in the posted version of the rollback-horizon paper (unrestricted miss 0.43, 1.03, 2.61, 7.37; horizon throughput 90.8, 56.3, 50.5, 50.5; volume-matched control 0.30, 0.22, 0.69, 3.12).
