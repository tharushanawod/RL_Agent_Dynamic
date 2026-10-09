# Raw-state tabular Q-learning for time-dependent bus network design (Mandl)

The agent builds 24 sequential hourly service plans P_1..P_24 (routes, frequencies,
buses) for Mandl's Swiss network. It uses tabular Q-learning on the **complete raw
state**: no compression, aggregation or discretisation. The point of the experiment
is to measure how badly this raw state space explodes.

## Run

```bash
pip install -r requirements.txt          # numpy required; matplotlib/scipy/psutil/pandas optional
# 1. insert real data (see transit_rl/data/input/README.md)
# 2. full experiment: 5 replications x 10,000 episodes
python -m transit_rl.main
# quick check
python -m transit_rl.main --episodes 50 --replications 1 --log-every 10
# code check on SYNTHETIC random data (not Mandl)
python tests/smoke_test.py
```

## Architecture

| # | Topic | Implementation |
|---|---|---|
| 1 | Episode | One full day of 24 hourly steps, starting at the morning peak: `od_hour_08` → … → `od_hour_23` → `od_hour_00` → … → `od_hour_07` (`OD_START_FILE_HOUR` in config). |
| 2 | RL step | One agent action (`TransitEnvironment.step`). An hour takes many steps. |
| 3 | State | `(t, OD_t, R_current, P_current, P_{t-1}, F_remaining)` (`environment/state.py`). OD is the full 15×15 tuple, the route is the full stop sequence, both plans are full `(route, frequency[, kept])` records. |
| 4 | Action | `Action(type, arg)`: `ADD_STOP(j)`, `REMOVE_LAST_STOP`, `END_ROUTE`, `INCREASE_FREQUENCY(k)`, `DECREASE_FREQUENCY(k)`, `KEEP_ROUTE(k)`, `REMOVE_ROUTE(k)`, `FINISH_PLAN`; `k` = index in the current plan. |
| 5 | Passenger assignment | Re-run after `END_ROUTE`, `INCREASE/DECREASE_FREQUENCY`, `REMOVE_ROUTE` and at `FINISH_PLAN`. Not re-run during `ADD_STOP`/`REMOVE_LAST_STOP` unless `ASSIGN_AFTER_ROUTE_CONSTRUCTION_STEPS`. |
| 6 | OD never reduced | Matrices are loaded read-only. Every assignment starts from the original OD_t plus the current plan. Results go into separate `assigned/unserved/direct/transfer_demand` arrays. |
| 7 | Fleet | `v_k = ceil(f_k·T_k/60)`, `F_remaining = MAX_FLEET − Σv_k`, recomputed from the plan. Removing a route or lowering a frequency returns its buses automatically. Masking blocks increases that would exceed the fleet. |
| 8 | Previous plan | At hour start `previous_plan = P[t−1]`. Plans are immutable tuples (equivalent to deepcopy), so `P[t−1]` can never be changed. At `FINISH_PLAN`, `P[t] = current_plan`. |
| 9 | Transition cost | Hungarian matching on normalised (reversal-aware) Levenshtein distance. `C_route` and `C_freq` both lie in [0,1]. `C = w_r·C_route + w_f·C_freq` (`environment/transition_cost.py`). |
| 10 | Q-table | `dict[state_key][action] → q` plus `visits[state_key][action] → n`. A state is added only when an action is taken in it. `state_to_key` returns the raw tuples unchanged (lossless). Identical sub-objects are shared by reference to save memory. |
| 11 | ε-greedy | Chooses only from `env.valid_actions()`. Greedy ties are broken randomly with the seeded RNG. |
| 12 | Hour t → t+1 | `FINISH_PLAN` (or the step limit) → evaluate P_t → store the hourly record → `hour += 1` → `previous_plan = P_t`, `current_plan = copy of P_t` (kept flags cleared). The Q-update bootstraps across the hour boundary, so hours are linked through the return. Hour 24 is terminal. |

All modelling assumptions are marked `# ASSUMPTION:` in `transit_rl/config.py`.

## Outputs (`transit_rl/results/`)

```
config_used.json, replication_summary.csv, best_replication.json,
best_q_table.csv, best_hourly_service_plans.json, plots/
replication_<r>/
    q_table.csv, od_matrices.json        # load with utils.qtable_io.load_q_table_csv
    learning_curve.csv                   # per episode
    progress_log.csv                     # every LOG_EVERY episodes, incl. memory
    hourly_service_plans.csv / .json     # greedy evaluation episode
    state_space_statistics.json          # visit counts, revisits, size estimate
    plots/*.png                          # if matplotlib installed
```

Best replication = highest greedy evaluation reward (tie-break: mean training reward
over the last logging window).

## Resource warning

On synthetic test data, an episode takes about 0.3 s and creates about 750 new
states. A full replication is therefore about 1 h and about 7–8 M Q-table rows.
Expect several GB of RAM per replication. With `QTABLE_CSV_OD_MODE="inline"`
(the default, as specified) expect about 1.4 KB per CSV row, i.e. roughly 10 GB of
CSV per replication. `--od-mode reference` stores each OD matrix once
(`REF:<hour>` + `od_matrices.json`). It is equally lossless and about 5× smaller.
