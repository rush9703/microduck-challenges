# Hill Climb leaderboard: what the top entries do (read 2026-10-22)

Source: `GET /api/events/hill-climb-1m5/leaderboard` (85 rows, 22 people), each entry's public HF
repo (`manifest.json`, `README.md`) and, when it names one, the training fork.

Score distribution: 23 entries <= 0.8 s, 56 <= 1.0 s, 68 <= 1.5 s, 76 <= 5 s. Ours (base-s1/750,
5.064 s) is 76th. The official recipe walks at ~0.25 m/s actual; the board runs at 1-2 m/s.

## Three families

1. **Seed-overfit "branch tables" (#1-#10, ~0.75-0.78 s, cmd 0.3).** Lineage dashi/r760 ->
   jrubiosainz/game-hillclimber-remix* -> austindixson, DenRakEiw. The ONNX graph is edited, not
   trained: a step counter is smuggled through an action slot (head_roll), the start is
   "fingerprinted" to recognise which of the Arena's 5 fixed official seeds is running, and per-seed
   tables of joint-target offsets are searched by CMA-ES/SPSA **on those official seeds**. The
   descriptions say "simulation only", "in-sample". This tunes the evaluator's seeds, not a gait; it
   would not transfer to a robot or to new seeds. Within the rules as written, but not a training
   strategy. Not pursued unless the user asks.
2. **ES fine-tuning on the Arena scorer (#13-#55, ~0.78-0.98 s).** juicenv/es9, suraj-phanindra/hc*,
   namenu, eli-aifund, RhodWeo (793 ES generations from HannesVonEssen/microduck-running): take an
   already fast policy and perturb its weights, keeping what `arena score` likes. Also optimised on
   the official seeds, but on the network weights, so less brittle than (1).
3. **Plain PPO with a speed recipe (#64 TheSauceman, 1.236 s, cmd 0.15; trained from scratch,
   3000 iterations, 2048 envs).** Fork `SteveTarter/microduck-challenges`, commit `4c8874141`,
   documented commit by commit. This is the transferable part, below.

## TheSauceman's recipe ("climb1" + "climb2", ported from his sprint "dash" series)

Each change against this folder's recipe, with his reason:

| Change | From -> to | Why |
|---|---|---|
| new `forward_speed` reward (`microduck_mdp.forward_speed_reward`, exists in our locked package) | weight 12, `vel_ref` 3.0 | rewards raw forward speed whatever the command: the Arena keeps the best of the sweep, and `track_linear_velocity` caps reward at the command |
| `forward_reward` (track_linear_velocity) | 2.0 -> 0.5 | the speed term above drives speed now |
| `upright_reward` | 2.0 -> 1.0 | |
| upright std² | 0.05 -> 0.2 | a 20° forward lean keeps ~56% instead of ~10%; the Arena only calls a fall past 60° for 0.3 s |
| `step_reward` (air_time) | 3.0 -> 1.0 | long swings (0.125-0.3 s) were a brake on cadence |
| `track_angular_velocity` | 2.0 -> 0.5 | also prices roll/pitch rates a fast gait needs |
| `body_ang_vel`, `angular_momentum` | -0.05, -0.02 -> 0 | "motion blockers" |
| head_pose_tracking weight, head_pose_bias curriculum | 2.0, 0->3 -> 0, 0 | head free to pitch forward as a counterweight |
| standing envs | 2% -> 25% curriculum -> 0% | the Arena never commands 0, and the speed reward ignores the command |
| action_rate curriculum | to -1.0 by 1500 -> capped at -0.4 by 750 | strong smoothing fought the sharp motion; also his sprint "ar04" fixed a post-1500 slowdown |
| `ENABLE_SYMMETRY` | False -> True | mirror L/R gait, cancels steady yaw drift at speed |
| heading_hold std | 0.4 -> 0.15 | at 0.4 a 3° drift was free; at speed it reaches the walls |

His sequence on the hill: 7.71 s (walker) -> climb1 (speed recipe; finished on only 4 checkpoints,
yaw drift into walls) -> climb2 (+ symmetry + heading std 0.15) -> **1.236 s**.

Also learned on his sprint (same robot, flat): velocity pushes OFF was worse ("nopush"); widening
pose std for long strides was worse (reverted "dash5"); `upright` std² 0.2 helped.

Matches our own observation: base-s1's Arena score got worse after iteration 750 exactly as the
action_rate and standing-env curricula tightened.

## Possible next runs (one hypothesis each; user decides)

- `speed1`: port his climb2 recipe whole (bundled on purpose, it is a known-good set): the table
  above. Expected: ~1.2-1.5 s if it reproduces; risk: walls/falls, then judge every checkpoint.
- `ar04`: only cap the action_rate curriculum at -0.4 and drop standing envs (small, isolates the
  "post-750 slowdown" we saw). Expected: later checkpoints stop degrading, still ~5 s.
- Later: ES fine-tuning on the Arena scorer (family 2) from our best PPO policy; CPU-only, can run
  on Colab CPU or this PC.
