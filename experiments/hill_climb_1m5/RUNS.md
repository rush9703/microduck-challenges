# Runs: 1.5 m Hill Climb (`hill-climb-1m5`)

Reference (Pollen, checkpoint 1000 of the recipe): **5.876 s, 5/5 at 0.55 m/s** on the Space;
5.884 s on the Windows PC (CPU).
Board on 2026-10-22: 81 entries, #1 0.756 s (remix entries at 0.30 m/s); last 60 s (0/5).

One section per run, newest first, in the format of the challenge's AGENTS.md "Reporting".

<!-- template
## <name> (commit <sha>: <one change, and why>)
- started <date> on Colab <GPU>; HF <repo>; W&B <url>
- best: checkpoint N, X.XX s, k/5 at V m/s   (reference 5.876 s)
- others: 250 … · 500 … · …
- curves: fell_over …, episode length … of 1000
- published: no | <repo> rank R
- next: <proposal and why>
-->

## base-s1 (commit e463b80: the recipe unchanged, baseline on Colab)
- started 2026-10-22 04:40 UTC on Colab Pro L4 (~4.3 s/iteration, ~3.5 h); HF
  `rushzealot/mjlab-hillclimb1m5-microduck-base-s1`; W&B `.../microduck-arena/runs/eval-base-s1`
- best: checkpoint 750, **5.064 s, 5/5 at 0.55 m/s**   (reference 5.876 s); the Space re-timed it at 5.064 s
- others: 500 8.544 s 5/5 · 1000 5.816 s 5/5 · 1250 18.008 s 4/5 · 1500 6.344 s 5/5 ·
  1750 7.120 s 5/5 · 2000 7.020 s 5/5 · 2250 49.036 s 5/5 (at 0.90 m/s) · 2500 6.708 s 5/5 ·
  2750 17.772 s 4/5 · 2999 6.672 s 5/5. Finished 08:18 UTC; all 13 checkpoints + evals on HF.
- curves at 2254: episode length 978 of 1000, fell_over 0.17/iteration, track_linear_velocity 1.02
  and heading_hold 0.65, both still rising, nan_state 0. The curves keep improving while the Arena
  score worsens after 750: suspect the curricula that tighten after 750 (action_rate up to -1.0,
  CoM DR up to ±15 mm) trading speed for steadiness. Unverified.
- provenance says `dirty: true` only because of an untracked `logs_smoke.txt` in the Colab clone
  (git diff empty); published with `--allow-dirty`. Notebook fixed so it does not recur.
- published 2026-10-22 07:38 UTC: `rushzealot/microduck-hillclimb-base-s1` (public), entered on the
  board as `hillclimb-base-s1`: rank 75/83 policies, 19 by person; player points 110.
- next: candidates, one at a time: `max_slope_deg` 10 -> 12 (an untested lever the guide names);
  or the same commit with `--agent.seed 2`, to see how much of 750's result is luck.
