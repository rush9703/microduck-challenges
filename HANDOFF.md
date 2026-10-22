# HANDOFF

The single source of truth for any agent or machine picking this work up. **Read it whole, then
`src/microduck_challenges/hill_climb_1m5/AGENTS.md`.** Update the *State* and *Next step* sections
(and `experiments/hill_climb_1m5/RUNS.md`) at the end of every session, and commit them.

## Goal

Take part in the [Microduck Arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena),
challenge **1.5 m Hill Climb** (`hill-climb-1m5`, task `Mjlab-HillClimb1m5-MicroDuck`). Train on
**Google Colab Pro**, not a local GPU or HF Jobs. User's goal: "earn some points": a first 5/5
entry, then beat Pollen's reference (5.876 s, 5/5 at 0.55 m/s), then iterate. Other challenges may
follow later (same tooling, another `experiments/<folder>/`).

## State (update every session)

- 2026-10-22: Setup done by the agent. The repo is cloned from `pollen-robotics/microduck-challenges`
  (upstream at `6ff4070`), and the remote is named `upstream`. The tooling is in place: notebook, watcher,
  renderer, status tool, `.env`. colab-mcp is configured in `opencode.json`. uv is installed on the
  Windows PC. Pollen's reference scored **5.884 s 5/5** on the PC's CPU (Space: 5.876 s).
- 2026-10-22 (later): `.env` filled and checked (HF `rushzealot` write token OK; W&B entity is the
  team `rush000003-arizona-state-university`, project `microduck-arena` created; GitHub `rush9703`).
  Fork `rush9703/microduck-challenges` created, `origin` set, `main` pushed (push uses `GITHUB_TOKEN`
  from `.env` via an http extraheader, never stored in git config).
- colab-mcp: the browser opened Colab under the user's *other* (non-Pro) Google account; the MCP server
  then closed after an aborted call and needs an OpenCode restart. The Pro account must be the one in
  the Colab tab (`?authuser=<pro email>` in the URL, or make it the browser's first-signed-in account).
  Colab Secrets and Drive must be set up under the **Pro** account.
- No publication yet.
- **2026-10-22 04:40 UTC: run `base-s1` TRAINING on Colab Pro (L4)**, commit `e463b80`, launched by the
  agent through colab-mcp (Colab tab under the Pro account; Drive mounted; secrets set). ~4.3 s/iter,
  so ~3.5 h for 3000 iterations. Checks before it on Colab: pytest 66 passed, GPU smoke OK, reference
  scored 5.880 s 5/5. The first start crashed (rsl_rl W&B writer, see Gotchas); fixed by TensorBoard + mirror.
  - HF: `rushzealot/mjlab-hillclimb1m5-microduck-base-s1` (also holds the failed start's folder
    `2026-10-22_04-34-39_base-s1`, no checkpoints, ignored by status.py)
  - W&B: curves `.../microduck-arena/runs/train-base-s1`, Arena scores `.../runs/eval-base-s1`
    (entity `rush000003-arizona-state-university`)
  - Follow it: `tools/status.py` (below). Colab's notebook cell "7. Monitor loop" keeps the kernel busy.

## Next step

1. User:
   - fill `.env`;
   - create the W&B account;
   - fork `pollen-robotics/microduck-challenges` to their GitHub;
   - add these Colab secrets: `HF_TOKEN`, `WANDB_API_KEY`, `GITHUB_USERNAME`, optional `WANDB_ENTITY`.
2. ~~Agent: set `origin`, push `main`.~~ Done.
3. ~~Run `base-s1`~~ running (see State). When it ends: report per the challenge AGENTS.md, fill
   `experiments/hill_climb_1m5/RUNS.md`, propose the next single change. Path B for later runs if
   colab-mcp is not available:
   - **Path A**: colab-mcp from the agent.
   - **Path B**: the user opens `https://colab.research.google.com/github/rush9703/microduck-challenges/blob/main/colab/microduck_train.ipynb?authuser=<pro email>` and runs it.
4. Supervise with `tools/status.py`. Report each run in the format of the challenge's AGENTS.md.
5. On the user's yes: publish the best 5/5 checkpoint with `--enter`.

## Permissions granted by the user

- The agent may, without asking: edit knobs/env, run tests, commit, **push to the user's own fork**
  (`origin`), start Colab trainings (they are paid by the user's Colab Pro units, so state each
  run's hypothesis first), read W&B/HF/Arena.
- **Always ask first**: `publish` (makes a repo public), `--enter` (Arena entry), HF Jobs (billed),
  anything touching `upstream`/pollen-robotics, deleting HF repos or runs.
- Docs and commit messages in English. Talk to the user in Chinese.
- Never print, commit or paste secrets. `.env` is git-ignored. On Colab, secrets live in the Secrets panel.
- Ford PC: personal accounts only, no Ford data. If the network blocks something, use the
  `ford-network-proxy-vpn` skill. Never bypass logins: ask the user to log in.

## How this setup works

```
this PC / any PC (agent)          GitHub fork (origin)              Colab Pro (L4 GPU)
 edit next_run.toml + knobs  --->  commit = one run's recipe  --->  colab/microduck_train.ipynb
 tools/status.py  <---------------------------------------------   tools/colab_watch.py pushes:
   reads HF status.json, W&B, Arena board                          - HF private repo <hf>/<prefix>-<name>
                                                                   - W&B curves + Arena score + video
                                                                   - Drive MyDrive/microduck backup
```

- **`experiments/hill_climb_1m5/next_run.toml`**: the run the notebook launches. Its fields are `name`, `seed`,
  `max_iterations`, `num_envs` and `hypothesis`. Change it in the same commit as the change it tests.
- **`colab/microduck_train.ipynb`**: the steps, in order:
  1. secrets;
  2. Drive;
  3. clone the fork and check out `main`;
  4. `uv sync --frozen --group workshop`;
  5. pytest and a GPU smoke test (64 envs, 2 iterations);
  6. start training in the background, resuming from the HF repo if this run already has checkpoints;
  7. a monitor loop;
  8. show the best replay.
- **`tools/colab_watch.py`**: runs beside training. It evaluates each checkpoint through the
  workshop's own `evaluate_checkpoint`, which writes `<run>/workshop/model_N.json` and `.onnx`, the
  same files the official workshop UI reads. It logs to W&B: the `arena/score_s` and
  `arena/seeds_finished` curves, plus an `arena/best_replay` video. It writes `<run>/status.json` and
  `train.log`, then uploads the run folder to HF (mirroring `logs/rsl_rl/`) and Drive.
- **`tools/render_trace.py`**: offscreen MuJoCo render of an Arena trace, with the visual meshes put back.
- **`tools/status.py`**: works on any machine. It prints the run's state, every checkpoint score,
  the rank the best checkpoint would take on the live board, and the W&B curves, each with how to read it.
  With `--pull` it downloads the run into `logs/rsl_rl/` so `publish` can run locally.

### Commands on the Windows PC (PowerShell)

uv is at `%APPDATA%\Python\Python312\Scripts` (on the user PATH; new shells see it). `uv sync` of
the full project is **Linux-oriented** (torch, mujoco-warp). On Windows, use these lighter commands:

```powershell
# status of the current run (reads .env)
uv run --no-project --python 3.12 --with huggingface_hub --with wandb python tools/status.py
uv run --no-project --python 3.12 --with huggingface_hub --with wandb python tools/status.py --all
# score an ONNX with the Arena's rules on CPU (~1 min), using the Arena 0.5.0 wheel pinned in pyproject.toml
uv run --no-project --python 3.12 --with "microduck-arena @ https://github.com/pollen-robotics/microduck-arena/releases/download/v0.5.0/microduck_arena-0.5.0-py3-none-any.whl" arena score <policy.onnx> --event hill-climb-1m5
```

Publishing also runs on Colab. `uv run publish --run ... --checkpoint N --repo <hf>/microduck-<name> --no-private --enter`
needs the project environment, which is installed there. The status tool's `--pull` plus a
Linux/WSL box also works. **Publishing needs the user's yes.**

## Gotchas

- **Train with `--agent.logger tensorboard`, never `wandb`**: rsl_rl's W&B writer calls
  `wandb.Settings(start_method="thread")`, which the wandb in `uv.lock` (0.30) rejects with a pydantic
  `extra_forbidden` error at iteration 0. `tools/colab_watch.py` mirrors the TensorBoard scalars into a
  W&B run `<name>-train` (group `<name>`) instead; Arena scores go to `<name>-eval`.
- Colab Pro has no background execution. If the tab or the VM dies, rerun the notebook: cell 6
  resumes from the latest checkpoint in the HF repo. **On resume, the curricula restart from stage 0**
  (they are keyed on the step counter of the new process). Prefer finishing a run in one session:
  3000 iterations take about 1–2.5 h on an L4.
- Checkpoints come every 250 iterations, and the best one is often *not* the last. The watcher judges them all.
- Scores off the Space are indicative. The x86 Colab, and this PC (5.884 vs 5.876 s), are close.
- `publish` refuses a run started from uncommitted changes. The notebook asserts a clean clone.
- Every run's commit must be pushed before it is published, so the policy's code link resolves.
- Do not modify `pyproject.toml` or `uv.lock`, or other challenges' folders (challenge AGENTS.md rules).
- Upstream moves. To sync, run `git fetch upstream && git merge upstream/main`, and only between runs.
- colab-mcp (`opencode.json`) needs OpenCode to have been **restarted** after it was added. Its first tool
  is `open_colab_browser_connection`, which opens a Colab tab that the user must have logged into. It is
  officially tested on Gemini CLI, Claude Code and Windsurf, not OpenCode. If it fails, use Path B.
