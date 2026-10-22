"""Where do things stand? Run from ANY machine with the repo and a filled .env.

Reads, never writes (except --pull, which downloads into logs/rsl_rl/):
  * the private HF checkpoint repos <HF_USERNAME>/<HF_CKPT_PREFIX>-<name>: status.json written
    by tools/colab_watch.py on Colab (iteration, training alive, every checkpoint's Arena score);
  * W&B (optional): the training curves that matter, with a one-line reading of each;
  * the live Arena leaderboard: where the best checkpoint would rank, and our own entries.

    uv run --no-project --python 3.12 --with huggingface_hub --with wandb python tools/status.py [--name base-s1] [--pull]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
ARENA = "https://pollen-robotics-microduck-arena.hf.space"
CURVES = {  # tag -> (what it is, how to read it)
    "Train/mean_episode_length": ("episode length (steps, 1000 = full 20 s)", "healthy when climbing toward 1000"),
    "Episode_Termination/fell_over": ("episodes ending in a fall, per iteration (out of ~100 at the start)", "healthy when falling toward 0"),
    "Episode_Reward/track_linear_velocity": ("speed-command tracking reward", "should keep growing; flat while total rises = farming"),
    "Episode_Termination/nan_state": ("NaN physics terminations", "should be ~0"),
    "Train/mean_reward": ("total reward", "context only: the Arena score is what counts"),
}


def load_env() -> dict:
    env = {}
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                if v.strip():
                    env[k.strip()] = v.strip()
    for k, v in env.items():
        os.environ.setdefault(k, v)  # a real environment variable wins over .env
    return dict(os.environ)


def next_run(folder: str) -> dict:
    p = ROOT / "experiments" / folder / "next_run.toml"
    return tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def leaderboard(event: str) -> list[dict]:
    with urllib.request.urlopen(f"{ARENA}/api/events/{event}/leaderboard", timeout=60) as r:
        return json.load(r)["rows"]


def rank_of(score: float, finished: int, rows: list[dict]) -> int:
    return 1 + sum(1 for r in rows if (-r["seeds_finished"], r["score"]) < (-finished, score))


def hf_status(api, repo: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    out = []
    for f in api.list_repo_files(repo):
        if f.endswith("/status.json"):
            out.append(json.loads(Path(hf_hub_download(repo, f, force_download=True)).read_text()))
    return out


def wandb_curves(env: dict, name: str) -> None:
    if not env.get("WANDB_API_KEY"):
        print("  (W&B: no WANDB_API_KEY in .env)")
        return
    try:
        import wandb
    except ImportError:
        print("  (W&B: add `--with wandb` to read curves)")
        return
    api = wandb.Api()
    path = f"{env.get('WANDB_ENTITY') or api.default_entity}/{env.get('WANDB_PROJECT', 'microduck-arena')}"
    runs = [r for r in api.runs(path, filters={"group": name}) if r.job_type == "curves"]
    if not runs:
        print(f"  (W&B: no curves run in group {name!r} of {path})")
        return
    for run in runs:
        print(f"  W&B training curves {run.name} [{run.state}] {run.url}")
        hist = list(run.scan_history(keys=["iteration", *CURVES]))
        for tag, (what, how) in CURVES.items():
            pts = [(h["iteration"], h[tag]) for h in hist if h.get(tag) is not None]
            if not pts:
                continue
            idx = sorted({int(i * (len(pts) - 1) / 5) for i in range(6)})
            trend = " -> ".join(f"{pts[i][1]:.3g}@{pts[i][0]}" for i in idx)
            print(f"    {tag:<38} {trend}\n      {what}; {how}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default=None)
    p.add_argument("--name", default=None, help="run name; default: experiments/<folder>/next_run.toml, or all")
    p.add_argument("--all", action="store_true", help="every checkpoint repo with the prefix")
    p.add_argument("--pull", action="store_true", help="download the run(s) into logs/rsl_rl/ for publish")
    a = p.parse_args()
    env = load_env()
    folder = a.folder or env.get("CHALLENGE_FOLDER", "hill_climb_1m5")
    event = env.get("ARENA_EVENT", "hill-climb-1m5")
    user, prefix = env.get("HF_USERNAME"), env.get("HF_CKPT_PREFIX", "mjlab-hillclimb1m5-microduck")
    if not user or not env.get("HF_TOKEN"):
        sys.exit("fill HF_USERNAME and HF_TOKEN in .env first (see .env.example)")

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=env["HF_TOKEN"])
    name = None if a.all else (a.name or next_run(folder).get("name"))
    repos = sorted(m.id for m in api.list_models(author=user, search=prefix))
    if name:
        repos = [r for r in repos if r == f"{user}/{prefix}-{name}"]
    if not repos:
        print(f"no checkpoint repo yet for {name or prefix!r} under {user}")

    rows = leaderboard(event)
    print(f"Arena {event}: {len(rows)} entries, #1 {rows[0]['score']:.3f}s ({rows[0]['user']})" if rows else "board empty")
    mine = [r for r in rows if r["user"] == user]
    for r in mine:
        print(f"  our entry {r['policy_name']}: {r['score']:.3f}s {r['seeds_finished']}/5 rank {rank_of(r['score'], r['seeds_finished'], rows)}")

    for repo in repos:
        print(f"\n== {repo}")
        curves_shown = set()
        for st in hf_status(api, repo):
            if not st.get("checkpoints"):
                continue  # a start that failed before its first checkpoint
            alive = "TRAINING" if st["training_alive"] else "stopped"
            print(f"  run {st['run_id']} [{alive}] iteration {st['iteration']}/{st['iterations_planned']}  (updated {st['updated_utc']})")
            for e in st["evaluations"]:
                if "error" in e:
                    print(f"    ckpt {e['checkpoint']:>5}: FAILED {e['error'][:120]}")
                else:
                    print(f"    ckpt {e['checkpoint']:>5}: {e['score']:7.3f}s  {e['seeds_finished']}/{e['seeds_total']}  at {e['command_vx']:.2f} m/s")
            b = st.get("best")
            if b:
                print(f"  BEST ckpt {b['checkpoint']}: {b['score']:.3f}s {b['seeds_finished']}/5 -> would rank "
                      f"{rank_of(b['score'], b['seeds_finished'], rows)} of {len(rows) + 1} (reference 5.876s)")
            if st.get("wandb_eval_url"):
                print(f"  W&B eval: {st['wandb_eval_url']}")
            if st["run_name"] not in curves_shown:
                curves_shown.add(st["run_name"])
                wandb_curves(env, st["run_name"])
        if a.pull:
            snapshot_download(repo, local_dir=str(ROOT / "logs" / "rsl_rl"), token=env["HF_TOKEN"])
            print(f"  pulled into logs/rsl_rl/")


if __name__ == "__main__":
    main()
