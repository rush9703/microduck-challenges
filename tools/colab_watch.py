"""Watch a training run on Colab: judge every checkpoint, keep everything off the VM.

Runs next to `uv run train ...` (see colab/microduck_train.ipynb). Every POLL seconds it:

1. evaluates each new checkpoint with the Arena's own evaluator, through the workshop's
   `evaluate_checkpoint` -> `logs/rsl_rl/<folder>/<run>/workshop/model_<N>.json` (+ .onnx), the
   exact files `uv run --group workshop workshop` reads, so a workshop opened later on any
   machine shows these results without re-evaluating;
2. logs the score to W&B (one "eval" run per training run, grouped with it) and, when a
   checkpoint beats the run's best, a video of its median Arena run (tools/render_trace.py);
3. writes `<run>/status.json` (iteration, alive, best) and pushes the run folder to the private
   HF repo (mirrors logs/rsl_rl/, so `hf download <repo> --local-dir logs/rsl_rl` restores it)
   and, if given, copies it to Google Drive.

It exits after the training process is gone and a last pass has run.

    uv run --frozen --group workshop --with wandb --with "imageio[ffmpeg]" python tools/colab_watch.py \
        --name base-s1 --hf-repo <user>/mjlab-hillclimb1m5-microduck-base-s1 --train-pid 1234
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

FRESH_S = 30.0  # a checkpoint younger than this may still be being written


def log(msg: str) -> None:
    print(f"[watch {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # a zombie (exited, not reaped) still answers kill(0)
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True


def last_iteration(train_log: Path) -> tuple[int, int] | None:
    try:
        text = train_log.read_text(errors="ignore")[-200_000:]
    except OSError:
        return None
    found = re.findall(r"Learning iteration (\d+)/(\d+)", text)
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def better(a: dict, b: dict | None) -> bool:
    """Arena order: more seeds finished first, then the lower score."""
    if b is None:
        return True
    return (-a["seeds_finished"], a["score"]) < (-b["seeds_finished"], b["score"])


class Watcher:
    def __init__(self, args):
        from arena.workshop.challenges import read_challenge

        self.args = args
        self.challenge = read_challenge(ROOT / "src" / "microduck_challenges" / args.folder / "challenge.toml")
        self.event = self.challenge.event
        self.exp_dir = ROOT / "logs" / "rsl_rl" / args.folder
        self.train_log = ROOT / "logs" / f"{args.name}.log"
        self.best: dict | None = None
        self.results: dict[tuple[str, int], dict] = {}
        self.wb = None
        self.api = None
        if args.hf_repo:
            from huggingface_hub import HfApi

            self.api = HfApi()
            self.api.create_repo(args.hf_repo, repo_type="model", private=True, exist_ok=True)
            log(f"checkpoints -> https://huggingface.co/{args.hf_repo} (private)")
        if os.environ.get("WANDB_API_KEY") and not args.no_wandb:
            try:
                import wandb

                self.wb = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "microduck-arena"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    name=f"{args.name}-eval",
                    group=args.name,
                    job_type="arena-eval",
                    id=f"eval-{args.name}",
                    resume="allow",
                    config={"event": self.event, "run_name": args.name, "hf_repo": args.hf_repo},
                )
                self.wb.define_metric("checkpoint")
                self.wb.define_metric("arena/*", step_metric="checkpoint")
                log(f"W&B eval run: {self.wb.url}")
            except Exception as exc:  # noqa: BLE001 - monitoring must never stop the watcher
                log(f"W&B disabled: {exc}")

    # ---- discovery -------------------------------------------------------------------------
    def runs(self):
        from arena.workshop.runs import find_runs

        return [r for r in find_runs(ROOT, self.challenge.task) if r.name == self.args.name]

    # ---- one checkpoint --------------------------------------------------------------------
    def judge(self, run, n: int) -> None:
        from arena.workshop.evaluations import evaluate_checkpoint, stored

        body = stored(run, n, None)
        if body is None:
            log(f"{run.id} checkpoint {n}: evaluating with the Arena's rules ...")
            t0 = time.time()
            body = evaluate_checkpoint(ROOT, run, n, self.challenge)
            log(f"  done in {time.time() - t0:.0f}s")
        if "error" in body:
            log(f"  checkpoint {n} FAILED: {body['error'][:300]}")
            self.results[(run.id, n)] = {"checkpoint": n, "error": body["error"][:300]}
            return
        row = {k: body[k] for k in ("checkpoint", "score", "seeds_finished", "seeds_total", "command_vx")}
        row["run_id"] = run.id
        row["per_seed"] = body.get("per_seed")
        self.results[(run.id, n)] = row
        log(f"  checkpoint {n}: {row['score']:.3f}s  {row['seeds_finished']}/{row['seeds_total']}  at {row['command_vx']:.2f} m/s")
        is_best = better(row, self.best)
        if is_best:
            self.best = row
            log(f"  new best of this run: checkpoint {n}")
        self.to_wandb(run, n, row, body, is_best)

    def to_wandb(self, run, n, row, body, is_best) -> None:
        if self.wb is None:
            return
        try:
            import wandb

            data = {
                "checkpoint": n,
                "arena/score_s": row["score"],
                "arena/seeds_finished": row["seeds_finished"],
                "arena/command_vx": row["command_vx"],
            }
            if is_best and not self.args.no_video:
                from render_trace import render_body

                video = run.path / "workshop" / f"model_{n}.mp4"
                render_body(body, self.event, video)
                data["arena/best_replay"] = wandb.Video(str(video), format="mp4",
                                                        caption=f"ckpt {n}: {row['score']:.2f}s {row['seeds_finished']}/5")
            self.wb.log(data)
            self.wb.summary.update({"best_checkpoint": self.best["checkpoint"], "best_score_s": self.best["score"],
                                    "best_seeds_finished": self.best["seeds_finished"]})
        except Exception:  # noqa: BLE001
            log("W&B/video error (ignored):\n" + traceback.format_exc(limit=3))

    # ---- persistence -----------------------------------------------------------------------
    def write_status(self, run, alive: bool) -> None:
        it = last_iteration(self.train_log)
        status = {
            "run_name": self.args.name,
            "run_id": run.id,
            "event": self.event,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "training_alive": alive,
            "iteration": it[0] if it else None,
            "iterations_planned": it[1] if it else None,
            "checkpoints": sorted(run.checkpoints),
            "evaluations": sorted((r for (rid, _), r in self.results.items() if rid == run.id),
                                  key=lambda r: r["checkpoint"]),
            "best": self.best,
            "wandb_eval_url": getattr(self.wb, "url", None),
        }
        (run.path / "status.json").write_text(json.dumps(status, indent=1))
        if self.train_log.exists():
            shutil.copy2(self.train_log, run.path / "train.log")

    def sync(self, run) -> None:
        if self.api is not None:
            try:
                self.api.upload_folder(
                    repo_id=self.args.hf_repo, repo_type="model",
                    folder_path=str(self.exp_dir.parent),  # logs/rsl_rl: the repo mirrors it
                    allow_patterns=[f"{self.args.folder}/{run.id}/**"],
                    commit_message=f"sync {run.id}",
                )
            except Exception as exc:  # noqa: BLE001
                log(f"HF upload failed (will retry): {exc}")
        if self.args.drive:
            try:
                dest = Path(self.args.drive) / "rsl_rl" / self.args.folder / run.id
                shutil.copytree(run.path, dest, dirs_exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                log(f"Drive copy failed (will retry): {exc}")

    # ---- loop ------------------------------------------------------------------------------
    def pass_once(self, alive: bool) -> bool:
        """One pass; True when every checkpoint present has been judged."""
        runs = self.runs()
        done = True
        for run in runs:
            for n in run.checkpoints:
                if (run.id, n) in self.results:
                    continue
                pt = run.path / f"model_{n}.pt"
                if alive and time.time() - pt.stat().st_mtime < FRESH_S:
                    done = False
                    continue
                try:
                    self.judge(run, n)
                except Exception:  # noqa: BLE001
                    log(f"evaluation crashed on {run.id}/{n}:\n" + traceback.format_exc(limit=5))
                    self.results[(run.id, n)] = {"checkpoint": n, "error": "crash"}
            self.write_status(run, alive)
            self.sync(run)
        return done

    def forever(self) -> None:
        while True:
            alive = pid_alive(self.args.train_pid)
            try:
                done = self.pass_once(alive)
            except Exception:  # noqa: BLE001
                log("pass failed:\n" + traceback.format_exc(limit=5))
                done = False
            if not alive and done:
                log("training process gone and every checkpoint judged: final state synced, exiting")
                break
            time.sleep(self.args.poll)
        if self.best:
            log(f"BEST: checkpoint {self.best['checkpoint']} {self.best['score']:.3f}s "
                f"{self.best['seeds_finished']}/5 at {self.best['command_vx']:.2f} m/s")
        if self.wb is not None:
            self.wb.finish()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="the --agent.run-name of the training")
    p.add_argument("--folder", default="hill_climb_1m5", help="challenge folder under src/microduck_challenges")
    p.add_argument("--hf-repo", default=None, help="private HF model repo to mirror the run into")
    p.add_argument("--drive", default=None, help="Google Drive folder to copy the run into")
    p.add_argument("--train-pid", type=int, default=None, help="exit once this process is gone")
    p.add_argument("--poll", type=float, default=60.0)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--once", action="store_true", help="one pass over what is on disk, then exit")
    args = p.parse_args()
    os.chdir(ROOT)
    w = Watcher(args)
    if args.once:
        w.pass_once(alive=False)
        if w.wb is not None:
            w.wb.finish()
    else:
        w.forever()


if __name__ == "__main__":
    main()
