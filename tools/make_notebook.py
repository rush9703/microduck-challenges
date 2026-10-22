"""Generates colab/microduck_train.ipynb. Edit HERE, then: python tools/make_notebook.py colab/microduck_train.ipynb"""
import json, sys
from pathlib import Path

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})

md(r"""
# Microduck Arena - Colab training worker

Trains the run described in `experiments/<challenge>/next_run.toml` of **your fork**, judges every
checkpoint with the Arena's own evaluator, and keeps everything off this VM:

| Where | What |
|---|---|
| W&B project `microduck-arena` | live training curves (group = run name) + an `-eval` run: Arena score per checkpoint and a replay video of each new best |
| HF private repo `<you>/<prefix>-<name>` | checkpoints, TensorBoard files, `workshop/model_N.json` evaluations, `status.json`, `train.log` |
| Google Drive `MyDrive/microduck/` | the same run folder, as a backup |

**Before the first run** (once): Runtime -> Change runtime type -> **L4 GPU** (or A100).
Left bar -> key icon (Secrets) -> add, with *Notebook access* on:
`HF_TOKEN` (write token), `WANDB_API_KEY`, `GITHUB_USERNAME`; optional `WANDB_ENTITY`.

Then **Runtime -> Run all** and leave the tab open (the last cell keeps the session busy until
the training ends). Nothing here publishes or enters the Arena: that is done afterwards, on purpose.
""")

code(r'''
#@title 1. Settings and secrets
CHALLENGE = "hill_climb_1m5"   #@param {type:"string"}
BRANCH = "main"                #@param {type:"string"}
USE_DRIVE = True               #@param {type:"boolean"}
RUN_TESTS_AND_SMOKE = True     #@param {type:"boolean"}
SCORE_REFERENCE = False        #@param {type:"boolean"}
ALLOW_RESUME = True            #@param {type:"boolean"}

import os, subprocess
from google.colab import userdata

def secret(key, required=True):
    try:
        value = userdata.get(key)
    except Exception:
        value = None
    if required and not value:
        raise SystemExit(f"Missing Colab secret {key}: key icon in the left bar, add it, enable notebook access.")
    return value or ""

os.environ["HF_TOKEN"] = secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secret("WANDB_API_KEY", required=False)
entity = secret("WANDB_ENTITY", required=False)
if entity:
    os.environ["WANDB_ENTITY"] = entity
    os.environ["WANDB_USERNAME"] = entity   # rsl_rl's W&B writer reads the entity from here
os.environ["WANDB_PROJECT"] = "microduck-arena"
os.environ["MUJOCO_GL"] = "egl"
os.environ["UV_HTTP_TIMEOUT"] = "600"
GITHUB_USER = secret("GITHUB_USERNAME")
REPO = "/content/microduck-challenges"
print("W&B:", "on" if os.environ["WANDB_API_KEY"] else "OFF (TensorBoard only)")
!nvidia-smi --query-gpu=name,memory.total --format=csv
''')

code(r'''
#@title 2. Google Drive (backup)
DRIVE_DIR = ""
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_DIR = "/content/drive/MyDrive/microduck"
    os.makedirs(DRIVE_DIR, exist_ok=True)
print("Drive backup:", DRIVE_DIR or "off")
''')

code(r'''
#@title 3. Get the fork at the commit to train
url = f"https://github.com/{GITHUB_USER}/microduck-challenges"
if not os.path.isdir(f"{REPO}/.git"):
    !git clone -q {url} {REPO}
%cd {REPO}
!git fetch -q origin && git checkout -q {BRANCH} && git pull -q --ff-only origin {BRANCH}
!git log -1 --format="training commit %h: %s"
dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
assert not dirty, "the clone has local changes: publish would refuse this run\n" + dirty
''')

code(r'''
#@title 4. Install (uv + the locked environment; ~5 min the first time per VM)
!pip -q install uv
!uv sync --frozen --group workshop 2>&1 | tail -3
''')

code(r'''
#@title 5. Tests, GPU smoke test, reference score
import tomllib
cfg = tomllib.load(open(f"experiments/{CHALLENGE}/next_run.toml", "rb"))
TASK, EVENT = cfg["task"], cfg["event"]
if RUN_TESTS_AND_SMOKE:
    !uv run --frozen --group workshop pytest -q 2>&1 | tail -3
    !uv run --frozen train {TASK} --env.scene.num-envs 64 --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke 2>&1 | grep -E "Learning iteration|Error|error" | tail -3
    !rm -rf logs/rsl_rl/{CHALLENGE}/*_smoke
if SCORE_REFERENCE:
    ref = EVENT
    !uv run --frozen hf download pollen-robotics/microduck-{ref}-reference policy.onnx --local-dir logs/reference/{ref} -q
    !uv run --frozen --group workshop arena score logs/reference/{ref}/policy.onnx --event {ref} | tail -8
''')

code(r'''
#@title 6. Launch training + watcher (background processes)
import glob, re, sys
from huggingface_hub import HfApi, snapshot_download

NAME = cfg["name"]
HF_USER = HfApi().whoami()["name"]
CKPT_REPO = f"{HF_USER}/{cfg['hf_ckpt_prefix']}-{NAME}"
MAX_IT = int(cfg["max_iterations"])
print(f"run {NAME!r}: {cfg['hypothesis']}")

def latest_checkpoint():
    best = None
    for path in glob.glob(f"logs/rsl_rl/{CHALLENGE}/*_{NAME}/model_*.pt"):
        n = int(re.search(r"model_(\d+)\.pt$", path).group(1))
        if best is None or n > best[0]:
            best = (n, os.path.basename(os.path.dirname(path)))
    return best

resume = []
iterations = MAX_IT
if ALLOW_RESUME:
    if latest_checkpoint() is None:
        try:   # a previous VM may have pushed part of this run
            snapshot_download(CKPT_REPO, local_dir="logs/rsl_rl")
        except Exception:
            pass
    found = latest_checkpoint()
    if found:
        n, run_dir = found
        if n >= MAX_IT - 1:
            raise SystemExit(f"{NAME} already finished (checkpoint {n}). Set a new name in next_run.toml.")
        iterations = MAX_IT - n
        resume = ["--agent.resume", "True", "--agent.load-run", run_dir, "--agent.load-checkpoint", f"model_{n}.pt"]
        print(f"RESUMING {run_dir} from checkpoint {n} for {iterations} more iterations "
              "(note: curricula restart from stage 0 on resume)")

logger = "wandb" if os.environ["WANDB_API_KEY"] else "tensorboard"
os.makedirs("logs", exist_ok=True)
train_cmd = ["uv", "run", "--frozen", "train", TASK,
             "--env.scene.num-envs", str(cfg["num_envs"]), "--agent.max-iterations", str(iterations),
             "--agent.seed", str(cfg["seed"]), "--agent.logger", logger, "--agent.run-name", NAME,
             "--agent.wandb-project", os.environ["WANDB_PROJECT"], *resume]
train = subprocess.Popen(train_cmd, stdout=open(f"logs/{NAME}.log", "a"), stderr=subprocess.STDOUT,
                         start_new_session=True, env={**os.environ, "WANDB_RUN_GROUP": NAME})
watch_cmd = ["uv", "run", "--frozen", "--group", "workshop", "--with", "wandb", "--with", "imageio[ffmpeg]",
             "python", "tools/colab_watch.py", "--name", NAME, "--folder", CHALLENGE,
             "--hf-repo", CKPT_REPO, "--train-pid", str(train.pid)]
if DRIVE_DIR:
    watch_cmd += ["--drive", DRIVE_DIR]
watch = subprocess.Popen(watch_cmd, stdout=open(f"logs/{NAME}.watch.log", "a"), stderr=subprocess.STDOUT,
                         start_new_session=True)
print("training pid", train.pid, "| watcher pid", watch.pid)
print("checkpoints ->", f"https://huggingface.co/{CKPT_REPO}")
''')

code(r'''
#@title 7. Monitor (keeps the session alive; safe to stop and re-run, training continues)
import json, time
from IPython.display import clear_output

def tail(path, n=4):
    try:
        return open(path, errors="ignore").read().splitlines()[-n:]
    except OSError:
        return []

def wandb_url():
    text = "\n".join(tail(f"logs/{NAME}.log", 100000))
    m = re.findall(r"View run at (\S+)", text)
    return m[0] if m else None

t0 = time.time()
while train.poll() is None or watch.poll() is None:
    clear_output(wait=True)
    log = "\n".join(tail(f"logs/{NAME}.log", 400))
    it = re.findall(r"Learning iteration (\d+)/(\d+)", log)
    print(f"run {NAME}  |  {(time.time() - t0) / 60:.0f} min in this cell  |  "
          f"iteration {it[-1][0] + '/' + it[-1][1] if it else '-'}  |  training {'running' if train.poll() is None else 'ended ' + str(train.returncode)}")
    print("W&B training:", wandb_url())
    for st in glob.glob(f"logs/rsl_rl/{CHALLENGE}/*_{NAME}/status.json"):
        s = json.load(open(st))
        print("W&B eval:   ", s.get("wandb_eval_url"))
        for e in s["evaluations"]:
            print(f"  ckpt {e['checkpoint']:>5}: " + (f"{e['score']:.3f}s {e['seeds_finished']}/5 at {e['command_vx']:.2f} m/s" if "score" in e else "FAILED"))
        if s.get("best"):
            print(f"  best: ckpt {s['best']['checkpoint']} {s['best']['score']:.3f}s {s['best']['seeds_finished']}/5")
    print("\n--- watcher ---"); print("\n".join(tail(f"logs/{NAME}.watch.log", 5)))
    if train.poll() not in (None, 0):
        print("\n--- TRAINING FAILED, end of its log ---"); print("\n".join(tail(f"logs/{NAME}.log", 30)))
    time.sleep(60)
print("done: training exit", train.returncode, "| watcher exit", watch.returncode)
''')

code(r'''
#@title 8. Watch the best checkpoint's Arena replay
from IPython.display import HTML, display
import base64
videos = sorted(glob.glob(f"logs/rsl_rl/{CHALLENGE}/*_{NAME}/workshop/model_*.mp4"), key=os.path.getmtime)
if videos:
    print(videos[-1])
    display(HTML(f'<video width=640 controls autoplay loop src="data:video/mp4;base64,{base64.b64encode(open(videos[-1], "rb").read()).decode()}"></video>'))
else:
    print("no replay yet")
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out)
