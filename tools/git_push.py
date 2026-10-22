"""Push the current branch to `origin` (the user's fork) with GITHUB_TOKEN from .env.

The token goes in a one-off http header, never into git config or a remote URL, and is masked in
the output. Works on Windows, macOS, Linux:  python tools/git_push.py
"""

import base64
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def env() -> dict:
    out = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    token = env().get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is empty in .env")
    header = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    r = subprocess.run(
        ["git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {header}",
         "push", "-u", "origin", branch, *sys.argv[1:]],
        cwd=ROOT, capture_output=True, text=True,
    )
    print((r.stdout + r.stderr).replace(token, "***").replace(header, "***"))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
