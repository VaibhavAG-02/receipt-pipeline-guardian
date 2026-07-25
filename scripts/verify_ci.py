#!/usr/bin/env python3
"""Execute every `run:` step of the CI workflow locally, in order.

This is not GitHub Actions. It does not test the runner image, the `uses:`
actions, caching, or artifact upload. What it *does* test is the part that
actually breaks: whether the shell commands and inline assertion scripts in
ci.yml work against a clean checkout with a clean dependency install.

Usage:  python scripts/verify_ci.py [--python /path/to/venv/bin/python]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter whose environment the steps run in")
    ap.add_argument("--keep", action="store_true",
                    help="keep generated data instead of cleaning first")
    args = ap.parse_args()

    wf = yaml.safe_load(WORKFLOW.read_text())
    job = wf["jobs"]["build"]
    steps = job["steps"]

    if not args.keep:
        for p in ("data", "dbt/target", "dbt/logs"):
            shutil.rmtree(ROOT / p, ignore_errors=True)

    bindir = str(Path(args.python).parent)
    env = {
        **os.environ,
        "PATH": bindir + os.pathsep + os.environ.get("PATH", ""),
        "CI": "true",
    }

    ran = failed = skipped = 0
    for i, step in enumerate(steps, 1):
        name = step.get("name") or f"step {i}"
        if "run" not in step:
            print(f"[skip] {name}  (uses: {step['uses']} — needs the Actions runner)")
            skipped += 1
            continue

        script_preview = step["run"]
        # Steps that install system packages need root and a package manager the
        # Actions runner has but a dev machine may not. Skipping is honest; the
        # alternative is a red run that says nothing about the workflow.
        if "sudo " in script_preview and shutil.which("sudo") is None:
            print(f"[skip] {name}  (needs sudo — available on the Actions runner)")
            skipped += 1
            continue

        cwd = ROOT / step.get("working-directory", ".")
        step_env = {**env, **{k: str(v) for k, v in (step.get("env") or {}).items()}}
        script = step["run"]

        print(f"\n[run ] {name}")
        print(f"       cwd={cwd.relative_to(ROOT) if cwd != ROOT else '.'}")
        t0 = time.time()
        proc = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", script],
            cwd=cwd, env=step_env, capture_output=True, text=True,
        )
        dt = time.time() - t0
        out = (proc.stdout or "").strip().splitlines()
        for line in out[-12:]:
            print(f"       | {line}")
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            for line in err[-15:]:
                print(f"       ! {line}")
            print(f"[FAIL] {name}  ({dt:.1f}s, exit {proc.returncode})")
            failed += 1
            return 1
        print(f"[ ok ] {name}  ({dt:.1f}s)")
        ran += 1

    print(f"\n{ran} run-steps passed, {skipped} runner-only steps skipped, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
