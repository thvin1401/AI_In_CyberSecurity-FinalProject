#!/usr/bin/env python
"""
scripts/drive_sync.py
Thin wrapper around rclone for syncing data and checkpoints with Google Drive.

Drive folder layout:
    han-spam/
        data/processed/      ← preprocessed tensors + fasttext.bin
        checkpoints/         ← per-fold .weights.h5 files
        results/             ← metrics.json, plots

Usage (inside container):
    python scripts/drive_sync.py push-data        # local processed → Drive
    python scripts/drive_sync.py pull-data        # Drive → local processed
    python scripts/drive_sync.py push-checkpoints # local checkpoints → Drive
    python scripts/drive_sync.py pull-checkpoints # Drive → local checkpoints
    python scripts/drive_sync.py status           # show what's on Drive
"""
import argparse
import subprocess
import sys
from pathlib import Path

REMOTE = "gdrive:han-spam"

MAPPINGS = {
    "data":        ("data/processed",  f"{REMOTE}/data/processed"),
    "checkpoints": ("checkpoints",     f"{REMOTE}/checkpoints"),
    "results":     ("outputs",         f"{REMOTE}/results"),
}


def rclone(*args: str) -> int:
    cmd = ["rclone", *args, "--progress"]
    print(f"  → rclone {' '.join(args)}")
    result = subprocess.run(cmd)
    return result.returncode


def push(key: str) -> None:
    local, remote = MAPPINGS[key]
    Path(local).mkdir(parents=True, exist_ok=True)
    rc = rclone("copy", local, remote)
    if rc != 0:
        print(f"[drive_sync] ERROR: push {key} failed (exit {rc})")
        sys.exit(rc)
    print(f"[drive_sync] ✓ pushed {local} → {remote}")


def pull(key: str) -> None:
    local, remote = MAPPINGS[key]
    Path(local).mkdir(parents=True, exist_ok=True)
    rc = rclone("copy", remote, local)
    if rc != 0:
        print(f"[drive_sync] ERROR: pull {key} failed (exit {rc})")
        sys.exit(rc)
    print(f"[drive_sync] ✓ pulled {remote} → {local}")


def status() -> None:
    rclone("ls", REMOTE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "push-data", "pull-data",
            "push-checkpoints", "pull-checkpoints",
            "push-results", "pull-results",
            "status",
        ],
    )
    args = parser.parse_args()

    actions = {
        "push-data":        lambda: push("data"),
        "pull-data":        lambda: pull("data"),
        "push-checkpoints": lambda: push("checkpoints"),
        "pull-checkpoints": lambda: pull("checkpoints"),
        "push-results":     lambda: push("results"),
        "pull-results":     lambda: pull("results"),
        "status":           status,
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
