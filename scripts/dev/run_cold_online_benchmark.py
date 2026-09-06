from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
TIMESTAMP = time.strftime("%Y-%m-%d_%H%M%S")

STATUS_PATH = REPORTS / f"online_cold_accuracy_benchmark_{TIMESTAMP}.status.json"
STDOUT_PATH = REPORTS / f"online_cold_accuracy_benchmark_{TIMESTAMP}.txt"
STDERR_PATH = REPORTS / f"online_cold_accuracy_benchmark_{TIMESTAMP}.stderr.txt"
META_PATH = REPORTS / f"online_cold_accuracy_benchmark_{TIMESTAMP}.meta.json"


def write_status(**kwargs) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": TIMESTAMP,
        "status_path": str(STATUS_PATH),
        "stdout_path": str(STDOUT_PATH),
        "stderr_path": str(STDERR_PATH),
        "meta_path": str(META_PATH),
        **kwargs,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env", override=True)

    key = os.getenv("SERPAPI_KEY", "").strip().strip('"').strip("'")
    if not key:
        write_status(
            state="blocked_missing_serpapi_key",
            message="SERPAPI_KEY is not set in environment/.env; live benchmark not run.",
        )
        STDOUT_PATH.write_text("", encoding="utf-8")
        STDERR_PATH.write_text(
            "SERPAPI_KEY is not set in environment/.env; live benchmark not run.\n",
            encoding="utf-8",
        )
        return 2

    cache = ROOT / "cache"
    backup = ROOT / f"cache_backup_online_cold_{TIMESTAMP}"
    restored = False
    moved_cache = False

    write_status(state="starting", serpapi_key_set=True)
    try:
        if cache.exists():
            if backup.exists():
                raise RuntimeError(f"Backup path already exists: {backup}")
            cache.rename(backup)
            moved_cache = True
        cache.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/accuracy_eval.py",
            "--live",
            "--min",
            "90",
            "--json",
        ]
        write_status(
            state="running_live_benchmark",
            serpapi_key_set=True,
            command=cmd,
            cache_backup=str(backup) if moved_cache else None,
        )
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=9000,
            )
            runtime = round(time.perf_counter() - t0, 2)
            STDOUT_PATH.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
            STDERR_PATH.write_text(proc.stderr or "", encoding="utf-8", errors="replace")
            meta = {
                "timestamp": TIMESTAMP,
                "command": cmd,
                "returncode": proc.returncode,
                "seconds": runtime,
                "cache_mode": "cold_fresh_live",
                "serpapi_key_set": True,
                "stdout_path": str(STDOUT_PATH),
                "stderr_path": str(STDERR_PATH),
                "cache_backup": str(backup) if moved_cache else None,
            }
            META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            runtime = round(time.perf_counter() - t0, 2)
            STDOUT_PATH.write_text(exc.stdout or "", encoding="utf-8", errors="replace")
            STDERR_PATH.write_text(exc.stderr or "", encoding="utf-8", errors="replace")
            meta = {
                "timestamp": TIMESTAMP,
                "command": cmd,
                "returncode": None,
                "timeout": True,
                "seconds": runtime,
                "cache_mode": "cold_fresh_live",
                "serpapi_key_set": True,
                "stdout_path": str(STDOUT_PATH),
                "stderr_path": str(STDERR_PATH),
                "cache_backup": str(backup) if moved_cache else None,
            }
            META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            rc = 124
    except Exception as exc:
        STDERR_PATH.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        write_status(state="runner_error", error=f"{type(exc).__name__}: {exc}")
        rc = 1
    finally:
        # Preserve the fresh-run cache for audit, but restore the user's original
        # cache to its original path.
        fresh_cache_archive = ROOT / f"cache_online_cold_results_{TIMESTAMP}"
        try:
            if cache.exists():
                if fresh_cache_archive.exists():
                    shutil.rmtree(fresh_cache_archive)
                cache.rename(fresh_cache_archive)
            if moved_cache and backup.exists():
                backup.rename(cache)
                restored = True
        except Exception as exc:
            prev = {}
            if STATUS_PATH.exists():
                try:
                    prev = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
                except Exception:
                    prev = {}
            prev.update(
                {
                    "state": "restore_error",
                    "restore_error": f"{type(exc).__name__}: {exc}",
                    "original_cache_restored": restored,
                    "fresh_cache_archive": str(fresh_cache_archive),
                }
            )
            STATUS_PATH.write_text(json.dumps(prev, indent=2), encoding="utf-8")
            return rc or 1

    write_status(
        state="completed",
        returncode=rc,
        original_cache_restored=restored or not moved_cache,
        fresh_cache_archive=str(ROOT / f"cache_online_cold_results_{TIMESTAMP}"),
        meta_path=str(META_PATH),
        stdout_path=str(STDOUT_PATH),
        stderr_path=str(STDERR_PATH),
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
