from __future__ import annotations

import json
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PID = "29000"
STAMP = "2026-09-06_213327"
START = datetime.fromisoformat("2026-09-06T21:33:27")


def pid_running(pid: str) -> bool:
    try:
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        text = (r.stdout or "") + (r.stderr or "")
        return pid in text and "No tasks" not in text
    except Exception:
        return False


def python_processes() -> list[dict]:
    try:
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
            "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3"
        )
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        if not (r.stdout or "").strip():
            return []
        data = json.loads(r.stdout)
        if isinstance(data, dict):
            return [data]
        return data if isinstance(data, list) else []
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def file_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {
        "exists": True,
        "bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "path": str(path),
    }


def main() -> int:
    reports = sorted(
        [p for p in REPORTS.glob("report_*.json") if datetime.fromtimestamp(p.stat().st_mtime) >= START],
        key=lambda p: p.stat().st_mtime,
    )
    cache_files = sorted((ROOT / "cache").glob("*.json")) if (ROOT / "cache").exists() else []
    result = {
        "runner_pid": int(PID),
        "runner_running": pid_running(PID),
        "python_processes": python_processes(),
        "fresh_cache_files": len(cache_files),
        "new_report_jsons_since_start": len(reports),
        "latest_report": str(reports[-1]) if reports else "",
        "latest_report_mtime": datetime.fromtimestamp(reports[-1].stat().st_mtime).isoformat(timespec="seconds") if reports else "",
        "status": file_info(REPORTS / f"online_cold_accuracy_benchmark_{STAMP}.status.json"),
        "stdout": file_info(REPORTS / f"online_cold_accuracy_benchmark_{STAMP}.txt"),
        "stderr": file_info(REPORTS / f"online_cold_accuracy_benchmark_{STAMP}.stderr.txt"),
        "meta": file_info(REPORTS / f"online_cold_accuracy_benchmark_{STAMP}.meta.json"),
        "original_cache_path": file_info(ROOT / "cache"),
        "cache_backup_path": file_info(ROOT / f"cache_backup_online_cold_{STAMP}"),
        "fresh_cache_archive": file_info(ROOT / f"cache_online_cold_results_{STAMP}"),
    }
    out = REPORTS / f"online_cold_poll_{STAMP}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())