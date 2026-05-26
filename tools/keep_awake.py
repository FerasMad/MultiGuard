"""Prevent Windows from sleeping during the autonomous 12h run.

Uses ctypes SetThreadExecutionState (no admin required). The flag persists
as long as this process is alive.

Run:
    python tools/keep_awake.py --hours 13
"""

from __future__ import annotations

import argparse
import ctypes
import time
from datetime import datetime, timedelta

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=13.0)
    p.add_argument("--display", action="store_true",
                   help="also keep display awake (default: only system)")
    args = p.parse_args()

    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if args.display:
        flags |= ES_DISPLAY_REQUIRED

    ctypes.windll.kernel32.SetThreadExecutionState(flags)
    deadline = datetime.now() + timedelta(hours=args.hours)
    print(f"[keep-awake] system kept awake until {deadline.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[keep-awake] PID={ctypes.windll.kernel32.GetCurrentProcessId()}")

    try:
        while datetime.now() < deadline:
            time.sleep(60)
            remaining = deadline - datetime.now()
            mins = int(remaining.total_seconds() / 60)
            if mins % 30 == 0:
                print(f"[keep-awake] {mins} min remaining ({datetime.now().strftime('%H:%M:%S')})")
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[keep-awake] released; Windows can sleep normally now")


if __name__ == "__main__":
    main()
