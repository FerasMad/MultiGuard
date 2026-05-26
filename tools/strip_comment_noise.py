"""Strip decorative banner-comment noise across the codebase (P12.1).

Cleanups applied (line-by-line, no AST):
  1. Pure decorative banner (only dashes/equals/etc) -> deleted entirely.
       e.g.  # ------------------------------------------------------------
             # ============================================================
             # ************************************************************
  2. Labeled banner (`# ----- label` or `# === label ===`) -> `# label`.
       e.g.  # ----------------------------- phase mgmt   -> # phase mgmt
             # === Helper functions ===                    -> # Helper functions

Preserves:
  - Module / class / function docstrings (triple-quoted strings)
  - `# noqa: ...`, `# type: ignore`, `# pragma: ...` pragmas
  - WHY-comments (anything not matching the banner regex)
  - Shebangs (`#!/usr/bin/env python`)
  - Spec references (V3.1, F.xx, R-Hx), commit IDs (P9.1), TODO/FIXME

Excludes:
  - .venv*/ and __pycache__/ directories
  - phases/v3/.venv-qwen* (legacy V3 Qwen venv)
  - *.pyc

Usage:
    python tools/strip_comment_noise.py            # dry-run; prints diff stats
    python tools/strip_comment_noise.py --apply    # actually rewrite files
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PURE_BANNER = re.compile(r"^\s*#\s*[-=*_#]{6,}\s*$")

LABELED_BANNER = re.compile(
    r"^(\s*)#\s*[-=*_]{3,}\s*([^-=*_\s][^-=*_]*?[^-=*_\s]|\S)\s*[-=*_]*\s*$"
)


def is_pragma(line: str) -> bool:
    """Don't touch # noqa / # type: ignore / # pragma / shebangs."""
    stripped = line.strip()
    if stripped.startswith("#!"):
        return True
    lower = stripped.lower()
    return any(p in lower for p in ("# noqa", "# type:", "# pragma:", "type: ignore"))


def strip_line(line: str) -> tuple[str | None, str]:
    """Return (new_line_or_None, change_kind). None means delete the line."""
    if is_pragma(line):
        return line, "keep"
    if PURE_BANNER.match(line):
        return None, "deleted"
    m = LABELED_BANNER.match(line)
    if m:
        indent, label = m.group(1), m.group(2).strip()
        if label:
            return f"{indent}# {label}\n", "stripped"
        return None, "deleted"
    return line, "keep"


def process_file(path: Path, apply: bool) -> dict:
    """Return per-file stats; rewrite in place if apply=True."""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    deleted = stripped = 0
    in_docstring = False
    docstring_marker = ""
    for line in lines:
        if not in_docstring:
            for marker in ('"""', "'''"):
                count = line.count(marker)
                if count % 2 == 1:
                    in_docstring = True
                    docstring_marker = marker
                    break
            if in_docstring:
                out.append(line)
                continue
        else:
            if docstring_marker in line:
                in_docstring = False
                docstring_marker = ""
            out.append(line)
            continue

        new_line, kind = strip_line(line)
        if kind == "deleted":
            deleted += 1
            continue
        if kind == "stripped":
            stripped += 1
            out.append(new_line)
            continue
        out.append(line)

    new_src = "".join(out)
    if apply and new_src != src:
        path.write_text(new_src, encoding="utf-8")
    return {"path": str(path.relative_to(ROOT)), "deleted": deleted, "stripped": stripped,
            "changed": new_src != src}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="Rewrite files in place")
    p.add_argument("--roots", nargs="+", default=None,
                   help="Restrict to these subdirs (default: all .py except .venv*)")
    args = p.parse_args()

    if args.roots:
        roots = [ROOT / r for r in args.roots]
    else:
        roots = [ROOT]

    pyfiles: list[Path] = []
    for root in roots:
        for f in root.rglob("*.py"):
            parts = f.parts
            if any(p.startswith(".venv") or p.startswith(".venv-qwen") for p in parts):
                continue
            if "__pycache__" in parts:
                continue
            # Skip third-party clones (chandlerbing65nm/FakeImageDetection etc).
            # Modifying upstream code would break the ability to pull updates.
            if "external" in parts:
                continue
            pyfiles.append(f)

    print(f"[strip] scanning {len(pyfiles)} .py files (apply={args.apply})")
    total_deleted = total_stripped = changed_files = 0
    for f in pyfiles:
        stats = process_file(f, args.apply)
        if stats["changed"]:
            changed_files += 1
            print(f"  {stats['path']}: -{stats['deleted']} deleted, "
                  f"~{stats['stripped']} stripped")
        total_deleted += stats["deleted"]
        total_stripped += stats["stripped"]

    verb = "would change" if not args.apply else "changed"
    print(f"\n[strip] {verb} {changed_files} files: "
          f"{total_deleted} lines deleted, {total_stripped} lines stripped")
    if not args.apply:
        print("[strip] re-run with --apply to actually rewrite files")


if __name__ == "__main__":
    main()
