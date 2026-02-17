#!/usr/bin/env python3
"""Normalize line endings to LF for tracked text files.

Scans all .py, .yaml, .yml, .md, .txt, .sh, .toml, .json, .tsv files
and converts CR+LF or CR-only line endings to plain LF.

Usage:
    python scripts/normalize_line_endings.py          # dry-run
    python scripts/normalize_line_endings.py --fix    # actually fix
"""

import argparse
import os
import sys

TEXT_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".md", ".txt", ".sh",
    ".toml", ".json", ".tsv",
}

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".egg-info", "nst.egg-info", ".tox",
}


def _needs_fix(path: str) -> bool:
    """Return True if file contains CR (\\r) characters."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return b"\r" in data
    except (OSError, UnicodeDecodeError):
        return False


def _fix_file(path: str) -> int:
    """Replace CRLF/CR with LF.  Returns number of \\r chars removed."""
    with open(path, "rb") as f:
        data = f.read()
    n_cr = data.count(b"\r")
    if n_cr == 0:
        return 0
    fixed = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    with open(path, "wb") as f:
        f.write(fixed)
    return n_cr


def main():
    parser = argparse.ArgumentParser(description="Normalize line endings to LF")
    parser.add_argument("--fix", action="store_true", help="Actually fix files (default: dry-run)")
    parser.add_argument("--root", default=None, help="Project root (default: auto-detect)")
    args = parser.parse_args()

    root = args.root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Scanning: {root}")
    print(f"Mode: {'FIX' if args.fix else 'DRY-RUN'}")

    bad_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden / cache dirs
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TEXT_EXTENSIONS:
                continue
            fpath = os.path.join(dirpath, fname)
            if _needs_fix(fpath):
                rel = os.path.relpath(fpath, root)
                bad_files.append(fpath)
                if args.fix:
                    n = _fix_file(fpath)
                    print(f"  FIXED: {rel}  ({n} CR chars removed)")
                else:
                    print(f"  NEEDS FIX: {rel}")

    print(f"\nTotal files needing fix: {len(bad_files)}")
    if bad_files and not args.fix:
        print("Run with --fix to normalize line endings.")
    elif not bad_files:
        print("All files already use LF line endings. ✅")


if __name__ == "__main__":
    main()
