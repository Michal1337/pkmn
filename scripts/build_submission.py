"""Package agent/ into submission.tar.gz with files at the TOP LEVEL.

The competition requires main.py and deck.csv at the root of the archive
(NOT nested in a folder). This packs every file under agent/ flat.

Usage:
    python scripts/build_submission.py            # -> submission.tar.gz
    python scripts/build_submission.py -o out.tar.gz
"""

import argparse
import os
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_DIR = os.path.join(ROOT, "agent")
REQUIRED = ["main.py", "deck.csv"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--out", default=os.path.join(ROOT, "submission.tar.gz"))
    args = p.parse_args()

    files = []
    for dirpath, _dirs, names in os.walk(AGENT_DIR):
        if "__pycache__" in dirpath:
            continue
        for name in names:
            if name.endswith(".pyc"):
                continue
            full = os.path.join(dirpath, name)
            arc = os.path.relpath(full, AGENT_DIR).replace(os.sep, "/")
            files.append((full, arc))

    arcnames = {arc for _, arc in files}
    missing = [r for r in REQUIRED if r not in arcnames]
    if missing:
        raise SystemExit(f"ERROR: missing required file(s) at top level: {missing}")

    # Validate the deck has exactly 60 cards before shipping.
    with open(os.path.join(AGENT_DIR, "deck.csv")) as f:
        n = len([ln for ln in f if ln.strip()])
    if n != 60:
        raise SystemExit(f"ERROR: deck.csv has {n} cards, expected 60.")

    with tarfile.open(args.out, "w:gz") as tar:
        for full, arc in sorted(files, key=lambda x: x[1]):
            tar.add(full, arcname=arc)

    print(f"Wrote {args.out}")
    print("Contents (top level):")
    for _, arc in sorted(files, key=lambda x: x[1]):
        print(f"  {arc}")


if __name__ == "__main__":
    main()
