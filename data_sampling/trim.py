#!/usr/bin/env python3
"""
Batch-rename Fashion-MNIST‐style PNG files.

Steps
-----
1.  Remove the *second-last* token from each file name.
    e.g.  checked_pattern_button_front_clothing_shirt → checked_pattern_button_front_shirt
2.  With configurable probabilities, shorten the name so it keeps:
        • N_keep[0] tokens (e.g. 5)  – p_keep[0]  (e.g. 0.50)
        • N_keep[1] tokens (e.g. 3)  – p_keep[1]  (e.g. 0.30)
        • N_keep[2] tokens (e.g. 1)  – p_keep[2]  (e.g. 0.20)
    The last token (category) is always retained.
3.  Copy the image to an output directory with its new name.
"""
from __future__ import annotations
import os
import random
import shutil
from pathlib import Path
from typing import List

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
IN_DIR   = Path("data_new")            # source folder containing PNGs
OUT_DIR  = Path("data_4113_trimed")        # destination folder (created if absent)

N_keep: List[int]  = [5, 3, 1]         # how many tokens to keep
p_keep: List[float] = [0.5, 0.3, 0.2]  # corresponding probabilities (sum = 1)

# -----------------------------------------------------------------------------


def pick_token_count() -> int:
    """Return one of the N_keep choices according to p_keep probabilities."""
    return random.choices(N_keep, weights=p_keep, k=1)[0]


def new_stem(stem: str, n_tokens: int) -> str:
    """
    Remove 2nd-last token, then truncate to the last *n_tokens* tokens.
    Always keeps the last (category) token.
    """
    tokens = stem.split("_")
    if len(tokens) < 2:
        return stem  # too short; leave unchanged

    # Step 1: remove second-last token
    del tokens[-2]

    # Step 2: retain only the last n_tokens tokens (category must stay)
    n_tokens = min(n_tokens, len(tokens))
    return "_".join(tokens[-n_tokens:])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in IN_DIR.glob("*.png"):
        if not img_path.is_file():
            continue

        n_tokens = pick_token_count()
        new_name = new_stem(img_path.stem, n_tokens) + img_path.suffix
        dest = OUT_DIR / new_name

        # If a name collision occurs, append a counter
        counter = 1
        while dest.exists():
            dest = OUT_DIR / f"{new_stem(img_path.stem, n_tokens)}_{counter}{img_path.suffix}"
            counter += 1

        shutil.copy2(img_path, dest)
        print(f"{img_path.name}  →  {dest.name}")


if __name__ == "__main__":
    main()