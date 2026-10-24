#!/usr/bin/env python3
"""Validate the ggml_type id allocation of this fork.

This fork assigns explicit ids to private types: the TurboQuant KV cache types,
the TurboQuant weight types, the ConvRot weight types and Prism's PQ2_0/PTQ1_0.
Upstream grows its enum sequentially, so a newly added upstream type can land on
one of those ids. Duplicate enumerator values are legal C++, the build stays
green, and the backend type tables are then indexed by an ambiguous id.

Run this before rebasing on upstream and from CI.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "ggml" / "include" / "ggml.h"

# Private types and the ids documented in docs/turboquant-type-ids.md.
PRIVATE = {
    "GGML_TYPE_TURBO2_0": 43,
    "GGML_TYPE_TURBO3_0": 44,
    "GGML_TYPE_TQ3_1S":   45,
    "GGML_TYPE_TQ4_1S":   46,
    "GGML_TYPE_TURBO4_0": 47,
    "GGML_TYPE_Q8_CR":    48,
    "GGML_TYPE_Q5_CR":    49,
    "GGML_TYPE_Q6_CR":    50,
    "GGML_TYPE_PQ2_0":   142,
    "GGML_TYPE_PTQ1_0":  143,
}


def parse_enum(text: str) -> list[tuple[str, int]]:
    m = re.search(r"enum ggml_type\s*\{(.*?)\n\s*\};", text, re.S)
    if m is None:
        raise SystemExit("check-ggml-type-ids: could not find enum ggml_type")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    entries = []
    value = 0
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, expr = item.split("=", 1)
            name, expr = name.strip(), expr.strip()
            if not re.fullmatch(r"-?\d+|0x[0-9a-fA-F]+", expr):
                raise SystemExit(f"check-ggml-type-ids: unsupported value {name} = {expr}")
            value = int(expr, 0)
        else:
            name = item
        entries.append((name, value))
        value += 1
    return entries


def main() -> int:
    entries = parse_enum(HEADER.read_text(encoding="utf-8"))
    values = {name: value for name, value in entries}
    count = values.get("GGML_TYPE_COUNT")
    private_min = min(PRIVATE.values())
    problems = []

    by_id: dict[int, list[str]] = {}
    for name, value in entries:
        by_id.setdefault(value, []).append(name)
    for value, names in sorted(by_id.items()):
        if len(names) > 1:
            problems.append(f"id {value} is used by {', '.join(names)}")

    for name, value in entries:
        if name == "GGML_TYPE_COUNT":
            continue
        if count is not None and value >= count:
            problems.append(f"{name} = {value} is at or past GGML_TYPE_COUNT = {count}")
        if name not in PRIVATE and value >= private_min:
            problems.append(
                f"upstream type {name} took id {value}, which belongs to the fork's private "
                f"range starting at {private_min}; renumber the private types (see "
                f"docs/turboquant-type-ids.md)")

    for name, want in PRIVATE.items():
        got = values.get(name)
        if got is None:
            problems.append(f"{name} is missing from enum ggml_type")
        elif got != want:
            problems.append(f"{name} moved from {want} to {got}; update this script and the doc")

    print(f"ggml_type entries: {len(entries)}, GGML_TYPE_COUNT = {count}, "
          f"private range starts at {private_min}")
    for name, want in PRIVATE.items():
        print(f"  {name:22s} = {values.get(name, '<missing>')}")

    if problems:
        print("\nERROR:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
