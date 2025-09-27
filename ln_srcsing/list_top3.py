#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-processing module for Gemini VLM PDF extraction results.

Usage:
    python postprocess.py results.pkl
"""

import sys
import pickle

def list_top3(pkl_path: str):
    with open(pkl_path, "rb") as f:
        out = pickle.load(f)

    per_page = out.get("per_page", [])
    # filter to only "found" pages with required fields
    candidates = [
        r for r in per_page
        if r.get("found") and r.get("value_text") and r.get("cell_coordinate")
    ]

    if not candidates:
        print("No candidates found in results.")
        return

    # sort by confidence descending
    candidates.sort(key=lambda r: r.get("confidence", 0.0), reverse=True)

    print("Top 3 candidates (by confidence):")
    for idx, r in enumerate(candidates[:3], 1):
        print(f"\nCandidate {idx}:")
        print(f"  Page index        : {r.get('page_index')}")
        print(f"  Cell coordinate   : {r.get('cell_coordinate')}")
        print(f"  Value text        : {r.get('value_text')}")
        print(f"  Label seen        : {r.get('label_variant_seen')}")
        print(f"  Label normalized  : {r.get('label_normalized')}")
        print(f"  Confidence        : {r.get('confidence'):.3f}")
        if r.get("evidence"):
            print(f"  Evidence          : {r.get('evidence')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python postprocess.py results.pkl")
        sys.exit(1)

    list_top3(sys.argv[1])

########################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-processing for Gemini VLM PDF extraction results.

Usage:
    python postprocess.py results.pkl --topk 3
"""

import sys
import os
import pickle
import argparse

CAND_KEYS = [
    "page_index",
    "found",
    "value_text",
    "cell_coordinate",
    "label_variant_seen",
    "label_normalized",
    "confidence",
    "evidence",
]

def as_dict(maybe_obj):
    """Return a plain dict for either a dict or a dataclass-like object."""
    if isinstance(maybe_obj, dict):
        return maybe_obj
    # dataclass or simple object with attributes
    out = {}
    for k in CAND_KEYS:
        out[k] = getattr(maybe_obj, k, None)
    # Try to include bbox if present
    out["bbox"] = getattr(maybe_obj, "bbox", None) if "bbox" not in out else out.get("bbox")
    return out

def coerce_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def list_topk(pkl_path: str, topk: int = 3):
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        out = pickle.load(f)

    per_page_raw = out.get("per_page", [])
    per_page = [as_dict(r) for r in per_page_raw]

    total_pages = len(per_page)
    candidates = [
        r for r in per_page
        if (r.get("found") is True)
        and r.get("value_text")
        and r.get("cell_coordinate")
    ]

    # sort by confidence desc (coerce to float)
    candidates.sort(key=lambda r: coerce_float(r.get("confidence", 0.0)), reverse=True)

    print(f"Summary: pages={total_pages}, found={len(candidates)}")
    if not candidates:
        print("No candidates found in results.")
        return

    k = min(topk, len(candidates))
    print(f"\nTop {k} candidates (by confidence):")
    for idx, r in enumerate(candidates[:k], 1):
        conf = coerce_float(r.get("confidence", 0.0))
        print(f"\nCandidate {idx}:")
        print(f"  Page index        : {r.get('page_index')}")
        print(f"  Cell coordinate   : {r.get('cell_coordinate')}")
        print(f"  Value text        : {r.get('value_text')}")
        print(f"  Label seen        : {r.get('label_variant_seen')}")
        print(f"  Label normalized  : {r.get('label_normalized')}")
        print(f"  Confidence        : {conf:.3f}")
        ev = r.get("evidence")
        if ev:
            print(f"  Evidence          : {ev}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="List top-K most likely cell/values found by the model.")
    ap.add_argument("pkl_path", help="Path to pickled 'out' dict from the main pipeline.")
    ap.add_argument("--topk", type=int, default=3, help="How many to list (default: 3).")
    args = ap.parse_args()
    list_topk(args.pkl_path, topk=args.topk)

