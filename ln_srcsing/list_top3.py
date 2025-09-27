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
