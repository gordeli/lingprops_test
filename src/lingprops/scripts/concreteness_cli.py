#!/usr/bin/env python3
"""Compute concreteness for a text or file.

Usage:
  python -m lingprops.scripts.concreteness_cli --text "Some text"
  python -m lingprops.scripts.concreteness_cli --file path/to/file.txt
"""
import argparse, sys, json
from lingprops import compute_concreteness, ensure_nltk_data

def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Text to analyze")
    g.add_argument("--file", type=str, help="Path to a text file to analyze")
    args = p.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    ensure_nltk_data()
    out = compute_concreteness(text)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
