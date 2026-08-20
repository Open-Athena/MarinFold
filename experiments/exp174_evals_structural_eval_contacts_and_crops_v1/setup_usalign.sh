#!/usr/bin/env bash
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
#
# Fetch + build US-align, the reference TM-score implementation.
#
# TM-score is the one metric in this harness with no well-tested Python
# implementation for our case: biotite's `tm_score` scores an *already
# superimposed* pair, and every pip-installable wrapper (tmtools) exposes
# TM-align's sequence-*independent* structural alignment. What #174 needs is
# the sequence-*dependent* variant — equivalent residues are the ones with the
# same residue index, since prediction and ground truth are the same protein —
# with the TM-score-maximizing superposition search. US-align does exactly
# that under `-TMscore 1`, and it is the implementation CASP assessors use.
#
# Single translation unit, no dependencies beyond a C++ compiler.
#
# Usage:  bash setup_usalign.sh [dest_dir]      (default: ./_bin)

set -euo pipefail

DEST="${1:-$(dirname "$0")/_bin}"
SRC_URL="https://zhanggroup.org/US-align/bin/module/USalign.cpp"

mkdir -p "$DEST"
cd "$DEST"

echo "[usalign] downloading $SRC_URL"
curl -sSfL -o USalign.cpp "$SRC_URL"

echo "[usalign] compiling (g++ -O3)"
g++ -O3 -ffast-math -lm -o USalign USalign.cpp

./USalign 2>&1 | grep -m1 'US-align (Version' | tr -s ' ' | tee USalign.version
echo "[usalign] built $DEST/USalign"
