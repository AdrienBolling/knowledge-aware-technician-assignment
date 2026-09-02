#!/usr/bin/env bash
# Guard: manuscript edits are restricted to the Results section
# (user instruction 2026-09-02) until specific authorization.
#
# Verifies that, in the paper submodule, the working tree differs from
# HEAD only inside Manuscript.tex's Results span (from the line
# '\section{Results}' up to, excluding, the line '\section{Discussion}').
# Any other changed file in paper/ fails the check.
#
# Run from the repo root BEFORE every paper commit:
#     scripts/check_paper_results_only.sh
# Bypass (requires explicit user authorization):
#     scripts/check_paper_results_only.sh --bypass   (or GUARD_BYPASS=1)
set -u
cd "$(dirname "$0")/.."

if [ "${1:-}" = "--bypass" ] || [ "${GUARD_BYPASS:-0}" = 1 ]; then
  echo "GUARD BYPASSED (explicit) — allowing edits outside the Results section."
  exit 0
fi

# Build artifacts are not edits.
ARTIFACTS='\.(aux|log|out|bbl|blg|fls|fdb_latexmk|synctex\.gz|toc|spl|pdf)$'
CHANGED=$(git -C paper status --porcelain --untracked-files=no \
  | awk '{print $2}' | grep -Ev "$ARTIFACTS" || true)

FAIL=0
for f in $CHANGED; do
  if [ "$f" != "Manuscript.tex" ]; then
    echo "GUARD FAIL: changed file outside allowance: paper/$f"
    FAIL=1
  fi
done

if echo "$CHANGED" | grep -qx "Manuscript.tex"; then
  git -C paper show HEAD:Manuscript.tex > /tmp/guard_head.tex
  python3 - <<'EOF' || FAIL=1
import re, sys

def split(path):
    text = open(path, encoding="utf-8").read().splitlines(keepends=True)
    start = end = None
    for i, line in enumerate(text):
        if start is None and re.match(r"\s*\\section\{Results\}", line):
            start = i
        elif start is not None and re.match(r"\s*\\section\{Discussion\}", line):
            end = i
            break
    if start is None or end is None:
        print("GUARD FAIL: could not locate the Results span markers "
              "(\\section{Results} ... \\section{Discussion})")
        sys.exit(1)
    return "".join(text[:start]), "".join(text[end:])

h_pre, h_post = split("/tmp/guard_head.tex")
w_pre, w_post = split("paper/Manuscript.tex")
ok = True
if h_pre != w_pre:
    print("GUARD FAIL: Manuscript.tex changed BEFORE the Results section")
    ok = False
if h_post != w_post:
    print("GUARD FAIL: Manuscript.tex changed AT/AFTER \\section{Discussion}")
    ok = False
sys.exit(0 if ok else 1)
EOF
fi

if [ "$FAIL" = 1 ]; then
  echo "Guard failed — restrict edits to the Results section of"
  echo "Manuscript.tex, or rerun with --bypass under explicit authorization."
  exit 1
fi
echo "GUARD OK: all manuscript changes are inside the Results section."
