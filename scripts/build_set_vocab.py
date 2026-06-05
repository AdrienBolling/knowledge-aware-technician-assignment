"""Generate the canonical set-mode vocabulary JSON.

Enumerates *every* token the set-mode ``_SetEmitter`` can ever produce
from the full pool of machine, component, and technician templates
shipped with the repo, and writes the result to
``run_configs/vocab/set_vocab.json``.

Single source of truth for set-mode token IDs.  All training and eval
runs read this file and freeze the tokenizer against it, so checkpoints
produced by one run are loadable by any other regardless of which
templates the local scenario happens to sample.

Usage::

    python scripts/build_set_vocab.py
    # → wrote run_configs/vocab/set_vocab.json  (147 tokens)

Run this whenever the template pool changes (new machine template, new
component type, new technician profile).  Then retrain any agents whose
checkpoints were sized against the previous vocab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kata.core.tokenizer import StateTokenizer
from kata.EntityFactories import list_technician_templates
from kata.EntityFactories.machine_factory import (
    list_templates as list_machine_templates,
    get_template,
)


def _all_component_types() -> list[str]:
    """Return every component type declared across the machine template pool."""
    comp_types: set[str] = set()
    for mt_name in list_machine_templates():
        tpl = get_template(mt_name)
        for comp in (tpl.get("components") or {}).values():
            ct = comp.get("component_type")
            if ct:
                comp_types.add(str(ct))
    return sorted(comp_types)


def _all_machine_types() -> list[str]:
    """Return every ``machine_type`` string declared across the pool.

    NOT the template *name* — the ``machine_type`` field inside each
    template (e.g. CNC, Welder, …).  This matches what the emitter
    sees at runtime.
    """
    types: set[str] = set()
    for mt_name in list_machine_templates():
        mtype = get_template(mt_name).get("machine_type")
        if mtype:
            types.add(str(mtype))
    return sorted(types)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build_set_vocab",
        description="Generate or extend the canonical set-mode vocabulary JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("run_configs/vocab/set_vocab.json"),
        help="Destination JSON file (default: run_configs/vocab/set_vocab.json).",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=64,
        help="Carried in the tokenizer; set-mode does not actually use it (default: 64).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Force a from-scratch rebuild that may renumber existing "
            "tokens.  By default the script is APPEND-ONLY: it loads "
            "the existing canonical vocab and only adds new tokens at "
            "the end, preserving every existing ID so old checkpoints "
            "remain loadable.  Use --rebuild only when you have no "
            "checkpoints to preserve."
        ),
    )
    args = parser.parse_args(argv)

    machine_types = _all_machine_types()
    component_types = _all_component_types()
    technician_templates = sorted(list_technician_templates())

    # Enumerate every token the set emitter can produce, using the
    # same deterministic order as StateTokenizer.build_set_vocab.  We
    # produce the list manually rather than calling that helper so we
    # can then ADD entries to an existing tokenizer without renumbering.
    target_tokens = _enumerate_target_tokens(
        machine_types=machine_types,
        component_types=component_types,
        technician_templates=technician_templates,
    )

    # From-scratch generation uses the full canonical builder so the
    # token order matches StateTokenizer.build_set_vocab exactly.
    # Subsequent calls reload that canonical and only append new
    # tokens — the order of new tokens then depends on the order in
    # which they appear in ``target_tokens``, but the original ids
    # are preserved row-for-row.
    if args.out.is_file() and not args.rebuild:
        tok = StateTokenizer.from_json(
            args.out, seq_length=args.seq_length, freeze=False
        )
        existing_count = tok.vocab_size
    else:
        # Fresh-build path: use the canonical builder directly so the
        # initial token order is stable.  This is the only path that
        # can renumber an existing canonical — guard it behind --rebuild.
        tok = StateTokenizer.build_set_vocab(
            machine_types=machine_types,
            component_types=component_types,
            technician_templates=technician_templates,
            seq_length=args.seq_length,
        )
        tok._frozen = False
        existing_count = tok.vocab_size

    # Append-only: token_to_id assigns a new id only for unseen tokens.
    new_added = 0
    for t in target_tokens:
        before = tok.vocab_size
        tok.token_to_id(t)
        if tok.vocab_size > before:
            new_added += 1
    tok.freeze()
    tok.to_json(args.out)

    print(f"wrote {args.out}  ({tok.vocab_size} tokens, +{new_added} new)")
    print(f"  machine_types        : {len(machine_types)}  {machine_types}")
    print(f"  component_types      : {len(component_types)}  {component_types}")
    print(f"  technician_templates : {len(technician_templates)}  {technician_templates}")
    return 0


def _enumerate_target_tokens(
    machine_types: list[str],
    component_types: list[str],
    technician_templates: list[str],
) -> list[str]:
    """Return the canonical-order list of every set-mode token string.

    Mirrors :meth:`StateTokenizer.build_set_vocab` token-by-token so
    the append-only growth logic can decide what's already present
    and what's new without invoking the all-or-nothing builder.
    """
    out: list[str] = []
    out += [
        # tech slot
        "<RATIO:FATIGUE>", "<COUNT:ASSIGNS>", "<COUNT:KNOW_VOL>",
        "<COUNT:KNOW_MAX>", "<RATIO:KNOW_SPEC>", "<COUNT:KNOW_ENT>",
        "<RATIO:MATCH>", "<RATIO:MATCH_N1>", "<RATIO:MATCH_N2>",
        "<TIME:ETA>", "<TIME:LAST_AGE>",
        # machine slot
        "<COUNT:PROC_TOT>", "<COUNT:IN_BUF>", "<COUNT:OUT_BUF>",
        "<COUNT:BD_COUNT>", "<TIME:DOWNTIME>", "<TIME:MEAN_TBF>",
        # env slot
        "<FOUR:SIM_T>", "<TIME:T_AGE>", "<COUNT:Q_SIZE>",
        "<COUNT:BROKEN_N>", "<COUNT:PROC_N>",
        "<TIME:N1_AGE>", "<TIME:N2_AGE>",
    ]
    for key in ("BUSY", "DISRUPT", "BROKEN", "PROC", "IS_CURRENT", "HAS_T"):
        out += [f"{key}=T", f"{key}=F"]
    out += [f"TEMPLATE={t}" for t in technician_templates]
    for k in ("M_TYPE", "T_M_TYPE", "N1_M_TYPE", "N2_M_TYPE"):
        out += [f"{k}={mt}" for mt in machine_types]
    out += [f"{k}=NONE" for k in ("T_M_TYPE", "N1_M_TYPE", "N2_M_TYPE")]
    for k in ("CUR_COMP", "T_C_TYPE", "N1_C_TYPE", "N2_C_TYPE"):
        out += [f"{k}={ct}" for ct in component_types]
    out += [f"{k}=NONE" for k in ("CUR_COMP", "T_C_TYPE", "N1_C_TYPE", "N2_C_TYPE")]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
