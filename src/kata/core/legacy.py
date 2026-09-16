"""Switches that restore simulator behaviour from before a correctness fix.

Each switch is an environment variable, so it reaches every process that a
run starts (vectorised training workers, the evaluation harness, test
subprocesses) without a change to the run configs.  The value is read when
the simulated object is built, not at import time.

``KATA_LEGACY_BUFFER_INTERRUPT=1``
    ``Machine._run`` keeps the pre-fix buffer handling: a breakdown that
    interrupts ``input_buffer.get()`` leaves the request queued, so the
    request swallows the next product; a breakdown that interrupts
    ``output_buffer.put()`` skips ``total_processed``.

``KATA_LEGACY_MACHINE_TRACKING=1``
    ``KataEnv`` keeps the pre-fix machine tracking: breakdown counts and
    downtime intervals are sampled from ``machine.broken`` at decision
    boundaries only, instead of being recorded at the breakdown and repair
    events.

Unset, empty, ``0``, ``false``, ``no`` or ``off`` select the fixed
behaviour (the default).  ``1``, ``true``, ``yes`` or ``on`` select the
legacy behaviour.  Any other value raises ``ValueError``, so a typing error
cannot silently select the wrong simulator.
"""

from __future__ import annotations

import os

LEGACY_BUFFER_INTERRUPT_ENV = "KATA_LEGACY_BUFFER_INTERRUPT"
LEGACY_MACHINE_TRACKING_ENV = "KATA_LEGACY_MACHINE_TRACKING"

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def _flag(name: str) -> bool:
    raw = os.environ.get(name, "")
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    msg = f"{name}={raw!r}: use one of 1/true/yes/on or 0/false/no/off"
    raise ValueError(msg)


def legacy_buffer_interrupt() -> bool:
    """Return True when the pre-fix machine buffer handling is selected."""
    return _flag(LEGACY_BUFFER_INTERRUPT_ENV)


def legacy_machine_tracking() -> bool:
    """Return True when the pre-fix decision-boundary machine tracking is selected."""
    return _flag(LEGACY_MACHINE_TRACKING_ENV)


def legacy_switches() -> dict[str, bool]:
    """Return the state of every legacy switch (for run manifests)."""
    return {
        "legacy_buffer_interrupt": legacy_buffer_interrupt(),
        "legacy_machine_tracking": legacy_machine_tracking(),
    }
