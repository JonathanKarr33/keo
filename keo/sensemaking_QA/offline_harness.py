#!/usr/bin/env python3
"""Shared runner for the offline suites (no API calls anywhere in them).

Every suite used to carry its own copy of this loop, and the copies disagreed in two ways
that matter:

- Two suites printed a pass line without a count, so a suite whose check list was emptied
  by a bad edit would have reported success while testing nothing. That is the same shape
  as the bugs these suites exist to catch, so run_suite refuses to pass an empty list.
- All of them caught only AssertionError, so a *broken* check (KeyError, TypeError) aborted
  the run and silently skipped every check after it. Here a broken check is reported and
  the rest still run.
"""

from __future__ import annotations

import traceback
from typing import Callable, List, Sequence, Tuple

Check = Tuple[str, Callable[[], None]]


def run_suite(label: str, checks: Sequence[Check]) -> None:
    """Run every check, then exit non-zero if any failed. Never returns on failure."""
    if not checks:
        raise SystemExit(
            f"{label}: no checks registered — a suite that tests nothing must not pass"
        )

    failed: List[str] = []
    for name, fn in checks:
        print(f"{name}:")
        try:
            fn()
        except AssertionError as e:
            failed.append(name)
            print(f"  FAIL: {e}")
        except Exception:
            # The check itself is broken rather than the code under test. Report it as a
            # failure and keep going; aborting here would hide the remaining checks.
            failed.append(name)
            print("  ERROR: the check raised, so it never tested anything:")
            for line in traceback.format_exc().strip().splitlines():
                print(f"    {line}")

    print()
    if failed:
        raise SystemExit(
            f"{len(failed)}/{len(checks)} {label} checks failed: {', '.join(failed)}"
        )
    print(f"OK: {len(checks)} {label} checks passed")
