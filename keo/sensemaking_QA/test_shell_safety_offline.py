#!/usr/bin/env python3
"""Static checks on the shell entry points.

macOS ships bash 3.2, where expanding an *empty* array as `"${ARR[@]}"` under `set -u` aborts with
"unbound variable". That is easy to miss because it only fires on the code path where the array
happens to be empty — the sweep judged fine with `--smoke` (which appends an argument) and died on a
plain `--judge` run, after paying for three hours of generation. These checks catch the pattern
without needing to execute the scripts.
"""

import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from offline_harness import run_suite  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = sorted(glob.glob(os.path.join(HERE, "*.sh")))

# `${ARR[@]+"${ARR[@]}"}` expands to nothing when unset or empty, which is what we want.
GUARDED = re.compile(r"\$\{(\w+)\[@\]\+")
UNGUARDED = re.compile(r'"\$\{(\w+)\[@\]\}"')
EMPTY_INIT = re.compile(r"^\s*(\w+)=\(\)\s*$", re.MULTILINE)


def test_scripts_are_present():
    assert SCRIPTS, "no shell scripts found — this suite would silently check nothing"
    print(f"  {len(SCRIPTS)} shell scripts found")


def test_scripts_parse():
    for path in SCRIPTS:
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert proc.returncode == 0, f"{os.path.basename(path)} has a syntax error:\n{proc.stderr}"
    print(f"  all {len(SCRIPTS)} scripts pass bash -n")


def test_possibly_empty_arrays_are_guarded():
    problems = []
    for path in SCRIPTS:
        src = open(path).read()
        if "set -u" not in src and "set -euo" not in src:
            continue
        # Arrays that start out empty are the ones that can be empty at expansion time.
        may_be_empty = set(EMPTY_INIT.findall(src))
        if not may_be_empty:
            continue
        for lineno, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for name in UNGUARDED.findall(line):
                # The guarded form ${A[@]+"${A[@]}"} *contains* the bare form as a substring, so a
                # plain regex match is not enough to call a line unsafe.
                if f"${{{name}[@]+" in line:
                    continue
                if name in may_be_empty:
                    problems.append(
                        f"{os.path.basename(path)}:{lineno} expands possibly-empty {name} as "
                        f'"${{{name}[@]}}" — use ${{{name}[@]+"${{{name}[@]}}"}} for bash 3.2'
                    )
    assert not problems, "unguarded empty-array expansions:\n  " + "\n  ".join(problems)
    print("  every possibly-empty array is expanded with the bash 3.2 safe form")


def test_the_safe_form_actually_works_here():
    """Verify the idiom against the local bash rather than trusting it."""
    unsafe = subprocess.run(
        ["bash", "-c", 'set -u; A=(); printf "%s" "${A[@]}"'], capture_output=True, text=True
    )
    safe = subprocess.run(
        ["bash", "-c", 'set -u; A=(); printf "ok%s" ${A[@]+"${A[@]}"}'], capture_output=True, text=True
    )
    assert safe.returncode == 0 and safe.stdout == "ok", (
        f"the guarded form failed on this bash: rc={safe.returncode} out={safe.stdout!r} "
        f"err={safe.stderr!r}"
    )
    if unsafe.returncode == 0:
        print("  this bash tolerates empty arrays, but the guard is kept for bash 3.2 hosts")
    else:
        print("  confirmed: bare form fails on this bash, guarded form succeeds")


def test_judge_args_specifically():
    """The exact array that broke a three-hour run."""
    path = os.path.join(HERE, "run_ablation_matrix.sh")
    src = open(path).read()
    assert "JUDGE_ARGS=()" in src, "JUDGE_ARGS no longer initialised empty; update this check"
    uses = [l for l in src.splitlines() if "JUDGE_ARGS[@]" in l and not l.lstrip().startswith("#")]
    assert uses, "JUDGE_ARGS is never expanded — judging would take no extra flags"
    for line in uses:
        assert GUARDED.search(line), f"unguarded JUDGE_ARGS expansion: {line.strip()}"
    print(f"  JUDGE_ARGS expanded safely in {len(uses)} place(s)")


def test_run_id_and_judge_skip_are_wired():
    """A new answerer must not write into the previous generator's arm dirs or inherit its scores."""
    path = os.path.join(HERE, "run_ablation_matrix.sh")
    src = open(path).read()
    assert "--run-id" in src and "ARM_ROOT" in src, "--run-id is gone; gemma-4 and qwen arms would collide"
    assert "judge_skip.py" in src, "judge skip no longer checks the recorded answerer"
    print("  --run-id and judge_skip.py are wired into the sweep")


def main():
    run_suite("shell safety", [
        ("scripts present", test_scripts_are_present),
        ("scripts parse", test_scripts_parse),
        ("empty arrays guarded", test_possibly_empty_arrays_are_guarded),
        ("safe form works on this bash", test_the_safe_form_actually_works_here),
        ("JUDGE_ARGS guarded", test_judge_args_specifically),
        ("run-id and judge skip wired", test_run_id_and_judge_skip_are_wired),
    ])


if __name__ == "__main__":
    main()
