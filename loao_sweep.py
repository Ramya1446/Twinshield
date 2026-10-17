# -*- coding: utf-8 -*-
"""
Leave-one-attack-out over every attack class, repeated across seeds, with
confidence intervals.

train_zeroday.py runs one class at one seed, which illustrates the method but
does not characterise it. This sweep runs every (class, seed) combination the
dataset supports, archives each run, and reports the mean and a 95% interval
for the two numbers that matter: detection of the quarantined class, and the
false-positive rate on normal traffic.

    python loao_sweep.py                          # every class, 5 seeds
    python loao_sweep.py --seeds 1 2 3
    python loao_sweep.py --classes "Brake Tampering" "DoS"
    python loao_sweep.py --dry-run                # print the plan and stop

Each run is a separate invocation of train_zeroday.py, so the sweep cannot
drift from the experiment the paper describes. Runs already archived are
skipped, which makes the sweep resumable: stop it and start it again and it
picks up where it left off.

Expect this to take a while. One run is a digital twin plus four detectors;
the full sweep is that many times over, so plan on leaving it overnight.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
ARCHIVE = os.path.join(RESULTS, "loao_runs")
ZD_FILE = os.path.join(RESULTS, "zero_day.json")

ATTACKS = ["DoS", "Brake Tampering", "Steering Tampering"]


def run_path(cls: str, seed: int) -> str:
    return os.path.join(ARCHIVE, f"{cls.replace(' ', '_')}__seed{seed}.json")


def one_run(cls: str, seed: int, extra: List[str]) -> Dict:
    """Invoke train_zeroday.py and archive the zero_day.json it writes."""
    dest = run_path(cls, seed)
    if os.path.exists(dest):
        print(f"  [have] {cls}, seed {seed}")
        return json.load(open(dest, encoding="utf-8"))

    cmd = [sys.executable, os.path.join(ROOT, "train_zeroday.py"),
           "--unknown", cls, "--seed", str(seed)] + extra
    print(f"  [run ] {cls}, seed {seed}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"train_zeroday.py failed for {cls}, seed {seed}")
    if not os.path.exists(ZD_FILE):
        raise SystemExit(f"{ZD_FILE} was not written")

    data = json.load(open(ZD_FILE, encoding="utf-8"))
    data["_sweep"] = {"class": cls, "seed": seed,
                      "seconds": round(time.time() - t0, 1)}
    os.makedirs(ARCHIVE, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    print(f"  [done] {cls}, seed {seed} in {data['_sweep']['seconds']:.0f}s")
    return data


def mean_ci(xs: List[float], lo_bound: float = 0.0, hi_bound: float = 100.0):
    """Mean with a 95% interval from Student's t, which is the right
    distribution for the handful of seeds a sweep like this can afford.

    Rates are bounded, so the interval is clipped to the range the quantity
    can actually take; an unclipped t-interval on a near-ceiling detection
    rate otherwise prints a bound above 100%.
    """
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, None, None
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    # two-sided 95% critical values for n-1 degrees of freedom
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return mu, max(lo_bound, mu - half), min(hi_bound, mu + half)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", nargs="*", default=ATTACKS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS, "loao_sweep.json"))
    ap.add_argument("--pass-through", nargs=argparse.REMAINDER, default=[],
                    help="arguments forwarded to train_zeroday.py, after --")
    args = ap.parse_args()

    extra = [a for a in args.pass_through if a != "--"]
    plan = [(c, s) for c in args.classes for s in args.seeds]
    todo = [cs for cs in plan if not os.path.exists(run_path(*cs))]
    print("=" * 74)
    print(f"Leave-one-attack-out sweep: {len(args.classes)} classes x "
          f"{len(args.seeds)} seeds = {len(plan)} runs, {len(todo)} still to do")
    print("=" * 74)
    if args.dry_run:
        for c, s in plan:
            print(f"  {'todo' if (c, s) in todo else 'have'}  {c}, seed {s}")
        return

    # keep the single-run artifacts the app reads, so a sweep does not
    # silently replace the run the Live Monitor is showing
    backup = ZD_FILE + ".before_sweep"
    if os.path.exists(ZD_FILE) and not os.path.exists(backup):
        shutil.copy(ZD_FILE, backup)
        print(f"  saved the existing run to {os.path.basename(backup)}\n")

    runs: List[Dict] = []
    for cls in args.classes:
        for seed in args.seeds:
            runs.append(one_run(cls, seed, extra))

    # ---------------------------------------------------------------- report
    model_names: List[str] = []
    for r in runs:
        for n in r.get("models", {}):
            if n not in model_names:
                model_names.append(n)

    summary: Dict[str, Dict] = {}
    print("\n" + "=" * 74)
    for cls in args.classes:
        sel = [r for r in runs if r["_sweep"]["class"] == cls]
        if not sel:
            continue
        print(f"\n  '{cls}' quarantined -- {len(sel)} seeds")
        print("    {:<26}{:>26}{:>22}".format(
            "Model", "Detected (mean, 95% CI)", "False alarms"))
        summary[cls] = {"n_seeds": len(sel), "models": {}}
        for name in model_names:
            det = [r["models"][name]["unknown_detection_rate"] * 100
                   for r in sel if name in r.get("models", {})]
            fpr = [r["models"][name]["false_positive_rate"] * 100
                   for r in sel if name in r.get("models", {})]
            if not det:
                continue
            dm, dlo, dhi = mean_ci(det)
            fm, flo, fhi = mean_ci(fpr)
            ci = f"[{dlo:.2f}, {dhi:.2f}]" if dlo is not None else "--"
            print("    {:<26}{:>26}{:>22}".format(
                name, f"{dm:6.2f}%  {ci}", f"{fm:.2f}%"))
            summary[cls]["models"][name] = {
                "detection_mean": dm, "detection_ci95": [dlo, dhi],
                "detection_runs": det,
                "fpr_mean": fm, "fpr_ci95": [flo, fhi], "fpr_runs": fpr,
            }

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"classes": args.classes, "seeds": args.seeds,
                   "summary": summary}, fh, indent=2)
    print("\n  wrote", args.out)
    print("  per-run archives in", ARCHIVE)
    if os.path.exists(backup):
        print(f"  the pre-sweep run is in {os.path.basename(backup)}; copy it back "
              f"over zero_day.json to restore what the app was showing.")


if __name__ == "__main__":
    main()
