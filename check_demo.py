"""
Pre-demo readiness check.

Run this before presenting.  It verifies that every artifact the app needs is
present and consistent, and performs one real inference end to end, so a missing
file is discovered now rather than in front of the judges.

    python check_demo.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
os.environ.setdefault("STREAMLIT_GLOBAL_SUPPRESS_DEPRECATION_WARNINGS", "true")

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVED = os.path.join(ROOT, "saved_models")
RESULTS = os.path.join(ROOT, "results")

OK, WARN, BAD = "  [ ok ]", "  [warn]", "  [FAIL]"
problems: list = []
warnings_: list = []


def check(label: str, cond: bool, detail: str = "", fatal: bool = True) -> bool:
    if cond:
        print(f"{OK} {label} {detail}")
    elif fatal:
        print(f"{BAD} {label} {detail}")
        problems.append(label)
    else:
        print(f"{WARN} {label} {detail}")
        warnings_.append(label)
    return cond


def main() -> int:
    print("=" * 70)
    print("TwinShield-Fusion :: demo readiness check")
    print("=" * 70)

    # ---- dataset -----------------------------------------------------------
    print("\nDataset")
    from preprocessing.loader import dataset_available, dataset_dir
    if not check("GEM-CAN CSVs present", dataset_available(), f"({dataset_dir()})"):
        print("\nDownload GEM_CAN_Dataset.zip from https://zenodo.org/records/17834776")
        return 1

    from preprocessing.prepare import prepare, split_report
    t0 = time.time()
    p = prepare()
    check("dataset prepared", True, f"{len(p.df):,} frames in {time.time()-t0:.1f}s")

    rep = split_report(p)
    for _, r in rep.iterrows():
        empty = [c for c in ("Normal", "DoS", "Brake Tampering", "Steering Tampering")
                 if r[c] == 0]
        check(f"split '{r['Split']}' has all classes", not empty,
              f"({int(r['Frames']):,} frames)" + (f" missing {empty}" if empty else ""))

    # ---- artifacts ---------------------------------------------------------
    print("\nTrained artifacts")
    need = {
        "digital_twin.pt": "GRU Digital Twin",
        "residuals.npy": "per-frame residuals",
        "expected_payload.npy": "expected payloads",
        "with_id__twinshield__fusion.pt": "TwinShield detector",
    }
    for f, desc in need.items():
        check(f"{desc}", os.path.exists(os.path.join(SAVED, f)), f"({f})")

    rj = os.path.join(RESULTS, "results.json")
    if check("results.json", os.path.exists(rj)):
        with open(rj) as fh:
            res = json.load(fh)
        for v, models in res["results"].items():
            check(f"variant '{v}' metrics", len(models) > 0, f"({len(models)} models)")
        check("Digital Twin metrics", bool(res.get("digital_twin")),
              f"MAE {res.get('digital_twin', {}).get('MAE', float('nan')):.2f} bytes")

    # residual/frame count consistency
    rp = os.path.join(SAVED, "residuals.npy")
    if os.path.exists(rp):
        r = np.load(rp, mmap_mode="r")
        check("residuals match frame count", r.shape[0] == len(p.df),
              f"({r.shape[0]:,} vs {len(p.df):,})")

    check("SHAP values", os.path.exists(os.path.join(RESULTS, "shap.npz")),
          "(run python compute_shap.py)", fatal=False)
    check("ECU stream cache", os.path.exists(os.path.join(SAVED, "ecu_streams.npz")),
          "(built on first app load if absent)", fatal=False)

    # ---- zero-day artifacts ------------------------------------------------
    print("\nUnknown-attack experiment (Live Monitor)")
    import ui.data as D
    zd_ok = check("zero-day artifacts", D.zeroday_ready(),
                  "(twin, gate and four detectors, all refitted without one class)")
    if zd_ok:
        zd = D.get_zero_day.__wrapped__()
        u = zd["unknown_class"]
        check("quarantined class present in test split",
              zd["test_class_support"].get(u, 0) > 0,
              f"('{u}': {zd['test_class_support'].get(u, 0):,} test frames)")
        check("quarantined class absent from the label space",
              u not in zd["known_classes"],
              f"(models emit {len(zd['known_classes'])} classes: "
              f"{', '.join(zd['known_classes'])})")
        for m in zd["model_order"]:
            r = zd["models"][m]
            print(f"         - {m:<20} {r['unknown_detection_rate']*100:6.2f}% of '{u}' "
                  f"flagged  ·  {r['false_positive_rate']*100:5.2f}% false alarms")
        prop = zd["models"].get("TwinShield-Fusion", {}).get("unknown_detection_rate", 0)
        base = max((zd["models"][m]["unknown_detection_rate"]
                    for m in zd["model_order"] if m != "TwinShield-Fusion"),
                   default=0.0)
        # Not fatal: a negative result is still a valid result, and the app
        # reports it as one rather than pretending the demo is broken.
        check("proposed model beats every baseline on the unknown attack",
              prop > base, f"({prop*100:.2f}% vs {base*100:.2f}%)", fatal=False)

    # ---- live inference ----------------------------------------------------
    if not problems:
        print("\nLive inference")
        try:
            cases = D.get_demo_cases.__wrapped__()
            check("replay cases", len(cases) >= 1, f"({len(cases)} available)")
            for name, sc in cases.items():
                print(f"         - {name}: {len(sc['frames'])} frames, "
                      f"{sc['n_target']} {sc['target']} frames, "
                      f"{sc['test_share']}% held-out")
            if not cases:
                raise RuntimeError("no replay case could be built")

            first = next(iter(cases.values()))
            t0 = time.time()
            out = D.infer_all(first["frames"][:24])
            dt = (time.time() - t0) * 1000
            n_models = len(D.ZD_ARCHS)
            check("four real forward passes + gate",
                  all(out[m]["proba"].shape[0] == 24 for m in D.ZD_ARCHS),
                  f"({dt:.0f} ms for 24 frames x {n_models} models, "
                  f"{dt/24:.1f} ms/frame)")
            check("probabilities normalised",
                  all(bool(np.allclose(out[m]["proba"].sum(axis=1), 1.0, atol=1e-4))
                      for m in D.ZD_ARCHS))
            check("attention normalised over present streams",
                  bool(np.allclose(out["TwinShield-Fusion"]["attention"].sum(axis=1),
                                   1.0, atol=1e-3)))
            check("anomaly gate scored every frame",
                  out["gate"]["score"].shape[0] == 24,
                  f"(threshold z = {out['gate']['threshold']:.2f})")
            check("single-model forward pass (closed-set detector)",
                  D.infer(first["frames"][:8])[0].shape[0] == 8)
        except Exception as exc:                      # noqa: BLE001
            check("live inference", False, f"({type(exc).__name__}: {exc})")

    # ---- verdict -----------------------------------------------------------
    print("\n" + "=" * 70)
    if problems:
        print(f"NOT READY -- {len(problems)} problem(s): {', '.join(problems)}")
        print("Run:  python train.py   then   python train_zeroday.py   then   python compute_shap.py")
        return 1
    if warnings_:
        print(f"READY (with {len(warnings_)} warning(s): {', '.join(warnings_)})")
    else:
        print("READY -- all checks passed.  Start with:  streamlit run app.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    raise SystemExit(main())
