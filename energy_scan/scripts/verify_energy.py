#!/usr/bin/env python3
"""Strict checks for the per-frame energy files. Usage: verify_energy.py DIR"""
import sys, glob, os, numpy as np

def check(path):
    sid = os.path.basename(path).replace(".energy.csv", "")
    txt = open(path).read().splitlines()
    if not txt: return sid, ["empty"], {}
    has_header = txt[0].strip() == "step,pe_eV,press_bar"
    body = txt[1:] if has_header else txt
    try:
        d = np.array([[float(x) for x in l.split(",")] for l in body])
    except Exception as e:
        return sid, [f"parse: {e}"], {}
    errs = []
    if not has_header: errs.append("no header")
    if d.shape[1] != 3: errs.append(f"columns={d.shape[1]}")
    if len(d) != 2001: errs.append(f"rows={len(d)}")
    steps = d[:, 0]
    if not np.allclose(steps, np.arange(2001) * 1000): errs.append("step grid mismatch")
    if len(np.unique(steps)) != len(steps): errs.append("duplicate steps")
    if not np.all(np.isfinite(d)): errs.append("non-finite")
    # precision: smallest gap between distinct energies
    u = np.unique(d[:, 1]); gaps = np.diff(u); gaps = gaps[gaps > 0]
    gap = float(gaps.min()) if len(gaps) else float("nan")
    if not (gap < 1e-3): errs.append(f"energy quantised: min gap={gap:.3g} eV")
    return sid, errs, {"rows": len(d), "min_gap": gap, "header": has_header}

def main(d):
    files = sorted(glob.glob(os.path.join(d, "*.energy.csv")))
    if not files: print("no files in", d); return 1
    bad = []; gaps = []
    for f in files:
        sid, errs, info = check(f)
        gaps.append(info.get("min_gap", np.nan))
        if errs: bad.append((sid, errs))
    print(f"files={len(files)}  bad={len(bad)}")
    if gaps:
        g = np.array([x for x in gaps if np.isfinite(x)])
        print(f"energy min-gap: median={np.median(g):.3g} eV  max={g.max():.3g} eV")
    for sid, errs in bad[:8]:
        print(f"  FAIL {sid}: {', '.join(errs)}")
    print("PASS" if not bad else "FAIL")
    return 0 if not bad else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
