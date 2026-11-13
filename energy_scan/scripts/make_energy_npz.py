#!/usr/bin/env python3
"""Convert per-frame energy CSVs to float64 npz (per state) with verification."""
import glob, os, json, sys
import numpy as np

HP = "/publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_out_hp"
TR = "/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/traj"
OUT = os.path.join(HP, "npz")
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(HP, "*.energy.csv")))
print(f"csv files: {len(files)}")
worst = 0.0; n = 0
for f in files:
    sid = os.path.basename(f).replace(".energy.csv", "")
    d = np.loadtxt(f, delimiter=",", skiprows=1)
    step = d[:, 0].astype(np.int64); pe = d[:, 1].astype(np.float64); press = d[:, 2].astype(np.float64)
    meta = {}
    mp = os.path.join(TR, sid, "meta.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        meta = {k: m.get(k) for k in ("species", "temperature_K", "density_mol_A3",
                                       "model_sha256", "cubin_sha256", "steps", "dump_every")}
    np.savez_compressed(os.path.join(OUT, sid + ".npz"), step=step, pe=pe, press=press,
                        meta=json.dumps(meta))
    # verify round trip
    z = np.load(os.path.join(OUT, sid + ".npz"))
    dmax = max(np.abs(z["step"] - step).max() if len(step) else 0,
               np.abs(z["pe"] - pe).max(), np.abs(z["press"] - press).max())
    worst = max(worst, float(dmax)); n += 1
print(f"written {n} npz -> {OUT}")
print(f"max |npz - csv| = {worst:.3e}  ({'PASS' if worst < 1e-9 else 'FAIL'})")
