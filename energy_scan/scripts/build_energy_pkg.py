#!/usr/bin/env python3
"""Combined npz, summary, and 5%-cooling reweighting diagnostics."""
import glob, os, json, csv
import numpy as np

HP="/publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_out_hp"
NP=os.path.join(HP,"npz"); TR="/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/traj"
KB=8.617333262145e-5

def tau_int(x):
    x=np.asarray(x,float); x=x-x.mean(); n=len(x); v=(x*x).mean()
    if v<=0: return 0.5
    t=0.0
    for k in range(1,min(n//2,2000)):
        c=(x[:-k]*x[k:]).mean()/v
        if c<=0: break
        t+=c
    return 0.5+2*t

rows=[]; ids=[]; E=[]; P=[]
for f in sorted(glob.glob(os.path.join(HP,"*.energy.csv"))):
    sid=os.path.basename(f).replace(".energy.csv","")
    d=np.loadtxt(f,delimiter=",",skiprows=1); pe=d[:,1]; pr=d[:,2]
    m=json.load(open(os.path.join(TR,sid,"meta.json")))
    T=m["temperature_K"]; rho=m["density_mol_A3"]
    b=np.array_split(pe,5)
    beta0=1/(KB*T); beta=1/(KB*0.95*T); db=beta0-beta
    u=pe-pe.min(); w=np.exp(db*u); p=w/w.sum()
    ess=1.0/np.sum(p*p)
    rows.append(dict(sim_id=sid,species=m["species"],temperature_K=T,
        density_mol_A3=round(rho,6),frames=len(pe),
        PE_mean_eV=round(float(pe.mean()),6),PE_std_eV=round(float(pe.std()),6),
        PE_drift_eV=round(float(b[-1].mean()-b[0].mean()),6),
        P_virial_mean_bar=round(float(pr.mean()),2),
        dBeta_per_eV=round(db,4),
        weight_error_from_0p1eV=round(db*0.05,4),
        ESS_5pct_cooling=int(round(ess)),ESS_frac=round(float(ess/len(pe)),4),
        max_weight_share=round(float(p.max()),5),
        tau_int_frames=round(float(tau_int(w)),2)))
    ids.append(sid); E.append(pe); P.append(pr)
ids=np.array(ids); E=np.array(E); P=np.array(P)
np.savez_compressed(os.path.join(NP,"all_states.npz"), sim_ids=ids, energy=E, press=P,
    step=np.arange(2001,dtype=np.int64)*1000,
    species=np.array([r["species"] for r in rows]),
    temperature_K=np.array([r["temperature_K"] for r in rows]),
    density_mol_A3=np.array([r["density_mol_A3"] for r in rows]))
with open(os.path.join(HP,"summary.csv"),"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote all_states.npz", E.shape, "and summary.csv", len(rows),"rows")
print(f"\n{'species':8s}{'n':>3s}{'PE_std':>9s}{'drift':>9s}{'ESS':>7s}{'ESS/N':>8s}{'maxw':>8s}{'tau':>7s}")
import collections
agg=collections.defaultdict(list)
for r in rows: agg[r["species"]].append(r)
for sp in ["H2O","CO2","NH3","CH4","C2H4","C2H6"]:
    v=agg[sp]
    print(f"{sp:8s}{len(v):3d}{np.mean([x['PE_std_eV'] for x in v]):9.2f}"
          f"{max(abs(x['PE_drift_eV']) for x in v):9.1f}"
          f"{int(np.mean([x['ESS_5pct_cooling'] for x in v])):7d}"
          f"{np.mean([x['ESS_frac'] for x in v]):8.3f}"
          f"{max(x['max_weight_share'] for x in v):8.3f}"
          f"{np.mean([x['tau_int_frames'] for x in v]):7.1f}")
